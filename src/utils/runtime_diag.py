"""Process / phase diagnostics for crash forensics (no trading logic).

SIGKILL cannot be caught — the kernel terminates the process without delivering
a signal handler. If Railway logs stop mid-phase with no [DIAG] SIGNAL and no
Python traceback, suspect SIGKILL (OOM killer or platform hard-kill) or a
sudden container stop that only sent an unhandled path.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import resource
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_current_phase: str = "startup"
_exit_status: str = "running"
_handlers_installed: bool = False
_atexit_installed: bool = False
_boot_logged: bool = False


def current_phase() -> str:
    return _current_phase


def set_phase(phase: str) -> None:
    global _current_phase
    _current_phase = str(phase)


def mark_exit_status(status: str) -> None:
    global _exit_status
    _exit_status = str(status)


def is_railway_env() -> bool:
    return bool(
        os.environ.get("RAILWAY_ENVIRONMENT")
        or os.environ.get("RAILWAY_PROJECT_ID")
        or os.environ.get("RAILWAY_SERVICE_ID")
        or os.environ.get("RAILWAY_STATIC_URL")
    )


def rss_mb() -> float | None:
    """Best-effort current RSS in MiB."""
    # Linux: VmRSS in kB
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except OSError:
        pass
    # Fallback: ru_maxrss (peak). Linux=kB, macOS=bytes.
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(ru) / (1024.0 * 1024.0)
        return float(ru) / 1024.0
    except Exception:  # noqa: BLE001
        return None


def available_memory_mb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def resource_usage() -> dict[str, Any]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    maxrss = float(ru.ru_maxrss)
    if sys.platform == "darwin":
        maxrss_mb = maxrss / (1024.0 * 1024.0)
        maxrss_unit = "bytes_native"
    else:
        maxrss_mb = maxrss / 1024.0
        maxrss_unit = "kb_native"
    return {
        "ru_maxrss_native": maxrss,
        "ru_maxrss_unit": maxrss_unit,
        "ru_maxrss_mb": maxrss_mb,
        "ru_utime_sec": float(ru.ru_utime),
        "ru_stime_sec": float(ru.ru_stime),
    }


def _emit(message: str) -> None:
    """Write DIAG lines to stdout (Railway) and logger; flush immediately."""
    line = f"[DIAG] {message}"
    print(line, flush=True)
    logger.info(line)


def log_diag(message: str) -> None:
    rss = rss_mb()
    rss_s = f"{rss:.1f}" if rss is not None else "n/a"
    _emit(f"{message} phase={_current_phase} rss_mb={rss_s} pid={os.getpid()}")


def log_process_boot() -> None:
    global _boot_logged
    if _boot_logged:
        return
    _boot_logged = True
    set_phase("boot")
    avail = available_memory_mb()
    ru = resource_usage()
    _emit(
        "PROCESS START "
        f"pid={os.getpid()} "
        f"python={sys.version.split()[0]} "
        f"platform={platform.platform()} "
        f"cwd={os.getcwd()} "
        f"railway={is_railway_env()} "
        f"cpu_count={os.cpu_count()} "
        f"rss_mb={rss_mb() if rss_mb() is not None else 'n/a'} "
        f"available_memory_mb={avail if avail is not None else 'n/a'} "
        f"ru_maxrss_mb={ru['ru_maxrss_mb']:.1f} "
        f"ru_utime_sec={ru['ru_utime_sec']:.3f} "
        f"ru_stime_sec={ru['ru_stime_sec']:.3f}"
    )
    _emit(
        "NOTE SIGKILL cannot be caught by handlers; "
        "mid-phase silence without SIGNAL/traceback may indicate SIGKILL or hard stop"
    )


@contextmanager
def phase_span(phase: str) -> Iterator[None]:
    """Log START/END around a phase; updates current_phase."""
    set_phase(phase)
    t0 = time.perf_counter()
    rss0 = rss_mb()
    rss0_s = f"{rss0:.1f}" if rss0 is not None else "n/a"
    _emit(f"START {phase} rss_mb={rss0_s}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        rss1 = rss_mb()
        rss1_s = f"{rss1:.1f}" if rss1 is not None else "n/a"
        _emit(f"END {phase} rss_mb={rss1_s} elapsed_sec={elapsed:.3f}")


def _signal_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    try:
        name = signal.Signals(signum).name
    except Exception:  # noqa: BLE001
        name = str(signum)
    ru = resource_usage()
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    _emit(
        f"SIGNAL name={name} signum={signum} pid={os.getpid()} "
        f"phase={_current_phase} rss_mb={rss_mb() if rss_mb() is not None else 'n/a'} "
        f"ru_maxrss_mb={ru['ru_maxrss_mb']:.1f} timestamp_utc={ts}"
    )
    # Re-raise default termination for TERM/INT after logging.
    if signum in (getattr(signal, "SIGTERM", 0), getattr(signal, "SIGINT", 0)):
        mark_exit_status(f"signal:{name}")
        raise SystemExit(128 + int(signum))


def install_signal_handlers() -> None:
    """Register handlers for catchable termination signals.

    SIGKILL is not catchable (not registered). SIGSTOP is also not catchable.
    """
    global _handlers_installed
    if _handlers_installed:
        return
    for sig_name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _signal_handler)
        except Exception as exc:  # noqa: BLE001
            _emit(f"signal_handler_skip name={sig_name} reason={exc}")
    _handlers_installed = True
    _emit(
        "signal_handlers_installed "
        "catchable=SIGTERM,SIGINT,SIGHUP,SIGQUIT "
        "not_catchable=SIGKILL"
    )


def _atexit_hook() -> None:
    ru = resource_usage()
    _emit(
        f"PROCESS EXIT status={_exit_status} phase={_current_phase} "
        f"pid={os.getpid()} rss_mb={rss_mb() if rss_mb() is not None else 'n/a'} "
        f"ru_maxrss_mb={ru['ru_maxrss_mb']:.1f} "
        f"ru_utime_sec={ru['ru_utime_sec']:.3f} "
        f"ru_stime_sec={ru['ru_stime_sec']:.3f}"
    )


def install_atexit_hook() -> None:
    global _atexit_installed
    if _atexit_installed:
        return
    atexit.register(_atexit_hook)
    _atexit_installed = True


def log_top_level_exception(exc: BaseException) -> None:
    ru = resource_usage()
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _emit(
        f"EXCEPTION type={type(exc).__name__} message={exc!s} "
        f"phase={_current_phase} rss_mb={rss_mb() if rss_mb() is not None else 'n/a'} "
        f"ru_maxrss_mb={ru['ru_maxrss_mb']:.1f}"
    )
    print(tb, flush=True)
    logger.error("Top-level exception in phase=%s\n%s", _current_phase, tb)
