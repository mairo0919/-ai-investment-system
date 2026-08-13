"""Crash-forensics diagnostics for paper trading (no strategy changes)."""

from __future__ import annotations

import signal
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.utils import runtime_diag as rd


@pytest.fixture(autouse=True)
def _reset_diag_state() -> None:
    rd._current_phase = "startup"
    rd._exit_status = "running"
    rd._handlers_installed = False
    rd._atexit_installed = False
    rd._boot_logged = False
    yield
    rd._handlers_installed = False
    rd._atexit_installed = False


def test_signal_handlers_registered() -> None:
    registered: dict[int, object] = {}

    def _fake_signal(sig, handler):  # noqa: ANN001
        registered[int(sig)] = handler
        return signal.SIG_DFL

    with patch.object(rd.signal, "signal", side_effect=_fake_signal):
        rd.install_signal_handlers()
    for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            assert int(sig) in registered
            assert registered[int(sig)] is rd._signal_handler


def test_current_phase_updates_in_span(capsys: pytest.CaptureFixture[str]) -> None:
    assert rd.current_phase() == "startup"
    with rd.phase_span("panel_build"):
        assert rd.current_phase() == "panel_build"
    out = capsys.readouterr().out
    assert "[DIAG] START panel_build" in out
    assert "[DIAG] END panel_build" in out
    assert "elapsed_sec=" in out


def test_top_level_exception_logs_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    rd.set_phase("model_inference")
    try:
        raise ValueError("diag-boom")
    except ValueError as exc:
        rd.log_top_level_exception(exc)
    out = capsys.readouterr().out
    assert "EXCEPTION type=ValueError" in out
    assert "diag-boom" in out
    assert "phase=model_inference" in out
    assert "ValueError: diag-boom" in out


def test_import_paper_still_does_not_start_trading() -> None:
    for key in list(sys.modules):
        if key == "src.paper" or key.startswith("src.paper."):
            del sys.modules[key]
    import src.paper as paper

    assert "src.paper.runner" not in sys.modules
    assert paper.__all__ == ["PaperTradingRunner"]


def test_main_still_runs_once_with_diag() -> None:
    from src.paper import runner as runner_mod

    fake_runner = MagicMock()
    fake_runner.run.return_value = {"ok": True}
    with (
        patch.object(runner_mod, "PaperTradingRunner", return_value=fake_runner),
        patch.object(runner_mod, "get_settings"),
        patch.object(runner_mod, "setup_logging"),
        patch.object(runner_mod, "install_signal_handlers"),
        patch.object(runner_mod, "install_atexit_hook"),
        patch.object(runner_mod, "log_process_boot"),
    ):
        rc = runner_mod.main([])
    assert rc == 0
    assert fake_runner.run.call_count == 1


def test_main_exception_returns_nonzero_and_logs(capsys: pytest.CaptureFixture[str]) -> None:
    from src.paper import runner as runner_mod

    with (
        patch.object(runner_mod, "PaperTradingRunner", side_effect=RuntimeError("fail-run")),
        patch.object(runner_mod, "get_settings"),
        patch.object(runner_mod, "setup_logging"),
        patch.object(runner_mod, "install_signal_handlers"),
        patch.object(runner_mod, "install_atexit_hook"),
        patch.object(runner_mod, "log_process_boot"),
    ):
        rc = runner_mod.main([])
    assert rc == 1
    out = capsys.readouterr().out
    assert "EXCEPTION type=RuntimeError" in out or "fail-run" in out


def test_process_boot_emits_pid(capsys: pytest.CaptureFixture[str]) -> None:
    rd.log_process_boot()
    out = capsys.readouterr().out
    assert "[DIAG] PROCESS START" in out
    assert "pid=" in out
    assert "SIGKILL cannot be caught" in out
