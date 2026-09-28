"""Small Capital Paper entrypoint — isolated state, ¥10k capital, integer execution.

Does not modify ``src/paper/runner.py``. Reuses PaperTradingRunner after applying
capital / execution overlays and refusing the canonical 10M ``data/paper`` directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import PROJECT_ROOT, Settings, get_settings
from src.core.exceptions import TrainingError
from src.paper.execution_audit import ExecutionAuditStore
from src.paper.lineage import LOCKED_PAPER_MODEL_ID, STRATEGY_CONFIG_ID
from src.paper.observation_store import STATUS_SUCCESS
from src.paper.runner import PaperTradingRunner
from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine, SimulationResult
from src.utils.logging import setup_logging
from src.utils.runtime_diag import (
    current_phase,
    install_atexit_hook,
    install_signal_handlers,
    log_diag,
    log_process_boot,
    log_top_level_exception,
    mark_exit_status,
    phase_span,
    set_phase,
)

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "small_capital_10k"
DEFAULT_STATE_DIR = PROJECT_ROOT / "data" / "paper_experiments" / "small_capital_10k"
CANONICAL_10M_STATE_DIR = (PROJECT_ROOT / "data" / "paper").resolve()
DEFAULT_CONFIG = Path("config/paper_trading_small_capital_10k.json")


def parse_max_sessions(value: str) -> int:
    """CLI type: positive integer. Omitted flag stays ``None`` (full catch-up)."""
    text = str(value).strip()
    try:
        if text.startswith("+"):
            raise ValueError
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"--max-sessions must be a positive integer, got {value!r}"
        ) from exc
    if number < 1:
        raise argparse.ArgumentTypeError(
            f"--max-sessions must be a positive integer, got {value!r}"
        )
    return number


def unprocessed_market_sessions(
    market_dates: Any,
    *,
    forward_start: pd.Timestamp,
    last_processed: str | pd.Timestamp | None,
    asof: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Market sessions the paper runner would still process (inclusive as-of).

    Matches ``PaperTradingRunner`` resume: empty state starts at forward start;
    otherwise the next business day after ``last_processed_date``.
    """
    if last_processed:
        sim_start = pd.Timestamp(last_processed).normalize() + pd.tseries.offsets.BDay(1)
        sim_start = pd.Timestamp(sim_start).normalize()
    else:
        sim_start = pd.Timestamp(forward_start).normalize()
    asof_n = pd.Timestamp(asof).normalize()
    days = sorted({pd.Timestamp(d).normalize() for d in market_dates})
    return [d for d in days if sim_start <= d <= asof_n]


def limit_unprocessed_sessions(
    sessions: list[pd.Timestamp],
    max_sessions: int | None,
) -> list[pd.Timestamp]:
    """Keep the next ``max_sessions`` unprocessed sessions. ``None`` keeps all."""
    if max_sessions is None:
        return list(sessions)
    return list(sessions)[: int(max_sessions)]


def clamp_processed_session_days(
    session_days: list[Any],
    *,
    asof: pd.Timestamp,
    max_sessions: int | None,
) -> list[pd.Timestamp]:
    """Drop the next-open buffer day and apply the invocation session cap."""
    asof_n = pd.Timestamp(asof).normalize()
    market = [
        pd.Timestamp(d).normalize()
        for d in session_days
        if pd.Timestamp(d).normalize() <= asof_n
    ]
    return limit_unprocessed_sessions(market, max_sessions)


def assert_small_capital_state_dir(path: Path) -> Path:
    """Refuse writing Small Capital state into the live 10M paper directory."""
    resolved = path.resolve()
    if resolved == CANONICAL_10M_STATE_DIR:
        raise TrainingError(
            "Small Capital Paper must not use PAPER_STATE_DIR=data/paper "
            f"(canonical 10M state). Use {DEFAULT_STATE_DIR.relative_to(PROJECT_ROOT)} "
            "or another isolated path."
        )
    return resolved


def resolve_small_capital_state_dir(
    *,
    explicit: Path | None = None,
    env_value: str | None = None,
) -> Path:
    if explicit is not None:
        return assert_small_capital_state_dir(
            explicit if explicit.is_absolute() else PROJECT_ROOT / explicit
        )
    raw = env_value if env_value is not None else os.getenv("PAPER_STATE_DIR")
    if raw and str(raw).strip():
        p = Path(str(raw).strip())
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return assert_small_capital_state_dir(p)
    return assert_small_capital_state_dir(DEFAULT_STATE_DIR)


def apply_small_capital_overlays(
    raw: dict[str, Any], cfg: SimulationConfig
) -> SimulationConfig:
    """Apply capital / execution / SC portfolio overlays without retuning strategy signals."""
    overlay: dict[str, Any] = {}
    if "initial_capital" in raw:
        overlay["initial_capital"] = float(raw["initial_capital"])
    if isinstance(raw.get("execution_policy"), dict):
        overlay["execution_policy"] = dict(raw["execution_policy"])
    port = raw.get("portfolio") or {}
    if isinstance(port, dict) and "max_position_weight" in port:
        overlay["max_position_weight"] = float(port["max_position_weight"])
    if not overlay:
        return cfg
    return cfg.with_overrides(**overlay)


class SmallCapitalPaperRunner(PaperTradingRunner):
    """PaperTradingRunner with Small Capital capital + integer execution overlays."""

    def __init__(
        self,
        settings: Settings,
        universe_path: Path,
        config_path: Path,
        *,
        max_sessions: int | None = None,
    ) -> None:
        if max_sessions is not None and int(max_sessions) < 1:
            raise TrainingError(
                f"max_sessions must be a positive integer, got {max_sessions!r}"
            )
        self.max_sessions = None if max_sessions is None else int(max_sessions)
        assert_small_capital_state_dir(settings.paper_state_dir)
        super().__init__(settings, universe_path, config_path)
        exp = str(self.raw.get("experiment_id") or "")
        if exp and exp != EXPERIMENT_ID:
            raise TrainingError(
                f"Small Capital runner expects experiment_id={EXPERIMENT_ID}, got {exp!r}"
            )
        self.cfg = apply_small_capital_overlays(self.raw, self.cfg)
        if float(self.cfg.initial_capital) != 10_000.0:
            raise TrainingError(
                f"Small Capital initial_capital must be 10000, got {self.cfg.initial_capital}"
            )
        if not self.cfg.execution_policy.is_integer_affordable:
            raise TrainingError(
                "Small Capital requires execution_policy sizing_mode=integer_affordable "
                f"and share_mode=integer; got {self.cfg.execution_policy}"
            )
        # Isolated report tree so 10M paper reports are untouched.
        self.report_dir = settings.reports_dir / "paper_experiments" / EXPERIMENT_ID
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / "rankings").mkdir(parents=True, exist_ok=True)
        (self.report_dir / "monthly").mkdir(parents=True, exist_ok=True)
        self.execution_audit = ExecutionAuditStore(settings.paper_state_dir)
        self.execution_audit.ensure_file()
        # Momentum benchmark reuses SimulationEngine; do not audit that track.
        self._suppress_execution_audit = False

    def persist_execution_events(
        self,
        result: SimulationResult,
        *,
        model_id: str | None = None,
        config_id: str | None = None,
    ) -> int:
        """Persist engine execution_events into execution_decisions.csv (idempotent)."""
        if self._suppress_execution_audit:
            return 0
        events = list(result.meta.get("execution_events") or [])
        if not events:
            return 0
        mid = model_id or self.raw.get("model_id") or LOCKED_PAPER_MODEL_ID
        cid = config_id or STRATEGY_CONFIG_ID
        enriched: list[dict[str, Any]] = []
        for ev in events:
            row = dict(ev)
            row.setdefault("model_id", mid)
            row.setdefault("config_id", cid)
            row.setdefault("experiment_id", EXPERIMENT_ID)
            enriched.append(row)
        return self.execution_audit.append_events(enriched)

    def _persist_observations(
        self,
        *,
        run_health_ctx: dict[str, Any],
        status: str,
        asof: pd.Timestamp,
        ai_rankings: pd.DataFrame,
        price_panel: pd.DataFrame,
        session_days: list[pd.Timestamp],
        processed_sessions: int,
        skip_reason: str | None,
    ) -> None:
        """Record only sessions this invocation actually processed."""
        if self.max_sessions is not None and status == STATUS_SUCCESS:
            limited = clamp_processed_session_days(
                session_days,
                asof=asof,
                max_sessions=self.max_sessions,
            )
            if limited:
                session_days = limited
                processed_sessions = len(limited)
                asof = limited[-1]
                self._rewrite_last_processed_date(asof)
        super()._persist_observations(
            run_health_ctx=run_health_ctx,
            status=status,
            asof=asof,
            ai_rankings=ai_rankings,
            price_panel=price_panel,
            session_days=session_days,
            processed_sessions=processed_sessions,
            skip_reason=skip_reason,
        )

    def _last_equity_session(self) -> pd.Timestamp | None:
        """Latest equity row, which is the session the engine actually marked."""
        path = self.store.equity_history_path
        if not path.exists():
            return None
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return None
        if frame.empty or "Date" not in frame.columns:
            return None
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        if dates.empty:
            return None
        return pd.Timestamp(dates.max()).normalize()

    def _session_limited_asof(
        self,
        asof: pd.Timestamp,
        *,
        prefer_saved_processed_date: bool = False,
    ) -> pd.Timestamp:
        """Paper as-of for a capped run is the last processed session.

        Unlimited catch-up keeps the caller as-of (legacy market date).
        Benchmark tracks run before ``last_processed_date`` is rewritten, so
        they use the latest equity session. The validity gate runs after that
        rewrite and uses the saved processed date, then ignores later rows.
        """
        requested = pd.Timestamp(asof).normalize()
        if self.max_sessions is None:
            return requested
        if prefer_saved_processed_date:
            state = self.store.load()
            if state is not None and state.last_processed_date:
                saved = pd.Timestamp(state.last_processed_date).normalize()
                if saved < requested:
                    return saved
                return requested
        last = self._last_equity_session()
        if last is not None and last < requested:
            return last
        return requested

    def _update_benchmark_tracks(
        self,
        *,
        us_px: pd.DataFrame,
        mom_rankings: pd.DataFrame,
        fx: Any,
        asof: pd.Timestamp,
        state_model_id: str,
    ) -> None:
        """Benchmark curve is not an AI execution decision.

        The parent runner builds it with another ``SimulationEngine``. While the
        Small Capital engine class is patched for audit persistence, that second
        engine must not append momentum ranks into ``execution_decisions.csv``.
        Session-limited runs also stop the track at the last processed session
        so later market dates do not enter the files the validity gate reads.
        """
        self._suppress_execution_audit = True
        try:
            return super()._update_benchmark_tracks(
                us_px=us_px,
                mom_rankings=mom_rankings,
                fx=fx,
                asof=self._session_limited_asof(asof),
                state_model_id=state_model_id,
            )
        finally:
            self._suppress_execution_audit = False

    def _emit_validity_gate(
        self,
        *,
        asof: pd.Timestamp,
        model_id: str,
        run_health_ctx: dict[str, Any] | None = None,
    ) -> None:
        """Session-limited validity is as of the processed session, not market as-of."""
        gate_asof = self._session_limited_asof(asof, prefer_saved_processed_date=True)
        if self.max_sessions is None:
            return super()._emit_validity_gate(
                asof=asof,
                model_id=model_id,
                run_health_ctx=run_health_ctx,
            )
        parent_fn = super()._emit_validity_gate.__func__
        globs = parent_fn.__globals__
        real_eval = globs["evaluate_validity_gate"]

        def _eval_through_processed(**kwargs: Any) -> dict[str, Any]:
            kwargs["asof"] = gate_asof
            kwargs["history_end"] = gate_asof
            return real_eval(**kwargs)

        globs["evaluate_validity_gate"] = _eval_through_processed
        try:
            return parent_fn(
                self,
                asof=gate_asof,
                model_id=model_id,
                run_health_ctx=run_health_ctx,
            )
        finally:
            globs["evaluate_validity_gate"] = real_eval

    def _rewrite_last_processed_date(self, processed_asof: pd.Timestamp) -> None:
        """Parent runner stamps the market as-of; keep the real processed session."""
        state = self.store.load()
        if state is None:
            return
        state.last_processed_date = str(pd.Timestamp(processed_asof).date())
        self.store.save_atomic(state)

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """Run paper catch-up while persisting SC execution audit without editing runner.py."""
        import src.paper.runner as runner_mod

        outer = self

        class _AuditingEngine(SimulationEngine):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                # None → full catch-up. A positive cap stops after that many
                # sessions while still seeing the next session as next-open.
                self.process_session_limit = outer.max_sessions

            def run(self, *args: Any, **kwargs: Any) -> SimulationResult:  # type: ignore[override]
                result = SimulationEngine.run(self, *args, **kwargs)
                try:
                    outer.persist_execution_events(
                        result,
                        model_id=LOCKED_PAPER_MODEL_ID,
                        config_id=STRATEGY_CONFIG_ID,
                    )
                except Exception:  # noqa: BLE001 — audit must not break trading
                    logger.exception("execution audit persistence failed (non-fatal)")
                return result

        prev = runner_mod.SimulationEngine
        runner_mod.SimulationEngine = _AuditingEngine  # type: ignore[misc]
        try:
            return super().run(force_refresh=force_refresh)
        finally:
            runner_mod.SimulationEngine = prev


def build_small_capital_settings(
    *,
    state_dir: Path | None = None,
    base: Settings | None = None,
) -> Settings:
    settings = base or get_settings()
    resolved = resolve_small_capital_state_dir(explicit=state_dir)
    return replace(settings, paper_state_dir=resolved)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Small Capital forward paper trading (¥10k, isolated state, no brokerage)"
    )
    parser.add_argument("--universe", type=Path, default=Path("config/universe.global100.json"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Override PAPER_STATE_DIR (must not be data/paper)",
    )
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--max-sessions",
        type=parse_max_sessions,
        default=None,
        help="Max unprocessed market sessions for this invocation (default: all)",
    )
    args = parser.parse_args(argv)

    get_settings.cache_clear()
    settings = build_small_capital_settings(state_dir=args.state_dir)
    setup_logging(settings.log_dir, settings.log_level)
    install_signal_handlers()
    install_atexit_hook()
    log_process_boot()
    set_phase("main")
    try:
        with phase_span("small_capital_paper_main"):
            SmallCapitalPaperRunner(
                settings,
                args.universe,
                args.config,
                max_sessions=args.max_sessions,
            ).run(force_refresh=args.force_refresh)
        mark_exit_status("ok")
        log_diag("main_return_ok")
        return 0
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, SystemExit):
            mark_exit_status(f"system_exit:{exc.code}")
            raise
        mark_exit_status(f"error:{type(exc).__name__}")
        log_top_level_exception(exc)
        logger.exception("run-small-capital-paper failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        log_diag(f"main_finally status_pending_atexit phase={current_phase()}")


if __name__ == "__main__":
    raise SystemExit(main())
