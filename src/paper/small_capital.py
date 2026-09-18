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

from src.config.settings import PROJECT_ROOT, Settings, get_settings
from src.core.exceptions import TrainingError
from src.paper.execution_audit import ExecutionAuditStore
from src.paper.lineage import LOCKED_PAPER_MODEL_ID, STRATEGY_CONFIG_ID
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
    ) -> None:
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

    def persist_execution_events(
        self,
        result: SimulationResult,
        *,
        model_id: str | None = None,
        config_id: str | None = None,
    ) -> int:
        """Persist engine execution_events into execution_decisions.csv (idempotent)."""
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

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """Run paper catch-up while persisting SC execution audit without editing runner.py."""
        import src.paper.runner as runner_mod

        outer = self

        class _AuditingEngine(SimulationEngine):
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
            SmallCapitalPaperRunner(settings, args.universe, args.config).run(
                force_refresh=args.force_refresh
            )
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
