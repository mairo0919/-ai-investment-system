"""Read-only filesystem access for Paper Volume state (Dashboard Phase D1).

Never creates directories, never writes files, never mutates Paper state.
Does not instantiate PaperStore / PaperObservationStore (those mkdir on init).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import PROJECT_ROOT
from src.paper.observation_store import RANKING_SNAPSHOT_COLUMNS
from src.simulation.config import load_simulation_config

logger = logging.getLogger(__name__)

# Default config chain used by Phase 5 paper (extends → initial_capital SSOT).
_DEFAULT_PAPER_CONFIG = PROJECT_ROOT / "config" / "paper_trading.json"


def resolve_initial_capital(*, config_path: Path | None = None) -> float:
    """Load initial_capital from simulation config chain (no dashboard hardcode)."""
    path = config_path if config_path is not None else _DEFAULT_PAPER_CONFIG
    cfg = load_simulation_config(path)
    return float(cfg.initial_capital)


def _safe_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Dashboard: failed to read JSON %s: %s", path, exc)
        return None


def _safe_csv(path: Path) -> pd.DataFrame | None:
    """Return DataFrame, empty DataFrame for empty/header-only CSV, or None if missing/corrupt."""
    if not path.exists() or not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        logger.warning("Dashboard: failed to read CSV %s: %s", path, exc)
        return None
    return df


class DashboardDataSource:
    """Experiment-scoped read-only view over one PAPER_STATE_DIR tree."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        experiment_id: str | None = None,
        initial_capital: float | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.experiment_id = experiment_id
        cfg_path = Path(config_path) if config_path is not None else None
        if initial_capital is not None:
            self.initial_capital = float(initial_capital)
        else:
            self.initial_capital = resolve_initial_capital(config_path=cfg_path)

    # --- path helpers (no mkdir) ---

    @property
    def portfolio_path(self) -> Path:
        return self.state_dir / "portfolio.json"

    @property
    def positions_path(self) -> Path:
        return self.state_dir / "positions.json"

    @property
    def pending_path(self) -> Path:
        return self.state_dir / "pending_orders.json"

    @property
    def trade_history_path(self) -> Path:
        return self.state_dir / "trade_history.csv"

    @property
    def equity_history_path(self) -> Path:
        return self.state_dir / "equity_history.csv"

    @property
    def validity_latest_path(self) -> Path:
        return self.state_dir / "validity_gate_latest.json"

    @property
    def run_health_path(self) -> Path:
        return self.state_dir / "run_health.json"

    @property
    def rankings_dir(self) -> Path:
        return self.state_dir / "rankings"

    def ranking_path(self, day: pd.Timestamp | str) -> Path:
        d = pd.Timestamp(day).normalize().date().isoformat()
        return self.rankings_dir / f"{d}.csv"

    # --- loaders ---

    def load_portfolio(self) -> dict[str, Any] | None:
        raw = _safe_json(self.portfolio_path)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            logger.warning("Dashboard: portfolio.json is not an object: %s", self.portfolio_path)
            return None
        return raw

    def load_positions(self) -> list[dict[str, Any]] | None:
        raw = _safe_json(self.positions_path)
        if raw is None:
            return None
        if not isinstance(raw, list):
            logger.warning("Dashboard: positions.json is not a list: %s", self.positions_path)
            return None
        out: list[dict[str, Any]] = []
        for row in raw:
            if isinstance(row, dict):
                out.append(row)
        return out

    def load_pending_orders(self) -> list[dict[str, Any]] | None:
        raw = _safe_json(self.pending_path)
        if raw is None:
            return None
        if not isinstance(raw, list):
            logger.warning(
                "Dashboard: pending_orders.json is not a list: %s", self.pending_path
            )
            return None
        return [row for row in raw if isinstance(row, dict)]

    def load_trade_history(self) -> pd.DataFrame | None:
        return _safe_csv(self.trade_history_path)

    def load_equity_history(self) -> pd.DataFrame | None:
        return _safe_csv(self.equity_history_path)

    def load_latest_validity(self) -> dict[str, Any] | None:
        raw = _safe_json(self.validity_latest_path)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            logger.warning(
                "Dashboard: validity_gate_latest.json is not an object: %s",
                self.validity_latest_path,
            )
            return None
        return raw

    def load_run_health(self) -> dict[str, Any] | None:
        raw = _safe_json(self.run_health_path)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            logger.warning("Dashboard: run_health.json is not an object: %s", self.run_health_path)
            return None
        return raw

    def load_ranking_dates(self) -> list[str]:
        """ISO dates with ranking snapshot files, ascending."""
        return list_ranking_dates(self.state_dir)

    def load_rankings(self, date: pd.Timestamp | str) -> pd.DataFrame | None:
        return load_rankings(self.state_dir, date)


def list_ranking_dates(state_dir: Path | str) -> list[str]:
    root = Path(state_dir) / "rankings"
    if not root.is_dir():
        return []
    dates: list[str] = []
    for p in root.iterdir():
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        stem = p.stem
        try:
            dates.append(pd.Timestamp(stem).normalize().date().isoformat())
        except (ValueError, TypeError):
            logger.warning("Dashboard: skip non-date ranking file %s", p.name)
    return sorted(set(dates))


def load_rankings(state_dir: Path | str, date: pd.Timestamp | str) -> pd.DataFrame | None:
    d = pd.Timestamp(date).normalize().date().isoformat()
    path = Path(state_dir) / "rankings" / f"{d}.csv"
    df = _safe_csv(path)
    if df is None:
        return None
    # Align to known columns when present; do not invent values.
    cols = [c for c in RANKING_SNAPSHOT_COLUMNS if c in df.columns]
    if not cols:
        return df
    return df.loc[:, cols].copy()
