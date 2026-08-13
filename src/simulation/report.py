"""Persist simulation artifacts under reports/simulation/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.metrics import compute_performance
from src.simulation.orders import EquityPoint, TradeRecord


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if np.isnan(x) or np.isinf(x) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def equity_to_frame(equity_curve: list[EquityPoint]) -> pd.DataFrame:
    if not equity_curve:
        return pd.DataFrame(
            columns=["Date", "Cash", "Position Value", "Total Equity", "Drawdown"]
        )
    return pd.DataFrame([e.to_dict() for e in equity_curve])


def trades_to_frame(trades: list[TradeRecord]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Country",
                "Entry Date",
                "Entry Price",
                "Exit Date",
                "Exit Price",
                "Quantity",
                "Gross PnL",
                "Net PnL",
                "Return",
                "Holding Days",
                "Exit Reason",
                "Entry Score",
                "Entry Rank",
            ]
        )
    return pd.DataFrame([t.to_dict() for t in trades])


def write_run_artifacts(
    out_dir: Path,
    *,
    equity_curve: list[EquityPoint],
    trades: list[TradeRecord],
    summary: dict[str, Any],
    prefix: str = "",
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eq = equity_to_frame(equity_curve)
    tr = trades_to_frame(trades)
    eq_path = out_dir / f"{prefix}equity_curve.csv"
    tr_path = out_dir / f"{prefix}trades.csv"
    sum_path = out_dir / f"{prefix}summary.json"
    eq.to_csv(eq_path, index=False)
    tr.to_csv(tr_path, index=False)
    write_json(sum_path, summary)
    tr_json = out_dir / f"{prefix}trades.json"
    write_json(tr_json, {"trades": tr.to_dict(orient="records")})
    return {
        "equity_curve": str(eq_path),
        "trades_csv": str(tr_path),
        "trades_json": str(tr_json),
        "summary": str(sum_path),
    }


def summarize_result(
    *,
    equity_curve: list[EquityPoint],
    trades: list[TradeRecord],
    initial_capital: float,
    transaction_costs: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    perf = compute_performance(
        equity_curve,
        trades,
        initial_capital=initial_capital,
        transaction_costs_total=transaction_costs,
    )
    out = {"performance": perf}
    if extra:
        out.update(extra)
    return out
