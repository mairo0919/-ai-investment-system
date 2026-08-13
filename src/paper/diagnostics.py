"""Forward paper diagnostics: score inversion, MAE/MFE, risk (monitor only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.simulation.holdout_diagnostics import score_tertile_diagnostics, sortino_ratio
from src.simulation.position import Position


def spearman_score_return(trades: pd.DataFrame) -> float | None:
    if trades is None or trades.empty:
        return None
    need = {"Entry Score", "Return"}
    if not need.issubset(set(trades.columns)):
        # tolerate alternate names
        cols = {c.lower(): c for c in trades.columns}
        score_col = cols.get("entry score") or cols.get("entry_score")
        ret_col = cols.get("return") or cols.get("return_pct")
        if not score_col or not ret_col:
            return None
        s = trades[score_col].astype(float)
        r = trades[ret_col].astype(float)
    else:
        s = trades["Entry Score"].astype(float)
        r = trades["Return"].astype(float)
    mask = s.notna() & r.notna()
    if mask.sum() < 3:
        return None
    return float(s[mask].corr(r[mask], method="spearman"))


def score_diagnostics_from_trades(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "buckets": {},
            "spearman_entry_score_vs_return": None,
            "note": "insufficient trades",
            "statistically_insufficient": True,
        }
    # Normalize column names for tertile helper via TradeRecord-like frame
    tr = trades.copy()
    rename = {}
    for a, b in [
        ("entry_score", "Entry Score"),
        ("EntryScore", "Entry Score"),
        ("net_pnl", "Net PnL"),
        ("NetPnL", "Net PnL"),
        ("return", "Return"),
        ("ReturnPct", "Return"),
    ]:
        if a in tr.columns and b not in tr.columns:
            rename[a] = b
    tr = tr.rename(columns=rename)
    if "Net PnL" not in tr.columns and "Return" in tr.columns:
        tr["Net PnL"] = tr["Return"].astype(float)
    buckets = score_tertile_diagnostics(tr)
    corr = spearman_score_return(tr)
    return {
        **buckets,
        "spearman_entry_score_vs_return": corr,
        "n_trades": int(len(tr)),
        "statistically_insufficient": len(tr) < 30,
        "policy": "Diagnostics only — do not alter FINAL strategy from these results.",
    }


def position_mae_mfe(
    pos: Position,
    *,
    path: pd.DataFrame,
) -> dict[str, Any]:
    """MAE/MFE for an open position using High/Low since entry."""
    if path.empty or pos.entry_price <= 0:
        return {"mae": None, "mfe": None, "alerts": []}
    lows = path["Low"].astype(float)
    highs = path["High"].astype(float)
    mae = float(lows.min() / pos.entry_price - 1.0)
    mfe = float(highs.max() / pos.entry_price - 1.0)
    alerts = []
    for thr in (-0.10, -0.15, -0.20, -0.25):
        if mae <= thr:
            alerts.append(f"MAE_LE_{int(abs(thr)*100)}PCT")
    return {"mae": mae, "mfe": mfe, "alerts": alerts, "auto_sell": False}


def gap_alerts_for_path(path: pd.DataFrame) -> list[dict[str, Any]]:
    if len(path) < 2:
        return []
    events = []
    closes = path["Close"].astype(float).to_numpy()
    opens = path["Open"].astype(float).to_numpy()
    dates = path["Date"].to_numpy()
    for i in range(1, len(path)):
        prev = closes[i - 1]
        if prev <= 0:
            continue
        gap = float(opens[i] / prev - 1.0)
        flags = []
        for thr in (-0.05, -0.10, -0.15):
            if gap <= thr:
                flags.append(f"GAP_LE_{int(abs(thr)*100)}PCT")
        if flags:
            events.append(
                {
                    "date": str(pd.Timestamp(dates[i]).date()),
                    "gap": gap,
                    "flags": flags,
                    "auto_sell": False,
                }
            )
    return events


def daily_risk_snapshot(
    *,
    asof: pd.Timestamp,
    cash: float,
    positions: dict[str, Position],
    peak_equity: float,
    realized_pnl: float,
) -> dict[str, Any]:
    pos_val = float(sum(p.market_value_base for p in positions.values()))
    equity = float(cash + pos_val)
    exposure = pos_val / equity if equity > 0 else 0.0
    dd = equity / peak_equity - 1.0 if peak_equity > 0 else 0.0
    largest = None
    worst_u = None
    best_u = None
    for p in positions.values():
        mv = p.market_value_base
        upnl = p.unrealized_pnl_base
        if largest is None or mv > largest["market_value"]:
            largest = {"symbol": p.symbol, "market_value": mv}
        if worst_u is None or upnl < worst_u["unrealized_pnl"]:
            worst_u = {"symbol": p.symbol, "unrealized_pnl": upnl}
        if best_u is None or upnl > best_u["unrealized_pnl"]:
            best_u = {"symbol": p.symbol, "unrealized_pnl": upnl}
    return {
        "date": str(asof.date()),
        "portfolio_equity": equity,
        "cash": float(cash),
        "exposure": float(exposure),
        "n_positions": len(positions),
        "current_drawdown": float(dd),
        "largest_position": largest,
        "largest_unrealized_loss": worst_u,
        "largest_unrealized_gain": best_u,
        "total_unrealized_pnl": float(sum(p.unrealized_pnl_base for p in positions.values())),
        "total_realized_pnl": float(realized_pnl),
    }


def performance_summary(
    equity_history: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    initial_capital: float,
    benchmark_return: float | None = None,
    momentum_return: float | None = None,
) -> dict[str, Any]:
    from src.simulation.metrics import compute_performance
    from src.simulation.orders import EquityPoint, TradeRecord

    if equity_history is None or equity_history.empty:
        return {"statistically_insufficient": True, "note": "no equity history"}

    curve = [
        EquityPoint(
            date=pd.Timestamp(r["Date"]),
            cash=float(r.get("Cash", 0)),
            position_value=float(r.get("Position Value", 0)),
            total_equity=float(r["Total Equity"]),
            drawdown=float(r.get("Drawdown", 0)),
            realized_pnl=float(r.get("Realized PnL", 0)),
            unrealized_pnl=float(r.get("Unrealized PnL", 0)),
            n_positions=int(r.get("N Positions", 0)),
        )
        for _, r in equity_history.iterrows()
    ]
    trade_objs: list[TradeRecord] = []
    # Prefer metrics from equity; trades optional
    costs = float(equity_history["Transaction Costs"].iloc[-1]) if "Transaction Costs" in equity_history.columns else 0.0
    perf = compute_performance(
        curve,
        trade_objs,
        initial_capital=initial_capital,
        transaction_costs_total=costs,
    )
    if trades is not None and not trades.empty and "Net PnL" in trades.columns:
        nets = trades["Net PnL"].astype(float)
        wins = nets[nets > 0]
        losses = nets[nets <= 0]
        gp = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        perf["number_of_trades"] = int(len(trades))
        perf["win_rate"] = float((nets > 0).mean())
        perf["profit_factor"] = float(gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else None)
    rets = equity_history["Total Equity"].astype(float).pct_change().dropna()
    perf["sortino"] = sortino_ratio(rets)
    perf["benchmark_return"] = benchmark_return
    perf["benchmark_excess"] = (
        float(perf["total_return"] - benchmark_return)
        if benchmark_return is not None and perf.get("total_return") is not None
        else None
    )
    perf["momentum_return"] = momentum_return
    perf["momentum_excess"] = (
        float(perf["total_return"] - momentum_return)
        if momentum_return is not None and perf.get("total_return") is not None
        else None
    )
    perf["statistically_insufficient"] = int(perf.get("number_of_trades") or 0) < 30 or len(rets) < 60
    if perf["statistically_insufficient"]:
        perf["note"] = "統計的に不十分 — forward sample がまだ少ない"
    return perf
