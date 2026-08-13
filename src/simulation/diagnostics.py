"""Phase 4B diagnostics: cost drag, turnover, trade quality, exit reasons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.simulation.orders import EquityPoint, TradeRecord
from src.simulation.report import trades_to_frame


def cost_decomposition(
    *,
    gross_perf: dict[str, Any],
    net_perf: dict[str, Any],
    low_perf: dict[str, Any] | None,
    initial_capital: float,
    commission_total: float,
    slippage_total: float,
) -> dict[str, Any]:
    g_ret = gross_perf.get("mean_fold_total_return", gross_perf.get("total_return"))
    n_ret = net_perf.get("mean_fold_total_return", net_perf.get("total_return"))
    l_ret = None
    if low_perf is not None:
        l_ret = low_perf.get("mean_fold_total_return", low_perf.get("total_return"))
    cost_drag = _sub(g_ret, n_ret)
    # Approximate gross profit in currency using mean-fold return * capital
    gross_profit = float(g_ret) * initial_capital if g_ret is not None else None
    total_cost = float(commission_total + slippage_total)
    return {
        "gross_return": g_ret,
        "net_return": n_ret,
        "low_cost_return": l_ret,
        "cost_drag": cost_drag,
        "total_transaction_cost": total_cost,
        "commission": float(commission_total),
        "slippage": float(slippage_total),
        "cost_over_gross_profit": (
            float(total_cost / gross_profit)
            if gross_profit is not None and abs(gross_profit) > 1e-9
            else None
        ),
        "cost_over_initial_capital": float(total_cost / initial_capital) if initial_capital else None,
        "gross_profitable": bool(g_ret is not None and g_ret > 0),
        "net_profitable": bool(n_ret is not None and n_ret > 0),
    }


def turnover_analysis(
    trades: list[TradeRecord] | pd.DataFrame,
    equity_curve: list[EquityPoint] | pd.DataFrame,
    *,
    initial_capital: float,
    rebalance_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades
    if isinstance(equity_curve, list):
        eq = pd.DataFrame([e.to_dict() for e in equity_curve]) if equity_curve else pd.DataFrame()
    else:
        eq = equity_curve.copy()

    n_days = max(int(len(eq)) - 1, 0)
    years = n_days / 252.0 if n_days else 0.0
    months = n_days / 21.0 if n_days else 0.0
    n_trades = int(len(tr))
    avg_hold = float(tr["Holding Days"].mean()) if n_trades and "Holding Days" in tr.columns else None
    entry_notional = float(tr["Entry Cost Base"].sum()) if n_trades and "Entry Cost Base" in tr.columns else 0.0
    avg_equity = float(eq["Total Equity"].mean()) if not eq.empty else initial_capital
    turnover = float(entry_notional / avg_equity) if avg_equity > 0 else 0.0

    by_country: dict[str, Any] = {}
    if n_trades and "Country" in tr.columns:
        for cty, g in tr.groupby("Country"):
            by_country[str(cty)] = {
                "number_of_trades": int(len(g)),
                "average_holding_days": float(g["Holding Days"].mean()),
                "entry_notional": float(g["Entry Cost Base"].sum()) if "Entry Cost Base" in g else None,
            }

    return {
        "number_of_trades": n_trades,
        "trades_per_year": float(n_trades / years) if years > 0 else None,
        "average_trades_per_month": float(n_trades / months) if months > 0 else None,
        "portfolio_turnover": turnover,
        "average_holding_days": avg_hold,
        "average_entries_per_rebalance": (
            (rebalance_stats or {}).get("avg_entries_per_rebalance")
        ),
        "average_exits_per_rebalance": (
            (rebalance_stats or {}).get("avg_exits_per_rebalance")
        ),
        "by_country": by_country,
        "n_days": n_days,
    }


def trade_quality_by_entry_score(
    trades: list[TradeRecord] | pd.DataFrame,
) -> dict[str, Any]:
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty or "Entry Score" not in tr.columns:
        return {"buckets": {}, "note": "no trades"}
    tr = tr.dropna(subset=["Entry Score"]).copy()
    if tr.empty:
        return {"buckets": {}, "note": "no entry scores"}
    try:
        tr["score_bucket"] = pd.qcut(
            tr["Entry Score"].astype(float),
            q=3,
            labels=["bottom_1_3", "middle_1_3", "top_1_3"],
            duplicates="drop",
        )
    except ValueError:
        tr["score_bucket"] = "all"
    out: dict[str, Any] = {}
    for bucket, g in tr.groupby("score_bucket", observed=False):
        nets = g["Net PnL"].astype(float)
        # Approx gross from price move (prices already include slip in net runs)
        if "Entry Price" in g.columns and "Exit Price" in g.columns:
            gross_rets = (g["Exit Price"].astype(float) / g["Entry Price"].astype(float) - 1.0)
        else:
            gross_rets = g["Return"].astype(float)
        wins = nets[nets > 0]
        losses = nets[nets <= 0]
        gp = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        out[str(bucket)] = {
            "trade_count": int(len(g)),
            "win_rate": float((nets > 0).mean()),
            "average_gross_return": float(gross_rets.mean()),
            "average_net_return": float(g["Return"].astype(float).mean()),
            "profit_factor": float(gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else None),
            "average_net_pnl": float(nets.mean()),
        }
    return {"buckets": out}


def exit_reason_analysis(
    trades: list[TradeRecord] | pd.DataFrame,
) -> dict[str, Any]:
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty or "Exit Reason" not in tr.columns:
        return {}
    out: dict[str, Any] = {}
    for reason, g in tr.groupby("Exit Reason"):
        nets = g["Net PnL"].astype(float)
        out[str(reason)] = {
            "count": int(len(g)),
            "share": float(len(g) / len(tr)),
            "average_pnl": float(nets.mean()),
            "win_rate": float((nets > 0).mean()),
            "average_holding_days": float(g["Holding Days"].mean()),
            "average_return": float(g["Return"].astype(float).mean()),
        }
    return out


def fold_stability(fold_returns: list[float | None], fold_sharpes: list[float | None] | None = None) -> dict[str, Any]:
    vals = [float(x) for x in fold_returns if x is not None]
    if not vals:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "positive_fold_ratio": None,
            "worst_fold": None,
            "best_fold": None,
            "n": 0,
        }
    arr = np.asarray(vals, dtype=float)
    out: dict[str, Any] = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "positive_fold_ratio": float((arr > 0).mean()),
        "worst_fold": float(arr.min()),
        "best_fold": float(arr.max()),
        "n": int(len(arr)),
    }
    if fold_sharpes is not None:
        sh = [float(x) for x in fold_sharpes if x is not None]
        if sh:
            out["sharpe_mean"] = float(np.mean(sh))
            out["sharpe_median"] = float(np.median(sh))
    return out


def regime_trade_analysis(
    trades: list[TradeRecord] | pd.DataFrame,
    regime_by_date: pd.Series,
) -> dict[str, Any]:
    """Bucket trades by market regime on entry date."""
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty or regime_by_date is None or regime_by_date.empty:
        return {}
    tr["Entry Date"] = pd.to_datetime(tr["Entry Date"])
    reg = regime_by_date.copy()
    reg.index = pd.to_datetime(reg.index).normalize()
    # asof backward map
    mapped = []
    for dt in tr["Entry Date"]:
        pos = reg.index.searchsorted(pd.Timestamp(dt).normalize(), side="right") - 1
        mapped.append(str(reg.iloc[pos]) if pos >= 0 else "Unknown")
    tr["regime"] = mapped
    out: dict[str, Any] = {}
    for label, g in tr.groupby("regime"):
        nets = g["Net PnL"].astype(float)
        wins = nets[nets > 0]
        losses = nets[nets <= 0]
        gp = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        out[str(label)] = {
            "trade_count": int(len(g)),
            "average_return": float(g["Return"].astype(float).mean()),
            "win_rate": float((nets > 0).mean()),
            "profit_factor": float(gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else None),
            "average_net_pnl": float(nets.mean()),
        }
    return out


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)
