"""Performance metrics for paper-trading simulations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.simulation.orders import EquityPoint, TradeRecord


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def compute_performance(
    equity_curve: list[EquityPoint] | pd.DataFrame,
    trades: list[TradeRecord] | pd.DataFrame,
    *,
    initial_capital: float,
    transaction_costs_total: float,
) -> dict[str, Any]:
    eq = _equity_frame(equity_curve)
    tr = _trades_frame(trades)

    if eq.empty:
        return {
            "total_return": 0.0,
            "cagr": None,
            "volatility": None,
            "sharpe": None,
            "max_drawdown": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "average_profit": None,
            "average_loss": None,
            "number_of_trades": 0,
            "average_holding_days": None,
            "turnover": 0.0,
            "transaction_costs": float(transaction_costs_total),
        }

    equity = eq["Total Equity"].astype(float)
    rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    n_days = max(len(equity) - 1, 0)
    years = n_days / 252.0 if n_days > 0 else 0.0
    if years > 0 and equity.iloc[-1] > 0 and initial_capital > 0:
        cagr = float((equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)
    else:
        cagr = None
    vol = float(rets.std(ddof=0) * np.sqrt(252)) if len(rets) else None
    sharpe = None
    if vol is not None and vol > 1e-12 and len(rets):
        sharpe = float(rets.mean() / rets.std(ddof=0) * np.sqrt(252))

    mdd = max_drawdown(equity)

    win_rate = None
    profit_factor = None
    avg_profit = None
    avg_loss = None
    avg_hold = None
    n_trades = 0
    turnover = 0.0

    if not tr.empty:
        n_trades = int(len(tr))
        nets = tr["Net PnL"].astype(float)
        wins = nets[nets > 0]
        losses = nets[nets <= 0]
        win_rate = float((nets > 0).mean()) if n_trades else None
        gross_wins = float(wins.sum()) if len(wins) else 0.0
        gross_losses = float(-losses.sum()) if len(losses) else 0.0
        if gross_losses > 1e-12:
            profit_factor = float(gross_wins / gross_losses)
        elif gross_wins > 0:
            profit_factor = float("inf")
        else:
            profit_factor = None
        avg_profit = float(wins.mean()) if len(wins) else None
        avg_loss = float(losses.mean()) if len(losses) else None
        avg_hold = float(tr["Holding Days"].astype(float).mean())
        # Approximate turnover: sum of entry notionals / average equity
        if "Entry Cost Base" in tr.columns and equity.mean() > 0:
            turnover = float(tr["Entry Cost Base"].astype(float).sum() / equity.mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_profit": avg_profit,
        "average_loss": avg_loss,
        "number_of_trades": n_trades,
        "average_holding_days": avg_hold,
        "turnover": turnover,
        "transaction_costs": float(transaction_costs_total),
        "final_equity": float(equity.iloc[-1]),
        "initial_capital": float(initial_capital),
    }


def buy_and_hold_benchmark(
    prices: pd.Series,
    *,
    initial_capital: float,
) -> dict[str, Any]:
    """Buy & hold on a single price series (e.g. index Close in local terms).

    Uses first/last available closes in the series window.
    """
    s = prices.dropna().astype(float)
    if len(s) < 2:
        return {
            "total_return": 0.0,
            "cagr": None,
            "max_drawdown": 0.0,
            "volatility": None,
            "sharpe": None,
            "equity_curve": [],
        }
    units = initial_capital / float(s.iloc[0])
    equity = units * s
    rets = equity.pct_change().dropna()
    n_days = max(len(equity) - 1, 0)
    years = n_days / 252.0 if n_days > 0 else 0.0
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    cagr = (
        float((equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)
        if years > 0
        else None
    )
    vol = float(rets.std(ddof=0) * np.sqrt(252)) if len(rets) else None
    sharpe = (
        float(rets.mean() / rets.std(ddof=0) * np.sqrt(252))
        if vol and vol > 1e-12
        else None
    )
    curve = [
        {
            "Date": str(pd.Timestamp(idx).date()),
            "Total Equity": float(val),
            "Drawdown": float(val / equity.cummax().loc[idx] - 1.0),
        }
        for idx, val in equity.items()
    ]
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown(equity),
        "volatility": vol,
        "sharpe": sharpe,
        "equity_curve": curve,
    }


def _equity_frame(equity_curve: list[EquityPoint] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(equity_curve, pd.DataFrame):
        return equity_curve.copy()
    if not equity_curve:
        return pd.DataFrame()
    return pd.DataFrame([e.to_dict() for e in equity_curve])


def _trades_frame(trades: list[TradeRecord] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(trades, pd.DataFrame):
        return trades.copy()
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.to_dict() for t in trades])
