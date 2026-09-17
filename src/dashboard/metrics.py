"""Dashboard aggregations — reuse paper/simulation metrics; never write state."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.dashboard.data import DashboardDataSource
from src.paper.diagnostics import performance_summary
from src.simulation.holdout_diagnostics import monthly_performance
from src.simulation.metrics import compute_performance
from src.simulation.orders import EquityPoint


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if isinstance(v, (float, int, np.floating, np.integer)) and pd.isna(v):
            return None
        out = float(v)
        if not np.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _s(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    text = str(v).strip()
    return text if text else None


def _boolish(v: Any) -> bool | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    text = str(v).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _equity_points(equity: pd.DataFrame) -> list[EquityPoint]:
    points: list[EquityPoint] = []
    for _, r in equity.iterrows():
        te = _f(r.get("Total Equity"))
        if te is None:
            continue
        points.append(
            EquityPoint(
                date=pd.Timestamp(r["Date"]),
                cash=float(_f(r.get("Cash")) or 0.0),
                position_value=float(_f(r.get("Position Value")) or 0.0),
                total_equity=float(te),
                drawdown=float(_f(r.get("Drawdown")) or 0.0),
                realized_pnl=float(_f(r.get("Realized PnL")) or 0.0),
                unrealized_pnl=float(_f(r.get("Unrealized PnL")) or 0.0),
                n_positions=int(_i(r.get("N Positions")) or 0),
            )
        )
    return points


def _transaction_costs_total(equity: pd.DataFrame) -> float:
    if equity.empty or "Transaction Costs" not in equity.columns:
        return 0.0
    val = _f(equity["Transaction Costs"].iloc[-1])
    return float(val) if val is not None else 0.0


def _normalize_trades_for_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    """Map common aliases so compute_performance can consume the CSV frame."""
    tr = trades.copy()
    rename: dict[str, str] = {}
    aliases = [
        ("net_pnl", "Net PnL"),
        ("NetPnL", "Net PnL"),
        ("holding_days", "Holding Days"),
        ("HoldingDays", "Holding Days"),
        ("entry_cost_base", "Entry Cost Base"),
        ("EntryCostBase", "Entry Cost Base"),
    ]
    for src, dst in aliases:
        if src in tr.columns and dst not in tr.columns:
            rename[src] = dst
    return tr.rename(columns=rename)


def compute_dashboard_performance(
    equity: pd.DataFrame,
    trades: pd.DataFrame | None,
    *,
    initial_capital: float,
    benchmark_return: float | None = None,
    momentum_return: float | None = None,
) -> dict[str, Any]:
    """Glue around existing metrics.

    Uses ``compute_performance`` with trades (fixes average_win/loss gap in
    ``performance_summary``) and keeps summary fields (sortino / benchmark) from
    ``performance_summary`` without modifying Paper diagnostics.
    """
    if equity is None or equity.empty or "Total Equity" not in equity.columns:
        return {
            "statistically_insufficient": True,
            "note": "no equity history",
            "total_return": None,
            "cumulative_return": None,
            "sharpe": None,
            "max_drawdown": None,
            "win_rate": None,
            "completed_trades": 0,
            "profit_factor": None,
            "average_win": None,
            "average_loss": None,
        }

    curve = _equity_points(equity)
    if not curve:
        return {
            "statistically_insufficient": True,
            "note": "no usable equity rows",
            "total_return": None,
            "cumulative_return": None,
            "sharpe": None,
            "max_drawdown": None,
            "win_rate": None,
            "completed_trades": 0,
            "profit_factor": None,
            "average_win": None,
            "average_loss": None,
        }

    trade_frame = (
        _normalize_trades_for_metrics(trades)
        if trades is not None and not trades.empty
        else pd.DataFrame()
    )
    # Pass DataFrame into compute_performance so win/loss averages populate.
    core = compute_performance(
        curve,
        trade_frame,
        initial_capital=float(initial_capital),
        transaction_costs_total=_transaction_costs_total(equity),
    )

    # Reuse performance_summary for sortino / bench fields / insufficiency flag.
    summary = performance_summary(
        equity,
        trades if trades is not None else pd.DataFrame(),
        initial_capital=float(initial_capital),
        benchmark_return=benchmark_return,
        momentum_return=momentum_return,
    )

    n_trades = int(core.get("number_of_trades") or 0)
    if trade_frame is not None and not trade_frame.empty and "Net PnL" in trade_frame.columns:
        n_trades = int(len(trade_frame))

    total_return = core.get("total_return")
    return {
        "total_return": total_return,
        "cumulative_return": total_return,  # same definition as compute_performance
        "cagr": core.get("cagr"),
        "volatility": core.get("volatility"),
        "sharpe": core.get("sharpe"),
        "max_drawdown": core.get("max_drawdown"),
        "win_rate": core.get("win_rate"),
        "completed_trades": n_trades,
        "number_of_trades": n_trades,
        "profit_factor": core.get("profit_factor"),
        "average_win": core.get("average_profit"),
        "average_loss": core.get("average_loss"),
        "average_holding_days": core.get("average_holding_days"),
        "turnover": core.get("turnover"),
        "transaction_costs": core.get("transaction_costs"),
        "final_equity": core.get("final_equity"),
        "initial_capital": core.get("initial_capital"),
        "sortino": summary.get("sortino"),
        "benchmark_return": summary.get("benchmark_return"),
        "benchmark_excess": summary.get("benchmark_excess"),
        "momentum_return": summary.get("momentum_return"),
        "momentum_excess": summary.get("momentum_excess"),
        "statistically_insufficient": summary.get("statistically_insufficient"),
        "note": summary.get("note"),
    }


def get_equity_curve(source: DashboardDataSource) -> list[dict[str, Any]]:
    equity = source.load_equity_history()
    if equity is None or equity.empty or "Date" not in equity.columns:
        return []
    rows: list[dict[str, Any]] = []
    for _, r in equity.iterrows():
        date = _s(r.get("Date"))
        te = _f(r.get("Total Equity"))
        if date is None or te is None:
            continue
        rows.append(
            {
                "date": date,
                "total_equity": te,
                "cash": _f(r.get("Cash")),
                "position_value": _f(r.get("Position Value")),
                "drawdown": _f(r.get("Drawdown")),
            }
        )
    return rows


def _daily_returns(equity: pd.DataFrame | None) -> list[dict[str, Any]]:
    if equity is None or equity.empty or "Total Equity" not in equity.columns:
        return []
    eq = equity.copy()
    eq["Date"] = pd.to_datetime(eq["Date"], errors="coerce")
    eq = eq.dropna(subset=["Date"]).sort_values("Date")
    rets = eq["Total Equity"].astype(float).pct_change()
    out: list[dict[str, Any]] = []
    for i in range(len(eq)):
        if i == 0 or pd.isna(rets.iloc[i]):
            continue
        out.append(
            {
                "date": str(pd.Timestamp(eq["Date"].iloc[i]).date()),
                "daily_return": float(rets.iloc[i]),
            }
        )
    return out


def _monthly_returns(equity: pd.DataFrame | None) -> list[dict[str, Any]]:
    if equity is None or equity.empty:
        return []
    monthly = monthly_performance(equity)
    if monthly is None or monthly.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, r in monthly.iterrows():
        rows.append(
            {
                "month": str(r["month"]),
                "strategy_return": float(r["strategy_return"]),
            }
        )
    return rows


def build_overview(source: DashboardDataSource) -> dict[str, Any]:
    port = source.load_portfolio()
    health = source.load_run_health()
    gate = source.load_latest_validity()
    trades = source.load_trade_history()
    equity = source.load_equity_history()

    completed = None
    if trades is not None:
        completed = int(len(trades))

    current_equity = _f(port.get("total_equity")) if port else None
    initial = float(source.initial_capital)
    total_pnl = None
    total_return = None
    if current_equity is not None:
        total_pnl = float(current_equity - initial)
        if initial > 0:
            total_return = float(current_equity / initial - 1.0)

    # Prefer equity-based metrics when available (same formula as compute_performance).
    if equity is not None and not equity.empty and "Total Equity" in equity.columns:
        perf = compute_dashboard_performance(
            equity, trades, initial_capital=initial
        )
        if perf.get("total_return") is not None:
            total_return = perf.get("total_return")
        if perf.get("final_equity") is not None and current_equity is None:
            current_equity = perf.get("final_equity")
            total_pnl = (
                float(current_equity - initial) if current_equity is not None else None
            )

    latest_asof = None
    if health and health.get("last_asof") is not None:
        latest_asof = _s(health.get("last_asof"))
    elif port and port.get("last_processed_date") is not None:
        latest_asof = _s(port.get("last_processed_date"))
    elif gate and gate.get("asof") is not None:
        latest_asof = _s(gate.get("asof"))

    review = None
    if gate is not None:
        raw_reasons = gate.get("review_reasons")
        if isinstance(raw_reasons, list):
            review = [str(x) for x in raw_reasons]
        elif raw_reasons is None:
            review = []
        else:
            review = None

    return {
        "experiment_id": source.experiment_id,
        "base_currency": _s(port.get("base_currency")) if port else None,
        "initial_capital": initial,
        "current_equity": current_equity,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "realized_pnl": _f(port.get("realized_pnl")) if port else None,
        "unrealized_pnl": _f(port.get("unrealized_pnl")) if port else None,
        "position_value": _f(port.get("position_value")) if port else None,
        "cash": _f(port.get("cash")) if port else None,
        "n_positions": _i(port.get("n_positions")) if port else None,
        "total_commission": _f(port.get("total_commission")) if port else None,
        "total_slippage_impact": _f(port.get("total_slippage_impact")) if port else None,
        "peak_equity": _f(port.get("peak_equity")) if port else None,
        "latest_asof": latest_asof,
        "last_success_utc": _s(health.get("last_success_utc")) if health else None,
        "run_status": _s(health.get("status")) if health else None,
        "model_id": _s(port.get("model_id"))
        if port and port.get("model_id") is not None
        else (_s(health.get("model_id")) if health else None),
        "config_id": _s(port.get("config_id")) if port else None,
        "validity_status": _s(gate.get("status")) if gate else None,
        "validity_review_reasons": review,
        "completed_trades": completed,
    }


def build_performance(source: DashboardDataSource) -> dict[str, Any]:
    equity = source.load_equity_history()
    trades = source.load_trade_history()
    port = source.load_portfolio()

    if equity is None:
        equity = pd.DataFrame()
    if trades is None:
        trades = pd.DataFrame()

    perf = compute_dashboard_performance(
        equity, trades, initial_capital=float(source.initial_capital)
    )
    return {
        "experiment_id": source.experiment_id,
        "total_return": perf.get("total_return"),
        "cumulative_return": perf.get("cumulative_return"),
        "daily_returns": _daily_returns(equity if not equity.empty else None),
        "monthly_returns": _monthly_returns(equity if not equity.empty else None),
        "sharpe": perf.get("sharpe"),
        "max_drawdown": perf.get("max_drawdown"),
        "win_rate": perf.get("win_rate"),
        "completed_trades": perf.get("completed_trades"),
        "profit_factor": perf.get("profit_factor"),
        "average_win": perf.get("average_win"),
        "average_loss": perf.get("average_loss"),
        "realized_pnl": _f(port.get("realized_pnl")) if port else None,
        "unrealized_pnl": _f(port.get("unrealized_pnl")) if port else None,
        "statistically_insufficient": perf.get("statistically_insufficient"),
        "note": perf.get("note"),
        "sortino": perf.get("sortino"),
        "benchmark_return": perf.get("benchmark_return"),
        "benchmark_excess": perf.get("benchmark_excess"),
    }


def build_current_portfolio(source: DashboardDataSource) -> list[dict[str, Any]]:
    positions = source.load_positions()
    if positions is None:
        return []

    ranking: pd.DataFrame | None = None
    dates = source.load_ranking_dates()
    asof = None
    health = source.load_run_health()
    port = source.load_portfolio()
    if health and health.get("last_asof"):
        asof = _s(health.get("last_asof"))
    elif port and port.get("last_processed_date"):
        asof = _s(port.get("last_processed_date"))
    if asof and asof in dates:
        ranking = source.load_rankings(asof)
    elif asof is None and dates:
        # No asof known — join the chronologically latest snapshot only.
        ranking = source.load_rankings(dates[-1])
    # If asof is set but that day's snapshot is missing → leave ranks null.

    by_symbol: dict[str, dict[str, Any]] = {}
    if ranking is not None and not ranking.empty and "Symbol" in ranking.columns:
        for _, r in ranking.iterrows():
            sym = _s(r.get("Symbol"))
            if sym is None:
                continue
            by_symbol[sym] = {
                "current_score": _f(r.get("Score")),
                "current_rank": _i(r.get("Rank")),
                "current_percentile": _f(r.get("Percentile")),
                "selected": _boolish(r.get("Selected")),
            }

    rows: list[dict[str, Any]] = []
    for p in positions:
        sym = _s(p.get("symbol"))
        cur = by_symbol.get(sym or "", {})
        rows.append(
            {
                "symbol": sym,
                "country": _s(p.get("country")),
                "currency": _s(p.get("currency")),
                "quantity": _f(p.get("quantity")),
                "entry_date": _s(p.get("entry_date")),
                "entry_price": _f(p.get("entry_price")),
                "current_price": _f(p.get("current_price")),
                "market_value": _f(p.get("market_value")),
                "unrealized_pnl": _f(p.get("unrealized_pnl")),
                "holding_days": _i(p.get("holding_days")),
                "entry_score": _f(p.get("entry_score")),
                "entry_rank": _i(p.get("entry_rank")),
                "entry_percentile": _f(p.get("entry_percentile")),
                "current_score": cur.get("current_score"),
                "current_rank": cur.get("current_rank"),
                "current_percentile": cur.get("current_percentile"),
                "selected": cur.get("selected"),
                "pending_exit_reason": _s(p.get("pending_exit_reason")),
            }
        )
    return rows


def build_trade_history(source: DashboardDataSource) -> list[dict[str, Any]]:
    trades = source.load_trade_history()
    if trades is None or trades.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, r in trades.iterrows():
        rows.append(
            {
                "symbol": _s(r.get("Symbol")),
                "country": _s(r.get("Country")),
                "currency": _s(r.get("Currency")),
                "entry_date": _s(r.get("Entry Date")),
                "exit_date": _s(r.get("Exit Date")),
                "entry_price": _f(r.get("Entry Price")),
                "exit_price": _f(r.get("Exit Price")),
                "quantity": _f(r.get("Quantity")),
                "gross_pnl": _f(r.get("Gross PnL")),
                "net_pnl": _f(r.get("Net PnL")),
                "return": _f(r.get("Return")),
                "holding_days": _i(r.get("Holding Days")),
                "exit_reason": _s(r.get("Exit Reason")),
                "entry_rank": _i(r.get("Entry Rank")),
                "model_id": _s(r.get("model_id")),
                "config_id": _s(r.get("config_id")),
            }
        )

    def _exit_key(row: dict[str, Any]) -> tuple:
        d = row.get("exit_date")
        if not d:
            return ("",)
        try:
            return (pd.Timestamp(d).to_datetime64(),)
        except (ValueError, TypeError):
            return (str(d),)

    rows.sort(key=_exit_key, reverse=True)
    return rows


def get_rankings(source: DashboardDataSource, date: pd.Timestamp | str) -> list[dict[str, Any]]:
    df = source.load_rankings(date)
    if df is None or df.empty:
        return []
    date_s = pd.Timestamp(date).normalize().date().isoformat()
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "date": _s(r.get("Date")) or date_s,
                "symbol": _s(r.get("Symbol")),
                "score": _f(r.get("Score")),
                "rank": _i(r.get("Rank")),
                "percentile": _f(r.get("Percentile")),
                "selected": _boolish(r.get("Selected")),
                "model_id": _s(r.get("model_id")),
                "country": _s(r.get("Country")),
                "close": _f(r.get("Close")),
            }
        )
    return rows


def list_ranking_dates(source: DashboardDataSource) -> list[str]:
    return source.load_ranking_dates()


def build_system_status(source: DashboardDataSource) -> dict[str, Any]:
    health = source.load_run_health()
    gate = source.load_latest_validity()
    port = source.load_portfolio()

    review = None
    sample = None
    if gate is not None:
        raw_reasons = gate.get("review_reasons")
        if isinstance(raw_reasons, list):
            review = [str(x) for x in raw_reasons]
        elif raw_reasons is None:
            review = []
        sample = gate.get("sample") if isinstance(gate.get("sample"), dict) else None

    model_id = None
    config_id = None
    if port:
        model_id = _s(port.get("model_id"))
        config_id = _s(port.get("config_id"))
    if model_id is None and health:
        model_id = _s(health.get("model_id"))
    if model_id is None and gate:
        model_id = _s(gate.get("model_id"))

    return {
        "experiment_id": source.experiment_id,
        "run_status": _s(health.get("status")) if health else None,
        "last_run_started_utc": _s(health.get("last_run_started_utc")) if health else None,
        "last_run_finished_utc": _s(health.get("last_run_finished_utc")) if health else None,
        "last_success_utc": _s(health.get("last_success_utc")) if health else None,
        "last_asof": _s(health.get("last_asof")) if health else None,
        "error_type": _s(health.get("error_type")) if health else None,
        "error_message": _s(health.get("error_message")) if health else None,
        "processed_sessions": _i(health.get("processed_sessions")) if health else None,
        "validity_status": _s(gate.get("status")) if gate else None,
        "validity_review_reasons": review,
        "validity_sample": sample,
        "model_id": model_id,
        "config_id": config_id,
    }
