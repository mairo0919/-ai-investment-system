"""True-holdout diagnostics: monthly, contribution, MAE/MFE, gaps (read-only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.simulation.orders import EquityPoint, TradeRecord
from src.simulation.report import equity_to_frame, trades_to_frame


def sortino_ratio(daily_returns: pd.Series, *, periods: int = 252) -> float | None:
    r = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return None
    downside = r.clip(upper=0.0)
    dd = float(downside.std(ddof=0))
    if dd <= 1e-12:
        return None
    return float(r.mean() / dd * np.sqrt(periods))


def monthly_performance(
    equity_curve: list[EquityPoint] | pd.DataFrame,
    *,
    benchmark_equity: pd.Series | None = None,
    momentum_equity: pd.Series | None = None,
) -> pd.DataFrame:
    eq = equity_to_frame(equity_curve) if isinstance(equity_curve, list) else equity_curve.copy()
    if eq.empty:
        return pd.DataFrame()
    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.sort_values("Date")
    s = eq.set_index("Date")["Total Equity"].astype(float)
    monthly = s.resample("ME").last().dropna()
    strat = monthly.pct_change().dropna()
    out = pd.DataFrame({"month": strat.index.strftime("%Y-%m"), "strategy_return": strat.values})
    if benchmark_equity is not None and not benchmark_equity.empty:
        b = benchmark_equity.sort_index().resample("ME").last().dropna().pct_change().dropna()
        out = out.merge(
            pd.DataFrame({"month": b.index.strftime("%Y-%m"), "benchmark_return": b.values}),
            on="month",
            how="left",
        )
        out["benchmark_excess"] = out["strategy_return"] - out["benchmark_return"]
    if momentum_equity is not None and not momentum_equity.empty:
        m = momentum_equity.sort_index().resample("ME").last().dropna().pct_change().dropna()
        out = out.merge(
            pd.DataFrame({"month": m.index.strftime("%Y-%m"), "momentum_return": m.values}),
            on="month",
            how="left",
        )
        out["momentum_excess"] = out["strategy_return"] - out["momentum_return"]
    return out


def symbol_contribution(
    trades: list[TradeRecord] | pd.DataFrame,
    *,
    initial_capital: float,
) -> dict[str, Any]:
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty:
        return {"symbols": [], "top_dependency": {}}
    g = (
        tr.groupby("Symbol", sort=False)
        .agg(
            trade_count=("Net PnL", "count"),
            net_pnl=("Net PnL", "sum"),
            win_rate=("Net PnL", lambda s: float((s.astype(float) > 0).mean())),
        )
        .reset_index()
    )
    total = float(g["net_pnl"].sum())
    g["contribution_pct"] = g["net_pnl"] / total if abs(total) > 1e-12 else 0.0
    g = g.sort_values("net_pnl", ascending=False)
    rows = g.to_dict(orient="records")
    ranked = list(g["Symbol"])
    dep = {}
    for k in (1, 3, 5):
        top = set(ranked[:k])
        top_pnl = float(g.loc[g["Symbol"].isin(top), "net_pnl"].sum())
        residual = total - top_pnl
        dep[f"top{k}"] = {
            "symbols": list(ranked[:k]),
            "pnl_share": float(top_pnl / total) if abs(total) > 1e-12 else None,
            "residual_pnl": residual,
            "residual_pnl_over_capital": float(residual / initial_capital) if initial_capital else None,
        }
    return {
        "total_net_pnl": total,
        "symbols": rows,
        "top_dependency": dep,
    }


def sector_contribution(
    trades: list[TradeRecord] | pd.DataFrame,
    symbol_sector: dict[str, str],
) -> dict[str, Any]:
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty:
        return {}
    tr["Sector"] = tr["Symbol"].map(lambda s: symbol_sector.get(str(s), "Unknown"))
    out: dict[str, Any] = {}
    for sector, g in tr.groupby("Sector"):
        nets = g["Net PnL"].astype(float)
        wins = nets[nets > 0]
        losses = nets[nets <= 0]
        gp = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        out[str(sector)] = {
            "trade_count": int(len(g)),
            "net_pnl": float(nets.sum()),
            "win_rate": float((nets > 0).mean()),
            "profit_factor": float(gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else None),
        }
    return out


def mae_mfe_analysis(
    trades: list[TradeRecord] | pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    """Max adverse / favorable excursion during each trade (local prices)."""
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty:
        return {"n_trades": 0}
    px = prices.copy()
    px["Date"] = pd.to_datetime(px["Date"]).dt.normalize()
    maes: list[float] = []
    mfes: list[float] = []
    worst = None
    thresholds = (-0.10, -0.15, -0.20, -0.30)
    thr_counts = {f"hit_{int(abs(t)*100)}pct": 0 for t in thresholds}

    for _, row in tr.iterrows():
        symbol = str(row["Symbol"])
        entry = pd.Timestamp(row["Entry Date"]).normalize()
        exit_ = pd.Timestamp(row["Exit Date"]).normalize()
        entry_px = float(row["Entry Price"])
        if entry_px <= 0:
            continue
        path = px.loc[
            (px["Symbol"] == symbol) & (px["Date"] >= entry) & (px["Date"] <= exit_)
        ]
        if path.empty:
            continue
        lows = path["Low"].astype(float)
        highs = path["High"].astype(float)
        mae = float((lows.min() / entry_px) - 1.0)
        mfe = float((highs.max() / entry_px) - 1.0)
        maes.append(mae)
        mfes.append(mfe)
        for t in thresholds:
            if mae <= t:
                thr_counts[f"hit_{int(abs(t)*100)}pct"] += 1
        trade_ret = float(row["Return"])
        if worst is None or trade_ret < worst["return"]:
            worst = {
                "symbol": symbol,
                "entry_date": str(entry.date()),
                "exit_date": str(exit_.date()),
                "return": trade_ret,
                "mae": mae,
                "mfe": mfe,
            }

    arr = np.asarray(maes, dtype=float) if maes else np.array([])
    return {
        "n_trades": int(len(maes)),
        "mean_mae": float(arr.mean()) if len(arr) else None,
        "mean_mfe": float(np.mean(mfes)) if mfes else None,
        "worst_mae": float(arr.min()) if len(arr) else None,
        "p95_mae": float(np.quantile(arr, 0.05)) if len(arr) else None,  # left tail
        "p99_mae": float(np.quantile(arr, 0.01)) if len(arr) else None,
        "threshold_hit_counts": thr_counts,
        "worst_trade": worst,
        "policy": "Uses High/Low between entry and exit dates only (no post-exit path).",
    }


def gap_down_audit(
    trades: list[TradeRecord] | pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    """Count overnight gap-downs while a position is open."""
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty:
        return {"n_trades": 0}
    px = prices.copy()
    px["Date"] = pd.to_datetime(px["Date"]).dt.normalize()
    events: list[dict[str, Any]] = []
    counts = {"gap_le_m5": 0, "gap_le_m10": 0, "gap_le_m15": 0}
    max_gap = 0.0

    for _, row in tr.iterrows():
        symbol = str(row["Symbol"])
        entry = pd.Timestamp(row["Entry Date"]).normalize()
        exit_ = pd.Timestamp(row["Exit Date"]).normalize()
        path = px.loc[
            (px["Symbol"] == symbol) & (px["Date"] >= entry) & (px["Date"] <= exit_)
        ].sort_values("Date")
        if len(path) < 2:
            continue
        closes = path["Close"].astype(float).to_numpy()
        opens = path["Open"].astype(float).to_numpy()
        dates = path["Date"].to_numpy()
        for i in range(1, len(path)):
            prev_c = closes[i - 1]
            op = opens[i]
            if prev_c <= 0:
                continue
            gap = float(op / prev_c - 1.0)
            max_gap = min(max_gap, gap)
            if gap <= -0.05:
                counts["gap_le_m5"] += 1
            if gap <= -0.10:
                counts["gap_le_m10"] += 1
            if gap <= -0.15:
                counts["gap_le_m15"] += 1
            if gap <= -0.05:
                events.append(
                    {
                        "symbol": symbol,
                        "date": str(pd.Timestamp(dates[i]).date()),
                        "gap": gap,
                    }
                )

    worst = min(events, key=lambda e: e["gap"]) if events else None
    return {
        "n_trades": int(len(tr)),
        "counts": counts,
        "max_gap_down": float(max_gap) if events else None,
        "worst_event": worst,
        "n_gap_events_le_5pct": int(counts["gap_le_m5"]),
    }


def score_tertile_diagnostics(
    trades: list[TradeRecord] | pd.DataFrame,
) -> dict[str, Any]:
    """Entry-score tertiles; rank-based fallback when qcut collapses (tied scores)."""
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty or "Entry Score" not in tr.columns:
        return {"buckets": {}, "note": "no trades"}
    tr = tr.dropna(subset=["Entry Score"]).copy()
    if tr.empty:
        return {"buckets": {}, "note": "no entry scores"}
    scores = tr["Entry Score"].astype(float)
    method = "qcut"
    try:
        tr["score_bucket"] = pd.qcut(
            scores,
            q=3,
            labels=["bottom_1_3", "middle_1_3", "top_1_3"],
            duplicates="drop",
        )
        if tr["score_bucket"].nunique() < 3:
            raise ValueError("qcut collapsed")
    except ValueError:
        method = "rank_tertile"
        # Equal-count tertiles by rank (handles ties via average rank + cut)
        ranks = scores.rank(method="first")
        tr["score_bucket"] = pd.qcut(
            ranks,
            q=3,
            labels=["bottom_1_3", "middle_1_3", "top_1_3"],
        )
    out: dict[str, Any] = {}
    for bucket, g in tr.groupby("score_bucket", observed=False):
        nets = g["Net PnL"].astype(float)
        wins = nets[nets > 0]
        losses = nets[nets <= 0]
        gp = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        out[str(bucket)] = {
            "trade_count": int(len(g)),
            "average_return": float(g["Return"].astype(float).mean()),
            "win_rate": float((nets > 0).mean()),
            "profit_factor": float(gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else None),
            "average_entry_score": float(g["Entry Score"].astype(float).mean()),
        }
    return {"buckets": out, "method": method}


def success_checklist(
    *,
    net_return: float | None,
    sharpe: float | None,
    benchmark_excess: float | None,
    momentum_excess: float | None,
    profit_factor: float | None,
    top1_share: float | None,
    top_sector_share: float | None,
    positive_month_ratio: float | None,
    worst_mae: float | None,
) -> dict[str, Any]:
    return {
        "A_net_return_positive": bool(net_return is not None and net_return > 0),
        "B_sharpe_positive": bool(sharpe is not None and sharpe > 0),
        "C_benchmark_excess_positive": bool(
            benchmark_excess is not None and benchmark_excess > 0
        ),
        "D_momentum_excess_positive": bool(
            momentum_excess is not None and momentum_excess > 0
        ),
        "E_profit_factor_gt_1": bool(profit_factor is not None and profit_factor > 1),
        "F_not_single_symbol_dependent": bool(
            top1_share is not None and top1_share < 0.50
        ),
        "G_not_single_sector_dependent": bool(
            top_sector_share is not None and top_sector_share < 0.60
        ),
        "H_not_single_month_dependent": bool(
            positive_month_ratio is not None and positive_month_ratio >= 0.40
        ),
        "I_tail_risk_not_extreme": bool(worst_mae is not None and worst_mae > -0.40),
        "notes": {
            "F_threshold": "top1 pnl share < 50%",
            "G_threshold": "top sector pnl share < 60%",
            "H_threshold": "positive month ratio >= 40%",
            "I_threshold": "worst MAE > -40%",
        },
    }
