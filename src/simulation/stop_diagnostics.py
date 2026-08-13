"""Stop-loss aftermath and counterfactual diagnostics (not used for live decisions)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.simulation.orders import TradeRecord
from src.simulation.report import trades_to_frame


DEFAULT_STOP_CLASS_BAND = 0.02  # |cf - actual| below this => NEUTRAL
DEFAULT_FURTHER_DROP_PCT = -0.05
AFTERMATH_HORIZONS = (1, 3, 5, 10, 20)


def _symbol_calendar(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    g = prices.loc[prices["Symbol"] == symbol].copy()
    if g.empty:
        return g
    g["Date"] = pd.to_datetime(g["Date"]).dt.normalize()
    return g.sort_values("Date").reset_index(drop=True)


def _price_on_or_after(cal: pd.DataFrame, day: pd.Timestamp, *, col: str = "Close") -> float | None:
    day = pd.Timestamp(day).normalize()
    hit = cal.loc[cal["Date"] >= day]
    if hit.empty:
        return None
    val = hit.iloc[0][col]
    return float(val) if pd.notna(val) else None


def _nth_session_after(cal: pd.DataFrame, day: pd.Timestamp, n: int) -> pd.Timestamp | None:
    """Return the date of the n-th trading session strictly after ``day``."""
    day = pd.Timestamp(day).normalize()
    after = cal.loc[cal["Date"] > day]
    if len(after) < n:
        return None
    return pd.Timestamp(after.iloc[n - 1]["Date"])


def analyze_stop_aftermath(
    trades: list[TradeRecord] | pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = AFTERMATH_HORIZONS,
    further_drop_pct: float = DEFAULT_FURTHER_DROP_PCT,
) -> dict[str, Any]:
    """Track prices after stop exits. Diagnostic only — never feeds entry/exit."""
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty:
        return {"n_stop_trades": 0}
    stops = tr.loc[tr["Exit Reason"] == "stop_loss"].copy()
    if stops.empty:
        return {"n_stop_trades": 0, "note": "no stop_loss exits"}

    per_trade: list[dict[str, Any]] = []
    horizon_rets: dict[int, list[float]] = {h: [] for h in horizons}
    recovered: dict[int, list[bool]] = {h: [] for h in horizons if h in (5, 10, 20)}
    further_drop: dict[int, list[bool]] = {h: [] for h in (5, 10, 20)}

    for _, row in stops.iterrows():
        symbol = str(row["Symbol"])
        exit_date = pd.Timestamp(row["Exit Date"]).normalize()
        entry_px = float(row["Entry Price"])
        exit_px = float(row["Exit Price"])
        cal = _symbol_calendar(prices, symbol)
        if cal.empty:
            continue

        path_rets: dict[str, float | None] = {}
        max_recovery = 0.0
        max_further_drop = 0.0
        recovered_by: dict[str, bool] = {}
        drop_by: dict[str, bool] = {}

        for h in horizons:
            dt = _nth_session_after(cal, exit_date, h)
            if dt is None:
                path_rets[f"d{h}"] = None
                continue
            px = _price_on_or_after(cal, dt, col="Close")
            if px is None or exit_px <= 0:
                path_rets[f"d{h}"] = None
                continue
            ret = float(px / exit_px - 1.0)
            path_rets[f"d{h}"] = ret
            horizon_rets[h].append(ret)
            max_recovery = max(max_recovery, ret)
            max_further_drop = min(max_further_drop, ret)

        # Path through each horizon for recovery / further drop
        for h in (5, 10, 20):
            after = cal.loc[cal["Date"] > exit_date].head(h)
            if after.empty:
                recovered_by[f"d{h}"] = False
                drop_by[f"d{h}"] = False
                recovered[h].append(False)
                further_drop[h].append(False)
                continue
            closes = after["Close"].astype(float)
            hit_entry = bool((closes >= entry_px).any()) if entry_px > 0 else False
            worse = bool(((closes / exit_px - 1.0) <= further_drop_pct).any()) if exit_px > 0 else False
            recovered_by[f"d{h}"] = hit_entry
            drop_by[f"d{h}"] = worse
            recovered[h].append(hit_entry)
            further_drop[h].append(worse)

        per_trade.append(
            {
                "symbol": symbol,
                "exit_date": str(exit_date.date()),
                "entry_price": entry_px,
                "exit_price": exit_px,
                "aftermath_returns": path_rets,
                "max_recovery_from_stop": float(max_recovery),
                "max_further_drop_from_stop": float(max_further_drop),
                "recovered_to_entry": recovered_by,
                "further_drop_ge_5pct": drop_by,
            }
        )

    summary = {
        "n_stop_trades": int(len(per_trade)),
        "mean_aftermath_return": {
            f"d{h}": float(np.mean(v)) if v else None for h, v in horizon_rets.items()
        },
        "median_aftermath_return": {
            f"d{h}": float(np.median(v)) if v else None for h, v in horizon_rets.items()
        },
        "recovery_to_entry_rate": {
            f"within_{h}d": float(np.mean(recovered[h])) if recovered[h] else None
            for h in (5, 10, 20)
        },
        "further_drop_ge_5pct_rate": {
            f"within_{h}d": float(np.mean(further_drop[h])) if further_drop[h] else None
            for h in (5, 10, 20)
        },
        "mean_max_recovery_from_stop": (
            float(np.mean([t["max_recovery_from_stop"] for t in per_trade])) if per_trade else None
        ),
        "mean_max_further_drop_from_stop": (
            float(np.mean([t["max_further_drop_from_stop"] for t in per_trade]))
            if per_trade
            else None
        ),
        "policy": (
            "Diagnostic only. Uses post-exit Close path; never used for Entry/Exit decisions."
        ),
        "trades": per_trade,
    }
    return summary


def analyze_stop_counterfactual(
    trades: list[TradeRecord] | pd.DataFrame,
    prices: pd.DataFrame,
    *,
    holding_period_days: int,
    neutral_band: float = DEFAULT_STOP_CLASS_BAND,
) -> dict[str, Any]:
    """Compare actual stop PnL vs holding to original holding period (diagnostic)."""
    tr = trades_to_frame(trades) if not isinstance(trades, pd.DataFrame) else trades.copy()
    if tr.empty:
        return {"n_stop_trades": 0}
    stops = tr.loc[tr["Exit Reason"] == "stop_loss"].copy()
    if stops.empty:
        return {"n_stop_trades": 0, "note": "no stop_loss exits"}

    rows: list[dict[str, Any]] = []
    for _, row in stops.iterrows():
        symbol = str(row["Symbol"])
        entry_date = pd.Timestamp(row["Entry Date"]).normalize()
        entry_px = float(row["Entry Price"])
        exit_px = float(row["Exit Price"])
        actual_ret = float(row["Return"])
        cal = _symbol_calendar(prices, symbol)
        if cal.empty or entry_px <= 0:
            continue

        # Counterfactual: exit at Open of the session after holding_period_days from entry
        # Engine counts entry day as holding_days=1 after bump; exit when holding_days >= N
        # → signal on session N, fill next open ≈ N-th session after entry's open day.
        hold_end = _nth_session_after(cal, entry_date, int(holding_period_days))
        if hold_end is None:
            continue
        # Next open after hold_end signal day
        fill_day = _nth_session_after(cal, hold_end, 1)
        if fill_day is None:
            fill_day = hold_end
        cf_px = _price_on_or_after(cal, fill_day, col="Open")
        if cf_px is None:
            cf_px = _price_on_or_after(cal, fill_day, col="Close")
        if cf_px is None:
            continue
        cf_ret = float(cf_px / entry_px - 1.0)
        diff = float(cf_ret - actual_ret)
        if diff > neutral_band:
            label = "BAD_STOP"
        elif diff < -neutral_band:
            label = "GOOD_STOP"
        else:
            label = "NEUTRAL_STOP"
        rows.append(
            {
                "symbol": symbol,
                "entry_date": str(entry_date.date()),
                "actual_stop_return": actual_ret,
                "counterfactual_holding_return": cf_ret,
                "difference_cf_minus_actual": diff,
                "classification": label,
                "counterfactual_exit_date": str(pd.Timestamp(fill_day).date()),
                "counterfactual_exit_price": float(cf_px),
            }
        )

    if not rows:
        return {"n_stop_trades": 0}

    labels = [r["classification"] for r in rows]
    n = len(rows)
    return {
        "n_stop_trades": n,
        "neutral_band": float(neutral_band),
        "holding_period_days": int(holding_period_days),
        "mean_actual_stop_return": float(np.mean([r["actual_stop_return"] for r in rows])),
        "mean_counterfactual_return": float(
            np.mean([r["counterfactual_holding_return"] for r in rows])
        ),
        "mean_difference_cf_minus_actual": float(
            np.mean([r["difference_cf_minus_actual"] for r in rows])
        ),
        "classification_rates": {
            "GOOD_STOP": float(labels.count("GOOD_STOP") / n),
            "BAD_STOP": float(labels.count("BAD_STOP") / n),
            "NEUTRAL_STOP": float(labels.count("NEUTRAL_STOP") / n),
        },
        "classification_counts": {
            "GOOD_STOP": int(labels.count("GOOD_STOP")),
            "BAD_STOP": int(labels.count("BAD_STOP")),
            "NEUTRAL_STOP": int(labels.count("NEUTRAL_STOP")),
        },
        "interpretation": {
            "GOOD_STOP": "Holding to original period would have been worse (stop helped).",
            "BAD_STOP": "Holding to original period would have been better (premature stop).",
            "NEUTRAL_STOP": f"|cf - actual| <= {neutral_band:.2%}",
        },
        "policy": (
            "Diagnostic only. Counterfactual uses point-in-time Open after holding window; "
            "never used for portfolio decisions."
        ),
        "trades": rows,
    }


def pick_stop_setting(
    rows: list[dict[str, Any]],
    *,
    baseline_name: str = "STOP_5",
    max_dd_worsen_limit: float = 0.05,
) -> dict[str, Any]:
    """Pick stop setting with MaxDD guardrail vs baseline STOP_5.

    Reject candidates whose MaxDD is worse than baseline by more than
    ``max_dd_worsen_limit`` (absolute drawdown points), unless no alternative remains.
    """
    by_name = {r["name"]: r for r in rows}
    baseline = by_name.get(baseline_name)
    eligible = []
    rejected = []
    for r in rows:
        if baseline and r.get("max_dd") is not None and baseline.get("max_dd") is not None:
            # more negative max_dd = worse
            worsen = float(baseline["max_dd"]) - float(r["max_dd"])
            if worsen > max_dd_worsen_limit:
                rejected.append({"name": r["name"], "max_dd_worsen_vs_baseline": worsen})
                continue
        eligible.append(r)
    pool = eligible if eligible else rows

    def key(r: dict[str, Any]) -> tuple:
        return (
            float(r["net_return"]) if r.get("net_return") is not None else -1e9,
            float(r["positive_fold_ratio"]) if r.get("positive_fold_ratio") is not None else -1e9,
            -float(r.get("fold_stability", {}).get("std") or 1e9),
            float(r["sharpe"]) if r.get("sharpe") is not None else -1e9,
            float(r["max_dd"]) if r.get("max_dd") is not None else -1e9,
            float(r["profit_factor"]) if r.get("profit_factor") not in (None, float("inf")) else 1e9,
            -float(r["trades"]) if r.get("trades") is not None else -1e9,
            -float(r["total_cost"]) if r.get("total_cost") is not None else -1e9,
        )

    selected = max(pool, key=key)
    return {
        "selected": selected["name"],
        "selected_row": selected,
        "rejected_for_maxdd": rejected,
        "max_dd_worsen_limit": max_dd_worsen_limit,
        "baseline": baseline_name,
    }
