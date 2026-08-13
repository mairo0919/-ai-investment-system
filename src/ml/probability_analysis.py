"""Probability-to-future-return relationship analysis (not a backtest)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _bucket_stats(future_return: pd.Series) -> dict[str, Any]:
    values = future_return.astype(float)
    return {
        "count": int(len(values)),
        "mean_future_return": float(values.mean()) if len(values) else None,
        "median_future_return": float(values.median()) if len(values) else None,
        "positive_rate": float((values > 0).mean()) if len(values) else None,
    }


def probability_decile_analysis(
    probability_up: np.ndarray | pd.Series,
    future_return: np.ndarray | pd.Series,
    *,
    n_deciles: int = 10,
) -> list[dict[str, Any]]:
    """Split scores into deciles and summarize realized future returns."""
    frame = pd.DataFrame(
        {
            "probability_up": np.asarray(probability_up, dtype=float),
            "future_return": np.asarray(future_return, dtype=float),
        }
    ).dropna()
    if frame.empty:
        return []

    # qcut can fail on too many duplicate probabilities; rank fallback keeps order.
    try:
        frame["decile"] = pd.qcut(
            frame["probability_up"],
            q=n_deciles,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        frame["decile"] = (
            frame["probability_up"].rank(method="first") - 1
        ) * n_deciles // max(len(frame), 1)
        frame["decile"] = frame["decile"].clip(0, n_deciles - 1).astype(int)

    rows: list[dict[str, Any]] = []
    for decile, group in frame.groupby("decile", sort=True):
        stats = _bucket_stats(group["future_return"])
        rows.append(
            {
                "decile": int(decile) + 1,  # 1 = lowest prob, 10 = highest when 10 bins
                "probability_up_min": float(group["probability_up"].min()),
                "probability_up_max": float(group["probability_up"].max()),
                "probability_up_mean": float(group["probability_up"].mean()),
                **stats,
            }
        )
    return rows


def top_probability_analysis(
    probability_up: np.ndarray | pd.Series,
    future_return: np.ndarray | pd.Series,
    *,
    fractions: tuple[float, ...] = (0.10, 0.20, 0.30),
) -> dict[str, Any]:
    """Summarize realized returns for top probability fractions vs all rows."""
    frame = pd.DataFrame(
        {
            "probability_up": np.asarray(probability_up, dtype=float),
            "future_return": np.asarray(future_return, dtype=float),
        }
    ).dropna()
    overall = _bucket_stats(frame["future_return"])
    result: dict[str, Any] = {"overall": overall, "top_fractions": {}}

    if frame.empty:
        return result

    ranked = frame.sort_values("probability_up", ascending=False)
    for fraction in fractions:
        n = max(1, int(round(len(ranked) * fraction)))
        top = ranked.iloc[:n]
        stats = _bucket_stats(top["future_return"])
        mean_all = overall["mean_future_return"]
        mean_top = stats["mean_future_return"]
        stats["mean_future_return_vs_all"] = (
            None
            if mean_all is None or mean_top is None
            else float(mean_top - mean_all)
        )
        result["top_fractions"][f"top_{int(fraction * 100)}pct"] = stats
    return result
