"""Daily cross-sectional ranking metrics (evaluation only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ml.ranking import information_coefficient, top_k_average_return


def cross_sectional_metrics_by_date(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
    min_names: int = 5,
) -> pd.DataFrame:
    """Compute IC and top-k excess return for each Date cross-section."""
    required = {"Date", score_col, return_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns for cross-section metrics: {missing}")

    rows: list[dict[str, Any]] = []
    for dt, group in frame.groupby(pd.to_datetime(frame["Date"]), sort=True):
        clean = group.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[score_col, return_col]
        )
        if len(clean) < min_names:
            continue
        row: dict[str, Any] = {
            "Date": pd.Timestamp(dt),
            "n_names": int(len(clean)),
            "ic": information_coefficient(
                clean, score_col=score_col, return_col=return_col
            ),
        }
        for frac in top_fractions:
            stats = top_k_average_return(
                clean,
                score_col=score_col,
                return_col=return_col,
                fraction=frac,
            )
            prefix = f"top_{int(frac * 100)}pct"
            row[f"{prefix}_mean_future_return"] = stats["mean_future_return"]
            row[f"{prefix}_median_future_return"] = stats["median_future_return"]
            row[f"{prefix}_excess_return"] = stats["excess_return"]
            row[f"{prefix}_positive_rate"] = float(
                (clean.nlargest(max(1, int(round(len(clean) * frac))), score_col)[return_col] > 0).mean()
            )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def summarize_daily_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    """Aggregate daily cross-section metrics into fold-level summaries."""
    if daily.empty:
        return {"n_days": 0}

    def _series_stats(values: pd.Series) -> dict[str, float | None]:
        clean = values.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return {
                "mean": None,
                "median": None,
                "std": None,
                "positive_ratio": None,
            }
        return {
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std(ddof=0)),
            "positive_ratio": float((clean > 0).mean()),
        }

    summary: dict[str, Any] = {"n_days": int(len(daily)), "ic": _series_stats(daily["ic"])}
    for col in daily.columns:
        if col.endswith("_excess_return") or col.endswith("_mean_future_return"):
            summary[col] = _series_stats(daily[col])
    return summary
