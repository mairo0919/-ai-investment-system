"""Learning-to-Rank evaluation metrics (Country-internal; no trading)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ml.cross_section import cross_sectional_metrics_by_date, summarize_daily_metrics
from src.ml.stability_diagnostics import (
    daily_score_dispersion,
    quintile_bucket_analysis,
    summarize_dispersion,
    top_bottom_spread,
)


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted cumulative gain for a relevance list already in predicted order."""
    if k <= 0 or len(relevances) == 0:
        return 0.0
    rel = np.asarray(relevances[:k], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum((2.0**rel - 1.0) * discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@k for one ranking group."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    ranked = y_true[order]
    ideal = np.sort(y_true)[::-1]
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg <= 0:
        return float("nan")
    return dcg_at_k(ranked, k) / ideal_dcg


def ndcg_by_groups(
    frame: pd.DataFrame,
    group_sizes: np.ndarray,
    *,
    relevance_col: str = "relevance",
    score_col: str = "score",
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float | None]:
    """Mean NDCG@k across contiguous ranking groups."""
    start = 0
    values: dict[int, list[float]] = {k: [] for k in ks}
    for size in group_sizes:
        size_i = int(size)
        block = frame.iloc[start : start + size_i]
        start += size_i
        y_true = block[relevance_col].to_numpy()
        y_score = block[score_col].to_numpy()
        for k in ks:
            val = ndcg_at_k(y_true, y_score, k=min(k, size_i))
            if not np.isnan(val):
                values[k].append(float(val))
    out: dict[str, float | None] = {}
    for k in ks:
        arr = values[k]
        out[f"ndcg_at_{k}"] = float(np.mean(arr)) if arr else None
        out[f"ndcg_at_{k}_n_groups"] = float(len(arr))
    return out


def evaluate_country_ranking(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
    relevance_col: str = "relevance",
    date_col: str = "Date",
    region_col: str = "Region",
    ndcg_ks: tuple[int, ...] = (5, 10),
    tie_warning_ratio: float = 0.5,
) -> dict[str, Any]:
    """Country-internal ranking metrics for one validation frame (single region)."""
    if frame.empty:
        return {"n_rows": 0}

    # Prefer Date×Country groups when multiple regions are present.
    multi_region = (
        region_col in frame.columns and int(frame[region_col].nunique(dropna=False)) > 1
    )
    sort_keys = [date_col]
    if multi_region:
        sort_keys.append(region_col)
    if "Symbol" in frame.columns:
        sort_keys.append("Symbol")
    sorted_frame = frame.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    groupby_keys = [date_col, region_col] if multi_region else [date_col]
    group_sizes = (
        sorted_frame.groupby(groupby_keys, sort=False).size().to_numpy(dtype=np.int32)
    )
    ndcg = ndcg_by_groups(
        sorted_frame,
        group_sizes,
        relevance_col=relevance_col,
        score_col=score_col,
        ks=ndcg_ks,
    )

    # Cross-sectional IC / TopK are always Country-internal (never Global pool).
    daily_parts = []
    for _, group in sorted_frame.groupby(
        [date_col, region_col] if region_col in sorted_frame.columns else [date_col],
        sort=True,
    ):
        daily_parts.append(
            cross_sectional_metrics_by_date(
                group,
                score_col=score_col,
                return_col=return_col,
                top_fractions=(0.10, 0.20),
                min_names=3,
            )
        )
    daily = (
        pd.concat(daily_parts, ignore_index=True)
        if daily_parts
        else pd.DataFrame()
    )
    ranking = summarize_daily_metrics(daily)
    quintiles = quintile_bucket_analysis(
        sorted_frame, score_col=score_col, return_col=return_col, date_col=date_col
    )
    q = quintiles.get("quintiles", {})
    q5 = (q.get("Q5") or {}).get("mean_future_return")
    q1 = (q.get("Q1") or {}).get("mean_future_return")
    q5_q1 = None if q5 is None or q1 is None else float(q5 - q1)

    dispersion_daily = daily_score_dispersion(
        sorted_frame, score_col=score_col, date_col=date_col
    )
    dispersion = summarize_dispersion(dispersion_daily)
    mean_tie = dispersion.get("mean_tie_ratio")
    warnings: list[str] = []
    if mean_tie is not None and mean_tie >= tie_warning_ratio:
        warnings.append(
            f"high_tie_ratio: mean_tie_ratio={mean_tie:.3f} >= {tie_warning_ratio}"
        )
    if dispersion.get("low_dispersion_day_ratio", 0) and (
        dispersion["low_dispersion_day_ratio"] >= 0.5
    ):
        warnings.append(
            f"frequent_low_dispersion_days: ratio={dispersion['low_dispersion_day_ratio']:.3f}"
        )

    spread = top_bottom_spread(
        sorted_frame,
        score_col=score_col,
        return_col=return_col,
        date_col=date_col,
        fraction=0.20,
    )

    return {
        "n_rows": int(len(sorted_frame)),
        "n_symbols": int(sorted_frame["Symbol"].nunique())
        if "Symbol" in sorted_frame.columns
        else None,
        "n_groups": int(len(group_sizes)),
        "region": str(sorted_frame[region_col].iloc[0])
        if region_col in sorted_frame.columns
        else None,
        "ndcg": ndcg,
        "ranking": ranking,
        "quintiles": quintiles,
        "q5_q1_spread": q5_q1,
        "top20_bottom20_spread": spread,
        "score_dispersion": dispersion,
        "warnings": warnings,
    }
