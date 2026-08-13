"""Cross-sectional ranking evaluation metrics (no trading / no order generation).

These helpers answer: "Did higher scores correspond to higher future returns?"
Grouping keys enable Global / Country / Market / Sector evaluation later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    score_col: str,
    return_col: str,
) -> pd.DataFrame:
    required = {score_col, return_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns for ranking evaluation: {missing}")
    out = frame.loc[:, [score_col, return_col]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise ValueError("No finite rows available for ranking evaluation")
    return out


def information_coefficient(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
) -> float:
    """Spearman rank IC between score and future return."""
    clean = _prepare_frame(frame, score_col=score_col, return_col=return_col)
    corr = clean[score_col].corr(clean[return_col], method="spearman")
    if corr is None or np.isnan(corr):
        return float("nan")
    return float(corr)


def rank_correlation(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
) -> float:
    """Alias of information_coefficient for explicit naming in reports."""
    return information_coefficient(frame, score_col=score_col, return_col=return_col)


def top_k_average_return(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
    fraction: float = 0.10,
) -> dict[str, float]:
    """Average/median future return of the top score fraction."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    clean = _prepare_frame(frame, score_col=score_col, return_col=return_col)
    n = max(1, int(round(len(clean) * fraction)))
    top = clean.nlargest(n, score_col)
    universe_mean = float(clean[return_col].mean())
    top_mean = float(top[return_col].mean())
    return {
        "count": float(n),
        "mean_future_return": top_mean,
        "median_future_return": float(top[return_col].median()),
        "universe_mean_future_return": universe_mean,
        "excess_return": top_mean - universe_mean,
    }


def evaluate_ranking(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
    group_col: str | None = None,
    top_fractions: tuple[float, ...] = (0.10, 0.20),
) -> dict[str, Any]:
    """Evaluate ranking quality globally or within a metadata group.

    Args:
        frame: Must include score/return columns. Optional group column such as
            ``country``, ``exchange``, or ``sector`` for sliced rankings.
        group_col: When set, metrics are computed per group as well as overall.
    """
    overall = {
        "information_coefficient": information_coefficient(
            frame, score_col=score_col, return_col=return_col
        ),
        "rank_correlation": rank_correlation(
            frame, score_col=score_col, return_col=return_col
        ),
        "top_fractions": {
            f"top_{int(frac * 100)}pct": top_k_average_return(
                frame,
                score_col=score_col,
                return_col=return_col,
                fraction=frac,
            )
            for frac in top_fractions
        },
        "n_rows": int(
            _prepare_frame(frame, score_col=score_col, return_col=return_col).shape[0]
        ),
    }
    result: dict[str, Any] = {"scope": "global", "metrics": overall}

    if group_col is None:
        return result
    if group_col not in frame.columns:
        raise ValueError(f"group_col '{group_col}' not found in frame")

    by_group: dict[str, Any] = {}
    for key, group in frame.groupby(group_col, dropna=False):
        label = "null" if pd.isna(key) else str(key)
        try:
            by_group[label] = {
                "information_coefficient": information_coefficient(
                    group, score_col=score_col, return_col=return_col
                ),
                "top_fractions": {
                    f"top_{int(frac * 100)}pct": top_k_average_return(
                        group,
                        score_col=score_col,
                        return_col=return_col,
                        fraction=frac,
                    )
                    for frac in top_fractions
                },
                "n_rows": int(
                    _prepare_frame(
                        group, score_col=score_col, return_col=return_col
                    ).shape[0]
                ),
            }
        except ValueError:
            by_group[label] = {"error": "insufficient_rows"}

    result["by_group"] = {"group_col": group_col, "groups": by_group}
    return result
