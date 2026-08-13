"""Momentum ranking baselines for Learning-to-Rank comparisons."""

from __future__ import annotations

import pandas as pd

from src.core.exceptions import TrainingError


def momentum_scores(
    frame: pd.DataFrame,
    *,
    feature_col: str,
) -> pd.Series:
    """Use a past-return feature as ranking score (higher = stronger momentum).

    Does not peek at future returns.
    """
    if feature_col not in frame.columns:
        raise TrainingError(f"Momentum baseline missing feature column: {feature_col}")
    return frame[feature_col].astype(float)


MOMENTUM_BASELINES: tuple[tuple[str, str], ...] = (
    ("momentum_5d", "return_5d"),
    ("momentum_20d", "return_20d"),
)
