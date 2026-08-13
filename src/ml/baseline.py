"""Simple baselines used to sanity-check the Phase 3 ML model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselinePrediction:
    """Baseline hard labels and optional probability estimates."""

    name: str
    y_pred: np.ndarray
    y_prob: np.ndarray


def majority_class_baseline(
    y_train: pd.Series | np.ndarray,
    n_samples: int,
) -> BaselinePrediction:
    """Always predict the majority class observed in the training labels."""
    values = np.asarray(y_train, dtype=int)
    if values.size == 0:
        raise ValueError("y_train is empty")
    # Deterministic tie-break: prefer class 1 when counts are equal.
    counts = np.bincount(values, minlength=2)
    majority = int(np.argmax(counts))
    y_pred = np.full(n_samples, majority, dtype=int)
    y_prob = np.full(n_samples, float(majority), dtype=float)
    return BaselinePrediction(name="majority_class", y_pred=y_pred, y_prob=y_prob)


def previous_return_baseline(return_1d: pd.Series | np.ndarray) -> BaselinePrediction:
    """Predict up tomorrow if today's return_1d was positive."""
    values = np.asarray(return_1d, dtype=float)
    y_pred = (values > 0).astype(int)
    y_prob = y_pred.astype(float)
    return BaselinePrediction(name="previous_return", y_pred=y_pred, y_prob=y_prob)
