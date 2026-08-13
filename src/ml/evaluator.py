"""Evaluation metrics for binary up/down prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def class_distribution(y: pd.Series | np.ndarray) -> dict[str, Any]:
    """Return counts and ratios for target_up classes."""
    values = pd.Series(np.asarray(y).astype(int))
    counts = values.value_counts().to_dict()
    total = int(len(values))
    ratio = {
        str(cls): (counts.get(cls, 0) / total if total else 0.0) for cls in (0, 1)
    }
    return {
        "counts": {str(cls): int(counts.get(cls, 0)) for cls in (0, 1)},
        "ratios": ratio,
        "total": total,
    }


def evaluate_binary(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute classification metrics for a binary target."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, zero_division=0)
        ),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "support": int(len(y_true_arr)),
    }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        metrics["probability_up_min"] = float(np.min(y_prob_arr))
        metrics["probability_up_max"] = float(np.max(y_prob_arr))
        metrics["probability_up_mean"] = float(np.mean(y_prob_arr))
        # ROC-AUC requires both classes present.
        if len(np.unique(y_true_arr)) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        else:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def evaluate_by_symbol(
    meta: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Compute test metrics for each Symbol independently."""
    frame = meta.copy()
    frame["y_true"] = np.asarray(y_true).astype(int)
    frame["y_pred"] = np.asarray(y_pred).astype(int)
    frame["y_prob"] = np.asarray(y_prob, dtype=float)

    per_symbol: dict[str, dict[str, Any]] = {}
    for symbol, group in frame.groupby("Symbol", sort=True):
        per_symbol[str(symbol)] = evaluate_binary(
            group["y_true"],
            group["y_pred"].to_numpy(),
            group["y_prob"].to_numpy(),
        )
    return per_symbol
