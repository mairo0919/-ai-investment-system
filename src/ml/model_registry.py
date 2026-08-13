"""Shared model orchestration helpers for Phase 3A / 3B comparison."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import TrainingError
from src.ml.dataset import MLDataset
from src.ml.evaluator import evaluate_binary, evaluate_by_symbol


class ProbabilisticModel(Protocol):
    """Minimal interface shared by LogisticRegression Pipeline and LightGBM."""

    def predict(self, x: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


def build_logistic_pipeline(
    *,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """Create StandardScaler + LogisticRegression (Phase 3A)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def predict_binary(
    model: ProbabilisticModel,
    dataset: MLDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and probability_up for one dataset partition."""
    try:
        y_pred = model.predict(dataset.X)
        y_prob = model.predict_proba(dataset.X)[:, 1]
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Prediction failed: {exc}") from exc

    y_prob_arr = np.asarray(y_prob, dtype=float)
    if not np.all((y_prob_arr >= 0.0) & (y_prob_arr <= 1.0)):
        raise TrainingError("probability_up contains values outside [0, 1]")
    return np.asarray(y_pred, dtype=int), y_prob_arr


def evaluate_model_partitions(
    model: ProbabilisticModel,
    splits_train: MLDataset,
    splits_valid: MLDataset,
    splits_test: MLDataset,
) -> dict[str, Any]:
    """Evaluate a fitted model on train / validation / test + per-symbol test."""
    train_pred, train_prob = predict_binary(model, splits_train)
    valid_pred, valid_prob = predict_binary(model, splits_valid)
    test_pred, test_prob = predict_binary(model, splits_test)

    model_metrics = {
        "train": evaluate_binary(splits_train.y, train_pred, train_prob),
        "validation": evaluate_binary(splits_valid.y, valid_pred, valid_prob),
        "test": evaluate_binary(splits_test.y, test_pred, test_prob),
    }
    per_symbol = {
        "test": evaluate_by_symbol(splits_test.meta, splits_test.y, test_pred, test_prob)
    }
    validation_auc = model_metrics["validation"].get("roc_auc")
    test_auc = model_metrics["test"].get("roc_auc")
    auc_gap = None
    if validation_auc is not None and test_auc is not None:
        auc_gap = float(validation_auc) - float(test_auc)

    return {
        "model_metrics": model_metrics,
        "per_symbol_metrics": per_symbol,
        "validation_test_auc_gap": auc_gap,
    }


def extract_test_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize metric dict for comparison tables."""
    return {
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
    }
