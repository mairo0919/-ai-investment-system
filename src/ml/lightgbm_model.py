"""LightGBM-specific training helpers (Phase 3B).

Uses raw engineered features (no StandardScaler). Early stopping may use the
validation split only — never the test split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from src.core.exceptions import TrainingError

logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

DEFAULT_EARLY_STOPPING_ROUNDS = 50


@dataclass(frozen=True)
class LightGBMFitResult:
    """Fitted LightGBM classifier plus training diagnostics."""

    model: LGBMClassifier
    best_iteration: int | None
    params: dict[str, Any]


def build_lightgbm_classifier(params: dict[str, Any] | None = None) -> LGBMClassifier:
    """Create an ``LGBMClassifier`` with the Phase 3B default hyperparameters."""
    merged = dict(DEFAULT_LGBM_PARAMS)
    if params:
        merged.update(params)
    return LGBMClassifier(**merged)


def fit_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> LightGBMFitResult:
    """Fit LightGBM on train data with early stopping on validation only.

    Args:
        x_train: Training features (must be the 26 model columns).
        y_train: Training labels.
        x_valid: Validation features used only for early stopping / monitoring.
        y_valid: Validation labels.
        params: Optional overrides for ``DEFAULT_LGBM_PARAMS``.
        early_stopping_rounds: Patience for validation AUC early stopping.
    """
    _validate_feature_frame(x_train, name="x_train")
    _validate_feature_frame(x_valid, name="x_valid")
    if list(x_train.columns) != list(x_valid.columns):
        raise TrainingError("x_train and x_valid feature columns do not match")

    model = build_lightgbm_classifier(params)
    used_params = dict(DEFAULT_LGBM_PARAMS)
    if params:
        used_params.update(params)

    logger.info(
        "Fitting LightGBM on train only (rows=%d) with validation early stopping "
        "(valid_rows=%d, metric=auc, patience=%d)",
        len(x_train),
        len(x_valid),
        early_stopping_rounds,
    )

    try:
        # Prefer modern LightGBM sklearn API: eval_X/eval_y (not deprecated eval_set).
        model.fit(
            x_train,
            y_train,
            eval_X=x_valid,
            eval_y=y_valid,
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                log_evaluation(period=0),
            ],
        )
    except TypeError:
        # Fallback for older lightgbm builds that still require eval_set.
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                log_evaluation(period=0),
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"LightGBM fitting failed: {exc}") from exc

    best_iteration = getattr(model, "best_iteration_", None)
    logger.info("LightGBM best_iteration=%s", best_iteration)
    return LightGBMFitResult(model=model, best_iteration=best_iteration, params=used_params)


def predict_lightgbm(model: LGBMClassifier, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return hard labels and probability_up from a fitted LightGBM model."""
    _validate_feature_frame(features, name="features")
    try:
        y_pred = model.predict(features)
        y_prob = model.predict_proba(features)[:, 1]
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"LightGBM prediction failed: {exc}") from exc

    y_prob = np.asarray(y_prob, dtype=float)
    if not np.all((y_prob >= 0.0) & (y_prob <= 1.0)):
        raise TrainingError("LightGBM probability_up contains values outside [0, 1]")
    return np.asarray(y_pred, dtype=int), y_prob


def gain_feature_importance(model: LGBMClassifier) -> dict[str, float]:
    """Return gain-based feature importance for all model feature columns."""
    try:
        importances = model.booster_.feature_importance(importance_type="gain")
        names = list(model.booster_.feature_name())
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Failed to extract LightGBM feature importance: {exc}") from exc

    if len(names) == 0:
        raise TrainingError("LightGBM returned empty feature importance")
    return {name: float(score) for name, score in zip(names, importances, strict=True)}


def top_feature_importance(importance: dict[str, float], n: int = 10) -> list[dict[str, float | str]]:
    """Return the top-N features by gain importance."""
    ranked = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "gain": score} for name, score in ranked[:n]]


def _validate_feature_frame(frame: pd.DataFrame, *, name: str) -> None:
    if frame is None or frame.empty:
        raise TrainingError(f"{name} is empty")
    if frame.shape[1] < 1:
        raise TrainingError(f"{name} has no feature columns")


def lightgbm_version() -> str:
    """Return the installed lightgbm package version."""
    return str(lgb.__version__)
