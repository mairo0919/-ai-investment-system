"""LightGBM Learning-to-Rank (LGBMRanker / lambdarank)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping, log_evaluation

from src.core.exceptions import TrainingError

logger = logging.getLogger(__name__)

DEFAULT_RANKER_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

DEFAULT_EVAL_AT: tuple[int, ...] = (5, 10)

DEFAULT_EARLY_STOPPING_ROUNDS = 40


@dataclass(frozen=True)
class RankerFitResult:
    model: LGBMRanker
    best_iteration: int | None
    params: dict[str, Any]


def build_lgbm_ranker(params: dict[str, Any] | None = None) -> LGBMRanker:
    merged = dict(DEFAULT_RANKER_PARAMS)
    if params:
        merged.update(params)
    return LGBMRanker(**merged)


def fit_lgbm_ranker(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    group_train: np.ndarray,
    *,
    x_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | np.ndarray | None = None,
    group_valid: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    label_gain: tuple[float, ...] | list[float] | str | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> RankerFitResult:
    """Fit LGBMRanker with optional validation early stopping.

    ``label_gain`` must cover every integer relevance present in ``y``
    (length > max(label)). Pass an explicit sequence; do not rely on defaults
    when using >31 label levels.
    """
    if int(np.sum(group_train)) != len(x_train):
        raise TrainingError(
            f"Train group mismatch: sum={int(np.sum(group_train))} rows={len(x_train)}"
        )
    # LightGBM handles NaN natively (used for small-sector / sparse macro cells).

    y_arr = np.asarray(y_train).astype(int)
    merged_params = dict(params or {})
    if label_gain is not None:
        if isinstance(label_gain, str):
            gain_param = label_gain
            gain_len = len(label_gain.split(","))
        else:
            gain_param = ",".join(str(float(g)) for g in label_gain)
            gain_len = len(tuple(label_gain))
        max_y = int(y_arr.max()) if len(y_arr) else -1
        if max_y >= gain_len:
            raise TrainingError(
                f"label_gain length {gain_len} must be > max label {max_y}"
            )
        merged_params["label_gain"] = gain_param

    model = build_lgbm_ranker(merged_params)
    fit_kwargs: dict[str, Any] = {
        "X": x_train,
        "y": y_arr,
        "group": np.asarray(group_train, dtype=np.int32),
        "eval_at": list(DEFAULT_EVAL_AT),
    }

    callbacks = [log_evaluation(period=0)]
    if (
        x_valid is not None
        and y_valid is not None
        and group_valid is not None
        and len(x_valid) > 0
    ):
        if int(np.sum(group_valid)) != len(x_valid):
            raise TrainingError(
                f"Valid group mismatch: sum={int(np.sum(group_valid))} rows={len(x_valid)}"
            )
        fit_kwargs["eval_X"] = x_valid
        fit_kwargs["eval_y"] = np.asarray(y_valid).astype(int)
        fit_kwargs["eval_group"] = [np.asarray(group_valid, dtype=np.int32)]
        callbacks.append(early_stopping(early_stopping_rounds, verbose=False))

    fit_kwargs["callbacks"] = callbacks
    model.fit(**fit_kwargs)
    best = getattr(model, "best_iteration_", None)
    logger.info(
        "LGBMRanker fitted rows=%d groups=%d best_iteration=%s",
        len(x_train),
        len(group_train),
        best,
    )
    return RankerFitResult(model=model, best_iteration=best, params=dict(model.get_params()))


def predict_rank_scores(model: LGBMRanker, x: pd.DataFrame) -> np.ndarray:
    """Predict ranking scores (higher = more relevant)."""
    return np.asarray(model.predict(x), dtype=float)
