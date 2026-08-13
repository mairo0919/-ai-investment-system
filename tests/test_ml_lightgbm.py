"""Tests for Phase 3B LightGBM training."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.config.settings import Settings
from src.features.indicators import FEATURE_COLUMNS
from src.ml.dataset import MODEL_FEATURE_COLUMNS
from src.ml.lightgbm_model import (
    DEFAULT_LGBM_PARAMS,
    fit_lightgbm,
    gain_feature_importance,
    predict_lightgbm,
)
from src.ml.trainer import ModelTrainer, load_model


def _feature_matrix(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.normal(size=(n, len(MODEL_FEATURE_COLUMNS))),
        columns=list(MODEL_FEATURE_COLUMNS),
    )
    y = (x["return_1d"] + 0.3 * x["rsi_14"] > 0).astype(int)
    return x, y


def _write_processed(tmp_path: Path, symbol: str, n: int = 150, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    data = {
        "Date": dates,
        "Symbol": symbol,
        "Open": close,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Adj Close": close,
        "Volume": rng.integers(1000, 5000, size=n).astype(float),
    }
    for column in FEATURE_COLUMNS:
        data[column] = rng.normal(0, 1, size=n)
    data["return_1d"] = pd.Series(close).pct_change().fillna(0.0).to_numpy()
    frame = pd.DataFrame(data)
    out = tmp_path / "processed" / f"{symbol.replace('.', '_')}_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)


def test_lightgbm_training_and_proba_range() -> None:
    x_train, y_train = _feature_matrix(n=160, seed=1)
    x_valid, y_valid = _feature_matrix(n=40, seed=2)
    result = fit_lightgbm(
        x_train,
        y_train,
        x_valid,
        y_valid,
        params={**DEFAULT_LGBM_PARAMS, "n_estimators": 40},
        early_stopping_rounds=10,
    )
    assert isinstance(result.model, LGBMClassifier)
    assert x_train.shape[1] == 26
    _, y_prob = predict_lightgbm(result.model, x_valid)
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))


def test_lightgbm_feature_count_and_importance_are_26() -> None:
    x_train, y_train = _feature_matrix(n=100, seed=3)
    x_valid, y_valid = _feature_matrix(n=30, seed=4)
    result = fit_lightgbm(
        x_train,
        y_train,
        x_valid,
        y_valid,
        params={**DEFAULT_LGBM_PARAMS, "n_estimators": 30},
        early_stopping_rounds=5,
    )
    assert result.model.n_features_in_ == 26
    importance = gain_feature_importance(result.model)
    assert len(importance) == 26
    assert set(importance) == set(MODEL_FEATURE_COLUMNS)


def test_fit_lightgbm_api_excludes_test_partition() -> None:
    signature = inspect.signature(fit_lightgbm)
    assert "x_test" not in signature.parameters
    assert "y_test" not in signature.parameters
    assert {"x_train", "y_train", "x_valid", "y_valid"}.issubset(signature.parameters)


def test_early_stopping_receives_validation_only() -> None:
    x_train, y_train = _feature_matrix(n=80, seed=5)
    x_valid, y_valid = _feature_matrix(n=25, seed=6)
    captured: dict[str, object] = {}

    original_fit = LGBMClassifier.fit

    def wrapped_fit(self: LGBMClassifier, x: pd.DataFrame, y: pd.Series, **kwargs):  # type: ignore[no-untyped-def]
        captured["kwargs"] = kwargs
        captured["x"] = x
        return original_fit(self, x, y, **kwargs)

    with patch.object(LGBMClassifier, "fit", wrapped_fit):
        fit_lightgbm(
            x_train,
            y_train,
            x_valid,
            y_valid,
            params={**DEFAULT_LGBM_PARAMS, "n_estimators": 20},
            early_stopping_rounds=5,
        )

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    pd.testing.assert_frame_equal(captured["x"], x_train)  # type: ignore[arg-type]
    if "eval_X" in kwargs:
        pd.testing.assert_frame_equal(kwargs["eval_X"], x_valid)
        pd.testing.assert_series_equal(kwargs["eval_y"], y_valid)
    else:
        eval_set = kwargs["eval_set"]
        pd.testing.assert_frame_equal(eval_set[0][0], x_valid)
        pd.testing.assert_series_equal(eval_set[0][1], y_valid)


def test_lightgbm_save_load_and_comparison_report(tmp_path: Path) -> None:
    _write_processed(tmp_path, "7203.T", n=150, seed=11)
    _write_processed(tmp_path, "5333.T", n=150, seed=12)
    settings = Settings(
        tickers=("7203.T", "5333.T"),
        processed_data_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
    )
    report = ModelTrainer(settings).run()

    lgbm_path = Path(report["artifacts"]["lightgbm"]["model_path"])
    assert lgbm_path.exists()
    assert Path(report["artifacts"]["comparison_report_path"]).exists()
    assert Path(report["artifacts"]["logistic_regression"]["model_path"]).exists()

    loaded = load_model(lgbm_path)
    sample = pd.DataFrame(
        np.zeros((3, len(MODEL_FEATURE_COLUMNS))),
        columns=list(MODEL_FEATURE_COLUMNS),
    )
    proba = loaded.predict_proba(sample)[:, 1]
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert set(report["comparison"]["test"]) >= {
        "majority_class",
        "previous_return",
        "logistic_regression",
        "lightgbm",
    }
