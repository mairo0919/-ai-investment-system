"""Tests for Phase 3 model training / persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import Settings
from src.features.indicators import FEATURE_COLUMNS
from src.ml.dataset import MODEL_FEATURE_COLUMNS
from src.ml.model_registry import build_logistic_pipeline
from src.ml.trainer import ModelTrainer, load_model


def _write_processed(tmp_path: Path, symbol: str, n: int = 120, seed: int = 0) -> None:
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


def test_scaler_is_inside_pipeline_and_fits_with_model() -> None:
    pipeline = build_logistic_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "model"]

    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        rng.normal(size=(50, len(MODEL_FEATURE_COLUMNS))),
        columns=list(MODEL_FEATURE_COLUMNS),
    )
    y = pd.Series((rng.random(50) > 0.5).astype(int))

    assert not hasattr(pipeline.named_steps["scaler"], "mean_")
    pipeline.fit(x, y)
    assert hasattr(pipeline.named_steps["scaler"], "mean_")
    assert pipeline.named_steps["scaler"].n_features_in_ == len(MODEL_FEATURE_COLUMNS)


def test_training_predict_proba_and_persistence(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_processed(tmp_path, "7203.T", n=150, seed=1)
    _write_processed(tmp_path, "5333.T", n=150, seed=2)

    settings = Settings(
        tickers=("7203.T", "5333.T"),
        processed_data_dir=processed,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
    )
    report = ModelTrainer(settings).run()

    logistic_metrics = report["models"]["logistic_regression"]["model_metrics"]["test"]
    assert 0.0 <= logistic_metrics["accuracy"] <= 1.0
    assert (
        logistic_metrics["roc_auc"] is None
        or 0.0 <= logistic_metrics["roc_auc"] <= 1.0
    )

    model_path = Path(report["artifacts"]["logistic_regression"]["model_path"])
    report_path = Path(report["artifacts"]["logistic_regression"]["report_path"])
    assert model_path.exists()
    assert report_path.exists()

    loaded = load_model(model_path)
    sample = pd.DataFrame(
        np.zeros((3, len(MODEL_FEATURE_COLUMNS))),
        columns=list(MODEL_FEATURE_COLUMNS),
    )
    proba = loaded.predict_proba(sample)[:, 1]
    assert np.all((proba >= 0.0) & (proba <= 1.0))
