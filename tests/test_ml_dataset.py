"""Tests for Phase 3 dataset / target generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import DatasetError
from src.features.indicators import FEATURE_COLUMNS
from src.ml.dataset import MODEL_FEATURE_COLUMNS, add_target_up, build_dataset


def _feature_frame(n: int = 40, symbol: str = "7203.T", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    data = {
        "Date": dates,
        "Symbol": symbol,
        "Open": close,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Adj Close": close,
        "Volume": rng.integers(1000, 2000, size=n).astype(float),
    }
    for column in FEATURE_COLUMNS:
        data[column] = rng.normal(0, 1, size=n)
    return pd.DataFrame(data)


def test_target_up_generation() -> None:
    frame = _feature_frame(n=5)
    # Force known adj closes: up, down, up, flat-ish
    frame["Adj Close"] = [100.0, 110.0, 105.0, 120.0, 120.0]
    labeled = add_target_up(frame)

    assert list(labeled["target_up"]) == [1, 0, 1, 0]
    expected_returns = [
        110 / 100 - 1,
        105 / 110 - 1,
        120 / 105 - 1,
        120 / 120 - 1,
    ]
    assert labeled["future_return_1d"].tolist() == pytest.approx(expected_returns)


def test_last_row_excluded_from_target() -> None:
    frame = _feature_frame(n=10)
    labeled = add_target_up(frame)
    assert len(labeled) == len(frame) - 1
    assert labeled["Date"].max() < frame["Date"].max()


def test_features_do_not_include_future_columns() -> None:
    assert "future_return_1d" not in MODEL_FEATURE_COLUMNS
    assert "target_up" not in MODEL_FEATURE_COLUMNS
    assert "Adj Close" not in MODEL_FEATURE_COLUMNS


def test_missing_features_raise(tmp_path: Path) -> None:
    frame = _feature_frame(n=30).drop(columns=["rsi_14"])
    path = tmp_path / "7203_T_features.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DatasetError, match="Missing required model features"):
        build_dataset(tmp_path, ("7203.T",))
