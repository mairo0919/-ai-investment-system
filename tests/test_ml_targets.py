"""Tests for Phase 3C multi-horizon targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.dataset import add_future_returns, materialize_target
from src.ml.targets import TARGET_UP_1D, TARGET_UP_5D, TARGET_UP_10D, target_5d_threshold


def _price_frame(prices: list[float], symbol: str = "7203.T") -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=len(prices))
    return pd.DataFrame(
        {
            "Date": dates,
            "Symbol": symbol,
            "Adj Close": prices,
            "Close": prices,
        }
    )


def test_target_1d_accuracy() -> None:
    frame = add_future_returns(_price_frame([100, 110, 105, 120]), horizons=(1,))
    labeled = materialize_target(frame, TARGET_UP_1D)
    assert list(labeled["target_up_1d"]) == [1, 0, 1]
    assert labeled["future_return_1d"].tolist() == pytest.approx(
        [0.10, -5 / 110, 120 / 105 - 1]
    )


def test_target_5d_accuracy() -> None:
    prices = [100 + i for i in range(12)]
    frame = add_future_returns(_price_frame(prices), horizons=(5,))
    labeled = materialize_target(frame, TARGET_UP_5D)
    # First row: (105/100 - 1) > 0
    assert labeled.iloc[0]["future_return_5d"] == pytest.approx(prices[5] / prices[0] - 1)
    assert int(labeled.iloc[0]["target_up_5d"]) == 1
    assert len(labeled) == len(prices) - 5


def test_target_10d_accuracy() -> None:
    prices = list(range(100, 130))
    frame = add_future_returns(_price_frame(prices), horizons=(10,))
    labeled = materialize_target(frame, TARGET_UP_10D)
    assert labeled.iloc[0]["future_return_10d"] == pytest.approx(prices[10] / prices[0] - 1)
    assert len(labeled) == len(prices) - 10


def test_threshold_target_and_neutral_exclusion() -> None:
    # Returns around t: +3%, +1%, -3%, 0%
    prices = [100.0, 103.0, 104.03, 100.9089, 100.9089]
    # Build a longer series with controlled 5d returns via explicit future column.
    dates = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Symbol": "7203.T",
            "Adj Close": prices,
            "future_return_5d": [0.03, 0.01, -0.03, 0.0, np.nan],
        }
    )
    cfg = target_5d_threshold(0.02)
    labeled = materialize_target(frame, cfg)
    assert list(labeled["target_5d_threshold"]) == [1, 0]
    assert len(labeled) == 2
