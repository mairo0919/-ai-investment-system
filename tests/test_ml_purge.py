"""Tests for purged chronological splits (Phase 3C)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import SplitError
from src.features.indicators import FEATURE_COLUMNS
from src.ml.dataset import MLDataset, add_future_returns, materialize_target
from src.ml.split import assert_no_horizon_leakage, chronological_split
from src.ml.targets import TARGET_UP_5D, TARGET_UP_10D


def _dataset(n_dates: int = 120, horizon: int = 5) -> tuple[MLDataset, pd.DataFrame]:
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    rng = np.random.default_rng(0)
    rows = []
    for symbol in ("7203.T", "5333.T"):
        close = 100 + np.cumsum(rng.normal(0, 1, size=n_dates))
        data = {
            "Date": dates,
            "Symbol": symbol,
            "Adj Close": close,
            "Close": close,
        }
        for column in FEATURE_COLUMNS:
            data[column] = rng.normal(0, 1, size=n_dates)
        rows.append(pd.DataFrame(data))
    raw = pd.concat(rows, ignore_index=True)
    with_future = add_future_returns(raw, horizons=(horizon,))
    cfg = TARGET_UP_5D if horizon == 5 else TARGET_UP_10D
    labeled = materialize_target(with_future, cfg)
    dataset = MLDataset(
        frame=labeled,
        feature_columns=FEATURE_COLUMNS,
        target_column=cfg.target_column,
        future_return_column=cfg.future_return_column,
    )
    return dataset, raw.loc[:, ["Date", "Symbol", "Adj Close"]]


def test_purge_5d_removes_boundary_rows() -> None:
    dataset, prices = _dataset(horizon=5)
    no_purge = chronological_split(dataset, purge_days=0)
    purged = chronological_split(dataset, purge_days=5)
    assert purged.purged_train_rows > 0
    assert purged.purged_validation_rows > 0
    assert len(purged.train.frame) < len(no_purge.train.frame)
    assert_no_horizon_leakage(purged, horizon_days=5, source_prices=prices)


def test_purge_10d_removes_boundary_rows() -> None:
    dataset, prices = _dataset(n_dates=160, horizon=10)
    purged = chronological_split(dataset, purge_days=10)
    assert purged.purge_days == 10
    assert purged.purged_train_rows > 0
    assert_no_horizon_leakage(purged, horizon_days=10, source_prices=prices)


def test_train_labels_do_not_use_validation_prices() -> None:
    dataset, prices = _dataset(horizon=5)
    purged = chronological_split(dataset, purge_days=5)
    assert_no_horizon_leakage(purged, horizon_days=5, source_prices=prices)


def test_validation_labels_do_not_use_test_prices() -> None:
    dataset, prices = _dataset(horizon=5)
    purged = chronological_split(dataset, purge_days=5)
    # Directly ensure max validation label date < min test date.
    assert_no_horizon_leakage(purged, horizon_days=5, source_prices=prices)
    with pytest.raises(SplitError):
        # Force a failing case: pretend purge was skipped.
        leaky = chronological_split(dataset, purge_days=0)
        assert_no_horizon_leakage(leaky, horizon_days=5, source_prices=prices)
