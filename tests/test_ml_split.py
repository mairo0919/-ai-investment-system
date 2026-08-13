"""Tests for chronological train/validation/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.indicators import FEATURE_COLUMNS
from src.ml.dataset import MLDataset
from src.ml.split import chronological_split


def _dataset(n_dates: int = 100) -> MLDataset:
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    rows = []
    rng = np.random.default_rng(1)
    for symbol in ("7203.T", "5333.T"):
        close = 100 + np.cumsum(rng.normal(0, 1, size=n_dates))
        row = {
            "Date": dates,
            "Symbol": symbol,
            "Adj Close": close,
            "target_up": (rng.random(n_dates) > 0.5).astype(int),
            "future_return_1d": rng.normal(0, 0.01, size=n_dates),
            "return_1d": rng.normal(0, 0.01, size=n_dates),
        }
        for column in FEATURE_COLUMNS:
            row[column] = rng.normal(0, 1, size=n_dates)
        rows.append(pd.DataFrame(row))
    frame = pd.concat(rows, ignore_index=True)
    return MLDataset(frame=frame, feature_columns=FEATURE_COLUMNS)


def test_splits_are_date_disjoint() -> None:
    bundle = chronological_split(_dataset())
    assert bundle.train.frame["Date"].max() < bundle.validation.frame["Date"].min()
    assert bundle.validation.frame["Date"].max() < bundle.test.frame["Date"].min()

    train_dates = set(bundle.train.frame["Date"])
    valid_dates = set(bundle.validation.frame["Date"])
    test_dates = set(bundle.test.frame["Date"])
    assert train_dates.isdisjoint(valid_dates)
    assert valid_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(test_dates)


def test_split_is_not_shuffled() -> None:
    bundle = chronological_split(_dataset())
    for part in (bundle.train, bundle.validation, bundle.test):
        dates = pd.to_datetime(part.frame["Date"])
        assert dates.is_monotonic_increasing
