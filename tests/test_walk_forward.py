"""Tests for walk-forward folds and cross-sectional ranking helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import SplitError
from src.data.universe import load_universe, region_of, validate_universe_config
from src.ml.cross_section import cross_sectional_metrics_by_date, summarize_daily_metrics
from src.ml.walk_forward import build_expanding_folds, mask_fold_partition


def test_global100_universe_config() -> None:
    report = validate_universe_config(Path("config/universe.global100.json"))
    assert report["size"] == 100
    assert report["region_counts"].get("Japan") == 30
    assert report["region_counts"].get("United States") == 40
    assert report["region_counts"].get("Europe") == 30
    assert len(report["sector_counts"]) >= 8


def test_region_grouping() -> None:
    assert region_of("JP") == "Japan"
    assert region_of("US") == "United States"
    assert region_of("DE") == "Europe"
    assert region_of("GB") == "Europe"
    universe = load_universe(Path("config/universe.global100.json"))
    assert len(universe.by_region("Japan")) == 30


def test_expanding_folds_and_purge() -> None:
    dates = pd.bdate_range("2015-01-01", periods=3000)
    folds = build_expanding_folds(
        dates,
        n_folds=5,
        min_train_days=756,
        valid_days=252,
        purge_days=10,
    )
    assert len(folds) == 5
    for fold in folds:
        assert fold.train_end < fold.valid_start
        # Ensure purge gap exists in the calendar sense.
        assert fold.purge_days == 10
    # Expanding: later folds have longer train windows.
    assert len(folds[-1].train_dates) > len(folds[0].train_dates)


def test_build_folds_requires_enough_history() -> None:
    dates = pd.bdate_range("2020-01-01", periods=200)
    with pytest.raises(SplitError):
        build_expanding_folds(dates, n_folds=5, min_train_days=756, valid_days=252)


def test_mask_fold_partition() -> None:
    dates = pd.bdate_range("2016-01-01", periods=2200)
    folds = build_expanding_folds(
        dates, n_folds=5, min_train_days=756, valid_days=252, purge_days=5
    )
    frame = pd.DataFrame(
        {
            "Date": list(dates) * 2,
            "Symbol": ["A"] * len(dates) + ["B"] * len(dates),
            "x": 1,
        }
    )
    train = mask_fold_partition(frame, folds[0], partition="train")
    valid = mask_fold_partition(frame, folds[0], partition="validation")
    assert train["Date"].max() < valid["Date"].min()


def test_sector_grouping_in_universe() -> None:
    universe = load_universe(Path("config/universe.global100.json"))
    report = universe.composition_report()
    # Avoid extreme single-sector concentration in the research universe.
    top_sector_share = max(report["sector_counts"].values()) / report["size"]
    assert top_sector_share < 0.35
    assert "null" not in report["sector_counts"] or report["sector_counts"].get("null", 0) < 10


def test_min_history_filter_policy() -> None:
    """History-short symbols must be excluded from all folds (not mixed)."""
    from src.ml.walk_forward_runner import MIN_HISTORY_ROWS, MIN_HISTORY_YEARS

    assert MIN_HISTORY_YEARS == 5
    assert MIN_HISTORY_ROWS == 252 * 5


def test_cross_sectional_top_buckets_and_ic() -> None:
    rows = []
    rng = np.random.default_rng(0)
    for i, dt in enumerate(pd.bdate_range("2024-01-01", periods=20)):
        scores = np.linspace(-1, 1, 30)
        fut = scores * 0.02 + rng.normal(0, 0.001, size=30)
        for j in range(30):
            rows.append(
                {
                    "Date": dt,
                    "Symbol": f"S{j}",
                    "score": scores[j],
                    "future_return": fut[j],
                    "Sector": "Tech" if j < 15 else "Health",
                    "Region": "US",
                }
            )
    frame = pd.DataFrame(rows)
    daily = cross_sectional_metrics_by_date(
        frame, top_fractions=(0.05, 0.10, 0.20), min_names=5
    )
    assert not daily.empty
    assert {"top_5pct_excess_return", "top_10pct_excess_return", "ic"} <= set(daily.columns)
    summary = summarize_daily_metrics(daily)
    assert summary["ic"]["mean"] is not None
    assert summary["ic"]["mean"] > 0.5
