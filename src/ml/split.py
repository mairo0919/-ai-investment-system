"""Chronological train / validation / test splitting (no shuffle)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.core.exceptions import SplitError
from src.ml.dataset import MLDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitBundle:
    """Date-ordered dataset partitions with preserved metadata."""

    train: MLDataset
    validation: MLDataset
    test: MLDataset
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    validation_dates: tuple[pd.Timestamp, pd.Timestamp]
    test_dates: tuple[pd.Timestamp, pd.Timestamp]
    purged_train_rows: int = 0
    purged_validation_rows: int = 0
    purge_days: int = 0


def _date_range(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.to_datetime(frame["Date"])
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def _subset(dataset: MLDataset, mask: pd.Series) -> MLDataset:
    frame = dataset.frame.loc[mask].reset_index(drop=True)
    if frame.empty:
        raise SplitError("A chronological split produced an empty partition")
    return MLDataset(
        frame=frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        future_return_column=dataset.future_return_column,
    )


def _order(dataset: MLDataset) -> MLDataset:
    ordered = dataset.frame.sort_values(["Date", "Symbol"], kind="mergesort").reset_index(
        drop=True
    )
    return MLDataset(
        frame=ordered,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        future_return_column=dataset.future_return_column,
    )


def chronological_split(
    dataset: MLDataset,
    *,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    purge_days: int = 0,
) -> SplitBundle:
    """Split by unique calendar dates so the same Date never crosses folds.

    When ``purge_days > 0``, the last ``purge_days`` unique dates of the Train
    and Validation blocks are discarded so that a horizon-H label cannot read
    prices from the next fold.

    Rows are never shuffled. Boundaries satisfy:
        max(train.Date) < min(validation.Date)
        max(validation.Date) < min(test.Date)
    """
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"Split ratios must sum to 1.0, got {total}")
    if min(train_ratio, valid_ratio, test_ratio) <= 0:
        raise SplitError("All split ratios must be positive")
    if purge_days < 0:
        raise SplitError("purge_days must be >= 0")

    frame = dataset.frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    unique_dates = pd.Series(sorted(frame["Date"].unique()))
    n_dates = len(unique_dates)
    if n_dates < 3:
        raise SplitError(f"Need at least 3 unique dates to split, got {n_dates}")

    train_end = max(1, int(n_dates * train_ratio))
    valid_end = train_end + max(1, int(n_dates * valid_ratio))
    if valid_end >= n_dates:
        valid_end = n_dates - 1
    if train_end >= valid_end:
        raise SplitError(
            f"Unable to form non-empty chronological splits from {n_dates} dates"
        )

    if purge_days > 0:
        if train_end - purge_days < 1:
            raise SplitError(
                f"purge_days={purge_days} removes the entire train date block"
            )
        if valid_end - train_end - purge_days < 1:
            raise SplitError(
                f"purge_days={purge_days} removes the entire validation date block"
            )

    train_slice_end = train_end - purge_days
    valid_slice_start = train_end
    valid_slice_end = valid_end - purge_days
    test_slice_start = valid_end

    train_date_values = set(unique_dates.iloc[:train_slice_end])
    valid_date_values = set(unique_dates.iloc[valid_slice_start:valid_slice_end])
    test_date_values = set(unique_dates.iloc[test_slice_start:])
    purged_train_dates = set(unique_dates.iloc[train_slice_end:train_end])
    purged_valid_dates = set(unique_dates.iloc[valid_slice_end:valid_end])

    if not train_date_values or not valid_date_values or not test_date_values:
        raise SplitError("One or more date partitions are empty after purge")

    normalized = MLDataset(
        frame=frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        future_return_column=dataset.future_return_column,
    )
    train = _order(_subset(normalized, frame["Date"].isin(train_date_values)))
    validation = _order(_subset(normalized, frame["Date"].isin(valid_date_values)))
    test = _order(_subset(normalized, frame["Date"].isin(test_date_values)))

    purged_train_rows = int(frame["Date"].isin(purged_train_dates).sum())
    purged_valid_rows = int(frame["Date"].isin(purged_valid_dates).sum())

    train_range = _date_range(train.frame)
    valid_range = _date_range(validation.frame)
    test_range = _date_range(test.frame)

    if train.frame["Date"].max() >= validation.frame["Date"].min():
        raise SplitError(
            "Train and validation date ranges overlap: "
            f"train={train_range}, validation={valid_range}"
        )
    if validation.frame["Date"].max() >= test.frame["Date"].min():
        raise SplitError(
            "Validation and test date ranges overlap: "
            f"validation={valid_range}, test={test_range}"
        )

    missing_features = [
        column for column in train.feature_columns if column not in train.frame.columns
    ]
    if missing_features:
        raise SplitError(f"Split frames missing feature columns: {missing_features}")

    # Horizon integrity: at least purge_days unique dates between folds.
    if purge_days > 0:
        train_dates_sorted = sorted(train_date_values)
        valid_dates_sorted = sorted(valid_date_values)
        gap_train_valid = sum(
            1 for d in unique_dates if train_dates_sorted[-1] < d < valid_dates_sorted[0]
        )
        # Dates removed by purge sit between the kept blocks.
        if gap_train_valid + 1 < purge_days and len(purged_train_dates) < purge_days:
            raise SplitError("Purge gap between train and validation is insufficient")

    logger.info(
        "Chronological split by dates: total_dates=%d train_dates=%d "
        "valid_dates=%d test_dates=%d purge_days=%d "
        "| rows train=%d valid=%d test=%d purged_train_rows=%d purged_valid_rows=%d",
        n_dates,
        len(train_date_values),
        len(valid_date_values),
        len(test_date_values),
        purge_days,
        len(train.frame),
        len(validation.frame),
        len(test.frame),
        purged_train_rows,
        purged_valid_rows,
    )
    return SplitBundle(
        train=train,
        validation=validation,
        test=test,
        train_dates=train_range,
        validation_dates=valid_range,
        test_dates=test_range,
        purged_train_rows=purged_train_rows,
        purged_validation_rows=purged_valid_rows,
        purge_days=purge_days,
    )


def assert_no_horizon_leakage(
    split: SplitBundle,
    *,
    horizon_days: int,
    source_prices: pd.DataFrame,
) -> None:
    """Assert train/validation labels do not require prices from the next fold.

    ``source_prices`` must contain Symbol, Date, Adj Close for the full history
    used to build targets (before split purge).
    """
    if horizon_days <= 0:
        raise SplitError("horizon_days must be positive")

    prices = source_prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values(["Symbol", "Date"], kind="mergesort")

    def _max_label_date(part: MLDataset) -> pd.Timestamp:
        # For each row date, label uses the price horizon_days ahead on that symbol.
        # Approximate with unique trading calendar from the price history.
        unique = pd.Index(sorted(prices["Date"].unique()))
        part_dates = pd.to_datetime(part.frame["Date"])
        positions = unique.get_indexer(part_dates)
        if (positions < 0).any():
            raise SplitError("Split contains dates missing from source price calendar")
        future_positions = positions + horizon_days
        if (future_positions >= len(unique)).any():
            # Rows without a future price should already have been dropped.
            future_positions = future_positions[future_positions < len(unique)]
        if len(future_positions) == 0:
            raise SplitError("Unable to validate horizon leakage; no future dates")
        return pd.Timestamp(unique[future_positions.max()])

    train_label_max = _max_label_date(split.train)
    valid_min = pd.Timestamp(split.validation.frame["Date"].min())
    if train_label_max >= valid_min:
        raise SplitError(
            "Train labels reach into validation prices: "
            f"train_label_max={train_label_max.date()} valid_min={valid_min.date()}"
        )

    valid_label_max = _max_label_date(split.validation)
    test_min = pd.Timestamp(split.test.frame["Date"].min())
    if valid_label_max >= test_min:
        raise SplitError(
            "Validation labels reach into test prices: "
            f"valid_label_max={valid_label_max.date()} test_min={test_min.date()}"
        )
