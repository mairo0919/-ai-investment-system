"""Expanding-window walk-forward fold generation with label purge."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.core.exceptions import SplitError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window train/validation fold."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    train_dates: tuple[pd.Timestamp, ...]
    valid_dates: tuple[pd.Timestamp, ...]
    purge_days: int


def build_expanding_folds(
    dates: pd.Series | list[pd.Timestamp],
    *,
    n_folds: int = 5,
    min_train_days: int = 252 * 3,
    valid_days: int = 252,
    purge_days: int = 5,
) -> list[WalkForwardFold]:
    """Build expanding-window folds on a sorted unique trading calendar.

    Policy:
        - Train always starts at the first date and expands after each fold.
        - Validation blocks are contiguous and non-overlapping.
        - ``purge_days`` unique dates are skipped between train_end and valid_start
          so horizon labels cannot read the next fold's prices.
        - Symbols with insufficient history are excluded upstream (not mixed in).
    """
    if n_folds < 5:
        raise SplitError("Walk-forward requires at least 5 folds")
    if min_train_days < 1 or valid_days < 1:
        raise SplitError("min_train_days and valid_days must be positive")
    if purge_days < 0:
        raise SplitError("purge_days must be >= 0")

    unique = pd.Index(sorted(pd.to_datetime(pd.Series(list(dates))).unique()))
    n = len(unique)
    needed = min_train_days + n_folds * (valid_days + purge_days)
    if n < needed:
        raise SplitError(
            f"Not enough unique dates for walk-forward: have={n}, need>={needed} "
            f"(min_train={min_train_days}, folds={n_folds}, valid={valid_days}, purge={purge_days})"
        )

    folds: list[WalkForwardFold] = []
    train_end_idx = min_train_days
    for fold_id in range(1, n_folds + 1):
        valid_start_idx = train_end_idx + purge_days
        valid_end_idx = valid_start_idx + valid_days
        if valid_end_idx > n:
            raise SplitError(
                f"Fold {fold_id} validation window exceeds calendar "
                f"(valid_end_idx={valid_end_idx}, n={n})"
            )

        train_dates = tuple(pd.Timestamp(x) for x in unique[:train_end_idx])
        valid_dates = tuple(
            pd.Timestamp(x) for x in unique[valid_start_idx:valid_end_idx]
        )
        if train_dates[-1] >= valid_dates[0]:
            raise SplitError(f"Train/valid overlap for fold={fold_id}")

        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            valid_start=valid_dates[0],
            valid_end=valid_dates[-1],
            train_dates=train_dates,
            valid_dates=valid_dates,
            purge_days=purge_days,
        )
        folds.append(fold)
        logger.info(
            "Walk-forward fold %d: train=%s..%s (%d days) valid=%s..%s (%d days) purge=%d",
            fold_id,
            fold.train_start.date(),
            fold.train_end.date(),
            len(train_dates),
            fold.valid_start.date(),
            fold.valid_end.date(),
            len(valid_dates),
            purge_days,
        )
        # Expand train through the end of this validation block for the next fold.
        train_end_idx = valid_end_idx

    return folds


def mask_fold_partition(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    partition: str,
) -> pd.DataFrame:
    """Filter a panel to train or validation dates of a fold."""
    dates = set(fold.train_dates if partition == "train" else fold.valid_dates)
    out = frame.loc[pd.to_datetime(frame["Date"]).isin(dates)].copy()
    return out.sort_values(["Date", "Symbol"], kind="mergesort").reset_index(drop=True)
