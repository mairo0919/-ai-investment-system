"""Dataset loading and target generation for Phase 3 ML training.

``shift(-n)`` is used only here to create labels. Feature columns themselves
must never contain future information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.exceptions import DatasetError
from src.features.indicators import FEATURE_COLUMNS
from src.features.pipeline import ticker_to_features_filename
from src.ml.targets import TARGET_UP_1D, TargetConfig

logger = logging.getLogger(__name__)

# Explicit base model feature list: Phase 2 engineered features only.
MODEL_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS

FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "Date",
    "Symbol",
    "target_up",
    "target_up_1d",
    "target_up_5d",
    "target_up_10d",
    "target_5d_threshold",
    "future_return_1d",
    "future_return_5d",
    "future_return_10d",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)

META_COLUMNS: tuple[str, ...] = ("Date", "Symbol")


@dataclass(frozen=True)
class MLDataset:
    """Labeled dataset with explicit feature matrix and metadata columns."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str = "target_up"
    future_return_column: str = "future_return_1d"

    @property
    def X(self) -> pd.DataFrame:
        return self.frame.loc[:, list(self.feature_columns)]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target_column].astype(int)

    @property
    def future_return(self) -> pd.Series:
        return self.frame[self.future_return_column].astype(float)

    @property
    def meta(self) -> pd.DataFrame:
        return self.frame.loc[:, ["Date", "Symbol"]].copy()


def add_target_up(frame: pd.DataFrame) -> pd.DataFrame:
    """Create legacy ``future_return_1d`` / ``target_up`` for Phase 3A/3B.

    Also materializes ``target_up_1d`` as an alias of ``target_up``.
    """
    out = add_future_returns(frame, horizons=(1,))
    out["target_up"] = (out["future_return_1d"] > 0).astype(int)
    out["target_up_1d"] = out["target_up"]
    before = len(out)
    out = out.loc[out["future_return_1d"].notna()].reset_index(drop=True)
    logger.info("Dropped %d trailing row(s) without next-day target", before - len(out))
    if out.empty:
        raise DatasetError("No rows remain after target generation")
    return out


def add_future_returns(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    """Add ``future_return_{h}d`` columns using only Adj Close leads."""
    required = {"Date", "Symbol", "Adj Close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DatasetError(f"Cannot create future returns; missing columns: {missing}")

    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if out["Date"].isna().any():
        raise DatasetError("Date column contains unparseable values")

    out = out.sort_values(["Symbol", "Date"], kind="mergesort").reset_index(drop=True)
    grouped = out.groupby("Symbol", sort=False)["Adj Close"]
    for horizon in horizons:
        future_adj = grouped.shift(-horizon)
        out[f"future_return_{horizon}d"] = future_adj / out["Adj Close"] - 1.0
    return out


def materialize_target(
    frame: pd.DataFrame,
    config: TargetConfig,
) -> pd.DataFrame:
    """Create one target column according to ``TargetConfig``.

    For threshold targets, rows inside the neutral band are dropped only when
    ``config.drop_neutral`` is True.
    """
    out = frame.copy()
    ret_col = config.future_return_column
    if ret_col not in out.columns:
        out = add_future_returns(out, horizons=(config.horizon_days,))

    if config.uses_threshold:
        assert config.threshold is not None
        thr = float(config.threshold)
        labels = pd.Series(np.nan, index=out.index, dtype=float)
        labels = labels.mask(out[ret_col] > thr, 1.0)
        labels = labels.mask(out[ret_col] < -thr, 0.0)
        out[config.target_column] = labels
        before = len(out)
        out = out.loc[out[config.target_column].notna() & out[ret_col].notna()].copy()
        out[config.target_column] = out[config.target_column].astype(int)
        dropped_neutral = before - len(out)
        logger.info(
            "Target %s: dropped %d neutral/unavailable rows (threshold=±%.2f%%)",
            config.name,
            dropped_neutral,
            thr * 100,
        )
    else:
        out[config.target_column] = (out[ret_col] > 0).astype("Int64")
        before = len(out)
        out = out.loc[out[ret_col].notna()].copy()
        out[config.target_column] = out[config.target_column].astype(int)
        logger.info(
            "Target %s: dropped %d trailing rows without horizon=%d return",
            config.name,
            before - len(out),
            config.horizon_days,
        )

    if out.empty:
        raise DatasetError(f"No rows remain after materializing target {config.name}")
    return out.reset_index(drop=True)


def _load_processed_csv(path: Path, ticker: str) -> pd.DataFrame:
    if not path.exists():
        raise DatasetError(f"Processed CSV not found for {ticker}: {path}")
    try:
        frame = pd.read_csv(path)
    except OSError as exc:
        raise DatasetError(f"Failed to read processed CSV {path}: {exc}") from exc

    if frame.empty:
        raise DatasetError(f"Processed CSV is empty: {path}")

    if "Symbol" not in frame.columns:
        frame["Symbol"] = ticker
    else:
        frame["Symbol"] = frame["Symbol"].fillna(ticker)

    missing_features = [column for column in MODEL_FEATURE_COLUMNS if column not in frame.columns]
    if missing_features:
        raise DatasetError(
            f"Missing required model features in {path}: {missing_features}"
        )
    return frame


def load_stock_feature_frames(
    processed_data_dir: Path,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    """Load and vertically stack processed stock feature CSVs (no targets yet)."""
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        path = processed_data_dir / ticker_to_features_filename(ticker)
        logger.info("Loading processed features for %s from %s", ticker, path)
        frames.append(_load_processed_csv(path, ticker))
    return pd.concat(frames, ignore_index=True, sort=False)


def build_dataset(
    processed_data_dir: Path,
    tickers: tuple[str, ...],
) -> MLDataset:
    """Phase 3A/3B dataset builder (1d target, stock features only)."""
    combined = load_stock_feature_frames(processed_data_dir, tickers)
    labeled = add_target_up(combined)

    overlap = set(MODEL_FEATURE_COLUMNS) & set(FORBIDDEN_FEATURE_COLUMNS)
    if overlap:
        raise DatasetError(f"Model feature list overlaps forbidden columns: {sorted(overlap)}")

    ordered_columns = [
        column
        for column in (
            "Date",
            "Symbol",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            *MODEL_FEATURE_COLUMNS,
            "future_return_1d",
            "target_up",
            "target_up_1d",
        )
        if column in labeled.columns
    ]
    labeled = labeled.loc[:, ordered_columns]

    if labeled.loc[:, list(MODEL_FEATURE_COLUMNS)].isna().any().any():
        raise DatasetError("Model features contain NaN values")

    logger.info(
        "Built dataset: rows=%d symbols=%s features=%d",
        len(labeled),
        ",".join(tickers),
        len(MODEL_FEATURE_COLUMNS),
    )
    return MLDataset(
        frame=labeled,
        feature_columns=MODEL_FEATURE_COLUMNS,
        target_column="target_up",
        future_return_column="future_return_1d",
    )


def build_experiment_dataset(
    frame: pd.DataFrame,
    *,
    config: TargetConfig,
    feature_columns: tuple[str, ...],
) -> MLDataset:
    """Build a dataset for one Phase 3C experiment configuration."""
    labeled = materialize_target(frame, config)

    missing = [column for column in feature_columns if column not in labeled.columns]
    if missing:
        raise DatasetError(f"Missing feature columns for experiment: {missing}")

    overlap = set(feature_columns) & set(FORBIDDEN_FEATURE_COLUMNS)
    if overlap:
        raise DatasetError(f"Feature set overlaps forbidden columns: {sorted(overlap)}")

    feature_block = labeled.loc[:, list(feature_columns)]
    if feature_block.isna().any().any():
        before = len(labeled)
        labeled = labeled.loc[feature_block.notna().all(axis=1)].reset_index(drop=True)
        logger.info(
            "Dropped %d rows with NaN features for target=%s",
            before - len(labeled),
            config.name,
        )

    if labeled.empty:
        raise DatasetError(f"Empty dataset after feature filtering for {config.name}")

    return MLDataset(
        frame=labeled,
        feature_columns=feature_columns,
        target_column=config.target_column,
        future_return_column=config.future_return_column,
    )


def ensure_legacy_target_alias(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``target_up`` exists when ``target_up_1d`` is present."""
    out = frame.copy()
    if "target_up" not in out.columns and "target_up_1d" in out.columns:
        out["target_up"] = out["target_up_1d"]
    return out


# Re-export default 1d config for convenience.
DEFAULT_TARGET = TARGET_UP_1D
