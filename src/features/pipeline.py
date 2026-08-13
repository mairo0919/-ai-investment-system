"""Feature-generation pipeline: load raw CSV, validate, compute, save processed CSV."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import Settings, get_settings
from src.core.exceptions import (
    DataSaveError,
    DataValidationError,
    EmptyDataError,
    FeatureError,
    FeatureGenerationError,
)
from src.data.service import ticker_to_filename
from src.features.indicators import FEATURE_COLUMNS, add_features
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Adj Close",
)
OHLCV_OUTPUT_COLUMNS: tuple[str, ...] = (
    "Date",
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)


def ticker_to_features_filename(ticker: str) -> str:
    """Convert a ticker symbol to a processed features CSV filename.

    Example:
        ``7203.T`` -> ``7203_T_features.csv``
    """
    return f"{ticker.replace('.', '_')}_features.csv"


def validate_ohlcv(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Validate and normalize raw OHLCV data before feature generation.

    Rules:
        - Required columns must exist.
        - Data must not be empty.
        - Date must parse as datetime.
        - Duplicate dates are rejected (not silently dropped).
        - Rows are sorted ascending by Date when needed.
        - OHLCV fields must be numeric.

    Args:
        frame: Raw market data.
        source: Path or label used in error messages.

    Returns:
        Cleaned DataFrame ready for feature generation.
    """
    if frame.empty:
        raise EmptyDataError(f"Input data is empty: {source}")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns in {source}: {missing}")

    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if out["Date"].isna().any():
        raise DataValidationError(f"Date column contains unparseable values: {source}")

    if out["Date"].duplicated().any():
        dupes = out.loc[out["Date"].duplicated(), "Date"].dt.strftime("%Y-%m-%d").tolist()
        raise DataValidationError(
            f"Duplicate Date values are not allowed in {source}: {dupes[:5]}"
        )

    if not out["Date"].is_monotonic_increasing:
        logger.info("Sorting rows by Date ascending for %s", source)
        out = out.sort_values("Date").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    for column in NUMERIC_COLUMNS:
        if column not in out.columns:
            continue
        converted = pd.to_numeric(out[column], errors="coerce")
        introduced_nan = converted.isna() & out[column].notna()
        if introduced_nan.any():
            raise DataValidationError(
                f"Column '{column}' contains non-numeric values in {source}"
            )
        out[column] = converted

    if "Adj Close" not in out.columns:
        logger.warning(
            "Adj Close missing in %s; Close will be used for price-based features",
            source,
        )

    return out


def drop_warmup_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop leading warmup rows, then any residual invalid feature rows.

    Warmup rows are the leading block before every feature has a value at least once.
    Later NaNs (for example ``volume_change_1d`` after a zero-volume day) are dropped
    separately so raw history is not altered and processed output stays finite.

    Returns:
        Tuple of ``(cleaned_frame, warmup_dropped, invalid_dropped)``.
    """
    feature_block = frame.loc[:, list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    frame = frame.copy()
    frame.loc[:, list(FEATURE_COLUMNS)] = feature_block

    complete = feature_block.notna().all(axis=1)
    if not complete.any():
        raise FeatureGenerationError(
            "All rows were dropped during warmup filtering; input history is too short"
        )

    first_complete_pos = int(np.argmax(complete.to_numpy()))
    warmup_dropped = first_complete_pos
    after_warmup = frame.iloc[first_complete_pos:].copy()

    valid_mask = after_warmup.loc[:, list(FEATURE_COLUMNS)].notna().all(axis=1)
    invalid_dropped = int((~valid_mask).sum())
    cleaned = after_warmup.loc[valid_mask].reset_index(drop=True)

    if cleaned.empty:
        raise FeatureGenerationError(
            "All rows were dropped during warmup filtering; input history is too short"
        )

    values = cleaned.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if np.isnan(values).any() or np.isinf(values).any():
        raise FeatureGenerationError("NaN or inf values remain after warmup filtering")

    return cleaned, warmup_dropped, invalid_dropped

class FeaturePipeline:
    """Build processed feature CSVs from raw OHLCV files without modifying raw data."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> list[Path]:
        """Generate features for all configured tickers.

        Returns:
            Paths of successfully written processed CSV files.
        """
        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        for ticker in self.settings.tickers:
            try:
                path = self.process_ticker(ticker)
                saved_paths.append(path)
            except (EmptyDataError, DataValidationError, FeatureGenerationError, DataSaveError):
                logger.error("Feature generation failed for ticker=%s", ticker, exc_info=True)
            except Exception:
                logger.exception("Unexpected error while processing ticker=%s", ticker)

        if not saved_paths:
            raise FeatureError("No feature CSV files were saved. See logs for details.")

        logger.info(
            "Generated features for %d/%d tickers",
            len(saved_paths),
            len(self.settings.tickers),
        )
        return saved_paths

    def process_ticker(self, ticker: str) -> Path:
        """Load one raw CSV, generate features, and write a processed CSV."""
        raw_path = self.settings.raw_data_dir / ticker_to_filename(ticker)
        if not raw_path.exists():
            raise DataValidationError(f"Raw CSV not found for {ticker}: {raw_path}")

        logger.info("Loading raw data for %s from %s", ticker, raw_path)
        try:
            raw = pd.read_csv(raw_path)
        except OSError as exc:
            raise DataValidationError(f"Failed to read raw CSV {raw_path}: {exc}") from exc

        raw_rows = len(raw)
        validated = validate_ohlcv(raw, source=str(raw_path))
        validated["Symbol"] = ticker

        featured = add_features(validated)
        processed, warmup_dropped, invalid_dropped = drop_warmup_rows(featured)
        logger.info(
            "Row filter for %s: raw_rows=%d warmup_dropped=%d invalid_dropped=%d kept=%d",
            ticker,
            raw_rows,
            warmup_dropped,
            invalid_dropped,
            len(processed),
        )

        output_columns = [column for column in OHLCV_OUTPUT_COLUMNS if column in processed.columns]
        output_columns.extend(list(FEATURE_COLUMNS))
        output = processed.loc[:, output_columns]

        return self._save_csv(ticker, output)

    def _save_csv(self, ticker: str, frame: pd.DataFrame) -> Path:
        """Persist processed features. Never writes to the raw data directory."""
        path = self.settings.processed_data_dir / ticker_to_features_filename(ticker)
        try:
            frame.to_csv(path, index=False)
        except OSError as exc:
            raise DataSaveError(f"Failed to write features for {ticker} -> {path}: {exc}") from exc

        logger.info("Saved features %s (%d rows) -> %s", ticker, len(frame), path)
        return path


def main() -> int:
    """CLI entry point for Phase 2 feature generation."""
    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)

    logger.info(
        "Starting feature generation: tickers=%s raw_dir=%s processed_dir=%s",
        ",".join(settings.tickers),
        settings.raw_data_dir,
        settings.processed_data_dir,
    )

    try:
        pipeline = FeaturePipeline(settings)
        paths = pipeline.run()
    except FeatureError as exc:
        logger.error("Feature generation job failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected failure in feature generation job")
        return 1

    for path in paths:
        logger.info("Output: %s", path)
    logger.info("Feature generation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
