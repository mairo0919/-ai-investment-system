"""Tests for Phase 2 feature generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.settings import Settings
from src.core.exceptions import DataValidationError
from src.features.indicators import FEATURE_COLUMNS, add_features, sma
from src.features.pipeline import FeaturePipeline, validate_ohlcv


def _make_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create synthetic ascending OHLCV data long enough for warmup."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.cumsum(rng.normal(0.0, 1.0, size=n)) + 100.0
    open_ = close + rng.normal(0.0, 0.2, size=n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, size=n)
    volume = rng.integers(1000, 5000, size=n).astype(float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        }
    )


def test_sma_calculation() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    result = sma(series, 5)
    assert pd.isna(result.iloc[3])
    assert result.iloc[4] == pytest.approx(3.0)


def test_rsi_bounds() -> None:
    frame = add_features(_make_ohlcv())
    rsi = frame["rsi_14"].dropna()
    assert not rsi.empty
    assert rsi.ge(-1e-9).all()
    assert rsi.le(100.0 + 1e-9).all()


def test_macd_columns_exist() -> None:
    frame = add_features(_make_ohlcv())
    for column in ("macd", "macd_signal", "macd_hist"):
        assert column in frame.columns
        assert frame[column].notna().any()


def test_bollinger_band_ordering() -> None:
    frame = add_features(_make_ohlcv()).dropna(subset=["bb_middle", "bb_upper", "bb_lower"])
    assert (frame["bb_upper"] >= frame["bb_middle"] - 1e-9).all()
    assert (frame["bb_middle"] >= frame["bb_lower"] - 1e-9).all()


def test_atr_non_negative() -> None:
    frame = add_features(_make_ohlcv())
    atr = frame["atr_14"].dropna()
    assert not atr.empty
    assert atr.ge(-1e-9).all()


def test_no_lookahead_bias_when_tail_changes() -> None:
    base = _make_ohlcv(n=150)
    modified = base.copy()
    modified.loc[modified.index[-1], ["Open", "High", "Low", "Close", "Adj Close"]] *= 1.25

    features_base = add_features(base)
    features_modified = add_features(modified)

    compare_columns = list(FEATURE_COLUMNS)
    left = features_base.iloc[:-1][compare_columns].reset_index(drop=True)
    right = features_modified.iloc[:-1][compare_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_missing_required_column_raises() -> None:
    frame = _make_ohlcv().drop(columns=["Volume"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_ohlcv(frame, source="test")


def test_duplicate_date_raises() -> None:
    frame = _make_ohlcv(n=30)
    frame.loc[5, "Date"] = frame.loc[4, "Date"]
    with pytest.raises(DataValidationError, match="Duplicate Date"):
        validate_ohlcv(frame, source="test")


def test_processed_csv_generation(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    ticker = "7203.T"
    raw_path = raw_dir / "7203_T.csv"
    _make_ohlcv(n=120).to_csv(raw_path, index=False)

    settings = Settings(
        tickers=(ticker,),
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        log_dir=tmp_path / "logs",
    )
    output = FeaturePipeline(settings).process_ticker(ticker)

    assert output.exists()
    assert output.name == "7203_T_features.csv"

    processed = pd.read_csv(output)
    assert "Symbol" in processed.columns
    assert processed["Symbol"].eq(ticker).all()
    for column in FEATURE_COLUMNS:
        assert column in processed.columns
    assert processed.loc[:, list(FEATURE_COLUMNS)].isna().sum().sum() == 0
    assert np.isfinite(processed.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all()
    # raw files must remain untouched
    raw_again = pd.read_csv(raw_path)
    assert len(raw_again) == 120
