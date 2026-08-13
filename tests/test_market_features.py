"""Tests for Phase 3C market features and joins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.market_features import (
    add_relative_strength,
    build_market_features,
    join_market_features,
)


def _ohlcv(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1000, 2000, size=n).astype(float),
        }
    )


def test_market_features_and_relative_strength() -> None:
    market = build_market_features(_ohlcv(), prefix="nikkei")
    assert "nikkei_return_1d" in market.columns
    assert "nikkei_rsi_14" in market.columns
    assert market.isna().sum().sum() == 0

    stock = pd.DataFrame(
        {
            "Date": market["Date"],
            "Symbol": "7203.T",
            "return_5d": 0.01,
            "return_20d": 0.02,
        }
    )
    stock = stock.merge(
        market.loc[:, ["Date", "nikkei_return_5d", "nikkei_return_20d"]],
        on="Date",
        how="inner",
    )
    with_rel = add_relative_strength(stock, market_prefix="nikkei")
    assert with_rel["relative_return_5d_nikkei"].iloc[0] == pytest.approx(
        0.01 - stock["nikkei_return_5d"].iloc[0]
    )


def test_join_is_inner_and_has_no_backfill() -> None:
    market = build_market_features(_ohlcv(n=100), prefix="nikkei")
    # Stock has an extra date not present in market.
    extra_date = market["Date"].max() + pd.Timedelta(days=10)
    stock = pd.DataFrame(
        {
            "Date": list(market["Date"]) + [extra_date],
            "Symbol": ["7203.T"] * (len(market) + 1),
            "return_5d": [0.0] * (len(market) + 1),
            "return_20d": [0.0] * (len(market) + 1),
        }
    )
    joined = join_market_features(stock, market)
    assert extra_date not in set(joined["Date"])
    assert len(joined) == len(market)
    # No forward/back fill artifacts: every market column must be finite.
    mcols = [c for c in joined.columns if c.startswith("nikkei_")]
    assert np.isfinite(joined.loc[:, mcols].to_numpy(dtype=float)).all()
