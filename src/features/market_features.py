"""Market-context feature engineering for Phase 3C.

All features use only information available at or before each bar's close.
No forward fill into the future and no backfill are performed here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.indicators import (
    TRADING_DAYS_PER_YEAR,
    rsi_wilder,
    safe_divide,
    select_price,
    sma,
)

logger = logging.getLogger(__name__)

MARKET_FEATURE_SUFFIXES: tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_20_ratio",
    "sma_60_ratio",
    "volatility_20",
    "rsi_14",
)


def market_feature_columns(prefix: str) -> tuple[str, ...]:
    """Return prefixed market feature column names for one index."""
    return tuple(f"{prefix}_{suffix}" for suffix in MARKET_FEATURE_SUFFIXES)


def build_market_features(ohlcv: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Compute Phase 3C market features from a single index OHLCV frame.

    Args:
        ohlcv: Date-sorted OHLCV with Adj Close/Close.
        prefix: Column prefix such as ``nikkei`` or ``topix``.
    """
    frame = ohlcv.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").drop_duplicates("Date", keep="last")
    price = select_price(frame)

    out = pd.DataFrame({"Date": frame["Date"].to_numpy()})
    out[f"{prefix}_return_1d"] = price.pct_change(1).to_numpy()
    out[f"{prefix}_return_5d"] = price.pct_change(5).to_numpy()
    out[f"{prefix}_return_20d"] = price.pct_change(20).to_numpy()

    sma_20 = sma(price, 20)
    sma_60 = sma(price, 60)
    out[f"{prefix}_sma_20_ratio"] = (safe_divide(price, sma_20) - 1.0).to_numpy()
    out[f"{prefix}_sma_60_ratio"] = (safe_divide(price, sma_60) - 1.0).to_numpy()
    out[f"{prefix}_volatility_20"] = (
        price.pct_change(1).rolling(20, min_periods=20).std(ddof=0)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    ).to_numpy()
    out[f"{prefix}_rsi_14"] = rsi_wilder(price, 14).to_numpy()

    feature_cols = list(market_feature_columns(prefix))
    out.loc[:, feature_cols] = out.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
    before = len(out)
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info(
        "Market features prefix=%s: kept=%d dropped_warmup=%d",
        prefix,
        len(out),
        before - len(out),
    )
    return out


def add_relative_strength(
    stock_frame: pd.DataFrame,
    *,
    market_prefix: str,
) -> pd.DataFrame:
    """Add stock-minus-market relative return features."""
    out = stock_frame.copy()
    stock_5 = "return_5d"
    stock_20 = "return_20d"
    mkt_5 = f"{market_prefix}_return_5d"
    mkt_20 = f"{market_prefix}_return_20d"
    for column in (stock_5, stock_20, mkt_5, mkt_20):
        if column not in out.columns:
            raise KeyError(f"Missing column required for relative strength: {column}")

    out[f"relative_return_5d_{market_prefix}"] = out[stock_5] - out[mkt_5]
    out[f"relative_return_20d_{market_prefix}"] = out[stock_20] - out[mkt_20]
    return out


def relative_strength_columns(market_prefix: str) -> tuple[str, ...]:
    """Return relative-strength feature names for one market prefix."""
    return (
        f"relative_return_5d_{market_prefix}",
        f"relative_return_20d_{market_prefix}",
    )


def join_market_features(
    stock_frame: pd.DataFrame,
    market_features: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join stock rows with market features on Date (no backfill).

    Forward fill is intentionally not used. Rows without an exact market Date
    match are dropped via inner join.
    """
    left = stock_frame.copy()
    right = market_features.copy()
    left["Date"] = pd.to_datetime(left["Date"])
    right["Date"] = pd.to_datetime(right["Date"])

    before = len(left)
    merged = left.merge(right, on="Date", how="inner", validate="many_to_one")
    dropped = before - len(merged)
    logger.info(
        "Joined market features: stock_rows=%d kept=%d dropped_unmatched=%d (inner join, no fill)",
        before,
        len(merged),
        dropped,
    )
    return merged.sort_values(["Symbol", "Date"], kind="mergesort").reset_index(drop=True)
