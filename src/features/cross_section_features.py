"""Date×Country / Sector cross-sectional and breadth features (no look-ahead)."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.indicators import select_price

logger = logging.getLogger(__name__)

DEFAULT_CS_COLUMNS: tuple[str, ...] = (
    "return_5d",
    "return_20d",
    "rsi_14",
    "volatility_20",
    "volume_ratio_20",
    "close_sma_20_ratio",
    "close_sma_60_ratio",
)

DEFAULT_SECTOR_COLUMNS: tuple[str, ...] = (
    "return_5d",
    "return_20d",
    "rsi_14",
    "volatility_20",
)

CS_FEATURE_COLUMNS: tuple[str, ...] = tuple(f"cs_{c}_pct" for c in DEFAULT_CS_COLUMNS)
SECTOR_PCT_COLUMNS: tuple[str, ...] = (
    "sector_return_5d_pct",
    "sector_return_20d_pct",
    "sector_rsi_pct",
    "sector_volatility_pct",
)
RELATIVE_MARKET_COLUMNS: tuple[str, ...] = (
    "relative_return_1d_market",
    "relative_return_5d_market",
    "relative_return_20d_market",
    "relative_return_60d_market",
)
RELATIVE_SECTOR_COLUMNS: tuple[str, ...] = (
    "relative_return_5d_sector",
    "relative_return_20d_sector",
)
BREADTH_COLUMNS: tuple[str, ...] = (
    "breadth_above_sma20",
    "breadth_above_sma60",
    "breadth_positive_5d",
    "breadth_positive_20d",
    "median_return_5d",
    "median_return_20d",
)
DISPERSION_COLUMNS: tuple[str, ...] = (
    "market_dispersion_5d",
    "market_dispersion_20d",
)


def _group_percentile(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str],
    group_keys: list[str],
    prefix: str,
    min_group_size: int,
    name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Within-group percentile ranks in [0, 1]; small groups → NaN."""
    out = frame.copy()
    sizes = out.groupby(group_keys, sort=False)[out.columns[0]].transform("size")
    for col in columns:
        if col not in out.columns:
            dest = name_map[col] if name_map else f"{prefix}_{col}_pct"
            out[dest] = np.nan
            continue
        dest = name_map[col] if name_map else f"{prefix}_{col}_pct"
        ranked = out.groupby(group_keys, sort=False)[col].rank(method="average", pct=True)
        out[dest] = ranked.where(sizes >= min_group_size, np.nan)
    return out


def add_cross_sectional_percentiles(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = DEFAULT_CS_COLUMNS,
    date_col: str = "Date",
    region_col: str = "Region",
    min_group_size: int = 5,
) -> pd.DataFrame:
    """Date × Country percentile ranks for selected stock features."""
    return _group_percentile(
        frame,
        columns=columns,
        group_keys=[date_col, region_col],
        prefix="cs",
        min_group_size=min_group_size,
    )


def add_sector_relative_percentiles(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = DEFAULT_SECTOR_COLUMNS,
    date_col: str = "Date",
    region_col: str = "Region",
    sector_col: str = "Sector",
    min_group_size: int = 3,
) -> pd.DataFrame:
    """Date × Country × Sector percentile ranks (null when group too small)."""
    name_map = {
        "return_5d": "sector_return_5d_pct",
        "return_20d": "sector_return_20d_pct",
        "rsi_14": "sector_rsi_pct",
        "volatility_20": "sector_volatility_pct",
    }
    return _group_percentile(
        frame,
        columns=columns,
        group_keys=[date_col, region_col, sector_col],
        prefix="sector",
        min_group_size=min_group_size,
        name_map=name_map,
    )


def ensure_return_60d(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stock return_60d from price if missing (point-in-time pct_change)."""
    out = frame.copy()
    if "return_60d" in out.columns:
        return out
    if "Symbol" not in out.columns:
        raise ValueError("ensure_return_60d requires Symbol column")
    out = out.sort_values(["Symbol", "Date"], kind="mergesort")
    price = select_price(out)
    out["return_60d"] = price.groupby(out["Symbol"], sort=False).pct_change(60)
    return out


def add_relative_market_returns(
    frame: pd.DataFrame,
    *,
    market_return_cols: dict[int, str] | None = None,
) -> pd.DataFrame:
    """stock return − country benchmark return for configured horizons."""
    out = ensure_return_60d(frame)
    mapping = market_return_cols or {
        1: "mkt_return_1d",
        5: "mkt_return_5d",
        20: "mkt_return_20d",
    }
    stock_cols = {
        1: "return_1d",
        5: "return_5d",
        20: "return_20d",
        60: "return_60d",
    }
    for horizon, stock_col in stock_cols.items():
        dest = f"relative_return_{horizon}d_market"
        mkt_col = mapping.get(horizon)
        if mkt_col is None or mkt_col not in out.columns:
            # 60d market return may be supplied separately as mkt_return_60d
            mkt_col = f"mkt_return_{horizon}d"
        if stock_col not in out.columns or mkt_col not in out.columns:
            out[dest] = np.nan
            continue
        out[dest] = out[stock_col] - out[mkt_col]
    return out


def add_relative_sector_returns(
    frame: pd.DataFrame,
    *,
    date_col: str = "Date",
    region_col: str = "Region",
    sector_col: str = "Sector",
    min_group_size: int = 3,
    horizons: tuple[int, ...] = (5, 20),
) -> pd.DataFrame:
    """stock return − same-day Date×Country×Sector mean return."""
    out = frame.copy()
    keys = [date_col, region_col, sector_col]
    sizes = out.groupby(keys, sort=False)[out.columns[0]].transform("size")
    for h in horizons:
        col = f"return_{h}d"
        dest = f"relative_return_{h}d_sector"
        if col not in out.columns:
            out[dest] = np.nan
            continue
        sector_mean = out.groupby(keys, sort=False)[col].transform("mean")
        out[dest] = (out[col] - sector_mean).where(sizes >= min_group_size, np.nan)
    return out


def add_market_breadth_and_dispersion(
    frame: pd.DataFrame,
    *,
    date_col: str = "Date",
    region_col: str = "Region",
) -> pd.DataFrame:
    """Universe-internal breadth / dispersion by Date × Country (same-day only)."""
    out = frame.copy()
    keys = [date_col, region_col]

    def _ratio(mask: pd.Series) -> pd.Series:
        return (
            mask.astype(float)
            .groupby([out[date_col], out[region_col]], sort=False)
            .transform("mean")
        )

    if {"Close", "sma_20"} <= set(out.columns):
        out["breadth_above_sma20"] = _ratio(out["Close"] > out["sma_20"])
    else:
        out["breadth_above_sma20"] = np.nan
    if {"Close", "sma_60"} <= set(out.columns):
        out["breadth_above_sma60"] = _ratio(out["Close"] > out["sma_60"])
    else:
        out["breadth_above_sma60"] = np.nan

    if "return_5d" in out.columns:
        out["breadth_positive_5d"] = _ratio(out["return_5d"] > 0)
    else:
        out["breadth_positive_5d"] = np.nan
    if "return_20d" in out.columns:
        out["breadth_positive_20d"] = _ratio(out["return_20d"] > 0)
    else:
        out["breadth_positive_20d"] = np.nan
    out["median_return_5d"] = (
        out.groupby(keys, sort=False)["return_5d"].transform("median")
        if "return_5d" in out
        else np.nan
    )
    out["median_return_20d"] = (
        out.groupby(keys, sort=False)["return_20d"].transform("median")
        if "return_20d" in out
        else np.nan
    )
    out["market_dispersion_5d"] = (
        out.groupby(keys, sort=False)["return_5d"].transform("std")
        if "return_5d" in out
        else np.nan
    )
    out["market_dispersion_20d"] = (
        out.groupby(keys, sort=False)["return_20d"].transform("std")
        if "return_20d" in out
        else np.nan
    )
    return out


def add_cross_section_feature_block(
    frame: pd.DataFrame,
    *,
    min_sector_group_size: int = 3,
    min_cs_group_size: int = 5,
) -> pd.DataFrame:
    """Apply CS + sector-relative + relative strength + breadth/dispersion."""
    out = add_cross_sectional_percentiles(frame, min_group_size=min_cs_group_size)
    out = add_sector_relative_percentiles(out, min_group_size=min_sector_group_size)
    out = add_relative_market_returns(out)
    out = add_relative_sector_returns(out, min_group_size=min_sector_group_size)
    out = add_market_breadth_and_dispersion(out)
    logger.info(
        "Cross-section feature block applied rows=%d cs_cols=%d",
        len(out),
        len(CS_FEATURE_COLUMNS),
    )
    return out
