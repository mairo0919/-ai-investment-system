"""Tests for Phase 3G cross-sectional / macro feature expansion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.cross_section_features import (
    add_cross_sectional_percentiles,
    add_market_breadth_and_dispersion,
    add_relative_market_returns,
    add_relative_sector_returns,
    add_sector_relative_percentiles,
    ensure_return_60d,
)
from src.features.feature_sets import resolve_feature_sets
from src.features.macro_features import (
    build_macro_series_features,
    join_macro_and_currency_features,
    load_macro_config,
)
from src.ml.ltr_labels import load_relevance_config


def _panel(n_dates: int = 10) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for i, dt in enumerate(pd.bdate_range("2022-01-03", periods=n_dates)):
        for region, n in [("Japan", 8), ("United States", 10), ("Europe", 6)]:
            for j in range(n):
                sector = ["Technology", "Healthcare", "Financial Services"][j % 3]
                close = 100 + j + i
                rows.append(
                    {
                        "Date": dt,
                        "Symbol": f"{region[:2]}{j}",
                        "Region": region,
                        "Sector": sector,
                        "Close": float(close),
                        "Adj Close": float(close),
                        "sma_20": float(close - 1),
                        "sma_60": float(close - 2),
                        "return_1d": rng.normal(0, 0.01),
                        "return_5d": j * 0.01,
                        "return_20d": j * 0.02,
                        "rsi_14": 40 + j,
                        "volatility_20": 0.1 + j * 0.01,
                        "volume_ratio_20": 0.8 + j * 0.05,
                        "close_sma_20_ratio": 0.01 * j,
                        "close_sma_60_ratio": 0.02 * j,
                        "mkt_return_1d": 0.001,
                        "mkt_return_5d": 0.002,
                        "mkt_return_20d": 0.003,
                        "mkt_return_60d": 0.004,
                    }
                )
    return pd.DataFrame(rows)


def test_cross_sectional_percentile_within_country_date() -> None:
    frame = add_cross_sectional_percentiles(_panel(), min_group_size=5)
    assert "cs_return_5d_pct" in frame.columns
    g = frame.loc[
        (frame["Date"] == frame["Date"].iloc[0]) & (frame["Region"] == "United States")
    ]
    assert g["cs_return_5d_pct"].min() >= 0
    assert g["cs_return_5d_pct"].max() <= 1
    # Highest return_5d gets highest percentile.
    assert g.loc[g["return_5d"].idxmax(), "cs_return_5d_pct"] == pytest.approx(
        g["cs_return_5d_pct"].max()
    )


def test_sector_rank_respects_min_group_size() -> None:
    frame = _panel()
    # Force tiny sectors on one day/region
    mask = (frame["Region"] == "Europe") & (frame["Date"] == frame["Date"].iloc[0])
    frame.loc[mask, "Sector"] = [f"Solo{i}" for i in range(mask.sum())]
    out = add_sector_relative_percentiles(frame, min_group_size=3)
    tiny = out.loc[mask, "sector_return_5d_pct"]
    assert tiny.isna().all()


def test_relative_market_and_sector_returns() -> None:
    frame = add_relative_market_returns(_panel())
    assert "relative_return_5d_market" in frame.columns
    assert frame["relative_return_5d_market"].equals(
        frame["return_5d"] - frame["mkt_return_5d"]
    )
    assert "relative_return_60d_market" in frame.columns
    rel_s = add_relative_sector_returns(frame, min_group_size=3)
    assert "relative_return_5d_sector" in rel_s.columns


def test_breadth_and_dispersion() -> None:
    frame = add_market_breadth_and_dispersion(_panel())
    assert frame["breadth_above_sma20"].between(0, 1).all()
    assert frame["breadth_positive_5d"].between(0, 1).all()
    assert "market_dispersion_5d" in frame.columns
    # Same Date×Region share the same breadth value.
    g = frame.loc[
        (frame["Date"] == frame["Date"].iloc[0]) & (frame["Region"] == "Japan"),
        "breadth_above_sma20",
    ]
    assert g.nunique() == 1


def test_macro_features_and_currency_mapping() -> None:
    specs = load_macro_config(Path("config/macro_series.json"))
    assert any(s.ticker == "^VIX" for s in specs)
    assert any(s.ticker == "DX-Y.NYB" for s in specs)
    assert not any(s.ticker == "DX=F" for s in specs)

    dates = pd.bdate_range("2022-01-03", periods=80)
    ohlcv = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.linspace(10, 20, len(dates)),
            "High": np.linspace(10, 20, len(dates)) + 1,
            "Low": np.linspace(10, 20, len(dates)) - 1,
            "Close": np.linspace(10, 20, len(dates)),
            "Adj Close": np.linspace(10, 20, len(dates)),
            "Volume": 1000,
        }
    )
    vix = build_macro_series_features(ohlcv, key="vix", style="level")
    assert "macro_vix_level" in vix.columns
    assert "macro_vix_change_5d" in vix.columns
    ret = build_macro_series_features(ohlcv, key="gold", style="return")
    assert "macro_gold_return_20d" in ret.columns

    panel = _panel(n_dates=60)
    # Build minimal macro frames for join
    from src.features.macro_features import MacroSeriesSpec

    macro_frames = {
        "vix": (
            MacroSeriesSpec("^VIX", "vix", "risk", "level", "global"),
            vix,
        ),
        "jpy": (
            MacroSeriesSpec("JPY=X", "jpy", "fx", "return", "currency", ("Japan",)),
            build_macro_series_features(ohlcv, key="jpy", style="return"),
        ),
    }
    joined, cols = join_macro_and_currency_features(panel, macro_frames)
    assert "macro_vix_level" in joined.columns
    assert "fx_local_return_5d" in joined.columns
    # Japan rows should have fx filled once panel dates overlap macro warmup.
    jp = joined.loc[joined["Region"] == "Japan", "fx_local_return_5d"]
    assert jp.notna().any()
    us = joined.loc[joined["Region"] == "United States", "fx_local_return_5d"]
    assert us.isna().all()


def test_feature_set_sizes_and_no_future_leak() -> None:
    specs = load_macro_config()
    sets = resolve_feature_sets(specs)
    assert len(sets["A"]) < len(sets["B"]) < len(sets["C"])
    for name, cols in sets.items():
        assert not any(c.startswith("future_return") for c in cols)
        assert "Symbol" not in cols
        assert "Region" not in cols


def test_ensure_return_60d_no_lookahead_shape() -> None:
    frame = ensure_return_60d(_panel())
    assert "return_60d" in frame.columns
    # First 60 bars per symbol are NaN-ish; with only 10 dates all may be NaN — OK.
    assert len(frame) == len(_panel())


def test_relevance_config_still_loads() -> None:
    cfg = load_relevance_config()
    assert cfg.min_group_size >= 1
