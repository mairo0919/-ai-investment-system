"""Macro / FX / commodity context features from yfinance (point-in-time)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import TrainingError
from src.data.cache import OhlcvCache
from src.data.providers.base import BaseDataProvider
from src.features.indicators import TRADING_DAYS_PER_YEAR, safe_divide, select_price, sma

logger = logging.getLogger(__name__)

DEFAULT_MACRO_CONFIG = Path("config/macro_series.json")


@dataclass(frozen=True)
class MacroSeriesSpec:
    ticker: str
    key: str
    category: str
    style: str  # "return" | "level"
    attach: str  # "global" | "currency"
    regions: tuple[str, ...] = ()


def load_macro_config(path: Path | None = None) -> tuple[MacroSeriesSpec, ...]:
    path = path or DEFAULT_MACRO_CONFIG
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for item in payload.get("series", []):
        specs.append(
            MacroSeriesSpec(
                ticker=str(item["ticker"]),
                key=str(item["key"]),
                category=str(item.get("category", "other")),
                style=str(item.get("style", "return")),
                attach=str(item.get("attach", "global")),
                regions=tuple(item.get("regions") or ()),
            )
        )
    return tuple(specs)


def build_macro_series_features(ohlcv: pd.DataFrame, *, key: str, style: str) -> pd.DataFrame:
    """Build compact features for one macro series."""
    frame = ohlcv.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").drop_duplicates("Date", keep="last")
    price = select_price(frame)
    prefix = f"macro_{key}"
    out = pd.DataFrame({"Date": frame["Date"].to_numpy()})

    if style == "level":
        out[f"{prefix}_level"] = price.to_numpy()
        out[f"{prefix}_change_5d"] = price.diff(5).to_numpy()
        sma20 = sma(price, 20)
        out[f"{prefix}_sma_20_ratio"] = (safe_divide(price, sma20) - 1.0).to_numpy()
        out[f"{prefix}_return_1d"] = price.pct_change(1).to_numpy()
        out[f"{prefix}_return_5d"] = price.pct_change(5).to_numpy()
    else:
        out[f"{prefix}_return_1d"] = price.pct_change(1).to_numpy()
        out[f"{prefix}_return_5d"] = price.pct_change(5).to_numpy()
        out[f"{prefix}_return_20d"] = price.pct_change(20).to_numpy()
        sma20 = sma(price, 20)
        out[f"{prefix}_sma_20_ratio"] = (safe_divide(price, sma20) - 1.0).to_numpy()
        out[f"{prefix}_volatility_20"] = (
            price.pct_change(1).rolling(20, min_periods=20).std(ddof=0)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        ).to_numpy()

    feat_cols = [c for c in out.columns if c != "Date"]
    out.loc[:, feat_cols] = out.loc[:, feat_cols].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feat_cols).reset_index(drop=True)
    return out


def fetch_macro_feature_frames(
    provider: BaseDataProvider,
    cache: OhlcvCache,
    specs: tuple[MacroSeriesSpec, ...],
    *,
    lookback_years: int,
    interval: str = "1d",
    force_refresh: bool = False,
) -> dict[str, tuple[MacroSeriesSpec, pd.DataFrame]]:
    """Fetch/cache each macro ticker and build feature frames."""
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=365 * lookback_years + 5)
    out: dict[str, tuple[MacroSeriesSpec, pd.DataFrame]] = {}
    for spec in specs:
        try:
            raw = cache.get_or_fetch(
                provider,
                spec.ticker,
                start=start,
                end=end,
                interval=interval,
                force_refresh=force_refresh,
            )
            feats = build_macro_series_features(raw, key=spec.key, style=spec.style)
            out[spec.key] = (spec, feats)
            logger.info(
                "Macro ready key=%s ticker=%s rows=%d style=%s",
                spec.key,
                spec.ticker,
                len(feats),
                spec.style,
            )
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(
                f"Required macro series failed ticker={spec.ticker}: {exc}"
            ) from exc
    return out


def _asof_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time join: last available macro row on or before stock Date."""
    left_sorted = left.copy()
    right_sorted = right.copy()
    # Align datetime resolutions (us vs s) for merge_asof.
    left_sorted["Date"] = pd.to_datetime(left_sorted["Date"]).astype("datetime64[ns]")
    right_sorted["Date"] = pd.to_datetime(right_sorted["Date"]).astype("datetime64[ns]")
    left_sorted = left_sorted.sort_values("Date", kind="mergesort")
    right_sorted = right_sorted.sort_values("Date", kind="mergesort")
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        on="Date",
        direction="backward",
    )
    return merged


def join_macro_and_currency_features(
    panel: pd.DataFrame,
    macro_frames: dict[str, tuple[MacroSeriesSpec, pd.DataFrame]],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Attach global macros and region-mapped FX context columns."""
    out = panel.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    added: list[str] = []

    # Global macros
    for key, (spec, feats) in macro_frames.items():
        if spec.attach != "global":
            continue
        before_cols = set(out.columns)
        out = _asof_join(out, feats)
        new_cols = [c for c in out.columns if c not in before_cols]
        added.extend(new_cols)

    # Currency: map to generic fx_local_* columns by region
    fx_generic_suffixes = (
        "return_1d",
        "return_5d",
        "return_20d",
        "sma_20_ratio",
        "volatility_20",
    )
    for suffix in fx_generic_suffixes:
        out[f"fx_local_{suffix}"] = np.nan
        added.append(f"fx_local_{suffix}")

    parts: list[pd.DataFrame] = []
    for region, group in out.groupby("Region", sort=False):
        g = group.copy()
        matched_spec = None
        matched_feats = None
        for key, (spec, feats) in macro_frames.items():
            if spec.attach == "currency" and region in spec.regions:
                matched_spec = spec
                matched_feats = feats
                break
        if matched_spec is None or matched_feats is None:
            parts.append(g)
            continue
        rename = {
            f"macro_{matched_spec.key}_{suffix}": f"fx_local_{suffix}"
            for suffix in fx_generic_suffixes
            if f"macro_{matched_spec.key}_{suffix}" in matched_feats.columns
        }
        fx = matched_feats.rename(columns=rename)[
            ["Date", *rename.values()]
        ].sort_values("Date")
        # Avoid colliding with empty fx_local_* placeholders already on g.
        g = g.drop(columns=list(rename.values()), errors="ignore")
        g = _asof_join(g.sort_values("Date", kind="mergesort"), fx)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True, sort=False)

    # Deduplicate added list while preserving order
    seen: set[str] = set()
    unique_added = []
    for c in added:
        if c not in seen and c in out.columns:
            seen.add(c)
            unique_added.append(c)

    logger.info("Joined macro/fx features: +%d columns", len(unique_added))
    return out, tuple(unique_added)


def compute_mkt_return_60d(
    market_ohlcv_by_region: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build Date → mkt_return_60d frames per region from benchmark OHLCV."""
    out: dict[str, pd.DataFrame] = {}
    for region, raw in market_ohlcv_by_region.items():
        frame = raw.copy()
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.sort_values("Date").drop_duplicates("Date", keep="last")
        price = select_price(frame)
        out[region] = pd.DataFrame(
            {
                "Date": frame["Date"].to_numpy(),
                "mkt_return_60d": price.pct_change(60).to_numpy(),
            }
        )
    return out


def attach_mkt_return_60d(
    panel: pd.DataFrame,
    mkt_60_by_region: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Attach region-matched mkt_return_60d via point-in-time asof join."""
    parts = []
    for region, group in panel.groupby("Region", sort=False):
        if region not in mkt_60_by_region:
            g = group.copy()
            g["mkt_return_60d"] = np.nan
            parts.append(g)
            continue
        g = group.sort_values("Date", kind="mergesort")
        parts.append(_asof_join(g, mkt_60_by_region[str(region)]))
    return pd.concat(parts, ignore_index=True, sort=False)


def macro_feature_column_names(
    specs: tuple[MacroSeriesSpec, ...],
    style_lookup: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Expected macro + fx_local column names for Feature Set C."""
    cols: list[str] = []
    for spec in specs:
        if spec.attach != "global":
            continue
        if spec.style == "level":
            cols.extend(
                [
                    f"macro_{spec.key}_level",
                    f"macro_{spec.key}_change_5d",
                    f"macro_{spec.key}_sma_20_ratio",
                    f"macro_{spec.key}_return_1d",
                    f"macro_{spec.key}_return_5d",
                ]
            )
        else:
            cols.extend(
                [
                    f"macro_{spec.key}_return_1d",
                    f"macro_{spec.key}_return_5d",
                    f"macro_{spec.key}_return_20d",
                    f"macro_{spec.key}_sma_20_ratio",
                    f"macro_{spec.key}_volatility_20",
                ]
            )
    cols.extend(
        [
            "fx_local_return_1d",
            "fx_local_return_5d",
            "fx_local_return_20d",
            "fx_local_sma_20_ratio",
            "fx_local_volatility_20",
        ]
    )
    return tuple(cols)


def summarize_macro_fetch(
    macro_frames: dict[str, tuple[MacroSeriesSpec, pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows = []
    for key, (spec, feats) in macro_frames.items():
        rows.append(
            {
                "key": key,
                "ticker": spec.ticker,
                "category": spec.category,
                "style": spec.style,
                "attach": spec.attach,
                "regions": list(spec.regions),
                "rows": int(len(feats)),
                "date_start": str(pd.to_datetime(feats["Date"]).min().date()),
                "date_end": str(pd.to_datetime(feats["Date"]).max().date()),
            }
        )
    return rows
