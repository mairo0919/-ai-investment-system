"""Local OHLCV cache with incremental refresh support."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.core.exceptions import DataSaveError, EmptyDataError
from src.data.filenames import ticker_to_filename
from src.data.providers.base import REQUIRED_COLUMNS, BaseDataProvider

logger = logging.getLogger(__name__)

# Caches thinner than this are treated as incomplete stubs (e.g. a single-day
# first fetch). Incremental refresh must not freeze that stub forever.
MIN_ROWS_FOR_INCREMENTAL = 60


class OhlcvCache:
    """File-backed OHLCV cache stored under ``data/raw`` (or a custom directory)."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, ticker: str) -> Path:
        """Return the cache CSV path for a ticker."""
        return self.cache_dir / ticker_to_filename(ticker)

    def load(self, ticker: str) -> pd.DataFrame | None:
        """Load cached OHLCV if present and non-empty."""
        path = self.path_for(ticker)
        if not path.exists():
            return None
        try:
            frame = pd.read_csv(path)
        except OSError as exc:
            logger.warning("Failed to read cache for %s: %s", ticker, exc)
            return None
        if frame.empty:
            return None
        if "Date" not in frame.columns:
            logger.warning("Cache for %s missing Date column; ignoring", ticker)
            return None
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).sort_values("Date")
        frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        return frame

    def save(self, ticker: str, frame: pd.DataFrame) -> Path:
        """Persist OHLCV to cache."""
        if frame.empty:
            raise EmptyDataError(f"Cannot cache empty frame for {ticker}")
        path = self.path_for(ticker)
        out = frame.copy()
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        try:
            out.to_csv(path, index=False)
        except OSError as exc:
            raise DataSaveError(f"Failed to write cache {path}: {exc}") from exc
        logger.info("Cached %s (%d rows) -> %s", ticker, len(frame), path)
        return path

    def get_or_fetch(
        self,
        provider: BaseDataProvider,
        ticker: str,
        *,
        start: date,
        end: date,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return OHLCV using cache when possible; fetch only missing ranges.

        Behavior:
            - First call / force_refresh: full download ``[start, end)``
            - Later calls: reuse cache and append only bars after the last cached date
        """
        if force_refresh:
            logger.info("Force refresh for %s", ticker)
            frame = provider.fetch(ticker, start=start, end=end, interval=interval)
            self.save(ticker, frame)
            return frame

        cached = self.load(ticker)
        if cached is None:
            logger.info("Cache miss for %s; full download", ticker)
            frame = provider.fetch(ticker, start=start, end=end, interval=interval)
            self.save(ticker, frame)
            return frame

        # Incomplete stub cache (e.g. SAN.MC / ITX.MC with 1 row): always full
        # historical fetch for the requested window — never incremental-extend.
        if len(cached) < MIN_ROWS_FOR_INCREMENTAL:
            logger.info(
                "Cache for %s has only %d rows (< %d); full historical download",
                ticker,
                len(cached),
                MIN_ROWS_FOR_INCREMENTAL,
            )
            frame = provider.fetch(ticker, start=start, end=end, interval=interval)
            self.save(ticker, frame)
            return frame

        last_cached = pd.Timestamp(cached["Date"].max()).date()
        # Request bars strictly after the last cached session.
        incremental_start = last_cached + timedelta(days=1)
        if incremental_start >= end:
            logger.info("Cache fresh for %s through %s", ticker, last_cached.isoformat())
            return _normalize_cached(cached)

        logger.info(
            "Incremental fetch for %s from %s to %s",
            ticker,
            incremental_start.isoformat(),
            end.isoformat(),
        )
        try:
            fresh = provider.fetch(
                ticker,
                start=incremental_start,
                end=end,
                interval=interval,
            )
        except EmptyDataError:
            logger.info("No new rows for %s; returning cache", ticker)
            return _normalize_cached(cached)

        merged = pd.concat([cached, fresh], ignore_index=True, sort=False)
        merged["Date"] = pd.to_datetime(merged["Date"])
        merged = (
            merged.sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .reset_index(drop=True)
        )
        # Keep only the requested lookback window when possible.
        start_ts = pd.Timestamp(start)
        merged = merged.loc[merged["Date"] >= start_ts].reset_index(drop=True)
        self.save(ticker, merged)
        return _normalize_cached(merged)


def _normalize_cached(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in out.columns]
    if missing:
        raise EmptyDataError(f"Cached frame missing columns: {missing}")
    out = out.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out.sort_values("Date").reset_index(drop=True)
