"""Yahoo Finance market data provider."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

from src.core.exceptions import DataFetchError, EmptyDataError
from src.data.providers.base import REQUIRED_COLUMNS, BaseDataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(BaseDataProvider):
    """Fetch daily OHLCV bars from Yahoo Finance via ``yfinance``."""

    def fetch(
        self,
        ticker: str,
        *,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download OHLCV data and normalize to the project schema."""
        logger.info(
            "Fetching %s from yfinance (start=%s, end=%s, interval=%s)",
            ticker,
            start.isoformat(),
            end.isoformat(),
            interval,
        )

        try:
            raw = yf.download(
                tickers=ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                interval=interval,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001 - network / library failures
            raise DataFetchError(f"Failed to fetch data for {ticker}: {exc}") from exc

        if raw is None or raw.empty:
            raise EmptyDataError(f"No market data returned for {ticker}")

        try:
            frame = self._normalize(raw)
        except (DataFetchError, EmptyDataError):
            raise
        except Exception as exc:  # noqa: BLE001 - unexpected schema issues
            raise DataFetchError(f"Failed to normalize data for {ticker}: {exc}") from exc

        if frame.empty:
            raise EmptyDataError(f"Normalized market data is empty for {ticker}")

        logger.info("Fetched %d rows for %s", len(frame), ticker)
        return frame

    @staticmethod
    def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns and enforce the required schema."""
        frame = raw.copy()

        if isinstance(frame.columns, pd.MultiIndex):
            # yfinance may return (Price, Ticker) or (Ticker, Price).
            level0 = {str(value) for value in frame.columns.get_level_values(0)}
            price_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if level0 & price_names:
                frame.columns = frame.columns.get_level_values(0)
            else:
                frame.columns = frame.columns.get_level_values(1)

        frame = frame.reset_index()

        # Index column name differs across yfinance versions.
        if "Date" not in frame.columns:
            if "Datetime" in frame.columns:
                frame = frame.rename(columns={"Datetime": "Date"})
            elif "index" in frame.columns:
                frame = frame.rename(columns={"index": "Date"})
            else:
                first_col = frame.columns[0]
                frame = frame.rename(columns={first_col: "Date"})

        # Some versions expose adjusted close as Adj Close or Capital Gains-related aliases.
        rename_map = {
            "AdjClose": "Adj Close",
            "adj close": "Adj Close",
            "adj_close": "Adj Close",
        }
        frame = frame.rename(columns=rename_map)

        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise DataFetchError(f"Missing required columns: {missing}")

        frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None).dt.normalize()
        frame = frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        frame = frame.reset_index(drop=True)
        return frame
