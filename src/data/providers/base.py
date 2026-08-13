"""Abstract interface for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

# Canonical OHLCV columns persisted to CSV.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)


class BaseDataProvider(ABC):
    """Provider interface so implementations can be swapped without changing callers."""

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        *,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV market data for a ticker.

        Args:
            ticker: Symbol in provider-native format (e.g. ``7203.T``).
            start: Inclusive start date.
            end: Exclusive end date (provider-dependent; treated as upper bound).
            interval: Bar interval string such as ``1d``.

        Returns:
            DataFrame with columns defined in ``REQUIRED_COLUMNS``, sorted by Date.

        Raises:
            DataFetchError: When the remote request fails.
            EmptyDataError: When the response contains no rows.
        """
