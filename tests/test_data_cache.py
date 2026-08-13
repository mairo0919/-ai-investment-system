"""Tests for OHLCV cache incremental behavior."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.data.cache import OhlcvCache
from src.data.providers.base import BaseDataProvider


class _FakeProvider(BaseDataProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[date, date]] = []

    def fetch(
        self,
        ticker: str,
        *,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.calls.append((start, end))
        # Return one row on the start date.
        return pd.DataFrame(
            {
                "Date": [pd.Timestamp(start)],
                "Open": [1.0],
                "High": [1.0],
                "Low": [1.0],
                "Close": [1.0],
                "Adj Close": [1.0],
                "Volume": [100.0],
            }
        )


def test_cache_miss_then_hit_skips_fetch(tmp_path: Path) -> None:
    cache = OhlcvCache(tmp_path)
    provider = _FakeProvider()
    first = cache.get_or_fetch(
        provider,
        "AAPL",
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
    )
    assert len(first) == 1
    assert len(provider.calls) == 1

    # End equals next day after cached date => cache considered fresh.
    second = cache.get_or_fetch(
        provider,
        "AAPL",
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
    )
    assert len(second) == 1
    assert len(provider.calls) == 1


def test_force_refresh_refetches(tmp_path: Path) -> None:
    cache = OhlcvCache(tmp_path)
    provider = _FakeProvider()
    cache.get_or_fetch(
        provider,
        "MSFT",
        start=date(2024, 1, 1),
        end=date(2024, 1, 5),
    )
    cache.get_or_fetch(
        provider,
        "MSFT",
        start=date(2024, 1, 1),
        end=date(2024, 1, 5),
        force_refresh=True,
    )
    assert len(provider.calls) == 2
