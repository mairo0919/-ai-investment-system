"""Tests for OHLCV cache incremental behavior."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions import EmptyDataError
from src.data.cache import MIN_ROWS_FOR_INCREMENTAL, OhlcvCache
from src.data.providers.base import BaseDataProvider


class _FakeProvider(BaseDataProvider):
    def __init__(self, *, n_rows: int = 1) -> None:
        self.calls: list[tuple[date, date]] = []
        self.n_rows = n_rows

    def fetch(
        self,
        ticker: str,
        *,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.calls.append((start, end))
        dates = pd.bdate_range(start=start, periods=self.n_rows)
        return pd.DataFrame(
            {
                "Date": list(dates),
                "Open": [1.0] * self.n_rows,
                "High": [1.0] * self.n_rows,
                "Low": [1.0] * self.n_rows,
                "Close": [1.0] * self.n_rows,
                "Adj Close": [1.0] * self.n_rows,
                "Volume": [100.0] * self.n_rows,
            }
        )


def _seed_cache(cache: OhlcvCache, ticker: str, *, n_rows: int, last: date) -> None:
    dates = pd.bdate_range(end=last, periods=n_rows)
    frame = pd.DataFrame(
        {
            "Date": list(dates),
            "Open": [1.0] * n_rows,
            "High": [1.0] * n_rows,
            "Low": [1.0] * n_rows,
            "Close": [1.0] * n_rows,
            "Adj Close": [1.0] * n_rows,
            "Volume": [100.0] * n_rows,
        }
    )
    cache.save(ticker, frame)


def test_cache_miss_full_download(tmp_path: Path) -> None:
    cache = OhlcvCache(tmp_path)
    provider = _FakeProvider(n_rows=5)
    first = cache.get_or_fetch(
        provider,
        "AAPL",
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
    )
    assert len(first) == 5
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == date(2024, 1, 1)


def test_sufficient_cache_fresh_skips_fetch(tmp_path: Path) -> None:
    cache = OhlcvCache(tmp_path)
    provider = _FakeProvider()
    last = date(2024, 1, 10)
    _seed_cache(cache, "AAPL", n_rows=MIN_ROWS_FOR_INCREMENTAL, last=last)

    second = cache.get_or_fetch(
        provider,
        "AAPL",
        start=date(2020, 1, 1),
        end=last + timedelta(days=1),
    )
    assert len(second) == MIN_ROWS_FOR_INCREMENTAL
    assert len(provider.calls) == 0


def test_sufficient_cache_incremental_fetch(tmp_path: Path) -> None:
    cache = OhlcvCache(tmp_path)
    provider = _FakeProvider(n_rows=1)
    last = date(2024, 1, 10)
    _seed_cache(cache, "MSFT", n_rows=MIN_ROWS_FOR_INCREMENTAL, last=last)

    out = cache.get_or_fetch(
        provider,
        "MSFT",
        start=date(2020, 1, 1),
        end=date(2024, 1, 20),
    )
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == last + timedelta(days=1)
    assert len(out) >= MIN_ROWS_FOR_INCREMENTAL


def test_thin_stub_cache_triggers_full_historical_fetch(tmp_path: Path) -> None:
    """SAN.MC / ITX.MC style: 1-row stub must not stay on incremental forever."""
    cache = OhlcvCache(tmp_path)
    provider = _FakeProvider(n_rows=120)
    _seed_cache(cache, "SAN.MC", n_rows=1, last=date(2026, 8, 12))

    out = cache.get_or_fetch(
        provider,
        "SAN.MC",
        start=date(2016, 1, 1),
        end=date(2026, 8, 13),
    )
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == date(2016, 1, 1)
    assert len(out) == 120


def test_thin_stub_empty_full_fetch_raises(tmp_path: Path) -> None:
    class _EmptyProvider(BaseDataProvider):
        def fetch(self, ticker, *, start, end, interval="1d"):  # noqa: ANN001
            raise EmptyDataError("no data")

    cache = OhlcvCache(tmp_path)
    _seed_cache(cache, "ITX.MC", n_rows=1, last=date(2026, 8, 12))
    with pytest.raises(EmptyDataError):
        cache.get_or_fetch(
            _EmptyProvider(),
            "ITX.MC",
            start=date(2016, 1, 1),
            end=date(2026, 8, 13),
        )


def test_force_refresh_refetches(tmp_path: Path) -> None:
    cache = OhlcvCache(tmp_path)
    provider = _FakeProvider(n_rows=MIN_ROWS_FOR_INCREMENTAL)
    cache.get_or_fetch(
        provider,
        "MSFT",
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
    )
    cache.get_or_fetch(
        provider,
        "MSFT",
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
        force_refresh=True,
    )
    assert len(provider.calls) == 2

