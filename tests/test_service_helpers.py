"""Unit tests for Phase 1 helpers that do not require network access."""

from __future__ import annotations

from src.data.service import create_provider, ticker_to_filename
from src.data.providers.yfinance_provider import YFinanceProvider


def test_ticker_to_filename() -> None:
    assert ticker_to_filename("7203.T") == "7203_T.csv"
    assert ticker_to_filename("5333.T") == "5333_T.csv"


def test_create_yfinance_provider() -> None:
    provider = create_provider("yfinance")
    assert isinstance(provider, YFinanceProvider)
