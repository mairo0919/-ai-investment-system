"""Fetch and persist market-index OHLCV for Phase 3C context features."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from src.config.settings import Settings, get_settings
from src.core.exceptions import (
    DataFetchError,
    DataProviderError,
    DataSaveError,
    EmptyDataError,
)
from src.data.providers.base import BaseDataProvider
from src.data.filenames import ticker_to_filename
from src.data.service import create_provider
from src.features.market_features import build_market_features
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Nikkei 225 works on Yahoo Finance.
NIKKEI_TICKER = "^N225"
NIKKEI_PREFIX = "nikkei"

# Official TOPIX cash index tickers currently fail on Yahoo Finance.
# Do not silently substitute an ETF proxy.
TOPIX_TICKER_CANDIDATES: tuple[str, ...] = ("^TOPX", "TOPIX.T", "TOPX")
TOPIX_PREFIX = "topix"


@dataclass(frozen=True)
class MarketFetchResult:
    """Result of attempting to download one market index."""

    name: str
    ticker: str | None
    available: bool
    raw_path: Path | None
    processed_path: Path | None
    rows_raw: int = 0
    rows_processed: int = 0
    error: str | None = None


class MarketDataService:
    """Download market indices into data/raw/market and feature CSVs."""

    def __init__(
        self,
        settings: Settings,
        provider: BaseDataProvider | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider or create_provider(settings.data_provider)
        self.raw_market_dir = settings.raw_data_dir / "market"
        self.processed_market_dir = settings.processed_data_dir / "market"

    def run(self) -> dict[str, MarketFetchResult]:
        """Fetch Nikkei and attempt TOPIX; persist whatever succeeds."""
        self.raw_market_dir.mkdir(parents=True, exist_ok=True)
        self.processed_market_dir.mkdir(parents=True, exist_ok=True)

        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=365 * self.settings.lookback_years + 5)

        results: dict[str, MarketFetchResult] = {}
        results["nikkei"] = self._fetch_one(
            name="nikkei",
            ticker=NIKKEI_TICKER,
            prefix=NIKKEI_PREFIX,
            start=start,
            end=end,
        )
        results["topix"] = self._fetch_topix(start=start, end=end)
        return results

    def _fetch_topix(self, *, start: date, end: date) -> MarketFetchResult:
        errors: list[str] = []
        for ticker in TOPIX_TICKER_CANDIDATES:
            try:
                return self._fetch_one(
                    name="topix",
                    ticker=ticker,
                    prefix=TOPIX_PREFIX,
                    start=start,
                    end=end,
                )
            except (DataFetchError, EmptyDataError) as exc:
                errors.append(f"{ticker}: {exc}")
                logger.warning("TOPIX candidate failed (%s): %s", ticker, exc)

        message = (
            "TOPIX cash index is unavailable via yfinance for candidates "
            f"{list(TOPIX_TICKER_CANDIDATES)}. No ETF proxy was substituted. "
            f"Details: {' | '.join(errors)}"
        )
        logger.error(message)
        return MarketFetchResult(
            name="topix",
            ticker=None,
            available=False,
            raw_path=None,
            processed_path=None,
            error=message,
        )

    def _fetch_one(
        self,
        *,
        name: str,
        ticker: str,
        prefix: str,
        start: date,
        end: date,
    ) -> MarketFetchResult:
        frame = self.provider.fetch(ticker, start=start, end=end, interval=self.settings.interval)
        raw_path = self.raw_market_dir / ticker_to_filename(ticker)
        try:
            frame.to_csv(raw_path, index=False)
        except OSError as exc:
            raise DataSaveError(f"Failed to save market raw CSV {raw_path}: {exc}") from exc

        features = build_market_features(frame, prefix=prefix)
        processed_path = self.processed_market_dir / f"{prefix}_features.csv"
        try:
            features.to_csv(processed_path, index=False)
        except OSError as exc:
            raise DataSaveError(
                f"Failed to save market feature CSV {processed_path}: {exc}"
            ) from exc

        logger.info(
            "Saved market index %s (%s): raw=%s processed=%s",
            name,
            ticker,
            raw_path,
            processed_path,
        )
        return MarketFetchResult(
            name=name,
            ticker=ticker,
            available=True,
            raw_path=raw_path,
            processed_path=processed_path,
            rows_raw=len(frame),
            rows_processed=len(features),
        )


def main() -> int:
    """CLI entry point for market-index download."""
    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    logger.info("Starting market index fetch")
    try:
        results = MarketDataService(settings).run()
    except DataProviderError as exc:
        logger.error("Market fetch failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected market fetch failure")
        return 1

    for name, result in results.items():
        if result.available:
            logger.info(
                "%s OK ticker=%s raw_rows=%d processed_rows=%d",
                name,
                result.ticker,
                result.rows_raw,
                result.rows_processed,
            )
        else:
            logger.error("%s UNAVAILABLE: %s", name, result.error)
    # Success if at least Nikkei is available.
    return 0 if results["nikkei"].available else 1


if __name__ == "__main__":
    sys.exit(main())
