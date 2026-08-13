"""Orchestrate market data download and CSV persistence."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from src.config.settings import Settings, get_settings
from src.core.exceptions import (
    DataFetchError,
    DataProviderError,
    DataSaveError,
    EmptyDataError,
)
from src.data.cache import OhlcvCache
from src.data.filenames import ticker_to_filename
from src.data.providers.base import BaseDataProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.universe import Universe, load_universe
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Re-export for existing imports.
__all__ = [
    "MarketDataService",
    "create_provider",
    "main",
    "ticker_to_filename",
]


def create_provider(name: str) -> BaseDataProvider:
    """Instantiate a data provider by configured name.

    Args:
        name: Provider key such as ``yfinance``.

    Returns:
        Concrete ``BaseDataProvider`` implementation.

    Raises:
        DataProviderError: When the provider name is unknown.
    """
    providers: dict[str, type[BaseDataProvider]] = {
        "yfinance": YFinanceProvider,
        # Future providers (not implemented yet):
        # "jquants": JQuantsProvider,
        # "us_official": USOfficialProvider,
        # "global_market": GlobalMarketProvider,
    }
    try:
        provider_cls = providers[name]
    except KeyError as exc:
        known = ", ".join(sorted(providers))
        raise DataProviderError(
            f"Unknown data provider '{name}'. Available providers: {known}"
        ) from exc
    return provider_cls()


class MarketDataService:
    """Fetch market data through a provider with local cache support."""

    def __init__(
        self,
        settings: Settings,
        provider: BaseDataProvider | None = None,
        cache: OhlcvCache | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider or create_provider(settings.data_provider)
        self.cache = cache or OhlcvCache(settings.raw_data_dir)

    def resolve_symbols(self, universe: Universe | None = None) -> tuple[str, ...]:
        """Prefer universe symbols when provided; otherwise settings.tickers."""
        if universe is not None:
            return universe.symbols
        if self.settings.universe_path is not None and self.settings.universe_path.exists():
            return load_universe(self.settings.universe_path).symbols
        return self.settings.tickers

    def run(
        self,
        *,
        force_refresh: bool = False,
        universe: Universe | None = None,
        symbols: tuple[str, ...] | None = None,
    ) -> list[Path]:
        """Download configured tickers (cache-aware) and persist CSV files.

        Returns:
            Paths of successfully written CSV files.
        """
        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=365 * self.settings.lookback_years + 5)
        tickers = symbols or self.resolve_symbols(universe)

        self.settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        for ticker in tickers:
            try:
                frame = self.cache.get_or_fetch(
                    self.provider,
                    ticker,
                    start=start,
                    end=end,
                    interval=self.settings.interval,
                    force_refresh=force_refresh,
                )
                path = self.cache.path_for(ticker)
                if not path.exists():
                    path = self.cache.save(ticker, frame)
                saved_paths.append(path)
                logger.info(
                    "Ready %s (%d rows) force_refresh=%s -> %s",
                    ticker,
                    len(frame),
                    force_refresh,
                    path,
                )
            except EmptyDataError:
                logger.error("Data is empty for ticker=%s", ticker, exc_info=True)
            except DataFetchError:
                logger.error("Failed to fetch ticker=%s", ticker, exc_info=True)
            except DataSaveError:
                logger.error("Failed to save CSV for ticker=%s", ticker, exc_info=True)
            except Exception:
                logger.exception("Unexpected error while processing ticker=%s", ticker)

        if not saved_paths:
            raise DataProviderError("No CSV files were saved. See logs for details.")

        logger.info("Saved %d/%d tickers", len(saved_paths), len(tickers))
        return saved_paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for market data acquisition (global yfinance-backed)."""
    parser = argparse.ArgumentParser(description="Fetch OHLCV via configured provider")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore local cache and re-download the full history window",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Optional universe JSON path (JP/US/EU mix supported)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)

    universe = None
    universe_path = args.universe or settings.universe_path
    if universe_path is not None and Path(universe_path).exists():
        universe = load_universe(Path(universe_path))

    symbols = universe.symbols if universe is not None else settings.tickers
    logger.info(
        "Starting market data fetch: provider=%s symbols=%d lookback_years=%s "
        "force_refresh=%s",
        settings.data_provider,
        len(symbols),
        settings.lookback_years,
        args.force_refresh,
    )

    try:
        service = MarketDataService(settings)
        paths = service.run(force_refresh=args.force_refresh, universe=universe)
    except DataProviderError as exc:
        logger.error("Market data job failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected failure in market data job")
        return 1

    for path in paths:
        logger.info("Output: %s", path)
    logger.info("Market data fetch completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
