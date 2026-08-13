"""Shared filename helpers for market data artifacts."""


def ticker_to_filename(ticker: str) -> str:
    """Convert a ticker symbol to a CSV filename.

    Example:
        ``7203.T`` -> ``7203_T.csv``
        ``BRK-B`` -> ``BRK-B.csv``
    """
    return f"{ticker.replace('.', '_')}.csv"
