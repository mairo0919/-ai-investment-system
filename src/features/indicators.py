"""Pure feature-calculation helpers for daily OHLCV bars.

All calculations use only information available at or before the bar's close.
No future values (``shift(-1)``, lead windows, or forward-looking targets) are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "return_20d",
    "log_return_1d",
    "sma_5",
    "sma_20",
    "sma_60",
    "ema_12",
    "ema_26",
    "close_sma_20_ratio",
    "close_sma_60_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr_14",
    "volatility_20",
    "volume_change_1d",
    "volume_sma_20",
    "volume_ratio_20",
    "high_low_range",
    "open_close_return",
)

TRADING_DAYS_PER_YEAR = 252


def select_price(frame: pd.DataFrame) -> pd.Series:
    """Prefer ``Adj Close``; fall back to ``Close`` when unavailable."""
    if "Adj Close" in frame.columns and frame["Adj Close"].notna().any():
        return frame["Adj Close"].astype(float)
    return frame["Close"].astype(float)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN instead of ``inf`` on zero denominators."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average with a full-window warmup."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (``adjust=False``) with a full-span warmup."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Method:
        1. Compute one-day price changes.
        2. Separate average gains and losses.
        3. Smooth with Wilder's recursive average, equivalent to
           ``ewm(alpha=1/period, adjust=False, min_periods=period)``.
        4. ``RSI = 100 - 100 / (1 + RS)`` where ``RS = avg_gain / avg_loss``.

    Values are expected in roughly ``[0, 100]`` after warmup.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = safe_divide(avg_gain, avg_loss)
    rsi = 100.0 - safe_divide(pd.Series(100.0, index=series.index), 1.0 + rs)

    # When average loss is exactly 0 and gain > 0, RSI is defined as 100.
    no_loss = avg_loss.eq(0) & avg_gain.gt(0)
    rsi = rsi.mask(no_loss, 100.0)
    # When both are 0, RSI is neutral 50 (flat market).
    both_zero = avg_loss.eq(0) & avg_gain.eq(0)
    rsi = rsi.mask(both_zero, 50.0)
    return rsi


def macd_components(
    series: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(
    series: pd.Series,
    *,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands and normalized width ``(upper - lower) / middle``."""
    middle = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = safe_divide(upper - lower, middle)
    return middle, upper, lower, width


def atr_wilder(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range using Wilder's smoothing.

    True Range on day ``t`` uses ``High_t``, ``Low_t``, and ``Close_{t-1}`` only.
    """
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append Phase 2 feature columns to a validated OHLCV DataFrame.

    Args:
        frame: Date-sorted OHLCV data. Must include Open/High/Low/Close/Volume.
            Adj Close is preferred for return and trend features.

    Returns:
        Copy of ``frame`` with feature columns appended. Warmup NaNs are retained;
        callers should drop incomplete rows after generation.
    """
    out = frame.copy()
    price = select_price(out)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    close = out["Close"].astype(float)
    open_ = out["Open"].astype(float)
    volume = out["Volume"].astype(float)

    out["return_1d"] = price.pct_change(1)
    out["return_5d"] = price.pct_change(5)
    out["return_20d"] = price.pct_change(20)
    out["log_return_1d"] = np.log(safe_divide(price, price.shift(1)))

    out["sma_5"] = sma(price, 5)
    out["sma_20"] = sma(price, 20)
    out["sma_60"] = sma(price, 60)
    out["ema_12"] = ema(price, 12)
    out["ema_26"] = ema(price, 26)
    out["close_sma_20_ratio"] = safe_divide(price, out["sma_20"]) - 1.0
    out["close_sma_60_ratio"] = safe_divide(price, out["sma_60"]) - 1.0

    out["rsi_14"] = rsi_wilder(price, 14)

    macd_line, signal_line, hist = macd_components(price)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    bb_middle, bb_upper, bb_lower, bb_width = bollinger_bands(price)
    out["bb_middle"] = bb_middle
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower
    out["bb_width"] = bb_width

    # ATR uses raw High/Low/Close as specified (not Adj Close).
    out["atr_14"] = atr_wilder(high, low, close, 14)

    out["volatility_20"] = out["return_1d"].rolling(window=20, min_periods=20).std(
        ddof=0
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)

    out["volume_change_1d"] = volume.pct_change(1)
    out["volume_sma_20"] = sma(volume, 20)
    out["volume_ratio_20"] = safe_divide(volume, out["volume_sma_20"])

    out["high_low_range"] = safe_divide(high - low, close)
    out["open_close_return"] = safe_divide(close - open_, open_)

    # Replace any residual inf values with NaN so warmup filtering can remove them.
    feature_block = out.loc[:, list(FEATURE_COLUMNS)]
    out.loc[:, list(FEATURE_COLUMNS)] = feature_block.replace([np.inf, -np.inf], np.nan)
    return out
