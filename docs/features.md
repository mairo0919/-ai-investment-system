# Feature Specification (Phase 2)

Phase 2 generates daily features for AI training and backtesting.
Targets / labels (e.g. next-day return) are **not** created here.

## Principles

- Use only information available at or before each bar's close
- Never use future values (`shift(-1)`, leading windows, etc.)
- Prefer `Adj Close` for price-based features; fall back to `Close`
- Do not mutate files under `data/raw/`
- Write outputs to `data/processed/`
- Keep warmup NaNs until the end, then drop incomplete rows (no `fillna(0)`)

## Input / Output

| Item | Path |
|---|---|
| Input | `data/raw/{ticker with . -> _}.csv` |
| Output | `data/processed/{ticker with . -> _}_features.csv` |

Output columns = original OHLCV + `Symbol` + feature columns.

## Feature Catalog

| Feature | Window | Method | Price / Fields |
|---|---|---|---|
| `return_1d` | 1 | pct change | Adj Close |
| `return_5d` | 5 | pct change | Adj Close |
| `return_20d` | 20 | pct change | Adj Close |
| `log_return_1d` | 1 | `log(p_t / p_{t-1})` | Adj Close |
| `sma_5` | 5 | simple MA | Adj Close |
| `sma_20` | 20 | simple MA | Adj Close |
| `sma_60` | 60 | simple MA | Adj Close |
| `ema_12` | 12 | EMA (`adjust=False`) | Adj Close |
| `ema_26` | 26 | EMA (`adjust=False`) | Adj Close |
| `close_sma_20_ratio` | 20 | `price / sma_20 - 1` | Adj Close |
| `close_sma_60_ratio` | 60 | `price / sma_60 - 1` | Adj Close |
| `rsi_14` | 14 | Wilder RSI | Adj Close |
| `macd` | 12/26 | `EMA12 - EMA26` | Adj Close |
| `macd_signal` | 9 | EMA of MACD | Adj Close |
| `macd_hist` | - | `macd - signal` | Adj Close |
| `bb_middle` | 20 | SMA | Adj Close |
| `bb_upper` | 20 | middle + 2σ | Adj Close |
| `bb_lower` | 20 | middle - 2σ | Adj Close |
| `bb_width` | 20 | `(upper - lower) / middle` | Adj Close |
| `atr_14` | 14 | Wilder ATR | High / Low / Close |
| `volatility_20` | 20 | `std(return_1d) * sqrt(252)` | Adj Close |
| `volume_change_1d` | 1 | pct change | Volume |
| `volume_sma_20` | 20 | simple MA | Volume |
| `volume_ratio_20` | 20 | `Volume / volume_sma_20` | Volume |
| `high_low_range` | 1 | `(High - Low) / Close` | High / Low / Close |
| `open_close_return` | 1 | `(Close - Open) / Open` | Open / Close |

## RSI (Wilder)

1. Compute one-day price changes
2. Split into gains and losses
3. Smooth with Wilder average: `ewm(alpha=1/14, adjust=False, min_periods=14)`
4. `RSI = 100 - 100 / (1 + RS)`, `RS = avg_gain / avg_loss`

## ATR (Wilder)

True Range on day `t`:

```text
max(
  High_t - Low_t,
  abs(High_t - Close_{t-1}),
  abs(Low_t - Close_{t-1})
)
```

ATR is the Wilder smooth of True Range (`period=14`).

## Warmup / invalid rows

1. Compute all features without filling warmup NaNs (`fillna` is not used).
2. Drop the leading warmup block (dominated by `sma_60`, typically 59 rows).
3. Drop any later rows that still contain NaN/inf after safe division
   (for example `volume_change_1d` around zero-volume days in Yahoo data).
4. Log `warmup_dropped` and `invalid_dropped` separately.

Processed CSV must contain neither NaN nor inf in feature columns.
