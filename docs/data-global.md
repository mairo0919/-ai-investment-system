# Global Market Data (yfinance-first)

## Policy

- **Current provider**: `yfinance` only
- **Scope**: Japan / US / Europe / other yfinance-accessible markets
- **Not now**: J-Quants, paid market-data APIs, brokerage APIs
- Callers use `BaseDataProvider` — do not depend on yfinance types outside the provider module

## Universe

| File | Role |
|---|---|
| `config/universe.sample.json` | 8-name pilot |
| `config/universe.global100.json` | Research-fixed ~100 names (JP30 / US40 / EU30) |

Required fields per instrument: `symbol`, `country`, `market`, `currency`.  
Optional / enrichable: `exchange`, `sector`, `industry`, `market_cap`, `timezone`.

This Global100 set is a **research fixed universe**, not a live investable index.  
**Known limitation: survivorship bias** (names selected ex-post among currently liquid large/mid caps).

Do not hardcode the list in Python. Do not download the entire global listing.

```bash
fetch-market-data --universe config/universe.global100.json
fetch-market-data --universe config/universe.global100.json --force-refresh
```

Optional `.env`:

```bash
UNIVERSE_PATH=config/universe.global100.json
DATA_PROVIDER=yfinance
```

## Market benchmarks (country context)

Verified yfinance tickers used as regional market features:

| Region | Ticker |
|---|---|
| Japan | `^N225` |
| United States | `^GSPC` |
| Europe | `^STOXX50E` |

Currency metadata is retained. FX-converted portfolio returns are **not** computed yet.  
Global pooled absolute returns mix currencies — prefer **country-internal ranking** for interpretation.

## Walk-forward ranking study

```bash
run-walk-forward --universe config/universe.global100.json
```

Reports: `reports/walk_forward/`. History policy: symbols with &lt;5 years of data are excluded from **all** folds (no partial-fold mixing).

## Phase 3E stability diagnostics

```bash
run-stability-analysis --universe config/universe.global100.json
```

Writes `reports/stability/` (country ranking, score distribution, regimes, contributions, buckets, 1d IC diagnostics). No trading / no backtest.

## Phase 3F Learning-to-Rank

```bash
run-learning-to-rank --universe config/universe.global100.json
```

- Ranking unit: **Date × Country** (not Global raw)
- Labels: within-group future-return relevance (`config/ranking_relevance.json`)
- Model: LightGBM `LGBMRanker` / `lambdarank`
- Reports: `reports/ranking/`
- Primary metrics: NDCG@5/10, Country IC, Top10/20 excess, Q5−Q1 spread vs momentum baselines

## Phase 3H label / score resolution

```bash
run-label-resolution --universe config/universe.global100.json
```

Compares relevance schemes A/B/C with explicit `label_gain` (Label D skipped). Reports: `reports/label_resolution/`.

## Phase 3G feature expansion

```bash
run-feature-expansion --universe config/universe.global100.json
```

Adds cross-sectional percentiles, sector-relative features, macro/FX/commodity context, breadth and dispersion. Compares Feature Sets A/B/C under Country-internal LTR. Reports: `reports/feature_expansion/`. Point-in-time fundamentals remain forbidden.

## Cache

OHLCV is cached under `data/raw/`.

| Case | Behavior |
|---|---|
| First fetch | Full lookback download |
| Later fetch | Reuse cache; download only missing recent days |
| `--force-refresh` | Ignore cache and re-download |

## Ranking evaluation foundation

Module: `src/ml/ranking.py`

Metrics (evaluation only — not trading):

- Information Coefficient (Spearman)
- Rank Correlation
- Top10% / Top20% average future return
- Excess return vs universe mean
- Optional group slices: country / sector / exchange

This is **not** a buy/sell ranking engine.
