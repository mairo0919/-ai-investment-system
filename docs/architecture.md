# Architecture

## Product direction

This project is an **AI Investment Engine**, not a price-direction prediction toy.

- AI ranks / scores expected opportunity
- Rule Engine decides trades later
- See `docs/product-philosophy.md`

## Phase 1 overview

Phase 1 focuses on market data acquisition only. Callers depend on `BaseDataProvider`, not on a specific vendor SDK.

```text
Settings (.env)
    │
    ▼
MarketDataService
    │
    ├── create_provider()  →  YFinanceProvider (current)
    │                      →  JQuantsProvider (future)
    │
    └── save CSV to data/raw/
```

## Error handling

| Case | Exception | Behavior |
|---|---|---|
| Network / API failure | `DataFetchError` | Logged, continue next ticker |
| Zero rows | `EmptyDataError` | Logged, continue next ticker |
| CSV write failure | `DataSaveError` | Logged, continue next ticker |
| Unexpected exception | `Exception` | Logged with stacktrace |
| No files saved | `DataProviderError` | Process exits with code 1 |
