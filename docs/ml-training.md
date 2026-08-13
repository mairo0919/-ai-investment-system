# ML Training (Phase 3A / 3B)

Phase 3 predicts the probability that the **next trading day's Adj Close is
higher than today's**, with leakage-safe chronological evaluation.

## Target

Created only in `src/ml/dataset.py` (never inside Phase 2 feature code):

```text
future_return_1d = AdjClose[t+1] / AdjClose[t] - 1
target_up = 1 if future_return_1d > 0 else 0
```

- `shift(-1)` is allowed **only** for this label creation
- The final row of each symbol is dropped because no next-day label exists
- `future_return_1d` / `target_up` are never used as model inputs

## Features

Model inputs are the explicit Phase 2 list `MODEL_FEATURE_COLUMNS` (26 columns).

Excluded from the model:

- `Date`, `Symbol`
- raw OHLCV (`Open` ... `Volume`, `Adj Close`)
- `future_return_1d`, `target_up`

Symbols are stacked vertically. `Symbol` is kept for evaluation only and is not
fed to the estimator.

## Chronological split

Random `train_test_split` is forbidden.

Unique dates are sorted ascending and cut into:

| Split | Ratio |
|---|---|
| Train | 70% oldest dates |
| Validation | next 15% |
| Test | latest 15% |

Guarantees:

```text
max(train.Date) < min(validation.Date)
max(validation.Date) < min(test.Date)
```

No date appears in more than one split. Phase 3A and 3B share the exact same split.

## Models

### Phase 3A — LogisticRegression

```text
StandardScaler -> LogisticRegression
```

Scaler is fitted on **train only** via an sklearn `Pipeline`.

### Phase 3B — LightGBM

```text
LGBMClassifier (no scaler)
```

Default hyperparameters are intentionally conservative (`num_leaves=15`,
regularization, subsample/colsample). Validation ROC-AUC early stopping is
enabled. **Test is never used for early stopping or tuning.**

Gain feature importance is reported for diagnostics only. It is **not** a causal
ranking.

## Baselines

1. **majority_class**: always predict the majority `target_up` from train
2. **previous_return**: predict up if today's `return_1d > 0`

All ML models are compared against these baselines on the same Test partition.

## Metrics

- Accuracy / Precision / Recall / F1
- ROC-AUC
- Confusion matrix
- Class distribution (`target_up` 0/1 ratios)
- Overall and per-symbol test metrics
- `validation_auc - test_auc` gap
- `probability_up` from `predict_proba` (reserved for future Phase 4 thresholds)

## CLI

```bash
train-model
python -m src.ml.trainer
```

Trains LogisticRegression and LightGBM, then writes a comparison report.

## Artifacts

| Path | Content |
|---|---|
| `models/logistic_regression.joblib` | fitted LR pipeline |
| `models/logistic_regression.metadata.json` | LR metadata |
| `models/lightgbm.joblib` | fitted LightGBM classifier |
| `models/lightgbm.metadata.json` | LightGBM metadata / best_iteration |
| `reports/logistic_regression_metrics.json` | LR metrics |
| `reports/lightgbm_metrics.json` | LightGBM metrics + importance |
| `reports/model_comparison.json` | side-by-side Test comparison |

## Look-ahead bias controls

- Features come from Phase 2 (no future shifts)
- Labels use next-day return only as `y`
- Scaler / model fit use train rows exclusively
- LightGBM early stopping uses validation only
- Validation and test are future periods relative to train
- No shuffle at any stage
- No Test-driven hyperparameter search
