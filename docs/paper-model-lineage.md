# Phase 5 Paper Model Lineage

Strategy / rules / portfolio sizing are unchanged. This document only clarifies **model and experiment lineage**.

## Paper model (current)

| Field | Value |
|-------|-------|
| `model_id` | `paper_4ab12cb5ade1681f` |
| Training | 2016-11-01 → 2025-08-05 |
| Paper forward start | 2026-08-01 |
| Strategy config | `FINAL_US_PHASE4C` |
| Artifact | `models/paper_frozen/` |

Canonical lineage file: `models/paper_frozen/model_metadata.json`

## Not the True Holdout model

Phase 4D True Holdout used a **different fit**:

| | Phase 4D True Holdout | Phase 5 Paper |
|--|----------------------|---------------|
| Experiment | `phase4d_true_holdout` | `phase5_forward_paper` |
| Train end | 2023-09-12 | 2025-08-05 |
| Evaluation window | 2024-09-09 → 2026-07-31 | 2026-08-01 → |
| Persisted artifact | No reusable model file | `lgbm_ranker.joblib` |
| Same model? | — | **`is_same_model_as_true_holdout = false`** |

Do **not** treat holdout scores/weights as the paper model.

## Performance reporting policy

- Phase 4D holdout performance and Phase 5 forward paper performance are **separate experiments**.
- Do **not** concatenate them into one cumulative return series.
- `reports/paper/latest.json` and `reports/paper/performance.json` expose them under distinct experiment keys.

## Retrain policy

- Automatic retrain: **forbidden**
- Manual retrain: **forbidden** unless a future Phase explicitly changes policy
- After paper start, `model_id` stays fixed at `paper_4ab12cb5ade1681f`
