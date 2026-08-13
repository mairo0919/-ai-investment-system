"""Model / experiment lineage for Phase 5 paper trading (no strategy change)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.exceptions import TrainingError
from src.paper.atomic_io import atomic_write_json

STRATEGY_CONFIG_ID = "FINAL_US_PHASE4C"
PARENT_EXPERIMENT = "phase4d_true_holdout"
PAPER_EXPERIMENT = "phase5_forward_paper"

# Canonical frozen paper model (Phase 5). Distinct from Phase 4D holdout fit.
LOCKED_PAPER_MODEL_ID = "paper_4ab12cb5ade1681f"

TRUE_HOLDOUT_TRAIN = {
    "training_start": "2016-11-01",
    "training_end": "2023-09-12",
    "experiment": "phase4d_true_holdout",
    "note": (
        "Phase 4D trained only on pre-holdout data (train end 2023-09-12). "
        "That fit was not persisted as a reusable artifact and is NOT the Paper model."
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_model_metadata(
    *,
    model_id: str,
    training_start: str,
    training_end: str,
    feature_set: str,
    label_config: dict[str, Any],
    ranker_config: dict[str, Any],
    strategy_config_id: str = STRATEGY_CONFIG_ID,
    created_at: str | None = None,
    parent_experiment: str = PARENT_EXPERIMENT,
    is_same_model_as_true_holdout: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "model_id": model_id,
        "training_start": training_start,
        "training_end": training_end,
        "feature_set": feature_set,
        "label_config": label_config,
        "ranker_config": ranker_config,
        "strategy_config_id": strategy_config_id,
        "created_at": created_at or _utc_now(),
        "parent_experiment": parent_experiment,
        "is_same_model_as_true_holdout": bool(is_same_model_as_true_holdout),
        "paper_experiment": PAPER_EXPERIMENT,
        "lineage_note": (
            "Phase 5 Paper model is a separate frozen fit from Phase 4D True Holdout. "
            "Do not concatenate Phase 4D holdout returns with Phase 5 forward paper returns."
        ),
        "true_holdout_reference": TRUE_HOLDOUT_TRAIN,
        "retrain_forbidden": True,
        "manual_retrain_policy": (
            "Forbidden unless an explicit future Phase instruction changes this policy."
        ),
    }
    if extra:
        meta.update(extra)
    return meta


def metadata_from_freeze_meta(freeze_meta: dict[str, Any]) -> dict[str, Any]:
    """Derive canonical lineage metadata from internal freeze blob."""
    return build_model_metadata(
        model_id=str(freeze_meta["model_id"]),
        training_start=str(freeze_meta["train_start"]),
        training_end=str(freeze_meta["train_end"]),
        feature_set=str(freeze_meta.get("feature_set", "A")),
        label_config={
            "label_scheme": freeze_meta.get("label_scheme"),
            "gain_name": freeze_meta.get("gain_name"),
            "horizon_days": freeze_meta.get("horizon_days"),
        },
        ranker_config={
            "param_preset": freeze_meta.get("param_preset"),
            "best_iteration": freeze_meta.get("best_iteration"),
            "model_params": freeze_meta.get("model_params"),
        },
        strategy_config_id=STRATEGY_CONFIG_ID,
        created_at=freeze_meta.get("training_timestamp_utc") or _utc_now(),
        parent_experiment=PARENT_EXPERIMENT,
        is_same_model_as_true_holdout=False,
        extra={
            "forward_start": freeze_meta.get("forward_start"),
            "early_stopping_end": freeze_meta.get("early_stopping_end"),
            "feature_list": freeze_meta.get("feature_list"),
            "git_commit": freeze_meta.get("git_commit"),
            "library_versions": freeze_meta.get("library_versions"),
        },
    )


def write_model_metadata(path: Path, metadata: dict[str, Any]) -> None:
    required = {
        "model_id",
        "training_start",
        "training_end",
        "feature_set",
        "label_config",
        "ranker_config",
        "strategy_config_id",
        "created_at",
        "parent_experiment",
        "is_same_model_as_true_holdout",
    }
    missing = required - set(metadata)
    if missing:
        raise TrainingError(f"model_metadata missing keys: {sorted(missing)}")
    if metadata.get("is_same_model_as_true_holdout") is not False:
        raise TrainingError(
            "is_same_model_as_true_holdout must be false — Paper model ≠ True Holdout model"
        )
    atomic_write_json(path, metadata)


def load_model_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise TrainingError(f"Missing model metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_paper_model_id_locked(model_id: str, *, locked_id: str = LOCKED_PAPER_MODEL_ID) -> None:
    if model_id != locked_id:
        raise TrainingError(
            f"Paper Model ID must remain fixed at {locked_id}; got {model_id}. Retrain forbidden."
        )


def refuse_retrain(*, paper_started: bool, model_exists: bool, force: bool = False) -> None:
    """Reject auto/manual retrain once a frozen model exists or paper has started."""
    if force:
        raise TrainingError(
            "Manual retrain is forbidden in Phase 5 without an explicit Phase-change instruction."
        )
    if model_exists or paper_started:
        raise TrainingError(
            "Retrain refused: Paper Model is frozen after Phase 5 start "
            "(automatic and manual retrain forbidden)."
        )


def separate_experiment_performance(
    *,
    phase5_forward: dict[str, Any],
    phase4d_holdout: dict[str, Any] | None,
) -> dict[str, Any]:
    """Present Phase 4D and Phase 5 as separate experiments (never concatenated)."""
    return {
        "policy": (
            "Phase 4D True Holdout and Phase 5 Forward Paper are separate experiments. "
            "Do not stitch their returns into one cumulative performance series."
        ),
        "phase4d_true_holdout": {
            "experiment_id": PARENT_EXPERIMENT,
            "is_same_model_as_paper": False,
            "performance": phase4d_holdout,
        },
        "phase5_forward_paper": {
            "experiment_id": PAPER_EXPERIMENT,
            "model_id": LOCKED_PAPER_MODEL_ID,
            "strategy_id": STRATEGY_CONFIG_ID,
            "performance": phase5_forward,
        },
        "combined_cumulative_return": None,
        "combined_cumulative_forbidden": True,
    }
