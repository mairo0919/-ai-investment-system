"""Phase 5 model / experiment lineage (no strategy change)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.exceptions import TrainingError
from src.paper.lineage import (
    LOCKED_PAPER_MODEL_ID,
    STRATEGY_CONFIG_ID,
    assert_paper_model_id_locked,
    build_model_metadata,
    load_model_metadata,
    refuse_retrain,
    separate_experiment_performance,
    write_model_metadata,
)
from src.paper.model_freeze import FrozenRankerStore


def test_paper_model_id_locked() -> None:
    assert_paper_model_id_locked(LOCKED_PAPER_MODEL_ID)
    with pytest.raises(TrainingError, match="fixed"):
        assert_paper_model_id_locked("paper_other_model")


def test_model_metadata_persistence(tmp_path: Path) -> None:
    path = tmp_path / "model_metadata.json"
    meta = build_model_metadata(
        model_id=LOCKED_PAPER_MODEL_ID,
        training_start="2016-11-01",
        training_end="2025-08-05",
        feature_set="A",
        label_config={"label_scheme": "B", "gain_name": "moderate_exp", "horizon_days": 5},
        ranker_config={"param_preset": "default"},
        strategy_config_id=STRATEGY_CONFIG_ID,
        created_at="2026-08-12T19:13:40+00:00",
        parent_experiment="phase4d_true_holdout",
        is_same_model_as_true_holdout=False,
    )
    write_model_metadata(path, meta)
    loaded = load_model_metadata(path)
    for key in (
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
    ):
        assert key in loaded
    assert loaded["is_same_model_as_true_holdout"] is False
    assert loaded["model_id"] == LOCKED_PAPER_MODEL_ID


def test_true_holdout_and_paper_not_same_model() -> None:
    meta = build_model_metadata(
        model_id=LOCKED_PAPER_MODEL_ID,
        training_start="2016-11-01",
        training_end="2025-08-05",
        feature_set="A",
        label_config={"label_scheme": "B"},
        ranker_config={"param_preset": "default"},
    )
    assert meta["is_same_model_as_true_holdout"] is False
    holdout_train_end = meta["true_holdout_reference"]["training_end"]
    assert holdout_train_end == "2023-09-12"
    assert holdout_train_end != meta["training_end"]

    experiments = separate_experiment_performance(
        phase5_forward={"total_return": 0.02},
        phase4d_holdout={"total_return": 0.699},
    )
    assert experiments["combined_cumulative_forbidden"] is True
    assert experiments["combined_cumulative_return"] is None
    assert experiments["phase4d_true_holdout"]["is_same_model_as_paper"] is False

    with pytest.raises(TrainingError, match="must be false"):
        write_model_metadata(
            Path("models/paper_frozen/.should_not_write_lineage.json"),
            {**meta, "is_same_model_as_true_holdout": True},
        )


def test_retrain_refused_after_paper_start(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="Retrain refused"):
        refuse_retrain(paper_started=True, model_exists=True, force=False)
    with pytest.raises(TrainingError, match="Manual retrain"):
        refuse_retrain(paper_started=True, model_exists=True, force=True)

    store = FrozenRankerStore(tmp_path / "models")
    # Pretend frozen artifact exists
    store.meta_path.write_text(json.dumps({"model_id": LOCKED_PAPER_MODEL_ID, "retrain_forbidden": True}))
    store.model_path.write_text("placeholder")
    with pytest.raises(TrainingError, match="Retrain refused|forbids|forbidden"):
        store.request_retrain(paper_started=True, force=False)
    with pytest.raises(TrainingError, match="Manual retrain"):
        store.request_retrain(paper_started=True, force=True)


def test_deployed_paper_metadata_file() -> None:
    path = Path("models/paper_frozen/model_metadata.json")
    assert path.exists(), "model_metadata.json must exist for deployed paper model"
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["model_id"] == LOCKED_PAPER_MODEL_ID
    assert meta["training_start"] == "2016-11-01"
    assert meta["training_end"] == "2025-08-05"
    assert meta["is_same_model_as_true_holdout"] is False
    assert meta["strategy_config_id"] == STRATEGY_CONFIG_ID
