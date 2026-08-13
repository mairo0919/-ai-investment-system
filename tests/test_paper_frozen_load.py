"""Paper trading must load locked frozen model only — never auto-train."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import pytest

from src.core.exceptions import TrainingError
from src.paper.lineage import LOCKED_PAPER_MODEL_ID
from src.paper.model_freeze import FrozenRankerStore, train_and_freeze_ranker


def _write_meta(path: Path, model_id: str = LOCKED_PAPER_MODEL_ID) -> None:
    path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "train_start": "2016-11-01",
                "train_end": "2025-08-05",
                "feature_set": "A",
                "label_scheme": "B",
                "gain_name": "moderate_exp",
                "horizon_days": 5,
                "param_preset": "default",
                "feature_list": ["return_1d"],
                "early_stopping_end": "2026-07-24",
                "forward_start": "2026-08-01",
                "best_iteration": 17,
                "retrain_forbidden": True,
                "training_timestamp_utc": "2026-08-12T19:13:40+00:00",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _seed_store(store: FrozenRankerStore, *, model_id: str = LOCKED_PAPER_MODEL_ID) -> None:
    store.root.mkdir(parents=True, exist_ok=True)
    joblib.dump({"stub": True}, store.model_path)
    _write_meta(store.meta_path, model_id=model_id)


def test_frozen_model_present_loads_successfully(tmp_path: Path) -> None:
    store = FrozenRankerStore(tmp_path / "models")
    _seed_store(store)
    model, meta = store.load_for_paper_trading()
    assert model == {"stub": True}
    assert meta["model_id"] == LOCKED_PAPER_MODEL_ID


def test_frozen_model_missing_fail_fast_without_train(tmp_path: Path) -> None:
    store = FrozenRankerStore(tmp_path / "models")
    with (
        patch("src.paper.model_freeze.fit_lgbm_ranker") as fit_mock,
        patch("src.paper.model_freeze.train_and_freeze_ranker") as train_mock,
        pytest.raises(TrainingError, match="automatic retraining is forbidden"),
    ):
        store.load_for_paper_trading()
    fit_mock.assert_not_called()
    train_mock.assert_not_called()


def test_paper_ensure_frozen_does_not_call_train(tmp_path: Path) -> None:
    from src.paper.runner import PaperTradingRunner

    store = FrozenRankerStore(tmp_path / "models")
    runner = MagicMock(spec=PaperTradingRunner)
    runner.model_store = store
    with (
        patch("src.paper.model_freeze.fit_lgbm_ranker") as fit_mock,
        patch("src.paper.model_freeze.train_and_freeze_ranker") as train_mock,
        pytest.raises(TrainingError, match="expected model_id"),
    ):
        PaperTradingRunner._ensure_frozen_model(runner, panel=MagicMock())
    fit_mock.assert_not_called()
    train_mock.assert_not_called()
    # Paper runner must not reference train_and_freeze_ranker anymore.
    import src.paper.runner as runner_mod

    assert not hasattr(runner_mod, "train_and_freeze_ranker")


def test_meta_only_missing_fail_fast(tmp_path: Path) -> None:
    store = FrozenRankerStore(tmp_path / "models")
    store.root.mkdir(parents=True, exist_ok=True)
    joblib.dump({"stub": True}, store.model_path)
    with pytest.raises(TrainingError, match="model_meta.json") as exc:
        store.load_for_paper_trading()
    msg = str(exc.value)
    assert "automatic retraining is forbidden in paper trading" in msg
    assert LOCKED_PAPER_MODEL_ID in msg
    assert str(store.root) in msg


def test_joblib_only_missing_fail_fast(tmp_path: Path) -> None:
    store = FrozenRankerStore(tmp_path / "models")
    store.root.mkdir(parents=True, exist_ok=True)
    _write_meta(store.meta_path)
    with pytest.raises(TrainingError, match="lgbm_ranker.joblib") as exc:
        store.load_for_paper_trading()
    assert "automatic retraining is forbidden" in str(exc.value)


def test_model_id_mismatch_fails_lock_validation(tmp_path: Path) -> None:
    store = FrozenRankerStore(tmp_path / "models")
    _seed_store(store, model_id="paper_2843ca7c32df21b1")
    with pytest.raises(TrainingError, match="must remain fixed"):
        store.load_for_paper_trading()


def test_research_train_and_freeze_still_callable(tmp_path: Path) -> None:
    """Research path keeps train_and_freeze_ranker; only paper path forbids auto-train."""
    assert callable(train_and_freeze_ranker)
    store = FrozenRankerStore(tmp_path / "models")
    # force_retrain still refused by policy when requested
    with pytest.raises(TrainingError, match="Manual retrain|Retrain refused|forbidden"):
        train_and_freeze_ranker(
            panel=MagicMock(),
            cfg=MagicMock(),
            feature_cols=[],
            params={},
            forward_start=MagicMock(),
            purge_days=5,
            early_stopping_valid_days=252,
            store=store,
            project_root=tmp_path,
            force_retrain=True,
        )


def test_import_paper_no_side_effect() -> None:
    for key in list(sys.modules):
        if key == "src.paper" or key.startswith("src.paper."):
            del sys.modules[key]
    import src.paper as paper

    assert "src.paper.runner" not in sys.modules
    assert paper.__all__ == ["PaperTradingRunner"]


def test_main_still_once() -> None:
    from src.paper import runner as runner_mod

    fake = MagicMock()
    fake.run.return_value = {}
    with (
        patch.object(runner_mod, "PaperTradingRunner", return_value=fake),
        patch.object(runner_mod, "get_settings"),
        patch.object(runner_mod, "setup_logging"),
        patch.object(runner_mod, "install_signal_handlers"),
        patch.object(runner_mod, "install_atexit_hook"),
        patch.object(runner_mod, "log_process_boot"),
    ):
        assert runner_mod.main([]) == 0
    assert fake.run.call_count == 1
