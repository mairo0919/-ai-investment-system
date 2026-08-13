"""Freeze LGBMRanker once for forward paper trading (no auto-retrain)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

from src.core.exceptions import TrainingError
from src.ml.ltr_label_schemes import assign_label_scheme, load_label_scheme_config
from src.ml.ltr_labels import assert_group_integrity, build_ranking_groups
from src.ml.ltr_model import fit_lgbm_ranker, predict_rank_scores
from src.paper.atomic_io import atomic_write_json
from src.paper.lineage import (
    LOCKED_PAPER_MODEL_ID,
    assert_paper_model_id_locked,
    metadata_from_freeze_meta,
    refuse_retrain,
    write_model_metadata,
)
from src.simulation.config import SimulationConfig


def _git_commit(project_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return None


def _make_model_id(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "paper_" + hashlib.sha256(blob).hexdigest()[:16]


class FrozenRankerStore:
    def __init__(self, models_dir: Path) -> None:
        self.root = models_dir / "paper_frozen"
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "model_meta.json"
        self.lineage_path = self.root / "model_metadata.json"
        self.model_path = self.root / "lgbm_ranker.joblib"

    def exists(self) -> bool:
        return self.meta_path.exists() and self.model_path.exists()

    def ensure_lineage_metadata(self, freeze_meta: dict[str, Any]) -> dict[str, Any]:
        """Write/refresh canonical lineage metadata (does not retrain)."""
        lineage = metadata_from_freeze_meta(freeze_meta)
        # After paper start, model_id must stay locked to the deployed artifact.
        if freeze_meta.get("model_id") == LOCKED_PAPER_MODEL_ID:
            assert_paper_model_id_locked(str(lineage["model_id"]))
        write_model_metadata(self.lineage_path, lineage)
        return lineage

    def load(self) -> tuple[Any, dict[str, Any]]:
        if not self.exists():
            raise TrainingError("Frozen paper model not found; run paper trading init first")
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if not meta.get("retrain_forbidden", True):
            raise TrainingError("Frozen model metadata corrupted (retrain_forbidden)")
        self.ensure_lineage_metadata(meta)
        model = joblib.load(self.model_path)
        return model, meta

    def save(
        self,
        *,
        model: Any,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        if self.exists():
            refuse_retrain(paper_started=True, model_exists=True, force=False)
        joblib.dump(model, self.model_path)
        atomic_write_json(self.meta_path, meta)
        self.ensure_lineage_metadata(meta)
        return meta

    def request_retrain(self, *, paper_started: bool, force: bool = False) -> None:
        """Explicit retrain entrypoint — always refused in Phase 5 once started/frozen."""
        refuse_retrain(
            paper_started=paper_started,
            model_exists=self.exists(),
            force=force,
        )


def train_and_freeze_ranker(
    *,
    panel: pd.DataFrame,
    cfg: SimulationConfig,
    feature_cols: list[str],
    params: dict[str, Any],
    forward_start: pd.Timestamp,
    purge_days: int,
    early_stopping_valid_days: int,
    store: FrozenRankerStore,
    project_root: Path,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """Train once on pre-forward data and persist freeze artifact."""
    if force_retrain:
        refuse_retrain(paper_started=True, model_exists=store.exists(), force=True)
    if store.exists():
        # Model already frozen: return existing artifact (no retrain).
        _, meta = store.load()
        if meta.get("model_id") == LOCKED_PAPER_MODEL_ID:
            assert_paper_model_id_locked(str(meta["model_id"]))
        return meta

    scheme_cfg = load_label_scheme_config()
    labeled_pack = assign_label_scheme(
        panel,
        return_col=f"future_return_{cfg.horizon_days}d",
        scheme=cfg.label_scheme,  # type: ignore[arg-type]
        gain_name=cfg.gain_name,  # type: ignore[arg-type]
        min_group_size=int(scheme_cfg.get("min_group_size", 5)),
    )
    labeled = labeled_pack.frame.copy()
    labeled["Date"] = pd.to_datetime(labeled["Date"]).dt.normalize()
    calendar = sorted(labeled["Date"].unique())
    pre = [d for d in calendar if d < forward_start]
    if len(pre) <= purge_days + early_stopping_valid_days + 100:
        raise TrainingError("Insufficient pre-forward history for model freeze")
    fit_calendar = pre[:-purge_days] if purge_days > 0 else pre
    es_dates = set(fit_calendar[-early_stopping_valid_days:])
    train_dates = set(fit_calendar[:-early_stopping_valid_days])

    drop_cols = [c for c in feature_cols if c in labeled.columns] + ["relevance"]
    train_df = labeled.loc[labeled["Date"].isin(train_dates)].dropna(subset=drop_cols).copy()
    es_df = labeled.loc[labeled["Date"].isin(es_dates)].dropna(subset=drop_cols).copy()
    if (train_df["Date"] >= forward_start).any() or (es_df["Date"] >= forward_start).any():
        raise TrainingError("Forward leakage into freeze training")

    train_s, train_g = build_ranking_groups(train_df, group_keys=("Date", "Region"))
    es_s, es_g = build_ranking_groups(es_df, group_keys=("Date", "Region"))
    assert_group_integrity(train_s, train_g)
    assert_group_integrity(es_s, es_g)

    x_train = train_s.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    x_es = es_s.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    fit = fit_lgbm_ranker(
        x_train,
        train_s["relevance"],
        train_g,
        x_valid=x_es,
        y_valid=es_s["relevance"],
        group_valid=es_g,
        params=params,
        label_gain=labeled_pack.label_gain,
    )

    fingerprint = {
        "label_scheme": cfg.label_scheme,
        "gain_name": cfg.gain_name,
        "feature_set": cfg.feature_set,
        "param_preset": cfg.param_preset,
        "horizon_days": cfg.horizon_days,
        "feature_list": list(feature_cols),
        "train_start": str(min(train_dates).date()),
        "train_end": str(max(train_dates).date()),
        "early_stopping_end": str(max(es_dates).date()),
        "forward_start": str(forward_start.date()),
        "best_iteration": fit.best_iteration,
    }
    model_id = _make_model_id(fingerprint)
    meta = {
        **fingerprint,
        "model_id": model_id,
        "retrain_forbidden": True,
        "training_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": _git_commit(project_root),
        "library_versions": {
            "lightgbm": lightgbm.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "model_params": dict(fit.params),
        "train_rows": int(len(train_s)),
        "early_stopping_rows": int(len(es_s)),
        "model_path": str(store.model_path),
    }
    store.save(model=fit.model, meta=meta)
    return meta


def score_panel_with_frozen(
    *,
    model: Any,
    panel: pd.DataFrame,
    feature_cols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    frame = panel.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    mask = frame["Date"] >= start
    if end is not None:
        mask &= frame["Date"] <= end
    hold = frame.loc[mask].dropna(subset=[c for c in feature_cols if c in frame.columns]).copy()
    if hold.empty:
        return hold
    x = hold.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    hold = hold.copy()
    hold["score"] = predict_rank_scores(model, x)
    return hold
