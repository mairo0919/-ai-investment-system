"""Phase 3F: Country-internal Learning-to-Rank walk-forward study."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings, get_settings
from src.core.exceptions import MLError, TrainingError
from src.features.indicators import FEATURE_COLUMNS
from src.features.market_features import MARKET_FEATURE_SUFFIXES
from src.ml.ltr_baselines import MOMENTUM_BASELINES, momentum_scores
from src.ml.ltr_labels import (
    assert_group_integrity,
    assign_relevance_labels,
    build_ranking_groups,
    load_relevance_config,
)
from src.ml.ltr_metrics import evaluate_country_ranking
from src.ml.ltr_model import fit_lgbm_ranker, predict_rank_scores
from src.ml.walk_forward import WalkForwardFold, build_expanding_folds, mask_fold_partition
from src.ml.walk_forward_runner import GENERIC_MARKET_PREFIX, WalkForwardRunner
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

REGIONS = ("Japan", "United States", "Europe")
HORIZONS = (5, 10)


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if np.isnan(x) or np.isinf(x) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    return obj


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None, "positive_ratio": None, "n": 0}
    arr = pd.Series(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(arr.median()),
        "std": float(arr.std(ddof=0)),
        "positive_ratio": float((arr > 0).mean()),
        "n": int(len(arr)),
    }


class LearningToRankRunner:
    """Walk-forward Country-internal LGBMRanker experiments."""

    def __init__(
        self,
        settings: Settings,
        universe_path: Path,
        relevance_config_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.wf = WalkForwardRunner(settings, universe_path)
        self.relevance_config = load_relevance_config(relevance_config_path)
        self.report_dir = settings.reports_dir / "ranking"
        self.feature_cols = tuple(
            list(FEATURE_COLUMNS)
            + [f"{GENERIC_MARKET_PREFIX}_{s}" for s in MARKET_FEATURE_SUFFIXES]
        )

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Phase 3F LTR study starting universe=%s", self.universe_path)

        fetch_summary = self.wf._fetch_equities(force_refresh=force_refresh)
        usable_meta, quality = self.wf._select_usable_symbols(
            fetch_summary, enrich_metadata=False
        )
        market_frames = self.wf._fetch_benchmarks(force_refresh=force_refresh)
        panel = self.wf._build_panel(usable_meta, market_frames)
        if panel.empty:
            raise TrainingError("Empty panel for LTR study")

        # Forbidden feature leakage checks.
        forbidden = {"Symbol", "Country", "Region", "Market", "Currency", "Sector"}
        overlap = forbidden.intersection(self.feature_cols)
        if overlap:
            raise TrainingError(f"Forbidden identity features in model inputs: {overlap}")

        experiments: list[dict[str, Any]] = []
        for horizon in HORIZONS:
            return_col = f"future_return_{horizon}d"
            labeled = assign_relevance_labels(
                panel,
                return_col=return_col,
                config=self.relevance_config,
            )
            labeled = labeled.dropna(subset=list(self.feature_cols) + ["relevance"]).copy()
            labeled["future_return"] = labeled[return_col]

            # Ensure future return columns used as labels are not in features.
            if return_col in self.feature_cols:
                raise TrainingError("future_return leaked into features")

            folds = build_expanding_folds(
                labeled["Date"],
                n_folds=5,
                min_train_days=252 * 3,
                valid_days=252,
                purge_days=horizon,
            )

            # Common ranker
            common = self._run_model_variant(
                labeled=labeled,
                folds=folds,
                horizon=horizon,
                model_type="common_ranker",
                region_filter=None,
            )
            experiments.append(common)

            # Country-specific rankers (diagnostic)
            for region in REGIONS:
                regional = self._run_model_variant(
                    labeled=labeled,
                    folds=folds,
                    horizon=horizon,
                    model_type="country_ranker",
                    region_filter=region,
                )
                experiments.append(regional)

            # Momentum baselines (no training)
            for baseline_name, feat in MOMENTUM_BASELINES:
                base = self._run_baseline(
                    labeled=labeled,
                    folds=folds,
                    horizon=horizon,
                    baseline_name=baseline_name,
                    feature_col=feat,
                )
                experiments.append(base)

        overview = {
            "created_at_utc": _timestamp_iso(),
            "phase": "3F",
            "purpose": "Country-internal Learning-to-Rank (no trading)",
            "relevance_config": {
                "name": self.relevance_config.name,
                "description": self.relevance_config.description,
                "group_keys": list(self.relevance_config.group_keys),
                "buckets": [
                    {
                        "relevance": b.relevance,
                        "min_percentile": b.min_percentile,
                        "max_percentile": b.max_percentile,
                        "label": b.label,
                    }
                    for b in self.relevance_config.buckets
                ],
                "min_group_size": self.relevance_config.min_group_size,
                "ndcg_ks": list(self.relevance_config.ndcg_ks),
            },
            "feature_policy": {
                "features": list(self.feature_cols),
                "excluded_from_features": sorted(forbidden),
                "note": "Primary evaluation is Country-internal ranking (Date×Country groups).",
            },
            "universe": self.wf.universe.composition_report(),
            "fetch_summary": fetch_summary,
            "quality_filter": quality,
            "experiments": [
                {
                    "experiment_id": e["experiment_id"],
                    "stability": e.get("stability_across_folds"),
                    "report_path": e.get("report_path"),
                }
                for e in experiments
            ],
            "known_limitations": [
                "Research fixed universe may contain survivorship bias.",
                "Global raw ranking is not the primary metric.",
                "LightGBM Ranker early stopping uses validation folds (diagnostic).",
                "No trading / portfolio / FX conversion.",
            ],
        }
        path = self.report_dir / "ranking_overview.json"
        _write_json(path, _json_safe(overview))
        logger.info("Wrote LTR overview -> %s", path)
        return overview

    def _run_model_variant(
        self,
        *,
        labeled: pd.DataFrame,
        folds: list[WalkForwardFold],
        horizon: int,
        model_type: str,
        region_filter: str | None,
    ) -> dict[str, Any]:
        region_tag = region_filter or "ALL"
        experiment_id = f"ltr_{horizon}d__{model_type}__{region_tag.replace(' ', '_')}"
        logger.info("Running %s", experiment_id)

        fold_reports: list[dict[str, Any]] = []
        for fold in folds:
            train_df = mask_fold_partition(labeled, fold, partition="train")
            valid_df = mask_fold_partition(labeled, fold, partition="validation")
            if region_filter is not None:
                train_df = train_df.loc[train_df["Region"] == region_filter].copy()
                valid_df = valid_df.loc[valid_df["Region"] == region_filter].copy()
            if train_df.empty or valid_df.empty:
                raise TrainingError(f"Empty partition in {experiment_id} fold={fold.fold_id}")

            train_sorted, train_groups = build_ranking_groups(
                train_df, group_keys=("Date", "Region")
            )
            valid_sorted, valid_groups = build_ranking_groups(
                valid_df, group_keys=("Date", "Region")
            )
            assert_group_integrity(train_sorted, train_groups)
            assert_group_integrity(valid_sorted, valid_groups)

            x_train = train_sorted.loc[:, list(self.feature_cols)]
            y_train = train_sorted["relevance"]
            x_valid = valid_sorted.loc[:, list(self.feature_cols)]
            y_valid = valid_sorted["relevance"]

            fit = fit_lgbm_ranker(
                x_train,
                y_train,
                train_groups,
                x_valid=x_valid,
                y_valid=y_valid,
                group_valid=valid_groups,
            )
            scores = predict_rank_scores(fit.model, x_valid)
            scored = valid_sorted.copy()
            scored["score"] = scores

            fold_reports.append(
                self._evaluate_fold(
                    scored=scored,
                    fold=fold,
                    train_rows=len(train_sorted),
                    train_groups=len(train_groups),
                    best_iteration=fit.best_iteration,
                )
            )

        summary = self._finalize_experiment(
            experiment_id=experiment_id,
            horizon=horizon,
            model_type=model_type,
            region_filter=region_filter,
            fold_reports=fold_reports,
        )
        return summary

    def _run_baseline(
        self,
        *,
        labeled: pd.DataFrame,
        folds: list[WalkForwardFold],
        horizon: int,
        baseline_name: str,
        feature_col: str,
    ) -> dict[str, Any]:
        experiment_id = f"ltr_{horizon}d__baseline__{baseline_name}"
        logger.info("Running %s", experiment_id)
        fold_reports: list[dict[str, Any]] = []
        for fold in folds:
            valid_df = mask_fold_partition(labeled, fold, partition="validation")
            valid_sorted, valid_groups = build_ranking_groups(
                valid_df, group_keys=("Date", "Region")
            )
            assert_group_integrity(valid_sorted, valid_groups)
            scored = valid_sorted.copy()
            scored["score"] = momentum_scores(scored, feature_col=feature_col).to_numpy()
            fold_reports.append(
                self._evaluate_fold(
                    scored=scored,
                    fold=fold,
                    train_rows=0,
                    train_groups=0,
                    best_iteration=None,
                )
            )
        return self._finalize_experiment(
            experiment_id=experiment_id,
            horizon=horizon,
            model_type=f"baseline_{baseline_name}",
            region_filter=None,
            fold_reports=fold_reports,
        )

    def _evaluate_fold(
        self,
        *,
        scored: pd.DataFrame,
        fold: WalkForwardFold,
        train_rows: int,
        train_groups: int,
        best_iteration: int | None,
    ) -> dict[str, Any]:
        by_country: dict[str, Any] = {}
        for region in REGIONS:
            group = scored.loc[scored["Region"] == region]
            if group.empty:
                by_country[region] = {"n_rows": 0}
                continue
            by_country[region] = evaluate_country_ranking(
                group,
                score_col="score",
                return_col="future_return",
                relevance_col="relevance",
                ndcg_ks=self.relevance_config.ndcg_ks,
                tie_warning_ratio=self.relevance_config.tie_warning_ratio,
            )

        # Aggregate across Date×Country groups (not Global raw pool).
        all_metrics = evaluate_country_ranking(
            scored,
            score_col="score",
            return_col="future_return",
            relevance_col="relevance",
            ndcg_ks=self.relevance_config.ndcg_ks,
            tie_warning_ratio=self.relevance_config.tie_warning_ratio,
        )
        all_metrics["region"] = "ALL_COUNTRY_GROUPS"

        return {
            "fold_id": fold.fold_id,
            "train_period": {
                "start": str(fold.train_start.date()),
                "end": str(fold.train_end.date()),
                "rows": train_rows,
                "groups": train_groups,
            },
            "validation_period": {
                "start": str(fold.valid_start.date()),
                "end": str(fold.valid_end.date()),
                "rows": int(len(scored)),
                "symbols": int(scored["Symbol"].nunique()),
                "groups": int(scored.groupby(["Date", "Region"]).ngroups),
            },
            "purge_days": fold.purge_days,
            "best_iteration": best_iteration,
            "by_country": by_country,
            "all_country_groups": all_metrics,
        }

    def _finalize_experiment(
        self,
        *,
        experiment_id: str,
        horizon: int,
        model_type: str,
        region_filter: str | None,
        fold_reports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stability = self._aggregate_stability(fold_reports)
        summary = {
            "experiment_id": experiment_id,
            "horizon_days": horizon,
            "model_type": model_type,
            "region_filter": region_filter,
            "folds": fold_reports,
            "stability_across_folds": stability,
        }
        path = self.report_dir / f"{experiment_id}.json"
        _write_json(path, _json_safe(summary))
        summary["report_path"] = str(path)
        logger.info("Wrote %s", path)
        return summary

    @staticmethod
    def _aggregate_stability(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"by_country": {}, "all_country_groups": {}}

        def collect(scope_key: str, region: str | None = None) -> dict[str, Any]:
            ndcg5, ndcg10, ic, t10, t20, q51 = [], [], [], [], [], []
            tie_warn_folds = 0
            for fold in fold_reports:
                block = (
                    fold["all_country_groups"]
                    if scope_key == "all"
                    else fold["by_country"].get(region or "", {})
                )
                if not block or block.get("n_rows", 0) == 0:
                    continue
                ndcg = block.get("ndcg") or {}
                if ndcg.get("ndcg_at_5") is not None:
                    ndcg5.append(float(ndcg["ndcg_at_5"]))
                if ndcg.get("ndcg_at_10") is not None:
                    ndcg10.append(float(ndcg["ndcg_at_10"]))
                ranking = block.get("ranking") or {}
                if (ranking.get("ic") or {}).get("mean") is not None:
                    ic.append(float(ranking["ic"]["mean"]))
                if (ranking.get("top_10pct_excess_return") or {}).get("mean") is not None:
                    t10.append(float(ranking["top_10pct_excess_return"]["mean"]))
                if (ranking.get("top_20pct_excess_return") or {}).get("mean") is not None:
                    t20.append(float(ranking["top_20pct_excess_return"]["mean"]))
                if block.get("q5_q1_spread") is not None:
                    q51.append(float(block["q5_q1_spread"]))
                if block.get("warnings"):
                    tie_warn_folds += 1
            return {
                "ndcg_at_5": _stats(ndcg5),
                "ndcg_at_10": _stats(ndcg10),
                "mean_ic": _stats(ic),
                "top10_excess": _stats(t10),
                "top20_excess": _stats(t20),
                "q5_q1_spread": _stats(q51),
                "folds_with_score_warnings": tie_warn_folds,
            }

        out["all_country_groups"] = collect("all")
        for region in REGIONS:
            out["by_country"][region] = collect("country", region)
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3F Country-internal Learning-to-Rank study"
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("config/universe.global100.json"),
    )
    parser.add_argument(
        "--relevance-config",
        type=Path,
        default=Path("config/ranking_relevance.json"),
    )
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    try:
        overview = LearningToRankRunner(
            settings,
            args.universe,
            relevance_config_path=args.relevance_config,
        ).run(force_refresh=args.force_refresh)
    except MLError as exc:
        logger.error("LTR study failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected LTR failure")
        return 1

    logger.info(
        "Phase 3F complete: experiments=%d",
        len(overview.get("experiments", [])),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
