"""Phase 3H: ranking label / score-resolution improvement study."""

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
from src.features.cross_section_features import add_cross_section_feature_block
from src.features.feature_sets import resolve_feature_sets
from src.features.macro_features import (
    attach_mkt_return_60d,
    compute_mkt_return_60d,
    fetch_macro_feature_frames,
    join_macro_and_currency_features,
    load_macro_config,
)
from src.ml.ltr_baselines import MOMENTUM_BASELINES, momentum_scores
from src.ml.ltr_label_schemes import (
    GainName,
    SchemeName,
    assign_label_scheme,
    label_gain_to_param,
    load_label_scheme_config,
    skip_reason_label_d,
)
from src.ml.ltr_labels import (
    assert_group_integrity,
    build_ranking_groups,
)
from src.ml.ltr_metrics import evaluate_country_ranking
from src.ml.ltr_model import fit_lgbm_ranker, predict_rank_scores
from src.ml.score_resolution import fold_score_diagnostics
from src.ml.walk_forward import WalkForwardFold, build_expanding_folds, mask_fold_partition
from src.ml.walk_forward_runner import REGION_BENCHMARKS, WalkForwardRunner
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

REGIONS = ("Japan", "United States", "Europe")


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


class LabelResolutionRunner:
    """Compare label schemes / gains / limited ranker params for score resolution."""

    def __init__(self, settings: Settings, universe_path: Path) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.wf = WalkForwardRunner(settings, universe_path)
        self.scheme_cfg = load_label_scheme_config()
        self.macro_specs = load_macro_config()
        self.feature_sets = resolve_feature_sets(self.macro_specs)
        self.report_dir = settings.reports_dir / "label_resolution"
        self.param_presets: dict[str, dict[str, Any]] = self.scheme_cfg[
            "ranker_param_presets"
        ]
        self.low_res_thr = float(self.scheme_cfg.get("low_resolution_tie_threshold", 0.5))
        self.min_group = int(self.scheme_cfg.get("min_group_size", 5))
        self.ndcg_ks = tuple(int(k) for k in self.scheme_cfg.get("ndcg_ks", [5, 10]))

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Phase 3H label/score-resolution study starting")

        panel = self._build_enriched_panel(force_refresh=force_refresh)

        experiments: list[dict[str, Any]] = []
        # Predefined matrix (no post-hoc search).
        matrix_5d: list[tuple[SchemeName, GainName, str, str]] = [
            ("A", "linear", "default", "A"),
            ("B", "linear", "default", "A"),
            ("B", "moderate_exp", "default", "A"),
            ("C", "linear", "default", "A"),
            ("C", "moderate_exp", "default", "A"),
            ("B", "linear", "finer_leaves", "A"),
            ("B", "linear", "coarser_leaves", "A"),
            ("B", "linear", "default", "B"),
            ("B", "linear", "default", "C"),  # Feature Set C reference
        ]
        for scheme, gain, preset, fset in matrix_5d:
            experiments.append(
                self._run_experiment(
                    panel=panel,
                    horizon=5,
                    scheme=scheme,
                    gain_name=gain,
                    param_preset=preset,
                    feature_set=fset,
                )
            )

        matrix_10d: list[tuple[SchemeName, GainName, str, str]] = [
            ("A", "linear", "default", "A"),
            ("B", "linear", "default", "A"),
            ("C", "linear", "default", "A"),
        ]
        for scheme, gain, preset, fset in matrix_10d:
            experiments.append(
                self._run_experiment(
                    panel=panel,
                    horizon=10,
                    scheme=scheme,
                    gain_name=gain,
                    param_preset=preset,
                    feature_set=fset,
                )
            )

        for horizon in (5, 10):
            for baseline_name, feat in MOMENTUM_BASELINES:
                experiments.append(
                    self._run_baseline(panel, horizon=horizon, baseline_name=baseline_name, feature_col=feat)
                )

        comparison = self._compare(experiments)
        overview = {
            "created_at_utc": _timestamp_iso(),
            "phase": "3H",
            "purpose": "Improve LTR label design and score resolution (no new data/APIs)",
            "label_d_skipped": skip_reason_label_d(),
            "label_schemes": self.scheme_cfg["schemes"],
            "gains": self.scheme_cfg["gains"],
            "ranker_param_presets": self.param_presets,
            "feature_set_sizes": {k: len(v) for k, v in self.feature_sets.items()},
            "comparison": comparison,
            "experiments": [
                {
                    "experiment_id": e["experiment_id"],
                    "stability": e.get("stability_across_folds"),
                    "report_path": e.get("report_path"),
                }
                for e in experiments
            ],
            "known_limitations": [
                "Label D not implemented (unequal group sizes vs shared label_gain).",
                "Param presets are diagnostic only (not Optuna / not exhaustive).",
                "Feature Set C is reference due to prior high tie ratios.",
                "No trading / Phase 4.",
            ],
        }
        path = self.report_dir / "label_resolution_overview.json"
        _write_json(path, _json_safe(overview))
        logger.info("Wrote Phase 3H overview -> %s", path)
        return overview

    def _build_enriched_panel(self, *, force_refresh: bool) -> pd.DataFrame:
        # Diagnostic phase markers only (no feature / strategy changes).
        from src.utils.runtime_diag import phase_span

        with phase_span("market_data_update"):
            fetch_summary = self.wf._fetch_equities(force_refresh=force_refresh)
            usable_meta, _quality = self.wf._select_usable_symbols(
                fetch_summary, enrich_metadata=False
            )
            market_frames = self.wf._fetch_benchmarks(force_refresh=force_refresh)
        with phase_span("panel_build"):
            panel = self.wf._build_panel(usable_meta, market_frames)
            mkt_ohlcv = {
                region: self.wf.cache.load(ticker)
                for region, (ticker, _) in REGION_BENCHMARKS.items()
            }
            if any(v is None for v in mkt_ohlcv.values()):
                raise TrainingError("Missing benchmark OHLCV cache for 60d relative strength")
        with phase_span("feature_pipeline"):
            panel = attach_mkt_return_60d(
                panel,
                compute_mkt_return_60d({k: v for k, v in mkt_ohlcv.items() if v is not None}),
            )
        with phase_span("cross_section_features"):
            panel = add_cross_section_feature_block(panel, min_sector_group_size=3)
        with phase_span("macro_features"):
            macro_frames = fetch_macro_feature_frames(
                self.wf.provider,
                self.wf.cache,
                self.macro_specs,
                lookback_years=self.settings.lookback_years,
                interval=self.settings.interval,
                force_refresh=force_refresh,
            )
            panel, _ = join_macro_and_currency_features(panel, macro_frames)
        return panel

    def _run_experiment(
        self,
        *,
        panel: pd.DataFrame,
        horizon: int,
        scheme: SchemeName,
        gain_name: GainName,
        param_preset: str,
        feature_set: str,
    ) -> dict[str, Any]:
        experiment_id = (
            f"h{horizon}d__label{scheme}__gain_{gain_name}__params_{param_preset}__fs{feature_set}"
        )
        logger.info("Running %s", experiment_id)
        return_col = f"future_return_{horizon}d"
        labeled_pack = assign_label_scheme(
            panel,
            return_col=return_col,
            scheme=scheme,
            gain_name=gain_name,
            min_group_size=self.min_group,
        )
        labeled = labeled_pack.frame.copy()
        labeled["future_return"] = labeled[return_col]
        feature_cols = self.feature_sets[feature_set]
        folds = build_expanding_folds(
            labeled["Date"],
            n_folds=5,
            min_train_days=252 * 3,
            valid_days=252,
            purge_days=horizon,
        )
        params = dict(self.param_presets[param_preset])
        fold_reports = []
        for fold in folds:
            fold_reports.append(
                self._fit_eval_fold(
                    labeled=labeled,
                    fold=fold,
                    feature_cols=feature_cols,
                    params=params,
                    label_gain=labeled_pack.label_gain,
                )
            )

        summary = {
            "experiment_id": experiment_id,
            "horizon_days": horizon,
            "label_scheme": scheme,
            "label_description": labeled_pack.description,
            "gain_name": gain_name,
            "label_gain": list(labeled_pack.label_gain),
            "label_gain_param": label_gain_to_param(labeled_pack.label_gain),
            "max_relevance": labeled_pack.max_relevance,
            "param_preset": param_preset,
            "ranker_params": params,
            "feature_set": feature_set,
            "feature_count": len(feature_cols),
            "folds": fold_reports,
            "stability_across_folds": self._aggregate(fold_reports),
        }
        path = self.report_dir / f"{experiment_id}.json"
        _write_json(path, _json_safe(summary))
        summary["report_path"] = str(path)
        return summary

    def _run_baseline(
        self,
        panel: pd.DataFrame,
        *,
        horizon: int,
        baseline_name: str,
        feature_col: str,
    ) -> dict[str, Any]:
        experiment_id = f"h{horizon}d__baseline__{baseline_name}"
        # Use Label A only for NDCG relevance grades in evaluation.
        labeled_pack = assign_label_scheme(
            panel,
            return_col=f"future_return_{horizon}d",
            scheme="A",
            gain_name="linear",
            min_group_size=self.min_group,
        )
        labeled = labeled_pack.frame.copy()
        labeled["future_return"] = labeled[f"future_return_{horizon}d"]
        folds = build_expanding_folds(
            labeled["Date"],
            n_folds=5,
            min_train_days=252 * 3,
            valid_days=252,
            purge_days=horizon,
        )
        fold_reports = []
        for fold in folds:
            valid = mask_fold_partition(labeled, fold, partition="validation")
            valid = valid.dropna(subset=[feature_col, "relevance"]).copy()
            valid_s, g = build_ranking_groups(valid, group_keys=("Date", "Region"))
            assert_group_integrity(valid_s, g)
            scored = valid_s.copy()
            scored["score"] = momentum_scores(scored, feature_col=feature_col).to_numpy()
            fold_reports.append(
                self._evaluate_scored(scored, fold, train_rows=0, train_groups=0, best_iteration=None)
            )
        summary = {
            "experiment_id": experiment_id,
            "horizon_days": horizon,
            "label_scheme": "baseline",
            "gain_name": None,
            "feature_set": f"baseline_{baseline_name}",
            "feature_count": 1,
            "folds": fold_reports,
            "stability_across_folds": self._aggregate(fold_reports),
        }
        path = self.report_dir / f"{experiment_id}.json"
        _write_json(path, _json_safe(summary))
        summary["report_path"] = str(path)
        return summary

    def _fit_eval_fold(
        self,
        *,
        labeled: pd.DataFrame,
        fold: WalkForwardFold,
        feature_cols: tuple[str, ...],
        params: dict[str, Any],
        label_gain: tuple[float, ...],
    ) -> dict[str, Any]:
        train_df = mask_fold_partition(labeled, fold, partition="train")
        valid_df = mask_fold_partition(labeled, fold, partition="validation")
        required = list(self.feature_sets["A"]) + ["relevance"]
        train_df = train_df.dropna(subset=required).copy()
        valid_df = valid_df.dropna(subset=required).copy()
        if train_df.empty or valid_df.empty:
            raise TrainingError("Empty train/valid after dropna")

        train_s, train_g = build_ranking_groups(train_df, group_keys=("Date", "Region"))
        valid_s, valid_g = build_ranking_groups(valid_df, group_keys=("Date", "Region"))
        assert_group_integrity(train_s, train_g)
        assert_group_integrity(valid_s, valid_g)

        x_train = train_s.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
        x_valid = valid_s.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
        fit = fit_lgbm_ranker(
            x_train,
            train_s["relevance"],
            train_g,
            x_valid=x_valid,
            y_valid=valid_s["relevance"],
            group_valid=valid_g,
            params=params,
            label_gain=label_gain,
        )
        scored = valid_s.copy()
        scored["score"] = predict_rank_scores(fit.model, x_valid)
        return self._evaluate_scored(
            scored,
            fold,
            train_rows=len(train_s),
            train_groups=len(train_g),
            best_iteration=fit.best_iteration,
        )

    def _evaluate_scored(
        self,
        scored: pd.DataFrame,
        fold: WalkForwardFold,
        *,
        train_rows: int,
        train_groups: int,
        best_iteration: int | None,
    ) -> dict[str, Any]:
        by_country = {}
        for region in REGIONS:
            group = scored.loc[scored["Region"] == region]
            if group.empty:
                by_country[region] = {"n_rows": 0}
                continue
            metrics = evaluate_country_ranking(
                group,
                ndcg_ks=self.ndcg_ks,
                tie_warning_ratio=float(self.scheme_cfg.get("tie_warning_ratio", 0.5)),
            )
            diag = fold_score_diagnostics(
                group, low_resolution_tie_threshold=self.low_res_thr
            )
            metrics["score_resolution"] = diag["resolution"]
            metrics["top10_membership"] = diag["top10_membership"]["by_region"].get(
                region, {}
            )
            by_country[region] = metrics

        all_metrics = evaluate_country_ranking(
            scored,
            ndcg_ks=self.ndcg_ks,
            tie_warning_ratio=float(self.scheme_cfg.get("tie_warning_ratio", 0.5)),
        )
        all_diag = fold_score_diagnostics(
            scored, low_resolution_tie_threshold=self.low_res_thr
        )
        all_metrics["score_resolution"] = all_diag["resolution"]
        all_metrics["top10_membership"] = all_diag["top10_membership"]
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

    def _aggregate(self, fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
        def collect(getter) -> dict[str, Any]:
            ndcg5, ndcg10, ic, t10, t20, q51 = [], [], [], [], [], []
            med_tie, p90_tie, uniq, lowres = [], [], [], []
            turnover, ambi = [], []
            warn = 0
            for fold in fold_reports:
                block = getter(fold)
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
                res = block.get("score_resolution") or {}
                if res.get("median_tie_ratio") is not None:
                    med_tie.append(float(res["median_tie_ratio"]))
                if res.get("p90_tie_ratio") is not None:
                    p90_tie.append(float(res["p90_tie_ratio"]))
                if res.get("mean_unique_score_ratio") is not None:
                    uniq.append(float(res["mean_unique_score_ratio"]))
                if res.get("low_resolution_day_ratio") is not None:
                    lowres.append(float(res["low_resolution_day_ratio"]))
                mem = block.get("top10_membership") or {}
                # country block has flat membership; all has by_region
                if "mean_membership_turnover" in mem and mem["mean_membership_turnover"] is not None:
                    turnover.append(float(mem["mean_membership_turnover"]))
                if "mean_cutoff_ambiguity" in mem and mem["mean_cutoff_ambiguity"] is not None:
                    ambi.append(float(mem["mean_cutoff_ambiguity"]))
                elif isinstance(mem.get("by_region"), dict):
                    for rstats in mem["by_region"].values():
                        if rstats.get("mean_membership_turnover") is not None:
                            turnover.append(float(rstats["mean_membership_turnover"]))
                        if rstats.get("mean_cutoff_ambiguity") is not None:
                            ambi.append(float(rstats["mean_cutoff_ambiguity"]))
                if block.get("warnings"):
                    warn += 1
            return {
                "ndcg_at_5": _stats(ndcg5),
                "ndcg_at_10": _stats(ndcg10),
                "mean_ic": _stats(ic),
                "top10_excess": _stats(t10),
                "top20_excess": _stats(t20),
                "q5_q1_spread": _stats(q51),
                "median_tie_ratio": _stats(med_tie),
                "p90_tie_ratio": _stats(p90_tie),
                "unique_score_ratio": _stats(uniq),
                "low_resolution_day_ratio": _stats(lowres),
                "top10_membership_turnover": _stats(turnover),
                "top10_cutoff_ambiguity": _stats(ambi),
                "folds_with_score_warnings": warn,
            }

        out: dict[str, Any] = {
            "all_country_groups": collect(lambda f: f["all_country_groups"]),
            "by_country": {},
        }
        for region in REGIONS:
            out["by_country"][region] = collect(
                lambda f, r=region: f["by_country"].get(r, {})
            )
        return out

    def _compare(self, experiments: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize label/resolution tradeoffs without adaptive retuning."""
        focus = [
            e
            for e in experiments
            if e.get("horizon_days") == 5
            and e.get("feature_set") == "A"
            and e.get("param_preset") == "default"
            and e.get("label_scheme") in {"A", "B", "C"}
        ]
        rows = []
        for e in focus:
            st = e["stability_across_folds"]["all_country_groups"]
            rows.append(
                {
                    "experiment_id": e["experiment_id"],
                    "label_scheme": e["label_scheme"],
                    "gain_name": e["gain_name"],
                    "median_tie_ratio": st["median_tie_ratio"]["mean"],
                    "unique_score_ratio": st["unique_score_ratio"]["mean"],
                    "mean_ic": st["mean_ic"]["mean"],
                    "top10_excess": st["top10_excess"]["mean"],
                    "q5_q1_spread": st["q5_q1_spread"]["mean"],
                    "ndcg_at_5": st["ndcg_at_5"]["mean"],
                    "top10_membership_turnover": st["top10_membership_turnover"]["mean"],
                    "top10_cutoff_ambiguity": st["top10_cutoff_ambiguity"]["mean"],
                }
            )
        return {
            "label_d_skipped": skip_reason_label_d(),
            "primary_5d_fsA_default_params": rows,
            "success_criteria_note": (
                "Success requires lower tie ratio AND non-worse IC/Top10 AND multi-country "
                "support; NDCG alone is insufficient."
            ),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3H label/score-resolution LTR study")
    parser.add_argument(
        "--universe", type=Path, default=Path("config/universe.global100.json")
    )
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    try:
        overview = LabelResolutionRunner(settings, args.universe).run(
            force_refresh=args.force_refresh
        )
    except MLError as exc:
        logger.error("Phase 3H failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected Phase 3H failure")
        return 1
    logger.info("Phase 3H complete experiments=%d", len(overview.get("experiments", [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
