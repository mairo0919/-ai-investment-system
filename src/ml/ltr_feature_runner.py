"""Phase 3G: Feature Set A/B/C comparison with Country-internal LTR."""

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
from src.features.feature_sets import (
    ABLATION_CATEGORIES,
    ablation_feature_set,
    load_feature_set_config,
    resolve_feature_sets,
    without_macro_columns,
)
from src.features.macro_features import (
    attach_mkt_return_60d,
    compute_mkt_return_60d,
    fetch_macro_feature_frames,
    join_macro_and_currency_features,
    load_macro_config,
    summarize_macro_fetch,
)
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
from src.ml.walk_forward_runner import REGION_BENCHMARKS, WalkForwardRunner
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

REGIONS = ("Japan", "United States", "Europe")
HORIZONS = (5, 10)
FORBIDDEN_FEATURES = frozenset(
    {"Symbol", "Country", "Region", "Market", "Currency", "Sector", "Industry"}
)


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


class FeatureExpansionLTRRunner:
    """Compare Feature Sets A/B/C under Phase 3F LTR protocol."""

    def __init__(
        self,
        settings: Settings,
        universe_path: Path,
        *,
        relevance_config_path: Path | None = None,
        macro_config_path: Path | None = None,
        feature_set_config_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.wf = WalkForwardRunner(settings, universe_path)
        self.relevance_config = load_relevance_config(relevance_config_path)
        self.macro_specs = load_macro_config(macro_config_path)
        self.fs_config = load_feature_set_config(feature_set_config_path)
        self.report_dir = settings.reports_dir / "feature_expansion"
        self.feature_sets = resolve_feature_sets(self.macro_specs)

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Phase 3G feature expansion LTR starting")

        fetch_summary = self.wf._fetch_equities(force_refresh=force_refresh)
        usable_meta, quality = self.wf._select_usable_symbols(
            fetch_summary, enrich_metadata=False
        )
        market_frames = self.wf._fetch_benchmarks(force_refresh=force_refresh)
        panel = self.wf._build_panel(usable_meta, market_frames)
        if panel.empty:
            raise TrainingError("Empty panel for Phase 3G")

        # Benchmark OHLCV for 60d relative strength.
        mkt_ohlcv = {}
        for region, (ticker, _) in REGION_BENCHMARKS.items():
            raw = self.wf.cache.load(ticker)
            if raw is None:
                raise TrainingError(f"Missing benchmark cache for {ticker}")
            mkt_ohlcv[region] = raw
        panel = attach_mkt_return_60d(panel, compute_mkt_return_60d(mkt_ohlcv))

        min_sector = int(self.fs_config.get("min_sector_group_size", 3))
        panel = add_cross_section_feature_block(
            panel, min_sector_group_size=min_sector, min_cs_group_size=5
        )

        macro_frames = fetch_macro_feature_frames(
            self.wf.provider,
            self.wf.cache,
            self.macro_specs,
            lookback_years=self.settings.lookback_years,
            interval=self.settings.interval,
            force_refresh=force_refresh,
        )
        panel, macro_cols = join_macro_and_currency_features(panel, macro_frames)
        macro_fetch_report = summarize_macro_fetch(macro_frames)

        # Validate feature sets exist on panel (sector pct may be NaN — OK until dropna).
        for name, cols in self.feature_sets.items():
            missing = [c for c in cols if c not in panel.columns]
            if missing:
                raise TrainingError(f"Feature Set {name} missing columns: {missing[:20]}")
            leak = FORBIDDEN_FEATURES.intersection(cols)
            if leak:
                raise TrainingError(f"Feature Set {name} has forbidden cols: {leak}")
            future_leak = [c for c in cols if c.startswith("future_return")]
            if future_leak:
                raise TrainingError(f"Future leak in Feature Set {name}: {future_leak}")

        experiments: list[dict[str, Any]] = []
        for horizon in HORIZONS:
            return_col = f"future_return_{horizon}d"
            labeled = assign_relevance_labels(
                panel, return_col=return_col, config=self.relevance_config
            )
            labeled["future_return"] = labeled[return_col]
            folds = build_expanding_folds(
                labeled["Date"],
                n_folds=5,
                min_train_days=252 * 3,
                valid_days=252,
                purge_days=horizon,
            )

            for set_name, cols in self.feature_sets.items():
                exp = self._run_feature_set(
                    labeled=labeled,
                    folds=folds,
                    horizon=horizon,
                    set_name=set_name,
                    feature_cols=cols,
                )
                experiments.append(exp)

            for baseline_name, feat in MOMENTUM_BASELINES:
                experiments.append(
                    self._run_baseline(
                        labeled=labeled,
                        folds=folds,
                        horizon=horizon,
                        baseline_name=baseline_name,
                        feature_col=feat,
                    )
                )

        comparison = self._compare_feature_sets(experiments)
        ablation_exps: list[dict[str, Any]] = []
        if comparison.get("run_ablation"):
            logger.info("Feature Set C improved — running category ablations")
            for horizon in HORIZONS:
                return_col = f"future_return_{horizon}d"
                labeled = assign_relevance_labels(
                    panel, return_col=return_col, config=self.relevance_config
                )
                labeled["future_return"] = labeled[return_col]
                folds = build_expanding_folds(
                    labeled["Date"],
                    n_folds=5,
                    min_train_days=252 * 3,
                    valid_days=252,
                    purge_days=horizon,
                )
                full_c = self.feature_sets["C"]
                ablations = {
                    "without_macro": without_macro_columns(full_c, self.macro_specs),
                    **{
                        name: ablation_feature_set(full_c, drop_columns=cols)
                        for name, cols in ABLATION_CATEGORIES.items()
                    },
                }
                for abl_name, cols in ablations.items():
                    ablation_exps.append(
                        self._run_feature_set(
                            labeled=labeled,
                            folds=folds,
                            horizon=horizon,
                            set_name=f"C_{abl_name}",
                            feature_cols=cols,
                        )
                    )

        overview = {
            "created_at_utc": _timestamp_iso(),
            "phase": "3G",
            "purpose": "Cross-sectional / macro / market-context feature expansion for Country LTR",
            "macro_tickers": macro_fetch_report,
            "macro_columns_joined": list(macro_cols),
            "feature_set_sizes": {k: len(v) for k, v in self.feature_sets.items()},
            "feature_sets": {k: list(v) for k, v in self.feature_sets.items()},
            "min_sector_group_size": min_sector,
            "universe": self.wf.universe.composition_report(),
            "fetch_summary": fetch_summary,
            "quality_filter": quality,
            "comparison": comparison,
            "experiments": [
                {
                    "experiment_id": e["experiment_id"],
                    "stability": e.get("stability_across_folds"),
                    "report_path": e.get("report_path"),
                }
                for e in experiments + ablation_exps
            ],
            "known_limitations": [
                "No point-in-time fundamentals (PER/PBR/MarketCap as features forbidden).",
                "Macro joined with merge_asof(backward) for point-in-time availability.",
                "Research fixed universe may contain survivorship bias.",
                "Primary evaluation remains Country-internal ranking (not Global raw).",
            ],
            "look_ahead_controls": [
                "Cross-sectional ranks use same-day peers only.",
                "Relative returns use lagged price changes only.",
                "Relevance labels from future_return are excluded from features.",
                "Walk-forward purge equals target horizon.",
            ],
        }
        path = self.report_dir / "feature_expansion_overview.json"
        _write_json(path, _json_safe(overview))
        logger.info("Wrote Phase 3G overview -> %s", path)
        return overview

    def _run_feature_set(
        self,
        *,
        labeled: pd.DataFrame,
        folds: list[WalkForwardFold],
        horizon: int,
        set_name: str,
        feature_cols: tuple[str, ...],
    ) -> dict[str, Any]:
        experiment_id = f"fs{set_name}__ltr_{horizon}d__common"
        logger.info("Running %s features=%d", experiment_id, len(feature_cols))
        fold_reports = []
        for fold in folds:
            train_df = mask_fold_partition(labeled, fold, partition="train")
            valid_df = mask_fold_partition(labeled, fold, partition="validation")
            # Require Phase-3F base features; newer CS/sector/macro NaNs are OK for LGBM.
            required = list(self.feature_sets["A"]) + ["relevance"]
            train_df = train_df.dropna(subset=required).copy()
            valid_df = valid_df.dropna(subset=required).copy()
            if train_df.empty or valid_df.empty:
                raise TrainingError(f"Empty partition after dropna in {experiment_id}")

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
            )
            scored = valid_s.copy()
            scored["score"] = predict_rank_scores(fit.model, x_valid)
            fold_reports.append(
                self._evaluate_fold(
                    scored,
                    fold,
                    train_rows=len(train_s),
                    train_groups=len(train_g),
                    best_iteration=fit.best_iteration,
                )
            )

        summary = {
            "experiment_id": experiment_id,
            "horizon_days": horizon,
            "feature_set": set_name,
            "feature_count": len(feature_cols),
            "model_type": "common_ranker",
            "folds": fold_reports,
            "stability_across_folds": self._aggregate_stability(fold_reports),
        }
        path = self.report_dir / f"{experiment_id}.json"
        _write_json(path, _json_safe(summary))
        summary["report_path"] = str(path)
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
        experiment_id = f"baseline__{baseline_name}__{horizon}d"
        fold_reports = []
        for fold in folds:
            valid_df = mask_fold_partition(labeled, fold, partition="validation")
            valid_df = valid_df.dropna(subset=[feature_col, "relevance"]).copy()
            valid_s, valid_g = build_ranking_groups(valid_df, group_keys=("Date", "Region"))
            assert_group_integrity(valid_s, valid_g)
            scored = valid_s.copy()
            scored["score"] = momentum_scores(scored, feature_col=feature_col).to_numpy()
            fold_reports.append(
                self._evaluate_fold(
                    scored, fold, train_rows=0, train_groups=0, best_iteration=None
                )
            )
        summary = {
            "experiment_id": experiment_id,
            "horizon_days": horizon,
            "feature_set": f"baseline_{baseline_name}",
            "feature_count": 1,
            "model_type": "baseline",
            "folds": fold_reports,
            "stability_across_folds": self._aggregate_stability(fold_reports),
        }
        path = self.report_dir / f"{experiment_id}.json"
        _write_json(path, _json_safe(summary))
        summary["report_path"] = str(path)
        return summary

    def _evaluate_fold(
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
            by_country[region] = evaluate_country_ranking(
                group,
                ndcg_ks=self.relevance_config.ndcg_ks,
                tie_warning_ratio=self.relevance_config.tie_warning_ratio,
            )
        all_metrics = evaluate_country_ranking(
            scored,
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

    @staticmethod
    def _aggregate_stability(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"by_country": {}, "all_country_groups": {}}

        def collect(block_getter) -> dict[str, Any]:
            ndcg5, ndcg10, ic, t10, t20, q51 = [], [], [], [], [], []
            warn = 0
            for fold in fold_reports:
                block = block_getter(fold)
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
                    warn += 1
            return {
                "ndcg_at_5": _stats(ndcg5),
                "ndcg_at_10": _stats(ndcg10),
                "mean_ic": _stats(ic),
                "top10_excess": _stats(t10),
                "top20_excess": _stats(t20),
                "q5_q1_spread": _stats(q51),
                "folds_with_score_warnings": warn,
            }

        out["all_country_groups"] = collect(lambda f: f["all_country_groups"])
        for region in REGIONS:
            out["by_country"][region] = collect(
                lambda f, r=region: f["by_country"].get(r, {})
            )
        return out

    def _compare_feature_sets(self, experiments: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide whether Feature Set C improved enough to justify ablation."""
        by_key = {e["experiment_id"]: e for e in experiments}
        table: dict[str, Any] = {}
        wins_c = 0
        comparisons = 0
        for horizon in HORIZONS:
            table[str(horizon)] = {}
            for region in REGIONS:
                row = {}
                metrics = {}
                for set_name in ("A", "B", "C"):
                    eid = f"fs{set_name}__ltr_{horizon}d__common"
                    st = by_key[eid]["stability_across_folds"]["by_country"][region]
                    metrics[set_name] = {
                        "mean_ic": st["mean_ic"],
                        "top10_excess": st["top10_excess"],
                        "q5_q1_spread": st["q5_q1_spread"],
                        "ndcg_at_5": st["ndcg_at_5"],
                        "ndcg_at_10": st["ndcg_at_10"],
                        "folds_with_score_warnings": st["folds_with_score_warnings"],
                    }
                row["sets"] = metrics
                # C vs best(A,B) on Mean IC and Top10 (ignore if heavy tie warnings on C)
                a_ic = metrics["A"]["mean_ic"]["mean"]
                b_ic = metrics["B"]["mean_ic"]["mean"]
                c_ic = metrics["C"]["mean_ic"]["mean"]
                a_t = metrics["A"]["top10_excess"]["mean"]
                b_t = metrics["B"]["top10_excess"]["mean"]
                c_t = metrics["C"]["top10_excess"]["mean"]
                baseline_ic = max(
                    [x for x in (a_ic, b_ic) if x is not None], default=None
                )
                baseline_t = max(
                    [x for x in (a_t, b_t) if x is not None], default=None
                )
                c_warn = metrics["C"]["folds_with_score_warnings"]
                improved = False
                if (
                    c_ic is not None
                    and baseline_ic is not None
                    and c_t is not None
                    and baseline_t is not None
                    and c_warn <= 2
                ):
                    comparisons += 1
                    if c_ic > baseline_ic and c_t > baseline_t:
                        improved = True
                        wins_c += 1
                row["c_improved_vs_ab"] = improved
                table[str(horizon)][region] = row

        run_ablation = comparisons > 0 and wins_c >= max(2, comparisons // 2)
        return {
            "by_horizon_country": table,
            "c_win_count": wins_c,
            "comparisons": comparisons,
            "run_ablation": run_ablation,
            "rule": "Ablation if C beats best(A,B) on both Mean IC and Top10 excess in >= half of country×horizon cells (and C tie warnings <= 2).",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3G feature expansion LTR study")
    parser.add_argument(
        "--universe", type=Path, default=Path("config/universe.global100.json")
    )
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    try:
        overview = FeatureExpansionLTRRunner(settings, args.universe).run(
            force_refresh=args.force_refresh
        )
    except MLError as exc:
        logger.error("Phase 3G failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected Phase 3G failure")
        return 1

    logger.info(
        "Phase 3G complete experiments=%d ablation=%s",
        len(overview.get("experiments", [])),
        overview.get("comparison", {}).get("run_ablation"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
