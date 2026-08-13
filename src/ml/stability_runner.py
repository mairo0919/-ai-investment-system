"""Phase 3E: ranking edge stability analysis runner (diagnostics only)."""

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
from src.ml.lightgbm_model import fit_lightgbm
from src.ml.model_registry import build_logistic_pipeline
from src.ml.dataset import materialize_target
from src.ml.stability_diagnostics import (
    REGIONS,
    assign_volatility_regime,
    classify_market_regime,
    compare_raw_vs_normalized,
    country_ranking_report,
    daily_score_dispersion,
    leave_one_group_out,
    one_day_ic_diagnostics,
    quintile_bucket_analysis,
    ranking_metrics_bundle,
    regime_performance,
    score_distribution_summary,
    summarize_dispersion,
    top_bottom_spread,
    top_bucket_contribution,
)
from src.ml.targets import TARGET_UP_1D, TARGET_UP_5D, TARGET_UP_10D, TargetConfig
from src.ml.walk_forward import WalkForwardFold, build_expanding_folds, mask_fold_partition
from src.ml.walk_forward_runner import (
    GENERIC_MARKET_PREFIX,
    REGION_BENCHMARKS,
    WalkForwardRunner,
)
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Focused experiment matrix (no combinatorial search).
FOCUS_EXPERIMENTS: tuple[tuple[TargetConfig, str], ...] = (
    (TARGET_UP_5D, "logistic_regression"),
    (TARGET_UP_10D, "lightgbm"),
    (TARGET_UP_1D, "logistic_regression"),  # reference for IC artifact
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
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    return obj


class StabilityAnalysisRunner:
    """Re-score Walk-Forward validation folds and run Phase 3E diagnostics."""

    def __init__(self, settings: Settings, universe_path: Path) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.wf = WalkForwardRunner(settings, universe_path)
        self.report_dir = settings.reports_dir / "stability"

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Phase 3E stability analysis starting universe=%s", self.universe_path)

        fetch_summary = self.wf._fetch_equities(force_refresh=force_refresh)
        usable_meta, quality = self.wf._select_usable_symbols(
            fetch_summary, enrich_metadata=False
        )
        market_frames = self.wf._fetch_benchmarks(force_refresh=force_refresh)
        panel = self.wf._build_panel(usable_meta, market_frames)
        if panel.empty:
            raise TrainingError("Panel empty; cannot run stability analysis")

        feature_cols = tuple(
            list(FEATURE_COLUMNS)
            + [f"{GENERIC_MARKET_PREFIX}_{s}" for s in MARKET_FEATURE_SUFFIXES]
        )

        country_ranking: dict[str, Any] = {}
        score_distribution: dict[str, Any] = {}
        regime_analysis: dict[str, Any] = {}
        group_contribution: dict[str, Any] = {}
        bucket_analysis: dict[str, Any] = {}
        one_day_diag: dict[str, Any] = {}

        for target, model_name in FOCUS_EXPERIMENTS:
            experiment_id = f"{target.name}__{model_name}"
            logger.info("Scoring folds for %s", experiment_id)
            labeled = materialize_target(panel, target)
            labeled = labeled.dropna(
                subset=list(feature_cols) + [target.target_column]
            ).copy()
            folds = build_expanding_folds(
                labeled["Date"],
                n_folds=5,
                min_train_days=252 * 3,
                valid_days=252,
                purge_days=target.purge_days,
            )
            fold_scored = self._score_folds(
                labeled=labeled,
                target=target,
                feature_cols=feature_cols,
                folds=folds,
                model_name=model_name,
            )

            country_ranking[experiment_id] = self._analyze_country(fold_scored)
            score_distribution[experiment_id] = self._analyze_scores(fold_scored)
            regime_analysis[experiment_id] = self._analyze_regimes(
                fold_scored, market_frames
            )
            group_contribution[experiment_id] = self._analyze_contribution(fold_scored)
            bucket_analysis[experiment_id] = self._analyze_buckets(fold_scored)

            if target.name == "target_up_1d":
                one_day_diag = self._analyze_one_day_ic(fold_scored)

        overview = {
            "created_at_utc": _timestamp_iso(),
            "phase": "3E",
            "purpose": "Diagnose whether a weak ranking edge is real and where it appears",
            "universe": self.wf.universe.composition_report(),
            "quality_filter": quality,
            "fetch_summary": fetch_summary,
            "focus_experiments": [
                f"{t.name}__{m}" for t, m in FOCUS_EXPERIMENTS
            ],
            "reports": {
                "country_ranking": "country_ranking.json",
                "score_distribution": "score_distribution.json",
                "regime_analysis": "regime_analysis.json",
                "group_contribution": "group_contribution.json",
                "bucket_analysis": "bucket_analysis.json",
                "one_day_ic_diagnostics": "one_day_ic_diagnostics.json",
            },
            "known_limitations": [
                "Diagnostics use Walk-Forward validation folds only (no new test tuning).",
                "Country-normalized ranking is post-hoc score transform; models are not retrained.",
                "Leave-one-group-out is evaluation-only (no retrain).",
                "Top-Bottom spread is diagnostic, not a long/short backtest.",
                "Research fixed universe may contain survivorship bias.",
            ],
        }

        payloads = {
            "country_ranking.json": country_ranking,
            "score_distribution.json": score_distribution,
            "regime_analysis.json": regime_analysis,
            "group_contribution.json": group_contribution,
            "bucket_analysis.json": bucket_analysis,
            "one_day_ic_diagnostics.json": one_day_diag,
            "stability_overview.json": overview,
        }
        for name, payload in payloads.items():
            path = self.report_dir / name
            _write_json(path, _json_safe(payload))
            logger.info("Wrote %s", path)

        overview["paths"] = {k: str(self.report_dir / k) for k in payloads}
        return overview

    def _score_folds(
        self,
        *,
        labeled: pd.DataFrame,
        target: TargetConfig,
        feature_cols: tuple[str, ...],
        folds: list[WalkForwardFold],
        model_name: str,
    ) -> list[pd.DataFrame]:
        scored_folds: list[pd.DataFrame] = []
        for fold in folds:
            train_df = mask_fold_partition(labeled, fold, partition="train")
            valid_df = mask_fold_partition(labeled, fold, partition="validation")
            if train_df.empty or valid_df.empty:
                raise TrainingError(f"Empty partition fold={fold.fold_id}")

            x_train = train_df.loc[:, list(feature_cols)]
            y_train = train_df[target.target_column].astype(int)
            x_valid = valid_df.loc[:, list(feature_cols)]

            if model_name == "logistic_regression":
                model = build_logistic_pipeline()
                model.fit(x_train, y_train)
            elif model_name == "lightgbm":
                fit = fit_lightgbm(
                    x_train, y_train, x_valid, valid_df[target.target_column].astype(int)
                )
                model = fit.model
            else:
                raise TrainingError(f"Unknown model {model_name}")

            scored = valid_df.copy()
            scored["score"] = model.predict_proba(x_valid)[:, 1]
            scored["future_return"] = scored[target.future_return_column]
            scored["fold_id"] = fold.fold_id
            scored["train_start"] = fold.train_start
            scored["train_end"] = fold.train_end
            scored_folds.append(scored)
            logger.info(
                "Scored fold=%d rows=%d symbols=%d",
                fold.fold_id,
                len(scored),
                scored["Symbol"].nunique(),
            )
        return scored_folds

    def _analyze_country(self, fold_scored: list[pd.DataFrame]) -> dict[str, Any]:
        folds_out: list[dict[str, Any]] = []
        for scored in fold_scored:
            fold_id = int(scored["fold_id"].iloc[0])
            compare = compare_raw_vs_normalized(scored)
            country = country_ranking_report(scored)
            folds_out.append(
                {
                    "fold_id": fold_id,
                    "country_internal_ranking": country,
                    "raw_vs_normalized": compare,
                }
            )
        # Aggregate country Mean IC / Top10 across folds
        agg: dict[str, Any] = {"folds": folds_out, "across_folds": {}}
        for region in ("global", *REGIONS):
            ics: list[float] = []
            t10s: list[float] = []
            for f in folds_out:
                block = f["country_internal_ranking"]
                metrics = block["global"] if region == "global" else block["by_country"].get(
                    region, {}
                )
                ic = (metrics.get("ic") or {}).get("mean")
                t10 = (metrics.get("top_10pct_excess_return") or {}).get("mean")
                if ic is not None:
                    ics.append(float(ic))
                if t10 is not None:
                    t10s.append(float(t10))
            agg["across_folds"][region] = {
                "mean_ic": _series_stats(ics),
                "top10_excess": _series_stats(t10s),
            }
        # Normalized global comparison across folds
        raw_ics, z_ics = [], []
        for f in folds_out:
            r = f["raw_vs_normalized"]["A_raw_probability"]["global"]
            z = f["raw_vs_normalized"]["B_country_zscore"]["global"]
            if (r.get("ic") or {}).get("mean") is not None:
                raw_ics.append(float(r["ic"]["mean"]))
            if (z.get("ic") or {}).get("mean") is not None:
                z_ics.append(float(z["ic"]["mean"]))
        agg["raw_vs_country_z_global"] = {
            "raw_mean_ic": _series_stats(raw_ics),
            "country_z_mean_ic": _series_stats(z_ics),
        }
        return agg

    def _analyze_scores(self, fold_scored: list[pd.DataFrame]) -> dict[str, Any]:
        folds_out = []
        for scored in fold_scored:
            daily = daily_score_dispersion(scored)
            folds_out.append(
                {
                    "fold_id": int(scored["fold_id"].iloc[0]),
                    "overall": score_distribution_summary(scored),
                    "dispersion": summarize_dispersion(daily),
                }
            )
        return {"folds": folds_out}

    def _analyze_regimes(
        self,
        fold_scored: list[pd.DataFrame],
        market_frames: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Market & volatility regimes using country benchmarks (train thresholds)."""
        folds_out = []
        for scored in fold_scored:
            fold_id = int(scored["fold_id"].iloc[0])
            train_end = pd.Timestamp(scored["train_end"].iloc[0])
            frame = scored.copy()
            # Market regime from stock-joined mkt_* (region-matched benchmark).
            if not {"mkt_sma_60_ratio", "mkt_return_20d", "mkt_volatility_20"} <= set(
                frame.columns
            ):
                raise TrainingError("Panel missing mkt_* regime features")
            frame["market_regime"] = classify_market_regime(
                frame["mkt_sma_60_ratio"], frame["mkt_return_20d"]
            )

            # Volatility thresholds from train period of each region's benchmark.
            vol_labels = pd.Series(index=frame.index, dtype=object)
            thresholds_by_region: dict[str, dict[str, float]] = {}
            for region, (_, prefix) in REGION_BENCHMARKS.items():
                mkt = market_frames[region]
                vol_col = f"{prefix}_volatility_20"
                # Prefer renamed columns already on frame; train thresholds from market_frames.
                dates = pd.to_datetime(mkt["Date"])
                train_vol = mkt.loc[dates <= train_end, vol_col]
                region_mask = frame["Region"] == region
                valid_vol = frame.loc[region_mask, "mkt_volatility_20"]
                labels, thr = assign_volatility_regime(train_vol, valid_vol)
                vol_labels.loc[region_mask] = labels.to_numpy()
                thresholds_by_region[region] = thr
            frame["volatility_regime"] = vol_labels

            folds_out.append(
                {
                    "fold_id": fold_id,
                    "market_regime": regime_performance(frame, regime_col="market_regime"),
                    "volatility_regime": regime_performance(
                        frame, regime_col="volatility_regime"
                    ),
                    "volatility_thresholds_from_train": thresholds_by_region,
                }
            )
        return {"folds": folds_out, "regime_rule": {
            "Bull": "mkt_sma_60_ratio > 0 and mkt_return_20d > 0",
            "Bear": "mkt_sma_60_ratio < 0 and mkt_return_20d < 0",
            "Neutral": "otherwise",
            "volatility": "Low/Medium/High terciles from each fold's train period",
        }}

    def _analyze_contribution(self, fold_scored: list[pd.DataFrame]) -> dict[str, Any]:
        folds_out = []
        for scored in fold_scored:
            fold_id = int(scored["fold_id"].iloc[0])
            contrib = top_bucket_contribution(scored, fraction=0.10)
            loco = leave_one_group_out(scored, group_col="Region")
            # Major sectors only (skip tiny)
            sector_counts = scored["Sector"].fillna("null").value_counts()
            major = [s for s, c in sector_counts.items() if c >= 100]
            base = ranking_metrics_bundle(scored)
            loso_parts = {}
            for sector in major:
                keep = scored.loc[scored["Sector"].fillna("null") != sector]
                new = ranking_metrics_bundle(keep)
                base_ic = (base.get("ic") or {}).get("mean")
                new_ic = (new.get("ic") or {}).get("mean")
                base_t = (base.get("top_10pct_excess_return") or {}).get("mean")
                new_t = (new.get("top_10pct_excess_return") or {}).get("mean")
                loso_parts[str(sector)] = {
                    "metrics": new,
                    "delta_mean_ic": None
                    if base_ic is None or new_ic is None
                    else float(new_ic - base_ic),
                    "delta_top10_excess": None
                    if base_t is None or new_t is None
                    else float(new_t - base_t),
                }
            folds_out.append(
                {
                    "fold_id": fold_id,
                    "top10_composition": contrib,
                    "leave_one_country_out": loco,
                    "leave_one_major_sector_out": {
                        "baseline": base,
                        "excluded": loso_parts,
                    },
                }
            )
        return {"folds": folds_out}

    def _analyze_buckets(self, fold_scored: list[pd.DataFrame]) -> dict[str, Any]:
        folds_out = []
        for scored in fold_scored:
            folds_out.append(
                {
                    "fold_id": int(scored["fold_id"].iloc[0]),
                    "quintiles": quintile_bucket_analysis(scored),
                    "ranking_spread_top20_minus_bottom20": top_bottom_spread(
                        scored, fraction=0.20
                    ),
                    "top_fractions_reference": {
                        "note": "See walk_forward reports for Top5/10/20 daily metrics",
                    },
                }
            )
        return {"folds": folds_out}

    def _analyze_one_day_ic(self, fold_scored: list[pd.DataFrame]) -> dict[str, Any]:
        folds_out = []
        pooled = pd.concat(fold_scored, ignore_index=True)
        for scored in fold_scored:
            folds_out.append(
                {
                    "fold_id": int(scored["fold_id"].iloc[0]),
                    **one_day_ic_diagnostics(scored),
                }
            )
        return {
            "experiment_id": "target_up_1d__logistic_regression",
            "folds": folds_out,
            "pooled_all_validation_folds": one_day_ic_diagnostics(pooled),
        }


def _series_stats(values: list[float]) -> dict[str, float | None]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3E ranking edge stability diagnostics"
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("config/universe.global100.json"),
    )
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    try:
        overview = StabilityAnalysisRunner(settings, args.universe).run(
            force_refresh=args.force_refresh
        )
    except MLError as exc:
        logger.error("Stability analysis failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected stability analysis failure")
        return 1

    logger.info(
        "Phase 3E complete; reports under %s",
        overview.get("paths", {}).get("stability_overview.json"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
