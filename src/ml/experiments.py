"""Phase 3C experiment runner: targets × feature sets × models."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import Settings, get_settings
from src.core.exceptions import DatasetError, MLError, TrainingError
from src.data.market_service import MarketDataService
from src.features.market_features import (
    add_relative_strength,
    join_market_features,
    market_feature_columns,
    relative_strength_columns,
)
from src.ml.dataset import (
    MODEL_FEATURE_COLUMNS,
    build_experiment_dataset,
    load_stock_feature_frames,
)
from src.ml.evaluator import class_distribution, evaluate_binary
from src.ml.lightgbm_model import (
    fit_lightgbm,
    gain_feature_importance,
    top_feature_importance,
)
from src.ml.model_registry import build_logistic_pipeline, predict_binary
from src.ml.probability_analysis import probability_decile_analysis, top_probability_analysis
from src.ml.split import assert_no_horizon_leakage, chronological_split
from src.ml.targets import TargetConfig, all_target_configs
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSet:
    """Named feature bundle for controlled A/B comparisons."""

    name: str
    columns: tuple[str, ...]


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _date_pair_to_str(pair: tuple[Any, Any]) -> dict[str, str]:
    return {"start": str(pair[0])[:10], "end": str(pair[1])[:10]}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ExperimentRunner:
    """Run Phase 3C target / feature-set / model comparisons."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.topix_available = False
        self.nikkei_available = False
        self.market_notes: list[str] = []

    def run(self) -> dict[str, Any]:
        """Execute the full validation-first experiment matrix."""
        self.settings.experiments_dir.mkdir(parents=True, exist_ok=True)

        market_status = self._ensure_market_data()
        base_frame = self._build_enriched_frame()
        feature_sets = self._feature_sets()
        targets = all_target_configs(self.settings.target_threshold)

        experiment_summaries: list[dict[str, Any]] = []
        for target in targets:
            for feature_set in feature_sets:
                for model_name in ("logistic_regression", "lightgbm"):
                    summary = self._run_one(
                        base_frame=base_frame,
                        target=target,
                        feature_set=feature_set,
                        model_name=model_name,
                    )
                    experiment_summaries.append(summary)

        selection = self._select_by_validation(experiment_summaries)
        selected_test = None
        if selection is not None:
            selected_path = Path(selection["report_path"])
            selected_full = json.loads(selected_path.read_text(encoding="utf-8"))
            selected_test = {
                "experiment_id": selection["experiment_id"],
                "test_metrics": selected_full["metrics"]["test"],
                "test_probability_analysis": selected_full["probability_analysis"]["test"],
            }

        overview = {
            "created_at_utc": _timestamp_iso(),
            "market_status": market_status,
            "market_notes": self.market_notes,
            "target_threshold": self.settings.target_threshold,
            "experiments": experiment_summaries,
            "validation_selection": selection,
            "selected_test_confirmation": selected_test,
            "selection_rule": (
                "Choose the Validation configuration with the highest ROC-AUC; "
                "break ties by higher top_10pct mean_future_return. "
                "Test is reported only for the selected configuration."
            ),
        }
        overview_path = self.settings.experiments_dir / "phase3c_overview.json"
        _write_json(overview_path, overview)
        logger.info("Wrote experiment overview -> %s", overview_path)
        self._log_overview(overview)
        return overview

    def _ensure_market_data(self) -> dict[str, Any]:
        logger.info("Fetching / refreshing market index data for Phase 3C")
        results = MarketDataService(self.settings).run()
        status = {
            name: {
                "available": result.available,
                "ticker": result.ticker,
                "rows_raw": result.rows_raw,
                "rows_processed": result.rows_processed,
                "error": result.error,
            }
            for name, result in results.items()
        }
        self.nikkei_available = bool(results["nikkei"].available)
        self.topix_available = bool(results["topix"].available)
        if not self.nikkei_available:
            raise DatasetError("Nikkei (^N225) market data is required for Phase 3C")
        if not self.topix_available:
            note = (
                "TOPIX cash index unavailable via yfinance; no ETF proxy was used. "
                "Feature Set B includes Nikkei context + relative strength only."
            )
            self.market_notes.append(note)
            logger.warning(note)
        return status

    def _build_enriched_frame(self) -> pd.DataFrame:
        stocks = load_stock_feature_frames(
            self.settings.processed_data_dir, self.settings.tickers
        )
        nikkei = pd.read_csv(self.settings.processed_market_dir / "nikkei_features.csv")
        merged = join_market_features(stocks, nikkei)
        merged = add_relative_strength(merged, market_prefix="nikkei")

        if self.topix_available:
            topix = pd.read_csv(self.settings.processed_market_dir / "topix_features.csv")
            merged = join_market_features(merged, topix)
            merged = add_relative_strength(merged, market_prefix="topix")

        # Persist integrated dataset without mutating Phase 2 stock CSVs.
        out_dir = self.settings.processed_data_dir / "integrated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "stocks_with_market_features.csv"
        merged.to_csv(out_path, index=False)
        logger.info("Wrote integrated dataset -> %s rows=%d", out_path, len(merged))
        return merged

    def _feature_sets(self) -> tuple[FeatureSet, ...]:
        set_a = FeatureSet(name="feature_set_a_stock26", columns=MODEL_FEATURE_COLUMNS)
        market_cols = list(market_feature_columns("nikkei"))
        relative_cols = list(relative_strength_columns("nikkei"))
        if self.topix_available:
            market_cols.extend(market_feature_columns("topix"))
            relative_cols.extend(relative_strength_columns("topix"))
        set_b_cols = tuple(list(MODEL_FEATURE_COLUMNS) + market_cols + relative_cols)
        set_b = FeatureSet(name="feature_set_b_stock_market", columns=set_b_cols)
        return (set_a, set_b)

    def _run_one(
        self,
        *,
        base_frame: pd.DataFrame,
        target: TargetConfig,
        feature_set: FeatureSet,
        model_name: str,
    ) -> dict[str, Any]:
        experiment_id = f"{target.name}__{feature_set.name}__{model_name}"
        logger.info("Running experiment %s", experiment_id)

        dataset = build_experiment_dataset(
            base_frame,
            config=target,
            feature_columns=feature_set.columns,
        )
        # Price history for leakage assertions (pre-target-filter calendar).
        price_history = base_frame.loc[:, ["Date", "Symbol", "Adj Close"]].copy()

        splits = chronological_split(
            dataset,
            train_ratio=self.settings.train_ratio,
            valid_ratio=self.settings.valid_ratio,
            test_ratio=self.settings.test_ratio,
            purge_days=target.purge_days,
        )
        assert_no_horizon_leakage(
            splits,
            horizon_days=target.horizon_days,
            source_prices=price_history,
        )

        if model_name == "logistic_regression":
            model = build_logistic_pipeline()
            model.fit(splits.train.X, splits.train.y)
            importance = None
            best_iteration = None
        elif model_name == "lightgbm":
            fit_result = fit_lightgbm(
                splits.train.X,
                splits.train.y,
                splits.validation.X,
                splits.validation.y,
            )
            model = fit_result.model
            importance = gain_feature_importance(model)
            best_iteration = fit_result.best_iteration
        else:
            raise TrainingError(f"Unknown model_name: {model_name}")

        metrics: dict[str, Any] = {}
        probability_analysis: dict[str, Any] = {}
        for part_name, part in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        ):
            y_pred, y_prob = predict_binary(model, part)
            metrics[part_name] = evaluate_binary(part.y, y_pred, y_prob)
            probability_analysis[part_name] = {
                "deciles": probability_decile_analysis(y_prob, part.future_return),
                "top_probability": top_probability_analysis(y_prob, part.future_return),
            }

        report = {
            "experiment_id": experiment_id,
            "created_at_utc": _timestamp_iso(),
            "target": {
                "name": target.name,
                "horizon_days": target.horizon_days,
                "threshold": target.threshold,
                "purge_days": target.purge_days,
                "target_column": target.target_column,
                "future_return_column": target.future_return_column,
            },
            "feature_set": {
                "name": feature_set.name,
                "feature_count": len(feature_set.columns),
                "features": list(feature_set.columns),
            },
            "model": model_name,
            "best_iteration": best_iteration,
            "date_ranges": {
                "train": _date_pair_to_str(splits.train_dates),
                "validation": _date_pair_to_str(splits.validation_dates),
                "test": _date_pair_to_str(splits.test_dates),
            },
            "rows": {
                "dataset": int(len(dataset.frame)),
                "train": int(len(splits.train.frame)),
                "validation": int(len(splits.validation.frame)),
                "test": int(len(splits.test.frame)),
                "purged_train_rows": splits.purged_train_rows,
                "purged_validation_rows": splits.purged_validation_rows,
            },
            "class_distribution": {
                "train": class_distribution(splits.train.y),
                "validation": class_distribution(splits.validation.y),
                "test": class_distribution(splits.test.y),
            },
            "metrics": metrics,
            "probability_analysis": probability_analysis,
            "feature_importance_gain": importance,
            "feature_importance_top10": (
                top_feature_importance(importance, n=10) if importance else None
            ),
            "market_notes": list(self.market_notes),
        }

        report_path = self.settings.experiments_dir / f"{experiment_id}.json"
        _write_json(report_path, report)

        summary = {
            "experiment_id": experiment_id,
            "target": target.name,
            "feature_set": feature_set.name,
            "model": model_name,
            "feature_count": len(feature_set.columns),
            "rows": report["rows"],
            "date_ranges": report["date_ranges"],
            "validation_roc_auc": metrics["validation"].get("roc_auc"),
            "validation_top10_mean_future_return": (
                probability_analysis["validation"]["top_probability"]["top_fractions"]
                .get("top_10pct", {})
                .get("mean_future_return")
            ),
            "test_roc_auc": metrics["test"].get("roc_auc"),
            "report_path": str(report_path),
            "best_iteration": best_iteration,
        }
        return summary

    @staticmethod
    def _select_by_validation(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        scored = [row for row in summaries if row.get("validation_roc_auc") is not None]
        if not scored:
            return None

        def sort_key(row: dict[str, Any]) -> tuple[float, float]:
            auc = float(row["validation_roc_auc"])
            top = row.get("validation_top10_mean_future_return")
            top_val = float(top) if top is not None else float("-inf")
            return (auc, top_val)

        best = max(scored, key=sort_key)
        return best

    @staticmethod
    def _log_overview(overview: dict[str, Any]) -> None:
        logger.info("=== Phase 3C Validation ROC-AUC matrix ===")
        for row in overview["experiments"]:
            logger.info(
                "%-55s valid_auc=%s top10_ret=%s test_auc=%s",
                row["experiment_id"],
                row["validation_roc_auc"],
                row["validation_top10_mean_future_return"],
                row["test_roc_auc"],
            )
        selection = overview.get("validation_selection")
        if selection:
            logger.info(
                "Validation-selected configuration: %s (valid_auc=%s)",
                selection["experiment_id"],
                selection["validation_roc_auc"],
            )
            conf = overview.get("selected_test_confirmation")
            if conf:
                logger.info(
                    "Test confirmation ROC-AUC=%s",
                    conf["test_metrics"].get("roc_auc"),
                )


def main() -> int:
    """CLI entry point for Phase 3C experiments."""
    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    logger.info("Starting Phase 3C experiments")
    try:
        overview = ExperimentRunner(settings).run()
    except MLError as exc:
        logger.error("Experiment run failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected experiment failure")
        return 1

    logger.info(
        "Phase 3C completed. overview=%s",
        settings.experiments_dir / "phase3c_overview.json",
    )
    _ = overview
    return 0


if __name__ == "__main__":
    sys.exit(main())
