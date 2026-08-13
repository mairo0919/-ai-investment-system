"""Train and compare Phase 3A/3B models on a shared chronological split."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib

from src.config.settings import Settings, get_settings
from src.core.exceptions import MLError, TrainingError
from src.ml.baseline import majority_class_baseline, previous_return_baseline
from src.ml.dataset import MODEL_FEATURE_COLUMNS, build_dataset
from src.ml.evaluator import class_distribution, evaluate_binary
from src.ml.lightgbm_model import (
    fit_lightgbm,
    gain_feature_importance,
    lightgbm_version,
    top_feature_importance,
)
from src.ml.model_registry import (
    build_logistic_pipeline,
    evaluate_model_partitions,
    extract_test_metric_row,
)
from src.ml.split import SplitBundle, chronological_split
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

LOGISTIC_NAME = "logistic_regression"
LIGHTGBM_NAME = "lightgbm"
COMPARISON_REPORT_FILENAME = "model_comparison.json"


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _date_pair_to_str(pair: tuple[Any, Any]) -> dict[str, str]:
    return {"start": str(pair[0])[:10], "end": str(pair[1])[:10]}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise TrainingError(f"Failed to write JSON {path}: {exc}") from exc


class ModelTrainer:
    """Train LogisticRegression + LightGBM on one shared chronological split."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> dict[str, Any]:
        """Fit all Phase 3 models, evaluate, compare, and persist artifacts."""
        dataset = build_dataset(self.settings.processed_data_dir, self.settings.tickers)
        splits = chronological_split(
            dataset,
            train_ratio=self.settings.train_ratio,
            valid_ratio=self.settings.valid_ratio,
            test_ratio=self.settings.test_ratio,
        )

        dataset_info = self._dataset_info(dataset, splits)
        baseline_metrics = self._baseline_metrics(splits)

        logistic_report = self._train_logistic(dataset_info, baseline_metrics, splits)
        lightgbm_report = self._train_lightgbm(dataset_info, baseline_metrics, splits)

        comparison = self._build_comparison(
            dataset_info=dataset_info,
            baseline_metrics=baseline_metrics,
            logistic_report=logistic_report,
            lightgbm_report=lightgbm_report,
        )
        comparison_path = self.settings.reports_dir / COMPARISON_REPORT_FILENAME
        comparison["artifacts"] = {
            "comparison_report_path": str(comparison_path),
            "logistic_regression": logistic_report.get("artifacts", {}),
            "lightgbm": lightgbm_report.get("artifacts", {}),
        }
        _write_json(comparison_path, comparison)
        logger.info("Saved comparison report -> %s", comparison_path)

        self._log_summary(comparison)
        return comparison

    def _dataset_info(self, dataset: Any, splits: SplitBundle) -> dict[str, Any]:
        return {
            "tickers": list(self.settings.tickers),
            "rows_total": int(len(dataset.frame)),
            "rows_train": int(len(splits.train.frame)),
            "rows_validation": int(len(splits.validation.frame)),
            "rows_test": int(len(splits.test.frame)),
            "feature_count": len(MODEL_FEATURE_COLUMNS),
            "feature_list": list(MODEL_FEATURE_COLUMNS),
            "class_distribution": {
                "all": class_distribution(dataset.y),
                "train": class_distribution(splits.train.y),
                "validation": class_distribution(splits.validation.y),
                "test": class_distribution(splits.test.y),
            },
            "date_ranges": {
                "train": _date_pair_to_str(splits.train_dates),
                "validation": _date_pair_to_str(splits.validation_dates),
                "test": _date_pair_to_str(splits.test_dates),
            },
            "split": {
                "method": "chronological_by_date",
                "shuffle": False,
                "train_ratio": self.settings.train_ratio,
                "valid_ratio": self.settings.valid_ratio,
                "test_ratio": self.settings.test_ratio,
            },
        }

    def _baseline_metrics(self, splits: SplitBundle) -> dict[str, Any]:
        majority_test = majority_class_baseline(splits.train.y, len(splits.test.frame))
        momentum_test = previous_return_baseline(splits.test.frame["return_1d"])
        return {
            "majority_class": {
                "test": evaluate_binary(
                    splits.test.y, majority_test.y_pred, majority_test.y_prob
                )
            },
            "previous_return": {
                "test": evaluate_binary(
                    splits.test.y, momentum_test.y_pred, momentum_test.y_prob
                )
            },
        }

    def _train_logistic(
        self,
        dataset_info: dict[str, Any],
        baseline_metrics: dict[str, Any],
        splits: SplitBundle,
    ) -> dict[str, Any]:
        pipeline = build_logistic_pipeline()
        logger.info(
            "Fitting LogisticRegression on train only (rows=%d, features=%d)",
            len(splits.train.frame),
            len(MODEL_FEATURE_COLUMNS),
        )
        try:
            pipeline.fit(splits.train.X, splits.train.y)
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"LogisticRegression fitting failed: {exc}") from exc

        evaluated = evaluate_model_partitions(
            pipeline, splits.train, splits.validation, splits.test
        )
        trained_at = _timestamp_iso()
        report: dict[str, Any] = {
            "model_name": LOGISTIC_NAME,
            "trained_at_utc": trained_at,
            "feature_list": list(MODEL_FEATURE_COLUMNS),
            "model_config": {
                "pipeline": ["StandardScaler", "LogisticRegression"],
                "logistic_regression": {
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "random_state": 42,
                },
                "split": dataset_info["split"],
            },
            "dataset": dataset_info,
            "baseline_metrics": baseline_metrics,
            "model_metrics": evaluated["model_metrics"],
            "per_symbol_metrics": evaluated["per_symbol_metrics"],
            "validation_test_auc_gap": evaluated["validation_test_auc_gap"],
            "library_versions": {
                "scikit-learn": _package_version("scikit-learn"),
                "numpy": _package_version("numpy"),
                "pandas": _package_version("pandas"),
            },
        }

        model_path = self.settings.models_dir / f"{LOGISTIC_NAME}.joblib"
        meta_path = self.settings.models_dir / f"{LOGISTIC_NAME}.metadata.json"
        report_path = self.settings.reports_dir / f"{LOGISTIC_NAME}_metrics.json"
        self._save_estimator(pipeline, model_path)
        metadata = {
            "model_name": LOGISTIC_NAME,
            "trained_at_utc": trained_at,
            "feature_list": list(MODEL_FEATURE_COLUMNS),
            "date_ranges": dataset_info["date_ranges"],
            "model_config": report["model_config"],
            "class_distribution": dataset_info["class_distribution"],
            "tickers": dataset_info["tickers"],
            "library_versions": report["library_versions"],
            "model_path": str(model_path),
        }
        _write_json(meta_path, metadata)
        report["artifacts"] = {
            "model_path": str(model_path),
            "metadata_path": str(meta_path),
            "report_path": str(report_path),
        }
        _write_json(report_path, report)
        logger.info("Saved LogisticRegression artifacts under %s / %s", model_path, report_path)
        return report

    def _train_lightgbm(
        self,
        dataset_info: dict[str, Any],
        baseline_metrics: dict[str, Any],
        splits: SplitBundle,
    ) -> dict[str, Any]:
        fit_result = fit_lightgbm(
            splits.train.X,
            splits.train.y,
            splits.validation.X,
            splits.validation.y,
        )
        evaluated = evaluate_model_partitions(
            fit_result.model, splits.train, splits.validation, splits.test
        )
        importance = gain_feature_importance(fit_result.model)
        trained_at = _timestamp_iso()
        report: dict[str, Any] = {
            "model_name": LIGHTGBM_NAME,
            "trained_at_utc": trained_at,
            "feature_list": list(MODEL_FEATURE_COLUMNS),
            "model_config": {
                "scaler": None,
                "lightgbm": fit_result.params,
                "early_stopping": {
                    "metric": "auc",
                    "eval_partition": "validation",
                    "test_used_for_early_stopping": False,
                },
                "split": dataset_info["split"],
            },
            "best_iteration": fit_result.best_iteration,
            "dataset": dataset_info,
            "baseline_metrics": baseline_metrics,
            "model_metrics": evaluated["model_metrics"],
            "per_symbol_metrics": evaluated["per_symbol_metrics"],
            "validation_test_auc_gap": evaluated["validation_test_auc_gap"],
            "feature_importance_gain": importance,
            "feature_importance_top10": top_feature_importance(importance, n=10),
            "library_versions": {
                "lightgbm": lightgbm_version(),
                "scikit-learn": _package_version("scikit-learn"),
                "numpy": _package_version("numpy"),
                "pandas": _package_version("pandas"),
            },
            "notes": {
                "feature_importance": (
                    "Gain importance ranks predictive contribution inside the tree "
                    "ensemble; it is not a causal relationship."
                ),
                "validation_test_gap": (
                    "A large validation_auc - test_auc gap may indicate period "
                    "dependence, market-regime shift, or overfitting."
                ),
            },
        }

        model_path = self.settings.models_dir / f"{LIGHTGBM_NAME}.joblib"
        meta_path = self.settings.models_dir / f"{LIGHTGBM_NAME}.metadata.json"
        report_path = self.settings.reports_dir / f"{LIGHTGBM_NAME}_metrics.json"
        self._save_estimator(fit_result.model, model_path)
        metadata = {
            "model_name": LIGHTGBM_NAME,
            "created_at": trained_at,
            "trained_at_utc": trained_at,
            "model_parameters": fit_result.params,
            "best_iteration": fit_result.best_iteration,
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "date_ranges": dataset_info["date_ranges"],
            "class_distribution": dataset_info["class_distribution"],
            "library_versions": report["library_versions"],
            "model_path": str(model_path),
        }
        _write_json(meta_path, metadata)
        report["artifacts"] = {
            "model_path": str(model_path),
            "metadata_path": str(meta_path),
            "report_path": str(report_path),
        }
        _write_json(report_path, report)
        logger.info("Saved LightGBM artifacts under %s / %s", model_path, report_path)
        return report

    def _build_comparison(
        self,
        *,
        dataset_info: dict[str, Any],
        baseline_metrics: dict[str, Any],
        logistic_report: dict[str, Any],
        lightgbm_report: dict[str, Any],
    ) -> dict[str, Any]:
        test_comparison = {
            "majority_class": extract_test_metric_row(
                baseline_metrics["majority_class"]["test"]
            ),
            "previous_return": extract_test_metric_row(
                baseline_metrics["previous_return"]["test"]
            ),
            "logistic_regression": extract_test_metric_row(
                logistic_report["model_metrics"]["test"]
            ),
            "lightgbm": extract_test_metric_row(lightgbm_report["model_metrics"]["test"]),
        }
        return {
            "trained_at_utc": _timestamp_iso(),
            "dataset": dataset_info,
            "baseline_metrics": baseline_metrics,
            "models": {
                LOGISTIC_NAME: {
                    "model_metrics": logistic_report["model_metrics"],
                    "per_symbol_metrics": logistic_report["per_symbol_metrics"],
                    "validation_test_auc_gap": logistic_report["validation_test_auc_gap"],
                    "artifacts": logistic_report["artifacts"],
                },
                LIGHTGBM_NAME: {
                    "model_metrics": lightgbm_report["model_metrics"],
                    "per_symbol_metrics": lightgbm_report["per_symbol_metrics"],
                    "validation_test_auc_gap": lightgbm_report["validation_test_auc_gap"],
                    "best_iteration": lightgbm_report["best_iteration"],
                    "feature_importance_top10": lightgbm_report["feature_importance_top10"],
                    "artifacts": lightgbm_report["artifacts"],
                },
            },
            "comparison": {"test": test_comparison},
            # Backward-compatible aliases used by older Phase 3A tests/tools.
            "model_metrics": lightgbm_report["model_metrics"],
            "per_symbol_metrics": lightgbm_report["per_symbol_metrics"],
        }

    def _save_estimator(self, estimator: Any, path: Path) -> None:
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(estimator, path)
        except OSError as exc:
            raise TrainingError(f"Failed to save model {path}: {exc}") from exc
        logger.info("Saved model -> %s", path)

    @staticmethod
    def _log_summary(comparison: dict[str, Any]) -> None:
        ds = comparison["dataset"]
        logger.info(
            "Date ranges | train=%s validation=%s test=%s",
            ds["date_ranges"]["train"],
            ds["date_ranges"]["validation"],
            ds["date_ranges"]["test"],
        )
        table = comparison["comparison"]["test"]
        for name, metrics in table.items():
            logger.info(
                "Test %-20s | acc=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%s",
                name,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["roc_auc"],
            )
        lgbm = comparison["models"][LIGHTGBM_NAME]
        logger.info("LightGBM best_iteration=%s", lgbm.get("best_iteration"))
        logger.info(
            "LightGBM validation_auc - test_auc = %s",
            lgbm.get("validation_test_auc_gap"),
        )


def load_model(path: Path) -> Any:
    """Load a persisted estimator (sklearn Pipeline or LightGBM classifier)."""
    try:
        model = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Failed to load model from {path}: {exc}") from exc
    return model


def main() -> int:
    """CLI entry point: train LogisticRegression + LightGBM and compare."""
    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)

    logger.info(
        "Starting multi-model training: tickers=%s processed_dir=%s",
        ",".join(settings.tickers),
        settings.processed_data_dir,
    )

    try:
        trainer = ModelTrainer(settings)
        report = trainer.run()
    except MLError as exc:
        logger.error("Training job failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected failure in training job")
        return 1

    logger.info("Training + comparison completed successfully")
    logger.info("Artifacts: %s", report.get("artifacts"))
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Re-export for Phase 3A imports/tests.
__all__ = [
    "ModelTrainer",
    "build_logistic_pipeline",
    "load_model",
    "main",
]
