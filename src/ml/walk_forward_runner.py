"""Walk-forward + cross-sectional ranking experiment runner (no trading)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings, get_settings
from src.core.exceptions import MLError, TrainingError
from src.data.cache import OhlcvCache
from src.data.metadata import InstrumentMeta, enrich_metadata_from_yfinance
from src.data.service import MarketDataService as EquityDataService
from src.data.service import create_provider
from src.data.universe import load_universe, region_of
from src.features.indicators import FEATURE_COLUMNS, add_features
from src.features.market_features import (
    MARKET_FEATURE_SUFFIXES,
    build_market_features,
    join_market_features,
)
from src.ml.cross_section import cross_sectional_metrics_by_date, summarize_daily_metrics
from src.ml.dataset import add_future_returns, materialize_target
from src.ml.evaluator import evaluate_binary
from src.ml.lightgbm_model import fit_lightgbm
from src.ml.model_registry import build_logistic_pipeline
from src.ml.targets import TARGET_UP_1D, TARGET_UP_5D, TARGET_UP_10D, TargetConfig
from src.ml.walk_forward import WalkForwardFold, build_expanding_folds, mask_fold_partition
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

MIN_HISTORY_YEARS = 5
MIN_HISTORY_ROWS = 252 * MIN_HISTORY_YEARS
GENERIC_MARKET_PREFIX = "mkt"

REGION_BENCHMARKS: dict[str, tuple[str, str]] = {
    # region: (yahoo_ticker, feature_prefix_before_rename)
    "Japan": ("^N225", "nikkei"),
    "United States": ("^GSPC", "spx"),
    "Europe": ("^STOXX50E", "stoxx"),
}


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stats(values: list[float]) -> dict[str, float | None]:
    arr = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if arr.empty:
        return {"mean": None, "median": None, "std": None, "positive_ratio": None}
    return {
        "mean": float(arr.mean()),
        "median": float(arr.median()),
        "std": float(arr.std(ddof=0)),
        "positive_ratio": float((arr > 0).mean()),
    }


class WalkForwardRunner:
    """Run limited walk-forward ranking experiments on a research universe."""

    def __init__(self, settings: Settings, universe_path: Path) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.universe = load_universe(universe_path)
        self.provider = create_provider(settings.data_provider)
        self.cache = OhlcvCache(settings.raw_data_dir)
        self.report_dir = settings.reports_dir / "walk_forward"

    def run(self, *, force_refresh: bool = False, enrich_metadata: bool = False) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        composition = self.universe.composition_report()

        fetch_summary = self._fetch_equities(force_refresh=force_refresh)
        usable_meta, quality = self._select_usable_symbols(
            fetch_summary, enrich_metadata=enrich_metadata
        )
        market_frames = self._fetch_benchmarks(force_refresh=force_refresh)
        panel = self._build_panel(usable_meta, market_frames)
        if panel.empty:
            raise TrainingError("Panel is empty after quality filters")

        targets = (TARGET_UP_5D, TARGET_UP_10D, TARGET_UP_1D)
        models = ("logistic_regression", "lightgbm")
        experiment_summaries: list[dict[str, Any]] = []

        for target in targets:
            labeled = materialize_target(panel, target)
            feature_cols = tuple(
                list(FEATURE_COLUMNS)
                + [f"{GENERIC_MARKET_PREFIX}_{s}" for s in MARKET_FEATURE_SUFFIXES]
            )
            labeled = labeled.dropna(subset=list(feature_cols) + [target.target_column]).copy()
            folds = build_expanding_folds(
                labeled["Date"],
                n_folds=5,
                min_train_days=252 * 3,
                valid_days=252,
                purge_days=target.purge_days,
            )
            for model_name in models:
                # 1d is reference-only: LR only to keep matrix small.
                if target.name == "target_up_1d" and model_name != "logistic_regression":
                    continue
                summary = self._run_target_model(
                    labeled=labeled,
                    target=target,
                    feature_cols=feature_cols,
                    folds=folds,
                    model_name=model_name,
                )
                experiment_summaries.append(summary)

        overview = {
            "created_at_utc": _timestamp_iso(),
            "universe": composition,
            "fetch_summary": fetch_summary,
            "quality_filter": quality,
            "panel": {
                "rows": int(len(panel)),
                "symbols": int(panel["Symbol"].nunique()),
                "date_start": str(pd.to_datetime(panel["Date"]).min().date()),
                "date_end": str(pd.to_datetime(panel["Date"]).max().date()),
                "currency_note": (
                    "Global pooled absolute returns mix currencies without FX conversion. "
                    "Country rankings are preferred for interpretation."
                ),
            },
            "known_limitations": list(self.universe.known_limitations)
            + [
                "Research fixed universe; survivorship bias may exist.",
                "Symbols with <5y history are excluded (not partially mixed into folds).",
            ],
            "experiments": experiment_summaries,
        }
        out = self.report_dir / "walk_forward_overview.json"
        _write_json(out, overview)
        logger.info("Wrote walk-forward overview -> %s", out)
        return overview

    def _fetch_equities(self, *, force_refresh: bool) -> dict[str, Any]:
        service = EquityDataService(self.settings, provider=self.provider, cache=self.cache)
        success: list[str] = []
        failed: list[dict[str, str]] = []
        for symbol in self.universe.symbols:
            try:
                service.run(
                    force_refresh=force_refresh,
                    symbols=(symbol,),
                )
                success.append(symbol)
            except Exception as exc:  # noqa: BLE001
                logger.error("Fetch failed for %s: %s", symbol, exc)
                failed.append({"symbol": symbol, "error": str(exc)})
        summary = {
            "requested": len(self.universe.symbols),
            "success": len(success),
            "failed": len(failed),
            "success_symbols": success,
            "failed_symbols": failed,
        }
        logger.info(
            "Equity fetch summary: success=%d failed=%d",
            summary["success"],
            summary["failed"],
        )
        return summary

    def _select_usable_symbols(
        self,
        fetch_summary: dict[str, Any],
        *,
        enrich_metadata: bool,
    ) -> tuple[dict[str, InstrumentMeta], dict[str, Any]]:
        cutoff = pd.Timestamp(date.today() - timedelta(days=365 * MIN_HISTORY_YEARS + 5))
        meta_by_symbol = {m.symbol: m for m in self.universe.instruments}
        usable: dict[str, InstrumentMeta] = {}
        excluded: list[dict[str, Any]] = []

        for symbol in fetch_summary["success_symbols"]:
            frame = self.cache.load(symbol)
            if frame is None or frame.empty:
                excluded.append({"symbol": symbol, "reason": "empty_cache"})
                continue
            first = pd.Timestamp(frame["Date"].min())
            rows = len(frame)
            if first > cutoff or rows < MIN_HISTORY_ROWS:
                excluded.append(
                    {
                        "symbol": symbol,
                        "reason": "insufficient_history",
                        "first_date": str(first.date()),
                        "rows": rows,
                        "policy": "exclude_from_all_folds",
                    }
                )
                continue
            meta = meta_by_symbol[symbol]
            if enrich_metadata:
                meta = enrich_metadata_from_yfinance(meta)
            usable[symbol] = meta

        quality = {
            "min_history_years": MIN_HISTORY_YEARS,
            "min_history_rows": MIN_HISTORY_ROWS,
            "usable": len(usable),
            "excluded": excluded,
            "policy": (
                "Symbols below minimum history are excluded entirely "
                "(not allowed in selected folds only)."
            ),
        }
        logger.info(
            "Usable symbols=%d excluded=%d",
            quality["usable"],
            len(excluded),
        )
        if quality["usable"] < 50:
            raise TrainingError(
                f"Usable symbols after history filter too few: {quality['usable']}"
            )
        return usable, quality

    def _fetch_benchmarks(self, *, force_refresh: bool) -> dict[str, pd.DataFrame]:
        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=365 * self.settings.lookback_years + 5)
        out: dict[str, pd.DataFrame] = {}
        for region, (ticker, prefix) in REGION_BENCHMARKS.items():
            try:
                raw = self.cache.get_or_fetch(
                    self.provider,
                    ticker,
                    start=start,
                    end=end,
                    interval=self.settings.interval,
                    force_refresh=force_refresh,
                )
                # Also persist under market dir for inspectability.
                market_dir = self.settings.raw_market_dir
                market_dir.mkdir(parents=True, exist_ok=True)
                features = build_market_features(raw, prefix=prefix)
                processed = self.settings.processed_market_dir
                processed.mkdir(parents=True, exist_ok=True)
                features.to_csv(processed / f"{prefix}_features.csv", index=False)
                out[region] = features
                logger.info("Benchmark ready region=%s ticker=%s rows=%d", region, ticker, len(features))
            except Exception as exc:  # noqa: BLE001
                raise TrainingError(
                    f"Required benchmark failed region={region} ticker={ticker}: {exc}"
                ) from exc
        return out

    def _build_panel(
        self,
        usable_meta: dict[str, InstrumentMeta],
        market_frames: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for symbol, meta in usable_meta.items():
            raw = self.cache.load(symbol)
            if raw is None:
                continue
            featured = add_features(raw)
            featured = featured.dropna(subset=list(FEATURE_COLUMNS)).copy()
            featured["Symbol"] = symbol
            featured["Country"] = meta.country
            featured["Market"] = meta.market
            featured["Currency"] = meta.currency
            featured["Sector"] = meta.sector
            featured["Industry"] = meta.industry
            featured["Region"] = region_of(meta.country)
            region = featured["Region"].iloc[0]
            if region not in market_frames:
                logger.warning("No benchmark for region=%s symbol=%s; skip", region, symbol)
                continue
            mkt = market_frames[region]
            # Rename region-specific prefix to generic mkt_*.
            prefix = REGION_BENCHMARKS[region][1]
            renamed = mkt.rename(
                columns={
                    f"{prefix}_{suffix}": f"{GENERIC_MARKET_PREFIX}_{suffix}"
                    for suffix in MARKET_FEATURE_SUFFIXES
                }
            )
            joined = join_market_features(featured, renamed)
            parts.append(joined)

        if not parts:
            return pd.DataFrame()
        panel = pd.concat(parts, ignore_index=True, sort=False)
        panel = add_future_returns(panel, horizons=(1, 5, 10))
        logger.info(
            "Built panel rows=%d symbols=%d date=%s..%s",
            len(panel),
            panel["Symbol"].nunique(),
            pd.to_datetime(panel["Date"]).min().date(),
            pd.to_datetime(panel["Date"]).max().date(),
        )
        return panel

    def _run_target_model(
        self,
        *,
        labeled: pd.DataFrame,
        target: TargetConfig,
        feature_cols: tuple[str, ...],
        folds: list[WalkForwardFold],
        model_name: str,
    ) -> dict[str, Any]:
        experiment_id = f"{target.name}__{model_name}"
        logger.info("Running walk-forward experiment %s", experiment_id)
        fold_reports: list[dict[str, Any]] = []

        for fold in folds:
            train_df = mask_fold_partition(labeled, fold, partition="train")
            valid_df = mask_fold_partition(labeled, fold, partition="validation")
            if train_df.empty or valid_df.empty:
                raise TrainingError(f"Empty partition in {experiment_id} fold={fold.fold_id}")

            x_train = train_df.loc[:, list(feature_cols)]
            y_train = train_df[target.target_column].astype(int)
            x_valid = valid_df.loc[:, list(feature_cols)]
            y_valid = valid_df[target.target_column].astype(int)

            if model_name == "logistic_regression":
                model = build_logistic_pipeline()
                model.fit(x_train, y_train)
            elif model_name == "lightgbm":
                # Early stopping uses validation; acceptable for WF diagnostics,
                # but we report validation metrics only (no nested test tuning).
                fit = fit_lightgbm(x_train, y_train, x_valid, y_valid)
                model = fit.model
            else:
                raise TrainingError(f"Unknown model {model_name}")

            proba = model.predict_proba(x_valid)[:, 1]
            pred = (proba >= 0.5).astype(int)
            cls_metrics = evaluate_binary(y_valid, pred, proba)

            scored = valid_df.copy()
            scored["score"] = proba
            scored["future_return"] = scored[target.future_return_column]

            daily = cross_sectional_metrics_by_date(
                scored,
                score_col="score",
                return_col="future_return",
                top_fractions=(0.05, 0.10, 0.20),
            )
            ranking_summary = summarize_daily_metrics(daily)

            country_metrics = {}
            for region, group in scored.groupby("Region"):
                daily_c = cross_sectional_metrics_by_date(
                    group,
                    score_col="score",
                    return_col="future_return",
                    top_fractions=(0.10, 0.20),
                    min_names=3,
                )
                country_metrics[str(region)] = summarize_daily_metrics(daily_c)

            sector_metrics = {}
            for sector, group in scored.groupby(scored["Sector"].fillna("null")):
                if len(group) < 30:
                    continue
                try:
                    ic = float(
                        group["score"].corr(group["future_return"], method="spearman")
                    )
                except Exception:  # noqa: BLE001
                    ic = float("nan")
                sector_metrics[str(sector)] = {
                    "sample_count": int(len(group)),
                    "mean_future_return": float(group["future_return"].mean()),
                    "mean_score": float(group["score"].mean()),
                    "ic": None if np.isnan(ic) else ic,
                }

            fold_report = {
                "fold_id": fold.fold_id,
                "train_period": {
                    "start": str(fold.train_start.date()),
                    "end": str(fold.train_end.date()),
                    "n_dates": len(fold.train_dates),
                    "rows": int(len(train_df)),
                    "symbols": int(train_df["Symbol"].nunique()),
                },
                "validation_period": {
                    "start": str(fold.valid_start.date()),
                    "end": str(fold.valid_end.date()),
                    "n_dates": len(fold.valid_dates),
                    "rows": int(len(valid_df)),
                    "symbols": int(valid_df["Symbol"].nunique()),
                },
                "purge_days": fold.purge_days,
                "classification": cls_metrics,
                "ranking": ranking_summary,
                "country_metrics": country_metrics,
                "sector_metrics": sector_metrics,
            }
            fold_reports.append(fold_report)

        stability = self._aggregate_stability(fold_reports)
        summary = {
            "experiment_id": experiment_id,
            "target": {
                "name": target.name,
                "horizon_days": target.horizon_days,
                "future_return_column": target.future_return_column,
                "target_column": target.target_column,
                "purge_days": target.purge_days,
            },
            "model": model_name,
            "feature_count": len(feature_cols),
            "folds": fold_reports,
            "stability_across_folds": stability,
        }
        path = self.report_dir / f"{experiment_id}.json"
        _write_json(path, summary)
        summary["report_path"] = str(path)
        return summary

    @staticmethod
    def _aggregate_stability(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
        roc = [f["classification"]["roc_auc"] for f in fold_reports if f["classification"].get("roc_auc") is not None]
        top10 = []
        mean_ic = []
        for fold in fold_reports:
            ranking = fold.get("ranking", {})
            ic = ranking.get("ic", {})
            if ic.get("mean") is not None:
                mean_ic.append(float(ic["mean"]))
            t10 = ranking.get("top_10pct_excess_return", {})
            if t10.get("mean") is not None:
                top10.append(float(t10["mean"]))
        return {
            "roc_auc": _stats([float(x) for x in roc]),
            "top10_excess_return": _stats(top10),
            "mean_ic": _stats(mean_ic),
            "n_folds": len(fold_reports),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward cross-sectional ranking study")
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("config/universe.global100.json"),
        help="Universe JSON path",
    )
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--enrich-metadata",
        action="store_true",
        help="Best-effort yfinance metadata enrichment (slower)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    logger.info("Starting walk-forward ranking study universe=%s", args.universe)

    try:
        overview = WalkForwardRunner(settings, args.universe).run(
            force_refresh=args.force_refresh,
            enrich_metadata=args.enrich_metadata,
        )
    except MLError as exc:
        logger.error("Walk-forward study failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected walk-forward failure")
        return 1

    logger.info(
        "Walk-forward completed: usable_symbols=%s experiments=%d",
        overview.get("quality_filter", {}).get("usable"),
        len(overview.get("experiments", [])),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
