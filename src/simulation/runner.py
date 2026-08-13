"""Walk-forward out-of-sample simulation runner + CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings, get_settings
from src.core.exceptions import TrainingError
from src.ml.ltr_baselines import momentum_scores
from src.ml.ltr_label_schemes import assign_label_scheme, load_label_scheme_config
from src.ml.ltr_labels import assert_group_integrity, build_ranking_groups
from src.ml.ltr_model import fit_lgbm_ranker, predict_rank_scores
from src.ml.ltr_resolution_runner import LabelResolutionRunner
from src.ml.walk_forward import WalkForwardFold, build_expanding_folds, mask_fold_partition
from src.ml.walk_forward_runner import REGION_BENCHMARKS
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.engine import SimulationEngine
from src.simulation.fx import FxConfig, FxConverter, fetch_fx_frames
from src.simulation.metrics import buy_and_hold_benchmark, compute_performance
from src.simulation.ranking import attach_country_ranks
from src.simulation.report import summarize_result, write_json, write_run_artifacts
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

REGIONS = ("Japan", "United States", "Europe")
REGION_ALIASES = {
    "japan": "Japan",
    "jp": "Japan",
    "us": "United States",
    "united_states": "United States",
    "usa": "United States",
    "europe": "Europe",
    "eu": "Europe",
    "global": "Global",
    "all": "Global",
}


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class SimulationRunner:
    """Connect Phase-3H ranking scores to the paper-trading engine (OOS only)."""

    def __init__(
        self,
        settings: Settings,
        universe_path: Path,
        sim_cfg: SimulationConfig,
    ) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.sim_cfg = sim_cfg
        self.lr = LabelResolutionRunner(settings, universe_path)
        self.report_dir = settings.reports_dir / "simulation"
        self.scheme_cfg = load_label_scheme_config()

    def run(
        self,
        *,
        force_refresh: bool = False,
        countries: list[str] | None = None,
        strategies: list[str] | None = None,
        include_momentum: bool = True,
    ) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        print("Simulation Started")
        logger.info("Phase 4 simulation starting")

        panel = self.lr._build_enriched_panel(force_refresh=force_refresh)
        fx = self._build_fx(force_refresh=force_refresh)
        price_panel = self._price_panel(panel)

        scopes = self._resolve_scopes(countries)
        strat_names = strategies or list(self.sim_cfg.strategies.keys()) or ["A"]
        if not strat_names:
            strat_names = ["A"]

        # Build OOS AI rankings once (shared across strategies)
        print("Training model / building walk-forward rankings…")
        ai_rankings, fold_metas = self._build_ai_rankings(panel)
        mom_rankings = self._build_momentum_rankings(panel) if include_momentum else pd.DataFrame()

        country_results: dict[str, Any] = {}
        strategy_comparison: dict[str, Any] = {}
        fold_results: dict[str, Any] = {}

        for scope in scopes:
            print(f"\n=== Scope: {scope} ===")
            scope_countries = None if scope == "Global" else {scope}
            country_results[scope] = {}
            fold_results[scope] = {}

            bench = self._benchmark_metrics(panel, scope, fold_metas=fold_metas, fx=fx)

            for strat in strat_names:
                print(f"Strategy {strat}…")
                cfg = self.sim_cfg.with_strategy(strat) if strat in self.sim_cfg.strategies else self.sim_cfg
                ai_pack = self._simulate_ranked(
                    name=f"AI_strategy_{strat}",
                    rankings=ai_rankings,
                    price_panel=price_panel,
                    fx=fx,
                    cfg=cfg,
                    countries=scope_countries,
                    fold_metas=fold_metas,
                )
                country_results[scope][f"AI_{strat}"] = ai_pack["aggregate"]
                fold_results[scope][f"AI_{strat}"] = ai_pack["folds"]
                write_run_artifacts(
                    self.report_dir / scope / f"AI_{strat}",
                    equity_curve=ai_pack["equity_curve"],
                    trades=ai_pack["trades"],
                    summary=ai_pack["aggregate"],
                )

            if include_momentum and not mom_rankings.empty:
                cfg_a = (
                    self.sim_cfg.with_strategy("A")
                    if "A" in self.sim_cfg.strategies
                    else self.sim_cfg
                )
                print("Momentum baseline…")
                mom_pack = self._simulate_ranked(
                    name="momentum_top10",
                    rankings=mom_rankings,
                    price_panel=price_panel,
                    fx=fx,
                    cfg=cfg_a,
                    countries=scope_countries,
                    fold_metas=fold_metas,
                )
                country_results[scope]["momentum_top10"] = mom_pack["aggregate"]
                fold_results[scope]["momentum_top10"] = mom_pack["folds"]
                write_run_artifacts(
                    self.report_dir / scope / "momentum_top10",
                    equity_curve=mom_pack["equity_curve"],
                    trades=mom_pack["trades"],
                    summary=mom_pack["aggregate"],
                )

            country_results[scope]["benchmark_buy_hold"] = bench
            # Compare using mean fold returns (same OOS windows), not multi-year B&H vs reset capital.
            primary = country_results[scope].get("AI_A") or next(
                (v for k, v in country_results[scope].items() if k.startswith("AI_")),
                None,
            )
            if primary and bench:
                p = primary.get("performance", {})
                country_results[scope]["vs_benchmark"] = {
                    "basis": "mean_fold_total_return_vs_fold_benchmark",
                    "strategy_mean_fold_return": p.get("mean_fold_total_return"),
                    "benchmark_mean_fold_return": bench.get("mean_fold_total_return"),
                    "excess_return": _sub(
                        p.get("mean_fold_total_return"), bench.get("mean_fold_total_return")
                    ),
                    "strategy_mean_fold_max_drawdown": p.get("mean_fold_max_drawdown"),
                    "benchmark_mean_fold_max_drawdown": bench.get("mean_fold_max_drawdown"),
                    "strategy_pooled_note": (
                        "pooled total_return stitches fold curves with capital reset; "
                        "use mean_fold_* for strategy comparison"
                    ),
                }

        # Strategy comparison on Global or first scope
        compare_scope = "Global" if "Global" in country_results else next(iter(country_results))
        strategy_comparison = {
            k: v.get("performance")
            for k, v in country_results[compare_scope].items()
            if isinstance(v, dict) and "performance" in v
        }

        # Top-level convenience copies (Strategy A / Japan if available)
        primary_scope = "Japan" if "Japan" in country_results else compare_scope
        primary_key = "AI_A" if "AI_A" in country_results.get(primary_scope, {}) else None
        primary_run = (
            country_results[primary_scope].get(primary_key) if primary_key else None
        )

        # Write root equity/trades from primary for the requested paths
        if primary_run and primary_key:
            src_eq = self.report_dir / primary_scope / primary_key / "equity_curve.csv"
            src_tr = self.report_dir / primary_scope / primary_key / "trades.csv"
            if src_eq.exists():
                (self.report_dir / "equity_curve.csv").write_text(
                    src_eq.read_text(encoding="utf-8"), encoding="utf-8"
                )
            if src_tr.exists():
                (self.report_dir / "trades.csv").write_text(
                    src_tr.read_text(encoding="utf-8"), encoding="utf-8"
                )

        overview = {
            "created_at_utc": _timestamp_iso(),
            "phase": "4",
            "purpose": "Historical simulation / paper trading on OOS walk-forward rankings",
            "universe": str(self.universe_path),
            "config": self.sim_cfg.to_dict(),
            "ranking_spec": {
                "label_scheme": self.sim_cfg.label_scheme,
                "gain_name": self.sim_cfg.gain_name,
                "feature_set": self.sim_cfg.feature_set,
                "param_preset": self.sim_cfg.param_preset,
                "horizon_days": self.sim_cfg.horizon_days,
            },
            "timing": {
                "signal": "day_T_close",
                "execution": "day_T1_open",
                "stop_take": "trigger_on_T_high_low_fill_T1_conservative",
            },
            "fx_policy": self.sim_cfg.raw.get("fx", {}).get("policy"),
            "scopes": scopes,
            "strategies": strat_names,
            "country_results": country_results,
            "fold_results": fold_results,
            "strategy_comparison": strategy_comparison,
            "known_limitations": [
                "Paper trading only; no brokerage / live orders.",
                "Daily OHLC stop/take fills are conservative approximations.",
                "Research fixed universe may embed survivorship bias.",
                "Global requires convertible FX; unsupported currencies excluded.",
            ],
        }
        write_json(self.report_dir / "summary.json", overview)
        write_json(self.report_dir / "country_results.json", country_results)
        write_json(self.report_dir / "fold_results.json", fold_results)
        write_json(self.report_dir / "strategy_comparison.json", strategy_comparison)
        print("\nCompleted")
        logger.info("Wrote simulation reports -> %s", self.report_dir)
        return overview

    def _resolve_scopes(self, countries: list[str] | None) -> list[str]:
        if not countries:
            return ["Japan", "United States", "Europe", "Global"]
        out: list[str] = []
        for c in countries:
            key = REGION_ALIASES.get(c.strip().lower(), c.strip())
            if key not in out:
                out.append(key)
        return out

    def _build_fx(self, *, force_refresh: bool) -> FxConverter:
        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=365 * self.settings.lookback_years + 5)
        fx_cfg = FxConfig(tickers=self.sim_cfg.fx_tickers, base_currency=self.sim_cfg.base_currency)
        frames = fetch_fx_frames(
            self.lr.wf.provider,
            self.lr.wf.cache,
            fx_cfg,
            start=start,
            end=end,
            interval=self.settings.interval,
            force_refresh=force_refresh,
        )
        conv = FxConverter.from_frames(frames, base_currency=self.sim_cfg.base_currency)
        # Sanity: required legs for Global
        for ccy in ("USD", "EUR", "GBP", "CHF"):
            if not conv.can_convert(ccy):
                logger.warning("FX cannot convert %s — related symbols excluded where needed", ccy)
        return conv

    def _price_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        cols = ["Date", "Symbol", "Country", "Currency", "Open", "High", "Low", "Close"]
        # Country column on panel is JP/US/...; engine uses Region-like Japan/...
        # Prefer Region as Country for portfolio country limits.
        frame = panel.copy()
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
        if "Region" in frame.columns:
            frame["Country"] = frame["Region"]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise TrainingError(f"Price panel missing columns: {missing}")
        return frame.loc[:, cols].drop_duplicates(["Date", "Symbol"])

    def _build_ai_rankings(
        self, panel: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        horizon = self.sim_cfg.horizon_days
        return_col = f"future_return_{horizon}d"
        labeled_pack = assign_label_scheme(
            panel,
            return_col=return_col,
            scheme=self.sim_cfg.label_scheme,  # type: ignore[arg-type]
            gain_name=self.sim_cfg.gain_name,  # type: ignore[arg-type]
            min_group_size=int(self.scheme_cfg.get("min_group_size", 5)),
        )
        labeled = labeled_pack.frame.copy()
        feature_cols = self.lr.feature_sets[self.sim_cfg.feature_set]
        params = dict(self.lr.param_presets[self.sim_cfg.param_preset])
        folds = build_expanding_folds(
            labeled["Date"],
            n_folds=5,
            min_train_days=252 * 3,
            valid_days=252,
            purge_days=horizon,
        )
        parts: list[pd.DataFrame] = []
        metas: list[dict[str, Any]] = []
        for i, fold in enumerate(folds, start=1):
            print(f"Fold {i}/{len(folds)}  Training model…")
            scored = self._fit_score_fold(
                labeled=labeled,
                fold=fold,
                feature_cols=feature_cols,
                params=params,
                label_gain=labeled_pack.label_gain,
            )
            ranked = attach_country_ranks(scored, country_col="Region")
            parts.append(ranked)
            metas.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": str(fold.train_start.date()),
                    "train_end": str(fold.train_end.date()),
                    "valid_start": str(fold.valid_start.date()),
                    "valid_end": str(fold.valid_end.date()),
                    "purge_days": fold.purge_days,
                }
            )
            print(
                f"  Ranking… Simulating window {fold.valid_start.date()} → {fold.valid_end.date()}"
            )
        rankings = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return rankings, metas

    def _fit_score_fold(
        self,
        *,
        labeled: pd.DataFrame,
        fold: WalkForwardFold,
        feature_cols: tuple[str, ...],
        params: dict[str, Any],
        label_gain: tuple[float, ...],
    ) -> pd.DataFrame:
        train_df = mask_fold_partition(labeled, fold, partition="train")
        valid_df = mask_fold_partition(labeled, fold, partition="validation")
        required = list(self.lr.feature_sets["A"]) + ["relevance"]
        # Prefer configured feature set for dropna where possible
        drop_cols = [c for c in feature_cols if c in labeled.columns] + ["relevance"]
        train_df = train_df.dropna(subset=drop_cols).copy()
        valid_df = valid_df.dropna(subset=drop_cols).copy()
        if train_df.empty or valid_df.empty:
            raise TrainingError("Empty train/valid in simulation fold")

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
        return scored

    def _build_momentum_rankings(self, panel: pd.DataFrame) -> pd.DataFrame:
        horizon = self.sim_cfg.horizon_days
        # Momentum needs no fit; still restrict to OOS validation windows of same folds
        labeled_pack = assign_label_scheme(
            panel,
            return_col=f"future_return_{horizon}d",
            scheme="A",
            gain_name="linear",
            min_group_size=int(self.scheme_cfg.get("min_group_size", 5)),
        )
        labeled = labeled_pack.frame.copy()
        folds = build_expanding_folds(
            labeled["Date"],
            n_folds=5,
            min_train_days=252 * 3,
            valid_days=252,
            purge_days=horizon,
        )
        parts: list[pd.DataFrame] = []
        feat = "return_20d"
        for fold in folds:
            valid = mask_fold_partition(labeled, fold, partition="validation")
            valid = valid.dropna(subset=[feat]).copy()
            if valid.empty:
                continue
            valid = valid.copy()
            valid["score"] = momentum_scores(valid, feature_col=feat).to_numpy()
            parts.append(attach_country_ranks(valid, country_col="Region"))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _simulate_ranked(
        self,
        *,
        name: str,
        rankings: pd.DataFrame,
        price_panel: pd.DataFrame,
        fx: FxConverter,
        cfg: SimulationConfig,
        countries: set[str] | None,
        fold_metas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fold_summaries: list[dict[str, Any]] = []
        all_equity = []
        all_trades = []
        # Concatenate fold simulations sequentially with capital reset each fold,
        # then also stitch equity for reporting (fold-wise primary).
        stitched_equity = []
        stitched_trades = []

        for meta in fold_metas:
            v0 = pd.Timestamp(meta["valid_start"])
            v1 = pd.Timestamp(meta["valid_end"])
            # Include one extra price day after end for T+1 opens when possible
            px = price_panel
            rk = rankings
            if countries is not None:
                px = px.loc[px["Country"].isin(countries)]
                rk = rk.loc[rk["Country"].isin(countries)]
            rk_fold = rk.loc[(rk["Date"] >= v0) & (rk["Date"] <= v1)].copy()
            # Prices from valid_start through last available (for T+1 fills)
            px_fold = px.loc[(px["Date"] >= v0)].copy()
            # Cap prices a bit beyond valid_end (10 business days buffer)
            px_fold = px_fold.loc[px_fold["Date"] <= v1 + pd.tseries.offsets.BDay(10)]

            if rk_fold.empty or px_fold.empty:
                fold_summaries.append({**meta, "skipped": True, "reason": "empty"})
                continue

            print(f"  Simulating {v0.date()} → {v1.date()} ({name})")
            engine = SimulationEngine(cfg, fx=fx, countries=countries)
            result = engine.run(prices=px_fold, rankings=rk_fold, start=v0, end=None)
            # Trim equity reporting to validation end when possible
            eq = [e for e in result.equity_curve if pd.Timestamp(e.date) <= v1 + pd.tseries.offsets.BDay(5)]
            trades = result.trades
            costs = (
                float(result.portfolio.total_commission + result.portfolio.total_slippage_impact)
                if result.portfolio
                else 0.0
            )
            summary = summarize_result(
                equity_curve=eq,
                trades=trades,
                initial_capital=cfg.initial_capital,
                transaction_costs=costs,
                extra={"fold": meta, "name": name},
            )
            perf = summary["performance"]
            print(
                f"    Trades: {perf.get('number_of_trades')}  "
                f"Return: {_pct(perf.get('total_return'))}"
            )
            fold_summaries.append(summary)
            stitched_equity.extend(eq)
            stitched_trades.extend(trades)
            all_equity = stitched_equity
            all_trades = stitched_trades

        # Aggregate: concatenate fold returns via equity rebuild is imperfect with capital reset.
        # Report mean fold metrics + pooled trades / stitched curve for inspection.
        perf_list = [f.get("performance", {}) for f in fold_summaries if "performance" in f]
        agg_perf = _mean_perf(perf_list)
        # Pooled trade stats overwrite trade-related fields
        pooled_costs = float(sum(float(p.get("transaction_costs") or 0) for p in perf_list))
        pooled = compute_performance(
            all_equity,
            all_trades,
            initial_capital=cfg.initial_capital,
            transaction_costs_total=pooled_costs,
        )
        # Prefer mean of fold total returns for "strategy return" headline; keep pooled for trades
        aggregate = {
            "name": name,
            "performance": {
                **pooled,
                "mean_fold_total_return": agg_perf.get("total_return"),
                "mean_fold_sharpe": agg_perf.get("sharpe"),
                "mean_fold_max_drawdown": agg_perf.get("max_drawdown"),
                "mean_fold_cagr": agg_perf.get("cagr"),
            },
            "n_folds": len(perf_list),
            "countries": sorted(countries) if countries else "Global",
        }
        return {
            "aggregate": aggregate,
            "folds": fold_summaries,
            "equity_curve": all_equity,
            "trades": all_trades,
        }

    def _benchmark_metrics(
        self,
        panel: pd.DataFrame,
        scope: str,
        *,
        fold_metas: list[dict[str, Any]],
        fx: FxConverter,
    ) -> dict[str, Any] | None:
        if scope == "Global":
            return {
                "note": "No synthetic global benchmark; compare per-country benchmarks only.",
                "total_return": None,
                "max_drawdown": None,
                "mean_fold_total_return": None,
                "mean_fold_max_drawdown": None,
            }
        ticker = self.sim_cfg.benchmarks.get(scope) or REGION_BENCHMARKS.get(scope, (None,))[0]
        if not ticker:
            return None
        frame = self.lr.wf.cache.load(ticker)
        if frame is None or frame.empty:
            try:
                end = date.today() + timedelta(days=1)
                start = end - timedelta(days=365 * self.settings.lookback_years + 5)
                frame = self.lr.wf.cache.get_or_fetch(
                    self.lr.wf.provider,
                    ticker,
                    start=start,
                    end=end,
                    interval=self.settings.interval,
                    force_refresh=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Benchmark fetch failed %s: %s", ticker, exc)
                return None
        s = frame.copy()
        s["Date"] = pd.to_datetime(s["Date"])
        series_local = s.set_index("Date")["Close"].astype(float).sort_index()
        ccy = {"Japan": "JPY", "United States": "USD", "Europe": "EUR"}.get(scope, "JPY")

        def to_base(series: pd.Series) -> pd.Series:
            if ccy == "JPY":
                return series
            vals = []
            for idx, px in series.items():
                try:
                    vals.append(float(px) * fx.rate_to_base(ccy, idx))
                except Exception:  # noqa: BLE001
                    vals.append(np.nan)
            return pd.Series(vals, index=series.index, dtype=float).dropna()

        fold_stats: list[dict[str, Any]] = []
        for meta in fold_metas:
            v0 = pd.Timestamp(meta["valid_start"])
            v1 = pd.Timestamp(meta["valid_end"])
            window = series_local.loc[(series_local.index >= v0) & (series_local.index <= v1)]
            window = to_base(window)
            bh = buy_and_hold_benchmark(window, initial_capital=self.sim_cfg.initial_capital)
            fold_stats.append(
                {
                    "fold_id": meta["fold_id"],
                    "total_return": bh.get("total_return"),
                    "max_drawdown": bh.get("max_drawdown"),
                    "cagr": bh.get("cagr"),
                    "sharpe": bh.get("sharpe"),
                }
            )

        # Full OOS span B&H (informational)
        if fold_metas:
            full0 = pd.Timestamp(fold_metas[0]["valid_start"])
            full1 = pd.Timestamp(fold_metas[-1]["valid_end"])
            full = to_base(
                series_local.loc[(series_local.index >= full0) & (series_local.index <= full1)]
            )
            out = buy_and_hold_benchmark(full, initial_capital=self.sim_cfg.initial_capital)
        else:
            out = buy_and_hold_benchmark(to_base(series_local), initial_capital=self.sim_cfg.initial_capital)

        rets = [f["total_return"] for f in fold_stats if f.get("total_return") is not None]
        mdds = [f["max_drawdown"] for f in fold_stats if f.get("max_drawdown") is not None]
        out["ticker"] = ticker
        out["scope"] = scope
        out["folds"] = fold_stats
        out["mean_fold_total_return"] = float(np.mean(rets)) if rets else None
        out["mean_fold_max_drawdown"] = float(np.mean(mdds)) if mdds else None
        print(
            f"  Benchmark {ticker}: mean-fold Return {_pct(out.get('mean_fold_total_return'))}  "
            f"mean-fold MDD {_pct(out.get('mean_fold_max_drawdown'))}  "
            f"full-OOS {_pct(out.get('total_return'))}"
        )
        return out


def _mean_perf(perfs: list[dict[str, Any]]) -> dict[str, Any]:
    if not perfs:
        return {}
    keys = [
        "total_return",
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "number_of_trades",
        "transaction_costs",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        vals = [p[k] for p in perfs if p.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.2f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 historical simulation / paper trading")
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("config/universe.global100.json"),
        help="Universe JSON path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/simulation.json"),
        help="Simulation config JSON",
    )
    parser.add_argument(
        "--country",
        action="append",
        default=None,
        help="Country/region scope (Japan/US/Europe/Global). Repeatable.",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=None,
        help="Strategy key from config (A/B/C). Repeatable.",
    )
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--skip-momentum", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    sim_cfg = load_simulation_config(args.config)
    runner = SimulationRunner(settings, args.universe, sim_cfg)
    try:
        runner.run(
            force_refresh=args.force_refresh,
            countries=args.country,
            strategies=args.strategy,
            include_momentum=not args.skip_momentum,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Simulation failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
