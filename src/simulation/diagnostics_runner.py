"""Phase 4B: strategy diagnostics & turnover reduction study (fixed AI rankings)."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings, get_settings
from src.core.exceptions import TrainingError
from src.ml.ltr_label_schemes import assign_label_scheme, load_label_scheme_config
from src.ml.ltr_labels import assert_group_integrity, build_ranking_groups
from src.ml.ltr_model import fit_lgbm_ranker, predict_rank_scores
from src.ml.walk_forward import WalkForwardFold, build_expanding_folds, mask_fold_partition
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.diagnostics import (
    cost_decomposition,
    exit_reason_analysis,
    trade_quality_by_entry_score,
    turnover_analysis,
)
from src.simulation.engine import SimulationEngine
from src.simulation.metrics import compute_performance
from src.simulation.quality import calibrate_train_thresholds
from src.simulation.ranking import attach_country_ranks
from src.simulation.report import summarize_result, write_json
from src.simulation.runner import SimulationRunner, _mean_perf, _pct, _sub
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

CORE_STRATEGIES = ("BASE", "LONGER", "LONG20", "WEEKLY", "CONCENTRATED", "QUALITY_FILTER")


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class DiagnosticsRunner:
    """Compare trading rules with a fixed walk-forward AI ranking."""

    def __init__(
        self,
        settings: Settings,
        universe_path: Path,
        sim_cfg: SimulationConfig,
    ) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.sim_cfg = sim_cfg
        self.base = SimulationRunner(settings, universe_path, sim_cfg)
        self.report_dir = settings.reports_dir / "simulation_diagnostics"
        self.scheme_cfg = load_label_scheme_config()

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        print("Phase 4B Diagnostics Started")

        panel = self.base.lr._build_enriched_panel(force_refresh=force_refresh)
        fx = self.base._build_fx(force_refresh=force_refresh)
        price_panel = self.base._price_panel(panel)

        print("Building fixed AI rankings (walk-forward)…")
        ai_rankings, fold_metas = self._build_ai_rankings_with_train_thresholds(panel)
        mom_rankings = self.base._build_momentum_rankings(panel)

        scopes = ["Japan", "United States", "Europe", "Global"]
        strategy_names = [s for s in CORE_STRATEGIES if s in self.sim_cfg.strategies] or list(
            self.sim_cfg.strategies.keys()
        )

        country_comparison: dict[str, Any] = {}
        fold_results: dict[str, Any] = {}
        strategy_comparison: dict[str, Any] = {}
        cost_analysis: dict[str, Any] = {}
        turnover_all: dict[str, Any] = {}
        holding_period_analysis: dict[str, Any] = {}
        trade_quality_all: dict[str, Any] = {}
        exit_reason_all: dict[str, Any] = {}

        for scope in scopes:
            print(f"\n=== Scope: {scope} ===")
            countries = None if scope == "Global" else {scope}
            bench = self.base._benchmark_metrics(panel, scope, fold_metas=fold_metas, fx=fx)
            country_comparison[scope] = {"benchmark": bench, "strategies": {}}
            fold_results[scope] = {}
            cost_analysis[scope] = {}
            turnover_all[scope] = {}
            trade_quality_all[scope] = {}
            exit_reason_all[scope] = {}
            holding_period_analysis[scope] = {}

            for sname in strategy_names:
                print(f"Strategy {sname}…")
                cfg = self.sim_cfg.with_strategy(sname)
                pack_by_cost = {}
                for cost_name in ("gross", "low", "net"):
                    cfg_c = cfg.with_cost_preset(cost_name)
                    pack = self._simulate_ranked(
                        name=f"{sname}_{cost_name}",
                        rankings=ai_rankings,
                        price_panel=price_panel,
                        fx=fx,
                        cfg=cfg_c,
                        countries=countries,
                        fold_metas=fold_metas,
                    )
                    pack_by_cost[cost_name] = pack

                net = pack_by_cost["net"]
                gross = pack_by_cost["gross"]
                low = pack_by_cost["low"]
                net_perf = net["aggregate"]["performance"]
                gross_perf = gross["aggregate"]["performance"]
                low_perf = low["aggregate"]["performance"]

                # Prefer commission/slippage from net run (pooled)
                commission, slippage = self._costs_from_pack(net)
                cost = cost_decomposition(
                    gross_perf=gross_perf,
                    net_perf=net_perf,
                    low_perf=low_perf,
                    initial_capital=cfg.initial_capital,
                    commission_total=commission,
                    slippage_total=slippage,
                )
                turn = turnover_analysis(
                    net["trades"],
                    net["equity_curve"],
                    initial_capital=cfg.initial_capital,
                    rebalance_stats=net.get("rebalance_stats"),
                )
                tq = trade_quality_by_entry_score(net["trades"])
                er = exit_reason_analysis(net["trades"])

                vs_bench = {
                    "strategy_mean_fold_return": net_perf.get("mean_fold_total_return"),
                    "benchmark_mean_fold_return": (bench or {}).get("mean_fold_total_return"),
                    "excess_return": _sub(
                        net_perf.get("mean_fold_total_return"),
                        (bench or {}).get("mean_fold_total_return"),
                    ),
                    "gross_excess_return": _sub(
                        gross_perf.get("mean_fold_total_return"),
                        (bench or {}).get("mean_fold_total_return"),
                    ),
                }

                row = {
                    "strategy": sname,
                    "config": {
                        "holding_period_days": cfg.holding_period_days,
                        "rebalance_frequency": cfg.rebalance_frequency,
                        "max_positions": cfg.max_positions,
                        "ranking_exit_enabled": cfg.ranking_exit_enabled,
                        "block_low_resolution_entries": cfg.block_low_resolution_entries,
                        "use_train_dispersion_filter": cfg.use_train_dispersion_filter,
                    },
                    "gross": {
                        "mean_fold_return": gross_perf.get("mean_fold_total_return"),
                        "mean_fold_sharpe": gross_perf.get("mean_fold_sharpe"),
                        "mean_fold_max_drawdown": gross_perf.get("mean_fold_max_drawdown"),
                        "trades": gross_perf.get("number_of_trades"),
                    },
                    "net": {
                        "mean_fold_return": net_perf.get("mean_fold_total_return"),
                        "mean_fold_sharpe": net_perf.get("mean_fold_sharpe"),
                        "mean_fold_max_drawdown": net_perf.get("mean_fold_max_drawdown"),
                        "mean_fold_cagr": net_perf.get("mean_fold_cagr"),
                        "win_rate": net_perf.get("win_rate"),
                        "trades": net_perf.get("number_of_trades"),
                    },
                    "cost": cost,
                    "turnover": turn,
                    "vs_benchmark": vs_bench,
                }
                country_comparison[scope]["strategies"][sname] = row
                fold_results[scope][sname] = {
                    "net_folds": net["folds"],
                    "gross_mean_fold_return": gross_perf.get("mean_fold_total_return"),
                }
                cost_analysis[scope][sname] = cost
                turnover_all[scope][sname] = turn
                trade_quality_all[scope][sname] = tq
                exit_reason_all[scope][sname] = er

                print(
                    f"  Gross {_pct(gross_perf.get('mean_fold_total_return'))} | "
                    f"Net {_pct(net_perf.get('mean_fold_total_return'))} | "
                    f"Trades {net_perf.get('number_of_trades')} | "
                    f"Cost drag {_pct(cost.get('cost_drag'))}"
                )

            # Holding period slice
            for sname in ("BASE", "LONGER", "LONG20"):
                if sname in country_comparison[scope]["strategies"]:
                    holding_period_analysis[scope][sname] = country_comparison[scope][
                        "strategies"
                    ][sname]

            # Ranking-exit diagnostic on LONGER (net only)
            if "LONGER" in self.sim_cfg.strategies:
                print("Ranking Exit ON (LONGER overlay)…")
                cfg_re = self.sim_cfg.with_strategy("LONGER").with_overrides(
                    ranking_exit_enabled=True, ranking_exit_percentile=0.30
                ).with_cost_preset("net")
                pack_re = self._simulate_ranked(
                    name="LONGER_ranking_exit",
                    rankings=ai_rankings,
                    price_panel=price_panel,
                    fx=fx,
                    cfg=cfg_re,
                    countries=countries,
                    fold_metas=fold_metas,
                )
                country_comparison[scope]["strategies"]["LONGER_RANKING_EXIT"] = {
                    "strategy": "LONGER_RANKING_EXIT",
                    "net": {
                        "mean_fold_return": pack_re["aggregate"]["performance"].get(
                            "mean_fold_total_return"
                        ),
                        "trades": pack_re["aggregate"]["performance"].get("number_of_trades"),
                        "mean_fold_sharpe": pack_re["aggregate"]["performance"].get(
                            "mean_fold_sharpe"
                        ),
                    },
                    "turnover": turnover_analysis(
                        pack_re["trades"],
                        pack_re["equity_curve"],
                        initial_capital=cfg_re.initial_capital,
                        rebalance_stats=pack_re.get("rebalance_stats"),
                    ),
                    "cost": {
                        "commission": self._costs_from_pack(pack_re)[0],
                        "slippage": self._costs_from_pack(pack_re)[1],
                        "total_transaction_cost": sum(self._costs_from_pack(pack_re)),
                    },
                    "exit_reasons": exit_reason_analysis(pack_re["trades"]),
                }

            # Momentum baseline under WEEKLY-like rules if available
            if not mom_rankings.empty and "WEEKLY" in self.sim_cfg.strategies:
                print("Momentum baseline (WEEKLY rules)…")
                cfg_m = self.sim_cfg.with_strategy("WEEKLY").with_cost_preset("net")
                mom = self._simulate_ranked(
                    name="momentum_WEEKLY",
                    rankings=mom_rankings,
                    price_panel=price_panel,
                    fx=fx,
                    cfg=cfg_m,
                    countries=countries,
                    fold_metas=fold_metas,
                )
                country_comparison[scope]["momentum_weekly"] = {
                    "mean_fold_return": mom["aggregate"]["performance"].get(
                        "mean_fold_total_return"
                    ),
                    "trades": mom["aggregate"]["performance"].get("number_of_trades"),
                    "mean_fold_sharpe": mom["aggregate"]["performance"].get("mean_fold_sharpe"),
                }

        # Cross-country strategy comparison table (net)
        for sname in strategy_names:
            strategy_comparison[sname] = {
                scope: country_comparison[scope]["strategies"].get(sname, {}).get("net")
                for scope in scopes
                if sname in country_comparison[scope]["strategies"]
            }
            strategy_comparison[sname]["gross_by_country"] = {
                scope: country_comparison[scope]["strategies"].get(sname, {}).get("gross")
                for scope in scopes
                if sname in country_comparison[scope]["strategies"]
            }

        overview = {
            "created_at_utc": _timestamp_iso(),
            "phase": "4B",
            "purpose": "Diagnose cost/turnover/rule impact with fixed AI rankings",
            "universe": str(self.universe_path),
            "ranking_spec": {
                "label_scheme": self.sim_cfg.label_scheme,
                "gain_name": self.sim_cfg.gain_name,
                "feature_set": self.sim_cfg.feature_set,
                "param_preset": self.sim_cfg.param_preset,
                "horizon_days": self.sim_cfg.horizon_days,
            },
            "strategies": strategy_names,
            "country_comparison": country_comparison,
            "findings_hints": self._findings(country_comparison),
            "known_limitations": [
                "Gross/Net re-simulate with same rankings/rules; sizing may differ slightly with cash.",
                "Train dispersion thresholds use train-period model scores (no validation peek).",
                "No Optuna / no new features / no brokerage.",
            ],
        }

        write_json(self.report_dir / "strategy_comparison.json", strategy_comparison)
        write_json(self.report_dir / "cost_analysis.json", cost_analysis)
        write_json(self.report_dir / "turnover_analysis.json", turnover_all)
        write_json(self.report_dir / "holding_period_analysis.json", holding_period_analysis)
        write_json(self.report_dir / "trade_quality.json", trade_quality_all)
        write_json(self.report_dir / "exit_reason_analysis.json", exit_reason_all)
        write_json(self.report_dir / "country_comparison.json", country_comparison)
        write_json(self.report_dir / "fold_results.json", fold_results)
        write_json(self.report_dir / "summary.json", overview)
        print("\nDiagnostics Completed")
        return overview

    def _findings(self, country_comparison: dict[str, Any]) -> dict[str, Any]:
        best_net = None
        best_gross = None
        for scope, block in country_comparison.items():
            if scope == "Global":
                continue
            for sname, row in block.get("strategies", {}).items():
                if "net" not in row:
                    continue
                nr = row["net"].get("mean_fold_return")
                gr = (row.get("gross") or {}).get("mean_fold_return")
                if nr is not None and (best_net is None or nr > best_net[2]):
                    best_net = (scope, sname, nr)
                if gr is not None and (best_gross is None or gr > best_gross[2]):
                    best_gross = (scope, sname, gr)
        return {
            "best_net": (
                {"scope": best_net[0], "strategy": best_net[1], "mean_fold_return": best_net[2]}
                if best_net
                else None
            ),
            "best_gross": (
                {
                    "scope": best_gross[0],
                    "strategy": best_gross[1],
                    "mean_fold_return": best_gross[2],
                }
                if best_gross
                else None
            ),
        }

    def _costs_from_pack(self, pack: dict[str, Any]) -> tuple[float, float]:
        # Reconstruct from fold portfolios is heavy; use performance transaction_costs split approx
        # Prefer explicit aggregates if present
        comm = float(pack.get("total_commission") or 0.0)
        slip = float(pack.get("total_slippage") or 0.0)
        if comm or slip:
            return comm, slip
        # Fallback: all transaction_costs as commission proxy
        tc = float(pack["aggregate"]["performance"].get("transaction_costs") or 0.0)
        return tc * (0.001 / 0.0015), tc * (0.0005 / 0.0015) if tc else (0.0, 0.0)

    def _build_ai_rankings_with_train_thresholds(
        self, panel: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        horizon = self.sim_cfg.horizon_days
        labeled_pack = assign_label_scheme(
            panel,
            return_col=f"future_return_{horizon}d",
            scheme=self.sim_cfg.label_scheme,  # type: ignore[arg-type]
            gain_name=self.sim_cfg.gain_name,  # type: ignore[arg-type]
            min_group_size=int(self.scheme_cfg.get("min_group_size", 5)),
        )
        labeled = labeled_pack.frame.copy()
        feature_cols = self.base.lr.feature_sets[self.sim_cfg.feature_set]
        params = dict(self.base.lr.param_presets[self.sim_cfg.param_preset])
        folds = build_expanding_folds(
            labeled["Date"],
            n_folds=5,
            min_train_days=252 * 3,
            valid_days=252,
            purge_days=horizon,
        )
        parts: list[pd.DataFrame] = []
        metas: list[dict[str, Any]] = []
        q = self.sim_cfg.train_threshold_quantile
        for i, fold in enumerate(folds, start=1):
            print(f"Fold {i}/{len(folds)} Training…")
            scored_valid, scored_train = self._fit_score_fold(
                labeled=labeled,
                fold=fold,
                feature_cols=feature_cols,
                params=params,
                label_gain=labeled_pack.label_gain,
            )
            ranked = attach_country_ranks(scored_valid, country_col="Region")
            train_ranked = attach_country_ranks(scored_train, country_col="Region")
            thresholds = calibrate_train_thresholds(
                train_ranked,
                quantile=q,
                top_percentile=self.sim_cfg.top_percentile,
            )
            parts.append(ranked)
            metas.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": str(fold.train_start.date()),
                    "train_end": str(fold.train_end.date()),
                    "valid_start": str(fold.valid_start.date()),
                    "valid_end": str(fold.valid_end.date()),
                    "purge_days": fold.purge_days,
                    "train_thresholds": thresholds,
                }
            )
            print(
                f"  train thresholds std>={thresholds['min_score_std']:.4g} "
                f"gap>={thresholds['min_top_median_gap']:.4g}"
            )
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), metas

    def _fit_score_fold(
        self,
        *,
        labeled: pd.DataFrame,
        fold: WalkForwardFold,
        feature_cols: tuple[str, ...],
        params: dict[str, Any],
        label_gain: tuple[float, ...],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = mask_fold_partition(labeled, fold, partition="train")
        valid_df = mask_fold_partition(labeled, fold, partition="validation")
        drop_cols = [c for c in feature_cols if c in labeled.columns] + ["relevance"]
        train_df = train_df.dropna(subset=drop_cols).copy()
        valid_df = valid_df.dropna(subset=drop_cols).copy()
        if train_df.empty or valid_df.empty:
            raise TrainingError("Empty train/valid in diagnostics fold")

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
        valid_out = valid_s.copy()
        valid_out["score"] = predict_rank_scores(fit.model, x_valid)
        train_out = train_s.copy()
        train_out["score"] = predict_rank_scores(fit.model, x_train)
        return valid_out, train_out

    def _simulate_ranked(
        self,
        *,
        name: str,
        rankings: pd.DataFrame,
        price_panel: pd.DataFrame,
        fx: Any,
        cfg: SimulationConfig,
        countries: set[str] | None,
        fold_metas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fold_summaries: list[dict[str, Any]] = []
        all_equity = []
        all_trades = []
        total_commission = 0.0
        total_slippage = 0.0
        reb_acc = {
            "avg_entries_per_rebalance": [],
            "avg_exits_per_rebalance": [],
        }

        for meta in fold_metas:
            v0 = pd.Timestamp(meta["valid_start"])
            v1 = pd.Timestamp(meta["valid_end"])
            px = price_panel
            rk = rankings
            if countries is not None:
                px = px.loc[px["Country"].isin(countries)]
                rk = rk.loc[rk["Country"].isin(countries)]
            rk_fold = rk.loc[(rk["Date"] >= v0) & (rk["Date"] <= v1)].copy()
            px_fold = px.loc[(px["Date"] >= v0) & (px["Date"] <= v1 + pd.tseries.offsets.BDay(10))]

            cfg_fold = cfg
            if (
                cfg.use_train_dispersion_filter or cfg.use_score_separation_filter
            ) and meta.get("train_thresholds"):
                th = meta["train_thresholds"]
                cfg_fold = cfg.apply_runtime_thresholds(
                    min_score_std=th.get("min_score_std") if cfg.use_train_dispersion_filter else None,
                    min_top_median_gap=th.get("min_top_median_gap"),
                    min_unique_score_ratio=(
                        th.get("min_unique_score_ratio") if cfg.use_train_dispersion_filter else None
                    ),
                    min_cutoff_median_gap=(
                        th.get("min_cutoff_median_gap") if cfg.use_score_separation_filter else None
                    ),
                )

            if rk_fold.empty or px_fold.empty:
                fold_summaries.append({**meta, "skipped": True})
                continue

            engine = SimulationEngine(cfg_fold, fx=fx, countries=countries)
            result = engine.run(prices=px_fold, rankings=rk_fold, start=v0, end=None)
            eq = [
                e
                for e in result.equity_curve
                if pd.Timestamp(e.date) <= v1 + pd.tseries.offsets.BDay(5)
            ]
            trades = result.trades
            fold_cost = 0.0
            if result.portfolio is not None:
                total_commission += float(result.portfolio.total_commission)
                total_slippage += float(result.portfolio.total_slippage_impact)
                fold_cost = float(
                    result.portfolio.total_commission + result.portfolio.total_slippage_impact
                )
            summary = summarize_result(
                equity_curve=eq,
                trades=trades,
                initial_capital=cfg.initial_capital,
                transaction_costs=fold_cost,
                extra={"fold": meta, "name": name},
            )
            rs = (result.meta or {}).get("rebalance_stats") or {}
            if rs.get("avg_entries_per_rebalance") is not None:
                reb_acc["avg_entries_per_rebalance"].append(rs["avg_entries_per_rebalance"])
            if rs.get("avg_exits_per_rebalance") is not None:
                reb_acc["avg_exits_per_rebalance"].append(rs["avg_exits_per_rebalance"])
            fold_summaries.append(summary)
            all_equity.extend(eq)
            all_trades.extend(trades)

        perf_list = [f.get("performance", {}) for f in fold_summaries if "performance" in f]
        agg_perf = _mean_perf(perf_list)
        pooled = compute_performance(
            all_equity,
            all_trades,
            initial_capital=cfg.initial_capital,
            transaction_costs_total=float(total_commission + total_slippage),
        )
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
        }
        return {
            "aggregate": aggregate,
            "folds": fold_summaries,
            "equity_curve": all_equity,
            "trades": all_trades,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "rebalance_stats": {
                "avg_entries_per_rebalance": (
                    float(np.mean(reb_acc["avg_entries_per_rebalance"]))
                    if reb_acc["avg_entries_per_rebalance"]
                    else 0.0
                ),
                "avg_exits_per_rebalance": (
                    float(np.mean(reb_acc["avg_exits_per_rebalance"]))
                    if reb_acc["avg_exits_per_rebalance"]
                    else 0.0
                ),
            },
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4B simulation strategy diagnostics")
    parser.add_argument("--universe", type=Path, default=Path("config/universe.global100.json"))
    parser.add_argument(
        "--config", type=Path, default=Path("config/simulation_diagnostics.json")
    )
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    sim_cfg = load_simulation_config(args.config)
    try:
        DiagnosticsRunner(settings, args.universe, sim_cfg).run(force_refresh=args.force_refresh)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Diagnostics failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
