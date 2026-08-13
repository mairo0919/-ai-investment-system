"""Phase 4D: true holdout validation of frozen FINAL strategy (no retuning)."""

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
from src.core.exceptions import TrainingError
from src.ml.ltr_baselines import momentum_scores
from src.ml.ltr_label_schemes import assign_label_scheme, load_label_scheme_config
from src.ml.ltr_labels import assert_group_integrity, build_ranking_groups
from src.ml.ltr_model import fit_lgbm_ranker, predict_rank_scores
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.diagnostics import turnover_analysis
from src.simulation.engine import SimulationEngine
from src.simulation.holdout_diagnostics import (
    gap_down_audit,
    mae_mfe_analysis,
    monthly_performance,
    score_tertile_diagnostics,
    sector_contribution,
    sortino_ratio,
    success_checklist,
    symbol_contribution,
)
from src.simulation.metrics import buy_and_hold_benchmark, compute_performance
from src.simulation.ranking import attach_country_ranks
from src.simulation.report import equity_to_frame, trades_to_frame, write_json
from src.simulation.runner import SimulationRunner, _pct, _sub
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

PHASE4C_SELECTION_NET = 0.2957  # US mean-fold Net from Phase 4C (selection only)


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_true_holdout_bundle(path: Path) -> tuple[dict[str, Any], SimulationConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Merge extends for ranking/fx defaults, then force FINAL overrides.
    cfg = load_simulation_config(path)
    final = dict(raw["final_strategy"])
    final.pop("immutable", None)
    # Explicit null stop
    cfg = cfg.with_overrides(
        top_percentile=float(final["top_percentile"]),
        max_positions=int(final["max_positions"]),
        max_position_weight=float(final["max_position_weight"]),
        max_country_weight=float(final["max_country_weight"]),
        rebalance_frequency=str(final["rebalance_frequency"]),
        holding_period_days=int(final["holding_period_days"]),
        take_profit_pct=final["take_profit_pct"],
        stop_loss_pct=final["stop_loss_pct"],
        trailing_stop_pct=final.get("trailing_stop_pct"),
        cooldown_days=int(final["cooldown_days"]),
        ranking_exit_enabled=bool(final["ranking_exit_enabled"]),
        use_score_separation_filter=bool(final.get("use_score_separation_filter", False)),
        commission_rate=float(final["commission_rate"]),
        slippage_rate=float(final["slippage_rate"]),
    )
    return raw, cfg


def assert_holdout_no_overlap(
    holdout_start: pd.Timestamp,
    prior_valid_end: pd.Timestamp,
) -> None:
    if holdout_start <= prior_valid_end:
        raise TrainingError(
            f"Holdout start {holdout_start.date()} must be after prior selection "
            f"valid_end {prior_valid_end.date()}"
        )


class TrueHoldoutRunner:
    """Run frozen FINAL strategy once on an unseen holdout window."""

    def __init__(
        self,
        settings: Settings,
        universe_path: Path,
        bundle_path: Path,
    ) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.bundle_path = bundle_path
        self.raw, self.cfg = load_true_holdout_bundle(bundle_path)
        self.holdout_meta = dict(self.raw["holdout"])
        if not self.holdout_meta.get("locked"):
            raise TrainingError("Holdout definition must be locked=true before run")
        self.report_dir = settings.reports_dir / "true_holdout"
        self.scheme_cfg = load_label_scheme_config()
        self.base = SimulationRunner(settings, universe_path, self.cfg)

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        print("Phase 4D True Holdout Started")
        print("FINAL strategy is IMMUTABLE — no retuning allowed.")

        h0 = pd.Timestamp(self.holdout_meta["start"]).normalize()
        h1 = pd.Timestamp(self.holdout_meta["end"]).normalize()
        prior_end = pd.Timestamp(self.holdout_meta["prior_selection_last_valid_end"]).normalize()
        assert_holdout_no_overlap(h0, prior_end)

        definition = {
            "created_at_utc": _timestamp_iso(),
            "holdout_start": str(h0.date()),
            "holdout_end": str(h1.date()),
            "prior_selection_last_valid_end": str(prior_end.date()),
            "locked": True,
            "rationale": self.holdout_meta.get("rationale"),
            "no_overlap_check": "passed",
            "phase4c_selection_net_return_us": PHASE4C_SELECTION_NET,
            "phase4c_label": "strategy_selection_performance_not_holdout",
        }
        write_json(self.report_dir / "holdout_definition.json", definition)
        write_json(
            self.report_dir / "final_strategy.json",
            {
                "immutable": True,
                "config": self.raw["final_strategy"],
                "ranking": self.raw.get("ranking"),
                "note": "Do not modify after seeing holdout results.",
            },
        )
        print(f"Holdout locked: {h0.date()} → {h1.date()} (after selection valid_end {prior_end.date()})")

        panel = self.base.lr._build_enriched_panel(force_refresh=force_refresh)
        fx = self.base._build_fx(force_refresh=force_refresh)
        price_panel = self.base._price_panel(panel)

        # Sector map from panel
        sym_sector = (
            panel.sort_values("Date")
            .drop_duplicates("Symbol")
            .set_index("Symbol")["Sector"]
            .fillna("Unknown")
            .astype(str)
            .to_dict()
            if "Sector" in panel.columns
            else {}
        )

        print("Training model on pre-holdout data only…")
        ai_rankings, train_info = self._fit_and_score_holdout(panel, h0, h1)
        mom_rankings = self._momentum_holdout_rankings(panel, h0, h1)

        scopes = ["United States", "Japan", "Europe"]
        country_results: dict[str, Any] = {}

        for scope in scopes:
            print(f"\n=== Holdout Scope: {scope} ===")
            countries = {scope}
            pack = self._simulate_window(
                rankings=ai_rankings,
                price_panel=price_panel,
                fx=fx,
                cfg=self.cfg,
                countries=countries,
                start=h0,
                end=h1,
                name=f"FINAL_{scope}",
            )
            mom = self._simulate_window(
                rankings=mom_rankings,
                price_panel=price_panel,
                fx=fx,
                cfg=self.cfg,
                countries=countries,
                start=h0,
                end=h1,
                name=f"MOM_{scope}",
            )
            bench = self._benchmark_holdout(panel, scope, h0, h1, fx)

            metrics = self._metrics_pack(pack, initial=self.cfg.initial_capital)
            mom_metrics = self._metrics_pack(mom, initial=self.cfg.initial_capital)
            bench_ret = (bench or {}).get("total_return")
            metrics["benchmark_return"] = bench_ret
            metrics["benchmark_excess"] = _sub(metrics.get("total_return"), bench_ret)
            metrics["momentum_return"] = mom_metrics.get("total_return")
            metrics["momentum_excess"] = _sub(
                metrics.get("total_return"), mom_metrics.get("total_return")
            )

            # Equity series for monthly
            eq = equity_to_frame(pack["equity_curve"])
            eq_s = (
                eq.set_index(pd.to_datetime(eq["Date"]))["Total Equity"].astype(float)
                if not eq.empty
                else pd.Series(dtype=float)
            )
            mom_eq = equity_to_frame(mom["equity_curve"])
            mom_s = (
                mom_eq.set_index(pd.to_datetime(mom_eq["Date"]))["Total Equity"].astype(float)
                if not mom_eq.empty
                else pd.Series(dtype=float)
            )
            bench_s = None
            if bench and bench.get("equity_curve"):
                be = pd.DataFrame(bench["equity_curve"])
                bench_s = be.set_index(pd.to_datetime(be["Date"]))["Total Equity"].astype(float)

            monthly = monthly_performance(pack["equity_curve"], benchmark_equity=bench_s, momentum_equity=mom_s)
            sym = symbol_contribution(pack["trades"], initial_capital=self.cfg.initial_capital)
            sec = sector_contribution(pack["trades"], sym_sector)
            buckets = score_tertile_diagnostics(pack["trades"])
            tail = mae_mfe_analysis(pack["trades"], price_panel)
            gaps = gap_down_audit(pack["trades"], price_panel)

            top_sector_share = None
            if sec:
                pnls = {k: float(v.get("net_pnl") or 0) for k, v in sec.items()}
                tot = sum(pnls.values())
                if abs(tot) > 1e-12:
                    top_sector_share = max(pnls.values()) / tot

            pos_month = None
            if not monthly.empty and "strategy_return" in monthly.columns:
                pos_month = float((monthly["strategy_return"] > 0).mean())

            checklist = success_checklist(
                net_return=metrics.get("total_return"),
                sharpe=metrics.get("sharpe"),
                benchmark_excess=metrics.get("benchmark_excess"),
                momentum_excess=metrics.get("momentum_excess"),
                profit_factor=metrics.get("profit_factor"),
                top1_share=(sym.get("top_dependency") or {}).get("top1", {}).get("pnl_share"),
                top_sector_share=top_sector_share,
                positive_month_ratio=pos_month,
                worst_mae=tail.get("worst_mae"),
            )

            country_results[scope] = {
                "metrics": metrics,
                "momentum_metrics": mom_metrics,
                "benchmark": bench,
                "monthly": monthly.to_dict(orient="records") if not monthly.empty else [],
                "symbol_contribution": sym,
                "sector_contribution": sec,
                "score_buckets": buckets,
                "tail_risk": tail,
                "gap_risk": gaps,
                "success_checklist": checklist,
                "equity_curve": pack["equity_curve"],
                "trades": pack["trades"],
                "momentum_equity_curve": mom["equity_curve"],
            }

            print(
                f"  Net {_pct(metrics.get('total_return'))} | "
                f"Sharpe {metrics.get('sharpe')} | "
                f"BenchEx {_pct(metrics.get('benchmark_excess'))} | "
                f"MomEx {_pct(metrics.get('momentum_excess'))} | "
                f"Trades {metrics.get('number_of_trades')}"
            )

        # Cost stress on US only (same rankings/logic)
        print("\n=== Cost Stress (US, diagnostic) ===")
        us_rank = ai_rankings
        cost_stress = {}
        for name, rates in (self.raw.get("cost_stress") or {}).items():
            cfg_c = self.cfg.with_overrides(
                commission_rate=float(rates["commission_rate"]),
                slippage_rate=float(rates["slippage_rate"]),
            )
            pack_c = self._simulate_window(
                rankings=us_rank,
                price_panel=price_panel,
                fx=fx,
                cfg=cfg_c,
                countries={"United States"},
                start=h0,
                end=h1,
                name=f"STRESS_{name}",
            )
            m = self._metrics_pack(pack_c, initial=self.cfg.initial_capital)
            cost_stress[name] = {
                "commission_rate": rates["commission_rate"],
                "slippage_rate": rates["slippage_rate"],
                "total_return": m.get("total_return"),
                "sharpe": m.get("sharpe"),
                "max_drawdown": m.get("max_drawdown"),
                "transaction_costs": m.get("transaction_costs"),
                "trades": m.get("number_of_trades"),
            }
            print(f"  {name}: Net {_pct(m.get('total_return'))} cost={m.get('transaction_costs')}")

        us = country_results["United States"]
        # Persist US primary artifacts
        equity_to_frame(us["equity_curve"]).to_csv(self.report_dir / "equity_curve.csv", index=False)
        trades_to_frame(us["trades"]).to_csv(self.report_dir / "trades.csv", index=False)
        pd.DataFrame(us["monthly"]).to_csv(self.report_dir / "monthly_returns.csv", index=False)

        write_json(
            self.report_dir / "benchmark_comparison.json",
            {
                "scope": "United States",
                "strategy": us["metrics"],
                "benchmark": us["benchmark"],
                "excess": us["metrics"].get("benchmark_excess"),
            },
        )
        write_json(
            self.report_dir / "momentum_comparison.json",
            {
                "scope": "United States",
                "strategy_return": us["metrics"].get("total_return"),
                "momentum_return": us["metrics"].get("momentum_return"),
                "excess": us["metrics"].get("momentum_excess"),
                "momentum_metrics": us["momentum_metrics"],
                "same_rules_as_final": True,
            },
        )
        write_json(self.report_dir / "symbol_contribution.json", us["symbol_contribution"])
        write_json(self.report_dir / "sector_contribution.json", us["sector_contribution"])
        write_json(self.report_dir / "score_bucket_diagnostics.json", {
            "United States": us["score_buckets"],
            "Japan": country_results["Japan"]["score_buckets"],
            "Europe": country_results["Europe"]["score_buckets"],
        })
        write_json(self.report_dir / "tail_risk.json", us["tail_risk"])
        write_json(self.report_dir / "gap_risk.json", us["gap_risk"])
        write_json(self.report_dir / "cost_stress.json", cost_stress)

        monthly_df = pd.DataFrame(us["monthly"])
        best_month = worst_month = None
        if not monthly_df.empty:
            i_best = monthly_df["strategy_return"].idxmax()
            i_worst = monthly_df["strategy_return"].idxmin()
            best_month = monthly_df.loc[i_best].to_dict()
            worst_month = monthly_df.loc[i_worst].to_dict()

        summary = {
            "created_at_utc": _timestamp_iso(),
            "phase": "4D",
            "purpose": "True holdout validation of frozen FINAL strategy",
            "holdout": definition,
            "train_info": train_info,
            "final_strategy": self.raw["final_strategy"],
            "phase4c_selection_performance_us": {
                "net_return": PHASE4C_SELECTION_NET,
                "label": "NOT holdout — strategy selection only",
            },
            "holdout_performance": {
                "United States": us["metrics"],
                "Japan": country_results["Japan"]["metrics"],
                "Europe": country_results["Europe"]["metrics"],
            },
            "success_checklist_us": us["success_checklist"],
            "monthly_us": {
                "positive_month_ratio": float((monthly_df["strategy_return"] > 0).mean())
                if not monthly_df.empty
                else None,
                "best_month": best_month,
                "worst_month": worst_month,
                "n_months": int(len(monthly_df)),
            },
            "divergence_vs_phase4c_us": _sub(us["metrics"].get("total_return"), PHASE4C_SELECTION_NET),
            "known_limitations": [
                "Holdout results must not trigger strategy retuning in this phase.",
                "Phase 4C +29.6% is selection performance and is reported separately.",
                "Paper trading only; no brokerage.",
            ],
        }
        write_json(self.report_dir / "holdout_summary.json", summary)
        print("\nTrue Holdout Completed")
        return summary

    def _fit_and_score_holdout(
        self,
        panel: pd.DataFrame,
        holdout_start: pd.Timestamp,
        holdout_end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        horizon = self.cfg.horizon_days
        purge = int(self.raw.get("training", {}).get("purge_days", horizon))
        es_days = int(self.raw.get("training", {}).get("early_stopping_valid_days", 252))

        labeled_pack = assign_label_scheme(
            panel,
            return_col=f"future_return_{horizon}d",
            scheme=self.cfg.label_scheme,  # type: ignore[arg-type]
            gain_name=self.cfg.gain_name,  # type: ignore[arg-type]
            min_group_size=int(self.scheme_cfg.get("min_group_size", 5)),
        )
        labeled = labeled_pack.frame.copy()
        labeled["Date"] = pd.to_datetime(labeled["Date"]).dt.normalize()
        feature_cols = self.base.lr.feature_sets[self.cfg.feature_set]
        params = dict(self.base.lr.param_presets[self.cfg.param_preset])

        calendar = sorted(labeled["Date"].unique())
        # Last pre-holdout date index with purge gap
        pre = [d for d in calendar if d < holdout_start]
        if len(pre) <= purge + es_days + 100:
            raise TrainingError("Insufficient pre-holdout history for training")
        # Exclude last ``purge`` pre-holdout dates from any fit labels near the boundary
        fit_calendar = pre[:-purge] if purge > 0 else pre
        if len(fit_calendar) <= es_days:
            raise TrainingError("Not enough fit calendar after purge")
        es_dates = set(fit_calendar[-es_days:])
        train_dates = set(fit_calendar[:-es_days])

        drop_cols = [c for c in feature_cols if c in labeled.columns] + ["relevance"]
        train_df = labeled.loc[labeled["Date"].isin(train_dates)].dropna(subset=drop_cols).copy()
        es_df = labeled.loc[labeled["Date"].isin(es_dates)].dropna(subset=drop_cols).copy()
        hold_df = labeled.loc[
            (labeled["Date"] >= holdout_start) & (labeled["Date"] <= holdout_end)
        ].copy()
        # Score holdout using features only; dropna on features (not future labels)
        hold_df = hold_df.dropna(subset=[c for c in feature_cols if c in hold_df.columns]).copy()
        if train_df.empty or hold_df.empty:
            raise TrainingError("Empty train or holdout after filters")

        # Assert no holdout dates in train/es
        if (train_df["Date"] >= holdout_start).any() or (es_df["Date"] >= holdout_start).any():
            raise TrainingError("Holdout leakage into train/early-stopping sets")

        train_s, train_g = build_ranking_groups(train_df, group_keys=("Date", "Region"))
        es_s, es_g = build_ranking_groups(es_df, group_keys=("Date", "Region"))
        assert_group_integrity(train_s, train_g)
        assert_group_integrity(es_s, es_g)
        hold_s, hold_g = build_ranking_groups(hold_df, group_keys=("Date", "Region"))
        assert_group_integrity(hold_s, hold_g)

        x_train = train_s.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
        x_es = es_s.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
        x_hold = hold_s.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)

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
        scored = hold_s.copy()
        scored["score"] = predict_rank_scores(fit.model, x_hold)
        rankings = attach_country_ranks(scored, country_col="Region")

        info = {
            "train_start": str(min(train_dates).date()),
            "train_end": str(max(train_dates).date()),
            "early_stopping_start": str(min(es_dates).date()),
            "early_stopping_end": str(max(es_dates).date()),
            "purge_days_before_holdout": purge,
            "holdout_start": str(holdout_start.date()),
            "holdout_end": str(holdout_end.date()),
            "train_rows": int(len(train_s)),
            "early_stopping_rows": int(len(es_s)),
            "holdout_rows": int(len(hold_s)),
            "best_iteration": fit.best_iteration,
            "leakage_check": "holdout dates excluded from train and early stopping",
        }
        print(
            f"  Train {info['train_start']}→{info['train_end']} | "
            f"ES {info['early_stopping_start']}→{info['early_stopping_end']} | "
            f"Holdout rows={info['holdout_rows']}"
        )
        return rankings, info

    def _momentum_holdout_rankings(
        self,
        panel: pd.DataFrame,
        holdout_start: pd.Timestamp,
        holdout_end: pd.Timestamp,
    ) -> pd.DataFrame:
        frame = panel.copy()
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
        hold = frame.loc[
            (frame["Date"] >= holdout_start) & (frame["Date"] <= holdout_end)
        ].dropna(subset=["return_20d"]).copy()
        hold["score"] = momentum_scores(hold, feature_col="return_20d").to_numpy()
        return attach_country_ranks(hold, country_col="Region")

    def _simulate_window(
        self,
        *,
        rankings: pd.DataFrame,
        price_panel: pd.DataFrame,
        fx: Any,
        cfg: SimulationConfig,
        countries: set[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        name: str,
    ) -> dict[str, Any]:
        px = price_panel.loc[price_panel["Country"].isin(countries)].copy()
        rk = rankings.loc[rankings["Country"].isin(countries)].copy()
        rk = rk.loc[(rk["Date"] >= start) & (rk["Date"] <= end)]
        # Buffer for T+1 opens after last signal
        px = px.loc[(px["Date"] >= start) & (px["Date"] <= end + pd.tseries.offsets.BDay(25))]
        engine = SimulationEngine(cfg, fx=fx, countries=countries)
        result = engine.run(prices=px, rankings=rk, start=start, end=None)
        # Metrics/equity locked to holdout end (price buffer only for T+1 fills).
        eq = [e for e in result.equity_curve if pd.Timestamp(e.date) <= end]
        costs = 0.0
        if result.portfolio is not None:
            costs = float(
                result.portfolio.total_commission + result.portfolio.total_slippage_impact
            )
        return {
            "name": name,
            "equity_curve": eq,
            "trades": result.trades,
            "total_commission": float(result.portfolio.total_commission) if result.portfolio else 0.0,
            "total_slippage": float(result.portfolio.total_slippage_impact) if result.portfolio else 0.0,
            "transaction_costs": costs,
            "meta": result.meta,
        }

    def _metrics_pack(self, pack: dict[str, Any], *, initial: float) -> dict[str, Any]:
        perf = compute_performance(
            pack["equity_curve"],
            pack["trades"],
            initial_capital=initial,
            transaction_costs_total=float(pack.get("transaction_costs") or 0),
        )
        eq = equity_to_frame(pack["equity_curve"])
        sortino = None
        if not eq.empty:
            rets = eq["Total Equity"].astype(float).pct_change().dropna()
            sortino = sortino_ratio(rets)
        tr = trades_to_frame(pack["trades"])
        avg_win = avg_loss = None
        pf = perf.get("profit_factor")
        if not tr.empty:
            nets = tr["Net PnL"].astype(float)
            wins = nets[nets > 0]
            losses = nets[nets <= 0]
            avg_win = float(wins.mean()) if len(wins) else None
            avg_loss = float(losses.mean()) if len(losses) else None
        turn = turnover_analysis(
            pack["trades"],
            pack["equity_curve"],
            initial_capital=initial,
            rebalance_stats=(pack.get("meta") or {}).get("rebalance_stats"),
        )
        return {
            **perf,
            "sortino": sortino,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "profit_factor": pf,
            "turnover": turn.get("portfolio_turnover"),
            "commission": float(pack.get("total_commission") or 0),
            "slippage": float(pack.get("total_slippage") or 0),
        }

    def _benchmark_holdout(
        self,
        panel: pd.DataFrame,
        scope: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fx: Any,
    ) -> dict[str, Any] | None:
        ticker = self.cfg.benchmarks.get(scope)
        if not ticker:
            from src.ml.walk_forward_runner import REGION_BENCHMARKS

            ticker = REGION_BENCHMARKS.get(scope, (None,))[0]
        if not ticker:
            return None
        frame = self.base.lr.wf.cache.load(ticker)
        if frame is None or frame.empty:
            return None
        s = frame.copy()
        s["Date"] = pd.to_datetime(s["Date"])
        s = s.loc[(s["Date"] >= start) & (s["Date"] <= end)]
        series = s.set_index("Date")["Close"].astype(float).sort_index()
        ccy = {"Japan": "JPY", "United States": "USD", "Europe": "EUR"}.get(scope, "JPY")
        if ccy != "JPY":
            vals = []
            for idx, px in series.items():
                try:
                    vals.append(float(px) * fx.rate_to_base(ccy, idx))
                except Exception:  # noqa: BLE001
                    vals.append(np.nan)
            series = pd.Series(vals, index=series.index, dtype=float).dropna()
        out = buy_and_hold_benchmark(series, initial_capital=self.cfg.initial_capital)
        out["ticker"] = ticker
        out["scope"] = scope
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4D true holdout validation")
    parser.add_argument("--universe", type=Path, default=Path("config/universe.global100.json"))
    parser.add_argument("--config", type=Path, default=Path("config/true_holdout.json"))
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    try:
        TrueHoldoutRunner(settings, args.universe, args.config).run(
            force_refresh=args.force_refresh
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("True holdout failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
