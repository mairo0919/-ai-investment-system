"""Phase 4C: staged low-turnover / exit-rule optimization (fixed AI rankings)."""

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
from src.ml.stability_diagnostics import classify_market_regime
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.diagnostics import (
    cost_decomposition,
    exit_reason_analysis,
    fold_stability,
    regime_trade_analysis,
    trade_quality_by_entry_score,
    turnover_analysis,
)
from src.simulation.diagnostics_runner import DiagnosticsRunner
from src.simulation.quality import count_reentries
from src.simulation.report import trades_to_frame, write_json
from src.simulation.runner import _pct, _sub
from src.simulation.stop_diagnostics import (
    analyze_stop_aftermath,
    analyze_stop_counterfactual,
    pick_stop_setting,
)
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pick_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Multi-criteria: Net → Fold stability → Sharpe → MaxDD → PF → Turnover/Cost."""

    def key(r: dict[str, Any]) -> tuple:
        n = r.get("net_return")
        pos = r.get("positive_fold_ratio")
        stab_std = (r.get("fold_stability") or {}).get("std")
        s = r.get("sharpe")
        mdd = r.get("max_dd")
        pf = r.get("profit_factor")
        trades = r.get("trades")
        cost = r.get("total_cost")
        pf_v = float(pf) if pf not in (None, float("inf")) else (1e6 if pf == float("inf") else -1e9)
        return (
            float(n) if n is not None else -1e9,
            float(pos) if pos is not None else -1e9,
            -float(stab_std) if stab_std is not None else -1e9,
            float(s) if s is not None else -1e9,
            float(mdd) if mdd is not None else -1e9,
            pf_v,
            -float(trades) if trades is not None else -1e9,
            -float(cost) if cost is not None else -1e9,
        )

    return max(rows, key=key)


class RuleOptimizationRunner:
    """Staged exit/entry rule study around BASE_US_CONCENTRATED."""

    def __init__(
        self,
        settings: Settings,
        universe_path: Path,
        sim_cfg: SimulationConfig,
    ) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.sim_cfg = sim_cfg
        self.diag = DiagnosticsRunner(settings, universe_path, sim_cfg)
        self.report_dir = settings.reports_dir / "rule_optimization"

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        print("Phase 4C Rule Optimization Started")

        panel = self.diag.base.lr._build_enriched_panel(force_refresh=force_refresh)
        fx = self.diag.base._build_fx(force_refresh=force_refresh)
        price_panel = self.diag.base._price_panel(panel)
        print("Building fixed AI rankings…")
        ai_rankings, fold_metas = self.diag._build_ai_rankings_with_train_thresholds(panel)
        mom_rankings = self.diag.base._build_momentum_rankings(panel)

        scopes_primary = ["United States"]
        scopes_diag = ["Japan", "Europe"]

        base_cfg = self._base_cfg()
        ctx = {
            "ai_rankings": ai_rankings,
            "mom_rankings": mom_rankings,
            "price_panel": price_panel,
            "fx": fx,
            "fold_metas": fold_metas,
            "panel": panel,
        }

        stop_diag_cfg = dict(self.sim_cfg.raw.get("stop_diagnostics") or {})
        neutral_band = float(stop_diag_cfg.get("neutral_band", 0.02))
        further_drop = float(stop_diag_cfg.get("further_drop_pct", -0.05))
        max_dd_worsen = float(stop_diag_cfg.get("max_dd_worsen_limit", 0.05))

        # ---- Part 1: Stop Loss ----
        print("\n=== Part 1: Stop Loss ===")
        stop_rows = []
        for name, stop in [
            ("NO_STOP", None),
            ("STOP_5", -0.05),
            ("STOP_8", -0.08),
            ("STOP_10", -0.10),
        ]:
            cfg = base_cfg.with_overrides(stop_loss_pct=stop, take_profit_pct=0.10)
            stop_rows.append(
                self._eval_named(name, cfg, countries={"United States"}, ctx=ctx, with_gross=True)
            )
        stop_pick = pick_stop_setting(
            stop_rows, baseline_name="STOP_5", max_dd_worsen_limit=max_dd_worsen
        )
        stop_analysis = {
            "candidates": stop_rows,
            "selected": stop_pick["selected"],
            "selection": {
                "method": "net_fold_stability_sharpe_maxdd_pf_turnover_with_maxdd_guardrail",
                "rejected_for_maxdd": stop_pick["rejected_for_maxdd"],
                "max_dd_worsen_limit": max_dd_worsen,
                "baseline": "STOP_5",
            },
        }
        selected_stop = next(r for r in stop_rows if r["name"] == stop_analysis["selected"])
        print(f"Selected stop: {selected_stop['name']} net={_pct(selected_stop['net_return'])}")

        # ---- Parts 2–3: Stop aftermath + counterfactual (diagnostic; STOP_5 baseline) ----
        print("\n=== Stop Aftermath / Counterfactual (STOP_5 trades) ===")
        stop5_pack = self.diag._simulate_ranked(
            name="STOP_5_diag",
            rankings=ai_rankings,
            price_panel=price_panel,
            fx=fx,
            cfg=base_cfg.with_overrides(stop_loss_pct=-0.05, take_profit_pct=0.10).with_cost_preset(
                "net"
            ),
            countries={"United States"},
            fold_metas=fold_metas,
        )
        aftermath = analyze_stop_aftermath(
            stop5_pack["trades"],
            price_panel,
            further_drop_pct=further_drop,
        )
        # Compact report: drop per-trade path detail beyond summary stats in summary file
        aftermath_report = {k: v for k, v in aftermath.items() if k != "trades"}
        aftermath_report["n_trade_details"] = len(aftermath.get("trades") or [])
        counterfactual = analyze_stop_counterfactual(
            stop5_pack["trades"],
            price_panel,
            holding_period_days=int(base_cfg.holding_period_days),
            neutral_band=neutral_band,
        )
        cf_report = {k: v for k, v in counterfactual.items() if k != "trades"}
        cf_report["n_trade_details"] = len(counterfactual.get("trades") or [])
        write_json(self.report_dir / "stop_aftermath_analysis.json", aftermath_report)
        write_json(self.report_dir / "stop_counterfactual.json", cf_report)
        print(
            f"  Stop trades={aftermath_report.get('n_stop_trades')} "
            f"recover5={aftermath_report.get('recovery_to_entry_rate', {}).get('within_5d')} "
            f"BAD_STOP={cf_report.get('classification_rates', {}).get('BAD_STOP')}"
        )

        # ---- Part 4: Take Profit ----
        print("\n=== Part 2: Take Profit ===")
        stop_pct = selected_stop["config"]["stop_loss_pct"]
        tp_rows = []
        for name, tp in [("NO_TP", None), ("TP_10", 0.10), ("TP_15", 0.15)]:
            cfg = base_cfg.with_overrides(stop_loss_pct=stop_pct, take_profit_pct=tp)
            tp_rows.append(
                self._eval_named(name, cfg, countries={"United States"}, ctx=ctx, with_gross=True)
            )
        tp_analysis = {"candidates": tp_rows, "selected": _pick_best(tp_rows)["name"]}
        selected_tp = next(r for r in tp_rows if r["name"] == tp_analysis["selected"])
        print(f"Selected TP: {selected_tp['name']} net={_pct(selected_tp['net_return'])}")
        tp_pct = selected_tp["config"]["take_profit_pct"]

        # ---- Part 3: Holding ----
        print("\n=== Part 3: Holding Period ===")
        hold_rows = []
        for name, h in [("HOLD_10", 10), ("HOLD_15", 15), ("HOLD_20", 20)]:
            cfg = base_cfg.with_overrides(
                stop_loss_pct=stop_pct, take_profit_pct=tp_pct, holding_period_days=h
            )
            hold_rows.append(
                self._eval_named(name, cfg, countries={"United States"}, ctx=ctx, with_gross=True)
            )
        hold_analysis = {"candidates": hold_rows, "selected": _pick_best(hold_rows)["name"]}
        selected_hold = next(r for r in hold_rows if r["name"] == hold_analysis["selected"])
        hold_days = selected_hold["config"]["holding_period_days"]
        print(f"Selected hold: {selected_hold['name']} net={_pct(selected_hold['net_return'])}")

        # Working config after parts 1-3
        work = base_cfg.with_overrides(
            stop_loss_pct=stop_pct,
            take_profit_pct=tp_pct,
            holding_period_days=hold_days,
        )

        # ---- Part 4: Trailing ----
        print("\n=== Part 4: Trailing Stop ===")
        trail_rows = [
            self._eval_named(
                "FIXED_ONLY",
                work.with_overrides(trailing_stop_pct=None),
                countries={"United States"},
                ctx=ctx,
                with_gross=True,
            )
        ]
        for name, tr in [("TRAIL_5", 0.05), ("TRAIL_8", 0.08)]:
            # Trailing as alternative: keep fixed stop from work, add trailing
            trail_rows.append(
                self._eval_named(
                    name,
                    work.with_overrides(trailing_stop_pct=tr),
                    countries={"United States"},
                    ctx=ctx,
                    with_gross=True,
                )
            )
        trail_analysis = {"candidates": trail_rows, "selected": _pick_best(trail_rows)["name"]}
        selected_trail = next(r for r in trail_rows if r["name"] == trail_analysis["selected"])
        trail_pct = selected_trail["config"]["trailing_stop_pct"]
        work = work.with_overrides(trailing_stop_pct=trail_pct)
        print(f"Selected trail: {selected_trail['name']}")

        # ---- Part 5: Entry frequency ----
        print("\n=== Part 5: Entry Frequency ===")
        freq_rows = []
        for name, freq in [("WEEKLY", "weekly"), ("BIWEEKLY", "biweekly")]:
            freq_rows.append(
                self._eval_named(
                    name,
                    work.with_overrides(rebalance_frequency=freq),
                    countries={"United States"},
                    ctx=ctx,
                    with_gross=True,
                )
            )
        freq_analysis = {"candidates": freq_rows, "selected": _pick_best(freq_rows)["name"]}
        selected_freq = next(r for r in freq_rows if r["name"] == freq_analysis["selected"])
        work = work.with_overrides(
            rebalance_frequency=selected_freq["config"]["rebalance_frequency"]
        )
        print(f"Selected freq: {selected_freq['name']}")

        # ---- Part 6: Score separation ----
        print("\n=== Part 6: Score Separation ===")
        sep_rows = [
            self._eval_named(
                "NO_SEP",
                work.with_overrides(use_score_separation_filter=False),
                countries={"United States"},
                ctx=ctx,
                with_gross=True,
            ),
            self._eval_named(
                "SCORE_SEP",
                work.with_overrides(use_score_separation_filter=True),
                countries={"United States"},
                ctx=ctx,
                with_gross=True,
            ),
        ]
        sep_analysis = {"candidates": sep_rows, "selected": _pick_best(sep_rows)["name"]}
        selected_sep = next(r for r in sep_rows if r["name"] == sep_analysis["selected"])
        work = work.with_overrides(
            use_score_separation_filter=selected_sep["config"]["use_score_separation_filter"]
        )
        print(f"Selected separation: {selected_sep['name']}")

        # ---- Part 7: Score buckets (diagnostic only) ----
        print("\n=== Part 7: Score Bucket Diagnostic ===")
        bucket_pack = self.diag._simulate_ranked(
            name="bucket_diag",
            rankings=ai_rankings,
            price_panel=price_panel,
            fx=fx,
            cfg=work.with_cost_preset("net"),
            countries={"United States"},
            fold_metas=fold_metas,
        )
        score_bucket_analysis = {
            "United States": trade_quality_by_entry_score(bucket_pack["trades"]),
        }
        for scope in scopes_diag:
            pack = self.diag._simulate_ranked(
                name=f"bucket_{scope}",
                rankings=ai_rankings,
                price_panel=price_panel,
                fx=fx,
                cfg=work.with_cost_preset("net"),
                countries={scope},
                fold_metas=fold_metas,
            )
            score_bucket_analysis[scope] = trade_quality_by_entry_score(pack["trades"])

        # ---- Part 8: Re-entry / cooldown ----
        print("\n=== Part 8: Re-entry / Cooldown ===")
        tr_df = trades_to_frame(bucket_pack["trades"])
        reentry_counts = count_reentries(tr_df)
        cool_rows = [
            self._eval_named(
                "NO_COOLDOWN",
                work.with_overrides(cooldown_days=0),
                countries={"United States"},
                ctx=ctx,
                with_gross=True,
            ),
            self._eval_named(
                "COOLDOWN_5",
                work.with_overrides(cooldown_days=5),
                countries={"United States"},
                ctx=ctx,
                with_gross=True,
            ),
        ]
        cool_analysis = {
            "reentry_counts_baseline": reentry_counts,
            "candidates": cool_rows,
            "selected": _pick_best(cool_rows)["name"],
        }
        selected_cool = next(r for r in cool_rows if r["name"] == cool_analysis["selected"])
        work = work.with_overrides(cooldown_days=selected_cool["config"]["cooldown_days"])
        print(
            f"Reentry5={reentry_counts.get('reentry_within_5_days')} "
            f"selected={selected_cool['name']}"
        )

        # ---- Final strategy set (≤6) ----
        final_names = {
            "BASE_US_CONCENTRATED": base_cfg,
            selected_stop["name"]: base_cfg.with_overrides(
                stop_loss_pct=stop_pct, take_profit_pct=0.10
            ),
            selected_hold["name"]: base_cfg.with_overrides(
                stop_loss_pct=stop_pct, take_profit_pct=tp_pct, holding_period_days=hold_days
            ),
            selected_freq["name"]: work.with_overrides(
                # ensure labeled
            ),
            "FINAL": work,
        }
        # Deduplicate by config fingerprint
        final_strategies: dict[str, SimulationConfig] = {}
        for n, c in final_names.items():
            final_strategies[n] = c
        if selected_trail["name"] != "FIXED_ONLY":
            final_strategies[selected_trail["name"]] = work
        if selected_sep["name"] == "SCORE_SEP":
            final_strategies["SCORE_SEP"] = work

        print("\n=== Final Strategy Comparison (US) ===")
        final_comparison: dict[str, Any] = {}
        fold_stability_out: dict[str, Any] = {}
        for name, cfg in final_strategies.items():
            row = self._eval_named(
                name, cfg, countries={"United States"}, ctx=ctx, with_gross=True, deep=True
            )
            # Momentum excess
            mom = self._eval_named(
                f"mom_vs_{name}",
                cfg,
                countries={"United States"},
                ctx={**ctx, "ai_rankings": mom_rankings},
                with_gross=False,
                label_rankings="momentum",
            )
            bench = self.diag.base._benchmark_metrics(
                panel, "United States", fold_metas=fold_metas, fx=fx
            )
            row["momentum_weekly_return"] = mom.get("net_return")
            row["momentum_excess"] = _sub(row.get("net_return"), mom.get("net_return"))
            row["benchmark_return"] = (bench or {}).get("mean_fold_total_return")
            row["benchmark_excess"] = _sub(
                row.get("net_return"), (bench or {}).get("mean_fold_total_return")
            )
            final_comparison[name] = row
            fold_stability_out[name] = row.get("fold_stability")
            print(
                f"  {name}: Net {_pct(row.get('net_return'))} | "
                f"MomEx {_pct(row.get('momentum_excess'))} | "
                f"Trades {row.get('trades')}"
            )

        # ---- Part 9: Cost sensitivity on FINAL ----
        print("\n=== Part 9: Cost Sensitivity ===")
        cost_sens = {}
        for preset in ("gross", "low", "net", "high"):
            cfg = work.with_cost_preset(preset)
            cost_sens[preset] = self._eval_named(
                f"FINAL_{preset}",
                cfg,
                countries={"United States"},
                ctx=ctx,
                with_gross=False,
                force_net_preset=False,
            )
            print(
                f"  Cost {preset}: Net {_pct(cost_sens[preset].get('net_return'))} "
                f"Cost={cost_sens[preset].get('total_cost'):,.0f}"
            )
        write_json(self.report_dir / "cost_sensitivity.json", cost_sens)

        # ---- Part 11/12 fold + regime for FINAL ----
        final_pack = self.diag._simulate_ranked(
            name="FINAL",
            rankings=ai_rankings,
            price_panel=price_panel,
            fx=fx,
            cfg=work.with_cost_preset("net"),
            countries={"United States"},
            fold_metas=fold_metas,
        )
        regime_by_date = self._regime_series(panel, "United States")
        regime_analysis = {
            "United States": regime_trade_analysis(final_pack["trades"], regime_by_date),
            "rule": {
                "Bull": "mkt_sma_60_ratio > 0 and mkt_return_20d > 0",
                "Bear": "mkt_sma_60_ratio < 0 and mkt_return_20d < 0",
                "Neutral": "otherwise",
            },
        }

        # Japan / Europe same FINAL rules
        country_transfer: dict[str, Any] = {}
        for scope in scopes_diag:
            country_transfer[scope] = self._eval_named(
                "FINAL",
                work,
                countries={scope},
                ctx=ctx,
                with_gross=True,
                deep=True,
            )
            bench = self.diag.base._benchmark_metrics(
                panel, scope, fold_metas=fold_metas, fx=fx
            )
            country_transfer[scope]["benchmark_return"] = (bench or {}).get(
                "mean_fold_total_return"
            )
            country_transfer[scope]["benchmark_excess"] = _sub(
                country_transfer[scope].get("net_return"),
                (bench or {}).get("mean_fold_total_return"),
            )
            regime_analysis[scope] = regime_trade_analysis(
                self.diag._simulate_ranked(
                    name=f"FINAL_{scope}",
                    rankings=ai_rankings,
                    price_panel=price_panel,
                    fx=fx,
                    cfg=work.with_cost_preset("net"),
                    countries={scope},
                    fold_metas=fold_metas,
                )["trades"],
                self._regime_series(panel, scope),
            )

        best_final = _pick_best(list(final_comparison.values()))
        overview = {
            "created_at_utc": _timestamp_iso(),
            "phase": "4C",
            "purpose": "Low-turnover / exit-rule optimization with fixed AI rankings",
            "base_strategy": "BASE_US_CONCENTRATED",
            "selected_path": {
                "stop": stop_analysis["selected"],
                "take_profit": tp_analysis["selected"],
                "holding": hold_analysis["selected"],
                "trailing": trail_analysis["selected"],
                "frequency": freq_analysis["selected"],
                "score_separation": sep_analysis["selected"],
                "cooldown": cool_analysis["selected"],
            },
            "final_config": {
                "stop_loss_pct": work.stop_loss_pct,
                "take_profit_pct": work.take_profit_pct,
                "holding_period_days": work.holding_period_days,
                "trailing_stop_pct": work.trailing_stop_pct,
                "rebalance_frequency": work.rebalance_frequency,
                "use_score_separation_filter": work.use_score_separation_filter,
                "cooldown_days": work.cooldown_days,
                "max_positions": work.max_positions,
            },
            "best_final_us": best_final,
            "beats_momentum": bool(
                (best_final.get("momentum_excess") or -1) > 0
            ),
            "beats_benchmark": bool((best_final.get("benchmark_excess") or -1) > 0),
            "stop_diagnostics_summary": {
                "aftermath": aftermath_report,
                "counterfactual": cf_report,
                "stop_effective": bool(
                    (cf_report.get("classification_rates") or {}).get("GOOD_STOP", 0)
                    > (cf_report.get("classification_rates") or {}).get("BAD_STOP", 0)
                ),
            },
            "country_transfer": country_transfer,
            "known_limitations": [
                "Staged selection uses mean-fold metrics; not a full factorial search.",
                "Stop aftermath/counterfactual are diagnostic only (no look-ahead into decisions).",
                "No Optuna / no new AI features.",
                "Paper trading only.",
            ],
        }

        write_json(self.report_dir / "stop_loss_analysis.json", stop_analysis)
        write_json(self.report_dir / "take_profit_analysis.json", tp_analysis)
        write_json(self.report_dir / "holding_period_analysis.json", hold_analysis)
        write_json(self.report_dir / "trailing_stop_analysis.json", trail_analysis)
        write_json(self.report_dir / "entry_frequency_analysis.json", freq_analysis)
        write_json(self.report_dir / "score_separation_analysis.json", sep_analysis)
        write_json(self.report_dir / "score_bucket_analysis.json", score_bucket_analysis)
        write_json(self.report_dir / "reentry_analysis.json", cool_analysis)
        write_json(self.report_dir / "final_strategy_comparison.json", final_comparison)
        write_json(self.report_dir / "fold_stability.json", fold_stability_out)
        write_json(self.report_dir / "regime_analysis.json", regime_analysis)
        write_json(self.report_dir / "summary.json", overview)
        print("\nRule Optimization Completed")
        return overview

    def _base_cfg(self) -> SimulationConfig:
        if "BASE_US_CONCENTRATED" in self.sim_cfg.strategies:
            return self.sim_cfg.with_strategy("BASE_US_CONCENTRATED")
        return self.sim_cfg.with_overrides(
            top_percentile=0.10,
            holding_period_days=10,
            rebalance_frequency="weekly",
            max_positions=5,
            max_position_weight=0.20,
            max_country_weight=1.0,
            stop_loss_pct=-0.05,
            take_profit_pct=0.10,
            ranking_exit_enabled=False,
        )

    def _eval_named(
        self,
        name: str,
        cfg: SimulationConfig,
        *,
        countries: set[str],
        ctx: dict[str, Any],
        with_gross: bool = False,
        deep: bool = False,
        label_rankings: str = "ai",
        force_net_preset: bool = True,
    ) -> dict[str, Any]:
        rankings = ctx["ai_rankings"] if label_rankings == "ai" else ctx["mom_rankings"]
        run_cfg = cfg
        if force_net_preset:
            try:
                run_cfg = cfg.with_cost_preset("net")
            except Exception:  # noqa: BLE001
                run_cfg = cfg

        pack = self.diag._simulate_ranked(
            name=name,
            rankings=rankings,
            price_panel=ctx["price_panel"],
            fx=ctx["fx"],
            cfg=run_cfg,
            countries=countries,
            fold_metas=ctx["fold_metas"],
        )
        perf = pack["aggregate"]["performance"]
        fold_rets = []
        fold_sharpes = []
        fold_details = []
        for f in pack["folds"]:
            if "performance" not in f:
                continue
            p = f["performance"]
            fold_rets.append(p.get("total_return"))
            fold_sharpes.append(p.get("sharpe"))
            if deep:
                fold_details.append(
                    {
                        "fold": f.get("fold"),
                        "net_return": p.get("total_return"),
                        "sharpe": p.get("sharpe"),
                        "max_dd": p.get("max_drawdown"),
                        "trades": p.get("number_of_trades"),
                        "transaction_costs": p.get("transaction_costs"),
                    }
                )
        stab = fold_stability(fold_rets, fold_sharpes)
        tr = trades_to_frame(pack["trades"])
        nets = tr["Net PnL"].astype(float) if not tr.empty else pd.Series(dtype=float)
        wins = nets[nets > 0]
        losses = nets[nets <= 0]
        gp = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else None
        avg_win = float(wins.mean()) if len(wins) else None
        row: dict[str, Any] = {
            "name": name,
            "config": {
                "stop_loss_pct": cfg.stop_loss_pct,
                "take_profit_pct": cfg.take_profit_pct,
                "holding_period_days": cfg.holding_period_days,
                "trailing_stop_pct": cfg.trailing_stop_pct,
                "rebalance_frequency": cfg.rebalance_frequency,
                "use_score_separation_filter": cfg.use_score_separation_filter,
                "cooldown_days": cfg.cooldown_days,
                "max_positions": cfg.max_positions,
            },
            "gross_return_mean_fold": None,
            "net_return": perf.get("mean_fold_total_return"),
            "sharpe": perf.get("mean_fold_sharpe"),
            "max_dd": perf.get("mean_fold_max_drawdown"),
            "cagr": perf.get("mean_fold_cagr"),
            "win_rate": perf.get("win_rate"),
            "profit_factor": float(gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else None),
            "trades": perf.get("number_of_trades"),
            "average_win": avg_win,
            "average_loss": avg_loss,
            "average_holding_days": perf.get("average_holding_days"),
            "total_cost": float(pack.get("total_commission", 0) + pack.get("total_slippage", 0)),
            "commission": float(pack.get("total_commission") or 0),
            "slippage": float(pack.get("total_slippage") or 0),
            "positive_fold_ratio": stab.get("positive_fold_ratio"),
            "fold_stability": stab,
            "turnover": turnover_analysis(
                pack["trades"],
                pack["equity_curve"],
                initial_capital=cfg.initial_capital,
                rebalance_stats=pack.get("rebalance_stats"),
            ),
            "exit_reasons": exit_reason_analysis(pack["trades"]),
        }
        if deep:
            row["folds"] = fold_details
        if with_gross:
            g_cfg = cfg.with_cost_preset("gross")
            g_pack = self.diag._simulate_ranked(
                name=f"{name}_gross",
                rankings=rankings,
                price_panel=ctx["price_panel"],
                fx=ctx["fx"],
                cfg=g_cfg,
                countries=countries,
                fold_metas=ctx["fold_metas"],
            )
            g_perf = g_pack["aggregate"]["performance"]
            row["gross_return"] = g_perf.get("mean_fold_total_return")
            row["cost_decomposition"] = cost_decomposition(
                gross_perf=g_perf,
                net_perf=perf,
                low_perf=None,
                initial_capital=cfg.initial_capital,
                commission_total=float(pack.get("total_commission") or 0),
                slippage_total=float(pack.get("total_slippage") or 0),
            )
        return row

    def _regime_series(self, panel: pd.DataFrame, region: str) -> pd.Series:
        frame = panel.loc[panel["Region"] == region].copy()
        if frame.empty or "mkt_sma_60_ratio" not in frame.columns:
            return pd.Series(dtype=object)
        daily = (
            frame.groupby("Date", sort=True)[["mkt_sma_60_ratio", "mkt_return_20d"]]
            .mean()
            .dropna()
        )
        if daily.empty:
            return pd.Series(dtype=object)
        regimes = classify_market_regime(daily["mkt_sma_60_ratio"], daily["mkt_return_20d"])
        regimes.index = pd.to_datetime(daily.index)
        return regimes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4C low-turnover / exit rule optimization")
    parser.add_argument("--universe", type=Path, default=Path("config/universe.global100.json"))
    parser.add_argument("--config", type=Path, default=Path("config/rule_optimization.json"))
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    sim_cfg = load_simulation_config(args.config)
    try:
        RuleOptimizationRunner(settings, args.universe, sim_cfg).run(
            force_refresh=args.force_refresh
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Rule optimization failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
