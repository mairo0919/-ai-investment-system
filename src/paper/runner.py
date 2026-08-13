"""Phase 5: forward paper trading CLI (frozen FINAL US strategy, no brokerage)."""

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

from src.config.settings import PROJECT_ROOT, Settings, get_settings
from src.core.exceptions import TrainingError
from src.ml.ltr_baselines import momentum_scores
from src.paper.atomic_io import atomic_write_json
from src.paper.cost_audit import cost_monotonicity_report, path_dependence_note
from src.paper.diagnostics import (
    daily_risk_snapshot,
    gap_alerts_for_path,
    performance_summary,
    position_mae_mfe,
    score_diagnostics_from_trades,
)
from src.paper.lineage import (
    LOCKED_PAPER_MODEL_ID,
    STRATEGY_CONFIG_ID,
    assert_paper_model_id_locked,
    separate_experiment_performance,
)
from src.paper.model_freeze import (
    FrozenRankerStore,
    score_panel_with_frozen,
    train_and_freeze_ranker,
)
from src.paper.state import (
    PaperOrderRecord,
    PaperState,
    PaperStore,
    fill_idempotency_key,
    order_idempotency_key,
)
from src.simulation.config import load_simulation_config
from src.simulation.engine import SimulationEngine
from src.simulation.holdout_runner import load_true_holdout_bundle
from src.simulation.metrics import buy_and_hold_benchmark
from src.simulation.ranking import attach_country_ranks
from src.simulation.runner import SimulationRunner
from src.utils.logging import setup_logging
from src.utils.runtime_diag import (
    current_phase,
    install_atexit_hook,
    install_signal_handlers,
    log_diag,
    log_process_boot,
    log_top_level_exception,
    mark_exit_status,
    phase_span,
    set_phase,
)

logger = logging.getLogger(__name__)

CONFIG_ID = STRATEGY_CONFIG_ID
COUNTRY = "United States"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_paper_config(path: Path) -> tuple[dict[str, Any], Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Resolve FINAL via true_holdout chain
    holdout_path = PROJECT_ROOT / "config" / "true_holdout.json"
    _, cfg = load_true_holdout_bundle(holdout_path)
    # Ensure paper extends don't loosen strategy
    if not raw.get("immutable_strategy", True):
        raise TrainingError("paper_trading.json must keep immutable_strategy=true")
    return raw, cfg


class PaperTradingRunner:
    """Daily forward paper workflow for frozen FINAL US strategy."""

    def __init__(
        self,
        settings: Settings,
        universe_path: Path,
        config_path: Path,
    ) -> None:
        self.settings = settings
        self.universe_path = universe_path
        self.config_path = config_path
        self.raw, self.cfg = load_paper_config(config_path)
        self.forward_start = pd.Timestamp(self.raw["forward"]["start"]).normalize()
        if not self.raw.get("forward", {}).get("locked"):
            raise TrainingError("Forward start must be locked=true")
        self.country = str(self.raw.get("country") or COUNTRY)
        self.store = PaperStore(settings.project_root / "data" / "paper")
        self.report_dir = settings.reports_dir / "paper"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / "rankings").mkdir(parents=True, exist_ok=True)
        (self.report_dir / "monthly").mkdir(parents=True, exist_ok=True)
        self.model_store = FrozenRankerStore(settings.models_dir)
        self.base = SimulationRunner(settings, universe_path, self.cfg)
        self.max_stale_days = int(
            self.raw.get("data_freshness", {}).get("max_stale_calendar_days", 5)
        )

    def run(self, *, force_refresh: bool = False) -> dict[str, Any]:
        print("Paper Trading Started (NO BROKERAGE)")
        print("FINAL strategy is IMMUTABLE — no retuning / no retrain.")
        log_diag("paper_run_begin")

        try:
            print("[1/8] Updating market data...")
            # market_data_update / panel_build / cross_section / macro logged inside
            # LabelResolutionRunner._build_enriched_panel (diag only).
            with phase_span("panel_enrichment"):
                panel = self.base.lr._build_enriched_panel(force_refresh=force_refresh)
            with phase_span("fx_and_price_panel"):
                fx = self.base._build_fx(force_refresh=force_refresh)
                price_panel = self.base._price_panel(panel)

            print("[2/8] Generating features... (included in panel)")
            us_px = price_panel.loc[price_panel["Country"] == self.country].copy()
            if us_px.empty:
                raise TrainingError("No US price rows")

            asof = self._latest_usable_asof(us_px)
            print(f"  As-of close date: {asof.date()}")

            print("[3/8] Loading model...")
            with phase_span("model_load"):
                model, model_meta = self._ensure_frozen_model(panel)
            model_id = str(model_meta["model_id"])
            assert_paper_model_id_locked(model_id, locked_id=LOCKED_PAPER_MODEL_ID)

            print("[4/8] Creating rankings...")
            feature_cols = list(model_meta["feature_list"])
            with phase_span("model_inference"):
                scored = score_panel_with_frozen(
                    model=model,
                    panel=panel,
                    feature_cols=feature_cols,
                    start=self.forward_start,
                    end=asof,
                )
            if scored.empty:
                raise TrainingError("No scored rows in forward window")
            with phase_span("signal_generation"):
                country_col = "Region" if "Region" in scored.columns else "Country"
                ai_rankings = attach_country_ranks(scored, country_col=country_col)
                ai_rankings = ai_rankings.loc[ai_rankings["Country"] == self.country].copy()

                mom = panel.copy()
                mom["Date"] = pd.to_datetime(mom["Date"]).dt.normalize()
                region_col = "Region" if "Region" in mom.columns else "Country"
                mom = mom.loc[
                    (mom["Date"] >= self.forward_start)
                    & (mom["Date"] <= asof)
                    & (mom[region_col] == self.country)
                ].dropna(subset=["return_20d"]).copy()
                mom["score"] = momentum_scores(mom, feature_col="return_20d").to_numpy()
                mom_rankings = attach_country_ranks(mom, country_col=region_col)

            print("[5/8] Evaluating portfolio...")
            state = self._load_or_init_state(model_id=model_id)
            if state.last_processed_date and pd.Timestamp(state.last_processed_date) >= asof:
                print(f"  Idempotent skip: already processed through {state.last_processed_date}")
                with phase_span("state_save"):
                    latest = self._write_reports(
                        state=state,
                        asof=asof,
                        ai_rankings=ai_rankings,
                        mom_rankings=mom_rankings,
                        price_panel=us_px,
                        fx=fx,
                        model_meta=model_meta,
                        skipped=True,
                    )
                print("[8/8] Saving report... (unchanged)")
                with phase_span("run_completed"):
                    print("Completed.")
                    log_diag("paper_run_completed_idempotent_skip")
                return latest

            # Need price buffer before forward for calendars continuity when resuming
            sim_start = self.forward_start
            if state.last_processed_date:
                # Resume from next day after last processed
                sim_start = pd.Timestamp(state.last_processed_date) + pd.tseries.offsets.BDay(1)
                sim_start = pd.Timestamp(sim_start).normalize()

            # Include one prior day so open fills can process if pending from previous close
            px_window = us_px.loc[
                (us_px["Date"] >= self.forward_start - pd.tseries.offsets.BDay(5))
                & (us_px["Date"] <= asof + pd.tseries.offsets.BDay(1))
            ].copy()
            rk_window = ai_rankings.loc[
                (ai_rankings["Date"] >= self.forward_start) & (ai_rankings["Date"] <= asof)
            ].copy()

            if sim_start > asof:
                print("  No new trading days to process.")
                latest = self._write_reports(
                    state=state,
                    asof=asof,
                    ai_rankings=ai_rankings,
                    mom_rankings=mom_rankings,
                    price_panel=us_px,
                    fx=fx,
                    model_meta=model_meta,
                    skipped=True,
                )
                print("Completed.")
                return latest

            print("[6/8] Creating paper orders...")
            with phase_span("order_decision"):
                portfolio = state.to_portfolio()
                # Preserve closed trade history separately; engine portfolio.trades starts empty
                n_trades_before = 0
                if self.store.trade_history_path.exists():
                    n_trades_before = len(pd.read_csv(self.store.trade_history_path))

                pending_engine = [
                    o.to_engine_order() for o in state.pending_orders if o.status == "PENDING"
                ]
                cooldown = {
                    k: pd.Timestamp(v) for k, v in state.cooldown_until.items()
                }

                # Track order keys before run
                pre_keys = set(state.seen_order_keys)

                engine = SimulationEngine(self.cfg, fx=fx, countries={self.country})
                # Calendar must include days from sim_start; for first run need >=2 days ideally
                cal_days = sorted(px_window.loc[px_window["Date"] >= sim_start, "Date"].unique())
                min_days = 1 if state.last_processed_date else 2
                if len(cal_days) < min_days and len(cal_days) >= 1:
                    min_days = 1
                if not len(cal_days):
                    raise TrainingError("Empty simulation calendar for catch-up")

                result = engine.run(
                    prices=px_window,
                    rankings=rk_window,
                    start=sim_start,
                    end=asof,
                    portfolio=portfolio,
                    pending=pending_engine,
                    cooldown_until=cooldown,
                    min_calendar_days=min_days,
                )

            # Sync state
            assert result.portfolio is not None
            new_trades = result.portfolio.trades
            state.sync_from_portfolio(result.portfolio)
            state.model_id = model_id
            state.config_id = CONFIG_ID
            state.last_processed_date = str(asof.date())
            state.cooldown_until = dict(result.meta.get("cooldown_until") or {})

            # Rebuild pending paper orders from engine pending + idempotency
            new_pending: list[PaperOrderRecord] = []
            audit_rows: list[dict[str, Any]] = []
            for o in result.meta.get("pending_orders") or []:
                key = order_idempotency_key(
                    signal_date=o.signal_date,
                    symbol=o.symbol,
                    side=o.side,
                    reason=o.reason,
                )
                if key in pre_keys and key in state.seen_order_keys:
                    # already recorded previously
                    pass
                state.seen_order_keys.add(key)
                rec = PaperOrderRecord(
                    order_id=key,
                    symbol=o.symbol,
                    country=o.country,
                    currency=o.currency,
                    side=o.side,
                    quantity=float(o.quantity),
                    signal_date=str(pd.Timestamp(o.signal_date).date()),
                    reason=o.reason,
                    status="PENDING",
                    order_type=o.order_type,
                    limit_or_stop_price=o.limit_or_stop_price,
                    score=o.score,
                    rank=o.rank,
                    percentile=o.percentile,
                    entry_date=str(pd.Timestamp(o.entry_date).date()) if o.entry_date is not None else None,
                    entry_price=o.entry_price,
                    entry_score=o.entry_score,
                    entry_rank=o.entry_rank,
                    model_id=model_id,
                    config_id=CONFIG_ID,
                    idempotency_key=key,
                )
                new_pending.append(rec)
                audit_rows.append(
                    {
                        "Date": rec.signal_date,
                        "Symbol": rec.symbol,
                        "Action": rec.side.upper(),
                        "Reason": rec.reason,
                        "Score": rec.score,
                        "Rank": rec.rank,
                        "PriceReference": "CLOSE_SIGNAL_NEXT_OPEN_FILL",
                        "ModelID": model_id,
                        "ConfigID": CONFIG_ID,
                        "idempotency_key": key,
                        "Status": "PENDING",
                    }
                )
            state.pending_orders = new_pending

            # Persist new closed trades with fill keys
            trade_rows = []
            for t in new_trades:
                d = t.to_dict()
                fkey = fill_idempotency_key(
                    fill_date=t.exit_date,
                    symbol=t.symbol,
                    side="sell",
                    reason=t.exit_reason,
                )
                if fkey in state.seen_fill_keys:
                    continue
                state.seen_fill_keys.add(fkey)
                d["idempotency_key"] = fkey
                d["model_id"] = model_id
                d["config_id"] = CONFIG_ID
                trade_rows.append(d)

            print("[7/8] Updating equity...")
            for ep in result.equity_curve:
                self.store.append_equity(
                    {
                        "Date": str(pd.Timestamp(ep.date).date()),
                        "Cash": ep.cash,
                        "Position Value": ep.position_value,
                        "Total Equity": ep.total_equity,
                        "Drawdown": ep.drawdown,
                        "Realized PnL": ep.realized_pnl,
                        "Unrealized PnL": ep.unrealized_pnl,
                        "N Positions": ep.n_positions,
                        "Transaction Costs": state.total_commission + state.total_slippage_impact,
                    }
                )

            # Momentum + benchmark parallel tracking (equity series)
            self._update_benchmark_tracks(
                us_px=us_px,
                mom_rankings=mom_rankings,
                fx=fx,
                asof=asof,
                state_model_id=model_id,
            )

            print("[8/8] Saving report...")
            with phase_span("state_save"):
                # Atomic save of portfolio state LAST after reports built from memory
                self.store.append_trades(trade_rows)
                self.store.append_audit(audit_rows)
                self.store.save_atomic(state)

                latest = self._write_reports(
                    state=state,
                    asof=asof,
                    ai_rankings=ai_rankings,
                    mom_rankings=mom_rankings,
                    price_panel=us_px,
                    fx=fx,
                    model_meta=model_meta,
                    skipped=False,
                )
            self._print_recommendation(latest)
            with phase_span("run_completed"):
                print("Completed.")
                print("No actual brokerage orders were submitted.")
                log_diag("paper_run_completed")
            return latest

        except Exception as exc:
            logger.exception(
                "Paper trading failed in phase=%s; no partial state write attempted after error path",
                current_phase(),
            )
            # Failure safety: do not save partial — we only save_atomic after successful engine run
            raise TrainingError(f"Paper trading aborted safely: {exc}") from exc

    def _ensure_frozen_model(self, panel: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
        feature_cols = list(self.base.lr.feature_sets[self.cfg.feature_set])
        params = dict(self.base.lr.param_presets[self.cfg.param_preset])
        freeze_cfg = self.raw.get("model_freeze") or {}
        meta = train_and_freeze_ranker(
            panel=panel,
            cfg=self.cfg,
            feature_cols=feature_cols,
            params=params,
            forward_start=self.forward_start,
            purge_days=int(freeze_cfg.get("purge_days", 5)),
            early_stopping_valid_days=int(freeze_cfg.get("early_stopping_valid_days", 252)),
            store=self.model_store,
            project_root=self.settings.project_root,
        )
        model, meta2 = self.model_store.load()
        return model, meta2

    def _load_or_init_state(self, *, model_id: str) -> PaperState:
        existing = self.store.load()
        if existing is not None:
            if existing.model_id and existing.model_id != model_id:
                raise TrainingError(
                    f"State model_id {existing.model_id} != frozen {model_id} (retrain forbidden)"
                )
            return existing
        state = PaperState(
            cash=float(self.cfg.initial_capital),
            base_currency=self.cfg.base_currency,
            peak_equity=float(self.cfg.initial_capital),
            forward_start=str(self.forward_start.date()),
            model_id=model_id,
            config_id=CONFIG_ID,
            brokerage_orders_submitted=0,
        )
        self.store.save_atomic(state)
        self.store.ensure_history_files()
        return state

    def _latest_usable_asof(self, us_px: pd.DataFrame) -> pd.Timestamp:
        us_px = us_px.copy()
        us_px["Date"] = pd.to_datetime(us_px["Date"]).dt.normalize()
        # Require Open+Close present for majority of US symbols
        last = us_px["Date"].max()
        if pd.isna(last):
            raise TrainingError("No dates in US price panel")
        last = pd.Timestamp(last).normalize()
        # Stale check vs today
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        if (today - last).days > self.max_stale_days:
            raise TrainingError(
                f"STALE_DATA: last market date {last.date()} older than "
                f"{self.max_stale_days} calendar days — refusing orders"
            )
        # Per-symbol freshness on last day
        day = us_px.loc[us_px["Date"] == last]
        if day.empty:
            raise TrainingError("ERROR: empty as-of day")
        missing = day.loc[day["Close"].isna() | day["Open"].isna()]
        if len(missing) == len(day):
            raise TrainingError("ERROR: all symbols missing OHLC on as-of day")
        return last

    def _update_benchmark_tracks(
        self,
        *,
        us_px: pd.DataFrame,
        mom_rankings: pd.DataFrame,
        fx: Any,
        asof: pd.Timestamp,
        state_model_id: str,
    ) -> None:
        # S&P buy&hold from forward start
        ticker = self.cfg.benchmarks.get(self.country) or "^GSPC"
        frame = self.base.lr.wf.cache.load(ticker)
        bench_path = self.report_dir / "benchmark_equity.csv"
        mom_path = self.report_dir / "momentum_equity.csv"
        if frame is not None and not frame.empty:
            s = frame.copy()
            s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
            s = s.loc[(s["Date"] >= self.forward_start) & (s["Date"] <= asof)]
            series = s.set_index("Date")["Close"].astype(float).sort_index()
            vals = []
            for idx, px in series.items():
                try:
                    vals.append(float(px) * fx.rate_to_base("USD", idx))
                except Exception:  # noqa: BLE001
                    vals.append(np.nan)
            series_jpy = pd.Series(vals, index=series.index, dtype=float).dropna()
            bh = buy_and_hold_benchmark(series_jpy, initial_capital=self.cfg.initial_capital)
            eq = pd.DataFrame(bh.get("equity_curve") or [])
            if not eq.empty:
                atomic_write_json(
                    self.report_dir / "benchmark_track.json",
                    {"ticker": ticker, "total_return": bh.get("total_return"), "asof": str(asof.date())},
                )
                eq.to_csv(bench_path, index=False)

        # Momentum paper: full forward window each time from cash (tracking curve, not shared portfolio)
        mom_state_path = self.report_dir / "momentum_state.json"
        engine = SimulationEngine(self.cfg, fx=fx, countries={self.country})
        px_window = us_px.loc[
            (us_px["Date"] >= self.forward_start - pd.tseries.offsets.BDay(5))
            & (us_px["Date"] <= asof + pd.tseries.offsets.BDay(1))
        ]
        rk = mom_rankings.loc[
            (mom_rankings["Date"] >= self.forward_start) & (mom_rankings["Date"] <= asof)
        ]
        try:
            res = engine.run(
                prices=px_window,
                rankings=rk,
                start=self.forward_start,
                end=asof,
                min_calendar_days=1,
            )
            from src.simulation.report import equity_to_frame

            eqm = equity_to_frame(res.equity_curve)
            if not eqm.empty:
                eqm.to_csv(mom_path, index=False)
            atomic_write_json(
                mom_state_path,
                {
                    "total_return": (
                        float(eqm["Total Equity"].iloc[-1] / self.cfg.initial_capital - 1.0)
                        if not eqm.empty
                        else None
                    ),
                    "asof": str(asof.date()),
                    "model_id": "momentum_baseline",
                    "config_id": CONFIG_ID,
                    "ref_ai_model_id": state_model_id,
                },
            )
        except TrainingError as exc:
            logger.warning("Momentum track skipped: %s", exc)

    def _write_reports(
        self,
        *,
        state: PaperState,
        asof: pd.Timestamp,
        ai_rankings: pd.DataFrame,
        mom_rankings: pd.DataFrame,
        price_panel: pd.DataFrame,
        fx: Any,
        model_meta: dict[str, Any],
        skipped: bool,
    ) -> dict[str, Any]:
        # Ranking snapshot for asof (and rebalance days already processed — save asof always)
        day_rank = ai_rankings.loc[ai_rankings["Date"] == asof].copy()
        held = set(state.positions)
        if not day_rank.empty:
            # Percentile = rank/n; Top 10% <=> Percentile <= 0.10
            day_rank["Selected"] = day_rank["Percentile"] <= float(self.cfg.top_percentile)
            day_rank["CurrentPosition"] = day_rank["Symbol"].isin(held)
            out = day_rank[
                ["Date", "Symbol", "Score", "Rank", "Percentile", "Selected", "CurrentPosition"]
            ].copy()
            out["Date"] = out["Date"].map(lambda d: str(pd.Timestamp(d).date()))
            out.to_csv(self.report_dir / "rankings" / f"{asof.date()}.csv", index=False)

        # Risk + MAE/MFE
        risk = daily_risk_snapshot(
            asof=asof,
            cash=state.cash,
            positions=state.positions,
            peak_equity=state.peak_equity,
            realized_pnl=state.realized_pnl,
        )
        tail = {}
        for sym, pos in state.positions.items():
            path = price_panel.loc[
                (price_panel["Symbol"] == sym)
                & (price_panel["Date"] >= pos.entry_date)
                & (price_panel["Date"] <= asof)
            ].sort_values("Date")
            mm = position_mae_mfe(pos, path=path)
            gaps = gap_alerts_for_path(path)
            tail[sym] = {**mm, "gap_events": gaps}

        trades = (
            pd.read_csv(self.store.trade_history_path)
            if self.store.trade_history_path.exists()
            else pd.DataFrame()
        )
        score_diag = score_diagnostics_from_trades(trades)

        eq_hist = (
            pd.read_csv(self.store.equity_history_path)
            if self.store.equity_history_path.exists()
            else pd.DataFrame()
        )
        bench_ret = None
        mom_ret = None
        bt = self.report_dir / "benchmark_track.json"
        mt = self.report_dir / "momentum_state.json"
        if bt.exists():
            bench_ret = json.loads(bt.read_text()).get("total_return")
        if mt.exists():
            mom_ret = json.loads(mt.read_text()).get("total_return")
        perf = performance_summary(
            eq_hist,
            trades,
            initial_capital=self.cfg.initial_capital,
            benchmark_return=bench_ret,
            momentum_return=mom_ret,
        )

        # Cost audit on closed trades if raw-like columns exist
        cost_audit = {
            "path_dependence": path_dependence_note(),
            "fixed_quantity": None,
        }
        if not trades.empty:
            # Build synthetic raw prices from stored entry/exit (diagnostic)
            t2 = trades.copy()
            if "Entry Price" in t2.columns and "Exit Price" in t2.columns:
                t2["Entry Price Raw"] = t2["Entry Price"].astype(float) / (
                    1.0 + float(self.cfg.slippage_rate)
                )
                t2["Exit Price Raw"] = t2["Exit Price"].astype(float) / (
                    1.0 - float(self.cfg.slippage_rate)
                )
                if "Quantity" not in t2.columns and "quantity" in t2.columns:
                    t2["Quantity"] = t2["quantity"]
                cost_audit["fixed_quantity"] = cost_monotonicity_report(t2)

        # Monthly snapshot if month-end crossed
        if not eq_hist.empty:
            self._maybe_write_monthly(eq_hist, bench_ret, mom_ret, trades)

        # Current recommendation
        holdings = []
        for p in state.positions.values():
            rank = None
            if not day_rank.empty:
                hit = day_rank.loc[day_rank["Symbol"] == p.symbol]
                if not hit.empty:
                    rank = int(hit.iloc[0]["Rank"])
            pnl_pct = (
                p.current_price / p.entry_price - 1.0 if p.entry_price else None
            )
            holdings.append(
                {
                    "symbol": p.symbol,
                    "action": "HOLD" if not p.pending_exit_reason else f"EXIT_PENDING:{p.pending_exit_reason}",
                    "rank": rank,
                    "holding_days": p.holding_days,
                    "holding_limit": self.cfg.holding_period_days,
                    "pnl_pct": pnl_pct,
                    "entry_score": p.entry_score,
                }
            )
        next_orders = [
            {
                "side": o.side.upper(),
                "symbol": o.symbol,
                "reason": o.reason,
                "expected_execution": "Next Open",
                "status": o.status,
            }
            for o in state.pending_orders
            if o.status == "PENDING"
        ]

        holdout_perf = self._load_phase4d_holdout_reference()
        experiments = separate_experiment_performance(
            phase5_forward=perf,
            phase4d_holdout=holdout_perf,
        )
        latest = {
            "created_at_utc": _utc_now(),
            "asof": str(asof.date()),
            "forward_start": str(self.forward_start.date()),
            "paper_start_date": str(self.forward_start.date()),
            "skipped_idempotent": skipped,
            "model_id": state.model_id or LOCKED_PAPER_MODEL_ID,
            "strategy_id": CONFIG_ID,
            "config_id": CONFIG_ID,
            "is_same_model_as_true_holdout": False,
            "brokerage_orders_submitted": 0,
            "brokerage_enabled": False,
            "current_portfolio": holdings,
            "next_paper_orders": next_orders,
            "portfolio": state.portfolio_json(),
            # Phase 5 only — never a stitched Phase4D+Phase5 cumulative series
            "performance": perf,
            "experiments": experiments,
            "tail_risk_open": tail,
            "note": (
                "No actual brokerage orders were submitted. "
                "Phase 4D True Holdout and Phase 5 Forward Paper are separate experiments; "
                "do not concatenate cumulative returns. "
                "Paper model is not the True Holdout model."
            ),
        }
        atomic_write_json(self.report_dir / "latest.json", latest)
        atomic_write_json(
            self.report_dir / "performance.json",
            {
                "experiment_id": "phase5_forward_paper",
                "model_id": state.model_id or LOCKED_PAPER_MODEL_ID,
                "strategy_id": CONFIG_ID,
                "paper_start_date": str(self.forward_start.date()),
                "is_same_model_as_true_holdout": False,
                "combined_with_phase4d_forbidden": True,
                "phase5_forward_paper": perf,
                "phase4d_true_holdout_reference_only": holdout_perf,
                "experiments": experiments,
            },
        )
        atomic_write_json(self.report_dir / "score_diagnostics.json", score_diag)
        atomic_write_json(self.report_dir / "risk_snapshot.json", {"daily": risk, "positions": tail})
        atomic_write_json(self.report_dir / "cost_audit.json", cost_audit)
        atomic_write_json(
            self.report_dir / "model_freeze.json",
            {k: model_meta.get(k) for k in model_meta},
        )
        return latest

    def _load_phase4d_holdout_reference(self) -> dict[str, Any] | None:
        path = self.settings.reports_dir / "true_holdout" / "holdout_summary.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            us = (raw.get("holdout_performance") or {}).get("United States") or {}
            train = raw.get("train_info") or {}
            return {
                "label": "phase4d_true_holdout_reference_not_paper",
                "is_same_model_as_paper": False,
                "holdout_period": {
                    "start": (raw.get("holdout") or {}).get("holdout_start"),
                    "end": (raw.get("holdout") or {}).get("holdout_end"),
                },
                "training_period": {
                    "start": train.get("train_start"),
                    "end": train.get("train_end"),
                },
                "us_metrics": {
                    "total_return": us.get("total_return"),
                    "cagr": us.get("cagr"),
                    "sharpe": us.get("sharpe"),
                    "max_drawdown": us.get("max_drawdown"),
                    "profit_factor": us.get("profit_factor"),
                    "win_rate": us.get("win_rate"),
                    "number_of_trades": us.get("number_of_trades"),
                },
                "note": (
                    "Reference only. Different model fit (train end 2023-09-12) "
                    "from Paper model (train end 2025-08-05)."
                ),
            }
        except Exception:  # noqa: BLE001
            return None

    def _maybe_write_monthly(
        self,
        eq_hist: pd.DataFrame,
        bench_ret: float | None,
        mom_ret: float | None,
        trades: pd.DataFrame,
    ) -> None:
        eq = eq_hist.copy()
        eq["Date"] = pd.to_datetime(eq["Date"])
        eq = eq.sort_values("Date")
        monthly = eq.set_index("Date")["Total Equity"].resample("ME").last().dropna()
        if monthly.empty:
            return
        rets = monthly.pct_change().dropna()
        for dt, r in rets.items():
            month = pd.Timestamp(dt).strftime("%Y-%m")
            path = self.report_dir / "monthly" / f"{month}.json"
            # Approximate month trade count
            n_tr = 0
            costs = 0.0
            if trades is not None and not trades.empty and "Exit Date" in trades.columns:
                td = pd.to_datetime(trades["Exit Date"])
                m = trades.loc[td.dt.strftime("%Y-%m") == month]
                n_tr = int(len(m))
                if "Commission Base" in m.columns:
                    costs = float(m["Commission Base"].astype(float).sum())
            atomic_write_json(
                path,
                {
                    "month": month,
                    "ai_return": float(r),
                    "sp500_return": None,
                    "momentum_return": None,
                    "benchmark_excess": None,
                    "momentum_excess": None,
                    "trades": n_tr,
                    "costs": costs,
                    "note": "Benchmark month legs filled when monthly series available",
                    "forward_benchmark_total": bench_ret,
                    "forward_momentum_total": mom_ret,
                },
            )

    def _print_recommendation(self, latest: dict[str, Any]) -> None:
        print("\n===== Current Portfolio =====")
        for h in latest.get("current_portfolio") or []:
            pnl = h.get("pnl_pct")
            pnl_s = f"{pnl*100:+.1f}%" if pnl is not None else "n/a"
            print(
                f"{h['symbol']}\n"
                f"{h['action']}\n"
                f"Rank {h.get('rank')}\n"
                f"Holding {h.get('holding_days')}/{h.get('holding_limit')} days\n"
                f"PnL {pnl_s}\n"
            )
        print("===== Next Paper Orders =====")
        orders = latest.get("next_paper_orders") or []
        if not orders:
            print("(none)")
        for o in orders:
            print(f"{o['side']} {o['symbol']}  Reason: {o['reason']}  Expected: {o['expected_execution']}")
        print("\nNo actual brokerage orders were submitted.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 forward paper trading (no brokerage)")
    parser.add_argument("--universe", type=Path, default=Path("config/universe.global100.json"))
    parser.add_argument("--config", type=Path, default=Path("config/paper_trading.json"))
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level)
    install_signal_handlers()
    install_atexit_hook()
    log_process_boot()
    set_phase("main")
    try:
        with phase_span("paper_trading_main"):
            PaperTradingRunner(settings, args.universe, args.config).run(
                force_refresh=args.force_refresh
            )
        mark_exit_status("ok")
        log_diag("main_return_ok")
        return 0
    except BaseException as exc:  # noqa: BLE001 — diagnose then re-raise SystemExit path
        if isinstance(exc, SystemExit):
            mark_exit_status(f"system_exit:{exc.code}")
            raise
        mark_exit_status(f"error:{type(exc).__name__}")
        log_top_level_exception(exc)
        logger.exception("run-paper-trading failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        log_diag(f"main_finally status_pending_atexit phase={current_phase()}")


if __name__ == "__main__":
    raise SystemExit(main())
