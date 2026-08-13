"""Phase 4D true holdout: no overlap, frozen FINAL, diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions import TrainingError
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
from src.simulation.holdout_runner import (
    assert_holdout_no_overlap,
    load_true_holdout_bundle,
)
from src.simulation.orders import EquityPoint, TradeRecord


def test_holdout_period_no_overlap_with_selection() -> None:
    raw = json.loads(Path("config/true_holdout.json").read_text(encoding="utf-8"))
    h = raw["holdout"]
    start = pd.Timestamp(h["start"])
    prior = pd.Timestamp(h["prior_selection_last_valid_end"])
    assert_holdout_no_overlap(start, prior)
    assert h["locked"] is True
    with pytest.raises(TrainingError):
        assert_holdout_no_overlap(prior, prior)


def test_final_config_immutable_values() -> None:
    raw, cfg = load_true_holdout_bundle(Path("config/true_holdout.json"))
    final = raw["final_strategy"]
    assert final["immutable"] is True
    assert cfg.rebalance_frequency == "biweekly"
    assert cfg.holding_period_days == 20
    assert cfg.take_profit_pct == pytest.approx(0.15)
    assert cfg.stop_loss_pct is None
    assert cfg.cooldown_days == 5
    assert cfg.max_positions == 5
    assert cfg.commission_rate == pytest.approx(0.001)
    assert cfg.slippage_rate == pytest.approx(0.0005)
    assert cfg.ranking_exit_enabled is False


def test_holdout_not_in_fit_window_logic() -> None:
    """Pre-holdout train/ES windows must end before holdout start."""
    raw = json.loads(Path("config/true_holdout.json").read_text(encoding="utf-8"))
    h0 = pd.Timestamp(raw["holdout"]["start"])
    prior = pd.Timestamp(raw["holdout"]["prior_selection_last_valid_end"])
    purge = int(raw["training"]["purge_days"])
    # Synthetic calendar
    cal = list(pd.bdate_range("2020-01-01", periods=2000))
    pre = [d for d in cal if d < h0]
    fit_cal = pre[:-purge]
    es_days = int(raw["training"]["early_stopping_valid_days"])
    es = set(fit_cal[-es_days:])
    train = set(fit_cal[:-es_days])
    assert max(train) < h0
    assert max(es) < h0
    assert prior < h0


def test_monthly_return_and_sortino() -> None:
    dates = pd.bdate_range("2024-10-01", periods=60)
    curve = []
    eq = 100.0
    for i, d in enumerate(dates):
        eq *= 1.001
        curve.append(EquityPoint(d, eq, 0, eq, 0, 0, 0, 0))
    monthly = monthly_performance(curve)
    assert not monthly.empty
    assert "strategy_return" in monthly.columns
    rets = pd.Series([0.01, -0.02, 0.015, -0.005])
    assert sortino_ratio(rets) is not None


def test_symbol_and_sector_contribution() -> None:
    trades = [
        TradeRecord(
            symbol="A",
            country="United States",
            currency="USD",
            entry_date=pd.Timestamp("2024-10-01"),
            entry_price=10,
            exit_date=pd.Timestamp("2024-10-20"),
            exit_price=12,
            quantity=1,
            gross_pnl_base=2,
            net_pnl_base=100.0,
            return_pct=0.2,
            holding_days=15,
            exit_reason="holding_period",
            entry_score=0.9,
            entry_rank=1,
            entry_cost_base=10,
            exit_proceeds_base=12,
            commission_base=0,
            slippage_impact_base=0,
        ),
        TradeRecord(
            symbol="B",
            country="United States",
            currency="USD",
            entry_date=pd.Timestamp("2024-10-01"),
            entry_price=10,
            exit_date=pd.Timestamp("2024-10-20"),
            exit_price=11,
            quantity=1,
            gross_pnl_base=1,
            net_pnl_base=20.0,
            return_pct=0.1,
            holding_days=15,
            exit_reason="holding_period",
            entry_score=0.2,
            entry_rank=8,
            entry_cost_base=10,
            exit_proceeds_base=11,
            commission_base=0,
            slippage_impact_base=0,
        ),
    ]
    sym = symbol_contribution(trades, initial_capital=1000)
    assert sym["top_dependency"]["top1"]["pnl_share"] == pytest.approx(100 / 120)
    sec = sector_contribution(trades, {"A": "Tech", "B": "Health"})
    assert "Tech" in sec


def test_mae_mfe_and_gap_down() -> None:
    dates = pd.bdate_range("2024-10-01", periods=15)
    rows = []
    for i, d in enumerate(dates):
        # gap down on day 3: prev close 100 -> open 89
        close = 100.0 - i
        open_ = 89.0 if i == 3 else close
        rows.append(
            {
                "Date": d,
                "Symbol": "AAA",
                "Open": open_,
                "High": max(open_, close) + 1,
                "Low": min(open_, close) - 2,
                "Close": close,
            }
        )
    prices = pd.DataFrame(rows)
    trades = [
        TradeRecord(
            symbol="AAA",
            country="United States",
            currency="USD",
            entry_date=dates[0],
            entry_price=100.0,
            exit_date=dates[10],
            exit_price=95.0,
            quantity=1,
            gross_pnl_base=-5,
            net_pnl_base=-5,
            return_pct=-0.05,
            holding_days=10,
            exit_reason="holding_period",
            entry_score=1.0,
            entry_rank=1,
            entry_cost_base=100,
            exit_proceeds_base=95,
            commission_base=0,
            slippage_impact_base=0,
        )
    ]
    mae = mae_mfe_analysis(trades, prices)
    assert mae["n_trades"] == 1
    assert mae["worst_mae"] is not None
    assert mae["threshold_hit_counts"]["hit_10pct"] >= 0
    gaps = gap_down_audit(trades, prices)
    assert gaps["counts"]["gap_le_m5"] >= 1


def test_score_tertile_rank_fallback() -> None:
    trades = []
    for i, score in enumerate([0.1, 0.1, 0.5, 0.5, 0.9, 0.9]):
        trades.append(
            TradeRecord(
                symbol=f"S{i}",
                country="United States",
                currency="USD",
                entry_date=pd.Timestamp("2024-10-01"),
                entry_price=10,
                exit_date=pd.Timestamp("2024-10-20"),
                exit_price=10 * (1 + 0.01 * (i - 2)),
                quantity=1,
                gross_pnl_base=0,
                net_pnl_base=float(10 * (i - 2)),
                return_pct=0.01 * (i - 2),
                holding_days=15,
                exit_reason="holding_period",
                entry_score=score,
                entry_rank=i + 1,
                entry_cost_base=10,
                exit_proceeds_base=10,
                commission_base=0,
                slippage_impact_base=0,
            )
        )
    out = score_tertile_diagnostics(trades)
    assert set(out["buckets"]) == {"bottom_1_3", "middle_1_3", "top_1_3"}


def test_success_checklist_and_cost_stress_config() -> None:
    raw = json.loads(Path("config/true_holdout.json").read_text(encoding="utf-8"))
    assert set(raw["cost_stress"]) >= {"current", "high", "extreme"}
    cl = success_checklist(
        net_return=0.05,
        sharpe=0.8,
        benchmark_excess=0.01,
        momentum_excess=0.02,
        profit_factor=1.5,
        top1_share=0.2,
        top_sector_share=0.3,
        positive_month_ratio=0.55,
        worst_mae=-0.18,
    )
    assert cl["A_net_return_positive"] is True
    assert cl["D_momentum_excess_positive"] is True


def test_look_ahead_signal_timing_unchanged() -> None:
    """Holdout uses same engine timing contract."""
    from src.simulation.engine import SimulationEngine

    doc = SimulationEngine.__doc__ or ""
    assert "Day T close" in doc
    assert "T+1 open" in doc or "next trading day open" in doc
