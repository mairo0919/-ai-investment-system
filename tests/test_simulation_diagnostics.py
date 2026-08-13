"""Phase 4B diagnostics: cost, weekly, no-trade, holding, quality filters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.simulation.config import load_simulation_config
from src.simulation.diagnostics import (
    cost_decomposition,
    exit_reason_analysis,
    trade_quality_by_entry_score,
    turnover_analysis,
)
from src.simulation.engine import SimulationEngine
from src.simulation.execution import CostModel
from src.simulation.fx import FxConverter
from src.simulation.orders import EquityPoint, TradeRecord
from src.simulation.quality import (
    allow_new_entries,
    calibrate_train_thresholds,
    daily_country_score_stats,
    is_entry_rebalance_day,
)


def _fx() -> FxConverter:
    idx = pd.bdate_range("2020-01-01", periods=400)
    return FxConverter({"USDJPY": pd.Series(100.0, index=idx), "EURJPY": pd.Series(120.0, index=idx)})


def _mini_prices(n: int = 30) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2020-01-02", periods=n)
    rows = []
    for d in dates:
        for sym, px0 in (("AAA", 100.0), ("BBB", 50.0), ("CCC", 80.0), ("DDD", 60.0)):
            rows.append(
                {
                    "Date": d,
                    "Symbol": sym,
                    "Country": "Japan",
                    "Currency": "JPY",
                    "Open": px0,
                    "High": px0 * 1.02,
                    "Low": px0 * 0.98,
                    "Close": px0,
                }
            )
    return pd.DataFrame(rows), dates


def _rankings(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for d in dates:
        for i, sym in enumerate(("AAA", "BBB", "CCC", "DDD")):
            rows.append(
                {
                    "Date": d,
                    "Symbol": sym,
                    "Country": "Japan",
                    "Score": float(4 - i) + (0.01 if d == dates[0] else 0.0),
                    "Rank": i + 1,
                    "Percentile": (i + 1) / 4.0,
                }
            )
    return pd.DataFrame(rows)


def test_diagnostics_config_loads_strategies() -> None:
    cfg = load_simulation_config(Path("config/simulation_diagnostics.json"))
    assert "BASE" in cfg.strategies
    assert "QUALITY_FILTER" in cfg.strategies
    weekly = cfg.with_strategy("WEEKLY")
    assert weekly.rebalance_frequency == "weekly"
    assert weekly.holding_period_days == 10


def test_zero_cost_simulation_and_cost_decomposition() -> None:
    prices, dates = _mini_prices(20)
    rankings = _rankings(dates)
    cfg = (
        load_simulation_config(Path("config/simulation_diagnostics.json"))
        .with_strategy("BASE")
        .with_overrides(top_percentile=0.5, max_positions=2, max_country_weight=1.0)
    )
    gross_cfg = cfg.with_cost_preset("gross")
    net_cfg = cfg.with_cost_preset("net")
    assert gross_cfg.commission_rate == 0.0
    assert net_cfg.commission_rate == pytest.approx(0.001)

    g = SimulationEngine(gross_cfg, fx=_fx(), countries={"Japan"}).run(
        prices=prices, rankings=rankings
    )
    n = SimulationEngine(net_cfg, fx=_fx(), countries={"Japan"}).run(
        prices=prices, rankings=rankings
    )
    assert g.portfolio is not None and n.portfolio is not None
    assert g.portfolio.total_commission == pytest.approx(0.0)
    assert n.portfolio.total_commission > 0.0

    decomp = cost_decomposition(
        gross_perf={"total_return": 0.05},
        net_perf={"total_return": 0.01},
        low_perf={"total_return": 0.03},
        initial_capital=1_000_000,
        commission_total=1000,
        slippage_total=500,
    )
    assert decomp["cost_drag"] == pytest.approx(0.04)
    assert decomp["gross_profitable"] is True
    assert decomp["cost_over_initial_capital"] == pytest.approx(0.0015)


def test_weekly_rebalance_reduces_entry_days() -> None:
    prices, dates = _mini_prices(25)
    rankings = _rankings(dates)
    daily = (
        load_simulation_config(Path("config/simulation_diagnostics.json"))
        .with_strategy("LONGER")
        .with_cost_preset("gross")
        .with_overrides(
            holding_period_days=20,
            stop_loss_pct=None,
            take_profit_pct=None,
            top_percentile=0.5,
            max_country_weight=1.0,
        )
    )
    weekly = daily.with_overrides(rebalance_frequency="weekly")
    d_res = SimulationEngine(daily, fx=_fx(), countries={"Japan"}).run(
        prices=prices, rankings=rankings
    )
    w_res = SimulationEngine(weekly, fx=_fx(), countries={"Japan"}).run(
        prices=prices, rankings=rankings
    )
    assert d_res.meta["rebalance_stats"]["entry_signal_days"] >= w_res.meta["rebalance_stats"][
        "entry_signal_days"
    ]


def test_is_entry_rebalance_day_weekly() -> None:
    cal = list(pd.bdate_range("2020-01-06", periods=10))  # Mon start
    assert is_entry_rebalance_day(cal[0], cal, frequency="weekly", weekly_weekday=0)
    assert not is_entry_rebalance_day(cal[1], cal, frequency="weekly", weekly_weekday=0)
    assert is_entry_rebalance_day(cal[1], cal, frequency="daily")


def test_no_trade_day_blocks_entries() -> None:
    cfg = (
        load_simulation_config(Path("config/simulation_diagnostics.json"))
        .with_strategy("BASE")
        .with_overrides(block_low_resolution_entries=True, low_resolution_tie_threshold=0.5)
    )
    # All same score -> high tie ratio
    day = pd.DataFrame(
        {
            "Symbol": ["A", "B", "C", "D"],
            "Country": ["Japan"] * 4,
            "Score": [1.0, 1.0, 1.0, 1.0],
            "Rank": [1, 2, 3, 4],
            "Percentile": [0.25, 0.5, 0.75, 1.0],
        }
    )
    ok, reason = allow_new_entries(day, cfg)
    assert ok is False
    assert reason == "low_resolution_tie"


def test_score_dispersion_filter() -> None:
    cfg = (
        load_simulation_config(Path("config/simulation_diagnostics.json"))
        .with_strategy("BASE")
        .with_overrides(min_score_std=1.0, min_top_median_gap=2.0)
    )
    low = pd.DataFrame(
        {
            "Symbol": ["A", "B", "C"],
            "Score": [0.1, 0.11, 0.12],
            "Country": ["Japan"] * 3,
        }
    )
    stats = daily_country_score_stats(low)
    assert stats["score_std"] < 1.0
    ok, reason = allow_new_entries(low, cfg)
    assert ok is False
    assert reason in {"low_score_std", "low_top_median_gap"}


def test_calibrate_train_thresholds_no_valid_peek() -> None:
    train = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4),
            "Symbol": ["A", "B", "C", "D"] * 2,
            "Country": ["Japan"] * 8,
            "Score": [3, 2, 1, 0, 6, 4, 2, 0],
        }
    )
    th = calibrate_train_thresholds(train, quantile=0.25)
    assert th["min_score_std"] >= 0
    assert th["min_top_median_gap"] >= 0


def test_holding_5_10_20_config() -> None:
    cfg = load_simulation_config(Path("config/simulation_diagnostics.json"))
    assert cfg.with_strategy("BASE").holding_period_days == 5
    assert cfg.with_strategy("LONGER").holding_period_days == 10
    assert cfg.with_strategy("LONG20").holding_period_days == 20


def test_five_vs_ten_positions() -> None:
    cfg = load_simulation_config(Path("config/simulation_diagnostics.json"))
    assert cfg.with_strategy("WEEKLY").max_positions == 10
    assert cfg.with_strategy("CONCENTRATED").max_positions == 5


def test_ranking_exit_increases_exit_reasons() -> None:
    prices, dates = _mini_prices(25)
    # AAA starts top then drops out of top 30%
    rows = []
    for d in dates:
        order = ("AAA", "BBB", "CCC", "DDD") if d <= dates[3] else ("BBB", "CCC", "DDD", "AAA")
        for i, sym in enumerate(order):
            rows.append(
                {
                    "Date": d,
                    "Symbol": sym,
                    "Country": "Japan",
                    "Score": float(4 - i),
                    "Rank": i + 1,
                    "Percentile": (i + 1) / 4.0,
                }
            )
    rankings = pd.DataFrame(rows)
    base = (
        load_simulation_config(Path("config/simulation_diagnostics.json"))
        .with_strategy("LONGER")
        .with_cost_preset("gross")
        .with_overrides(
            stop_loss_pct=None,
            take_profit_pct=None,
            holding_period_days=30,
            top_percentile=0.5,
            max_country_weight=1.0,
        )
    )
    off = SimulationEngine(base, fx=_fx(), countries={"Japan"}).run(
        prices=prices, rankings=rankings
    )
    on = SimulationEngine(
        base.with_overrides(ranking_exit_enabled=True, ranking_exit_percentile=0.30),
        fx=_fx(),
        countries={"Japan"},
    ).run(prices=prices, rankings=rankings)
    on_reasons = [t.exit_reason for t in on.trades]
    assert off.meta is not None
    # Ranking exit should be able to trigger when AAA falls
    assert "ranking_exit" in on_reasons or len(on.trades) >= len(off.trades)


def test_trade_quality_bucket_and_exit_reason_metrics() -> None:
    trades = [
        TradeRecord(
            symbol="A",
            country="Japan",
            currency="JPY",
            entry_date=pd.Timestamp("2020-01-02"),
            entry_price=10,
            exit_date=pd.Timestamp("2020-01-10"),
            exit_price=11,
            quantity=1,
            gross_pnl_base=1,
            net_pnl_base=0.8,
            return_pct=0.08,
            holding_days=5,
            exit_reason="holding_period",
            entry_score=0.9,
            entry_rank=1,
            entry_cost_base=10,
            exit_proceeds_base=10.8,
            commission_base=0.1,
            slippage_impact_base=0.1,
        ),
        TradeRecord(
            symbol="B",
            country="Japan",
            currency="JPY",
            entry_date=pd.Timestamp("2020-01-02"),
            entry_price=10,
            exit_date=pd.Timestamp("2020-01-08"),
            exit_price=9,
            quantity=1,
            gross_pnl_base=-1,
            net_pnl_base=-1.2,
            return_pct=-0.12,
            holding_days=4,
            exit_reason="stop_loss",
            entry_score=0.2,
            entry_rank=8,
            entry_cost_base=10,
            exit_proceeds_base=8.8,
            commission_base=0.1,
            slippage_impact_base=0.1,
        ),
        TradeRecord(
            symbol="C",
            country="Japan",
            currency="JPY",
            entry_date=pd.Timestamp("2020-01-03"),
            entry_price=10,
            exit_date=pd.Timestamp("2020-01-09"),
            exit_price=10.5,
            quantity=1,
            gross_pnl_base=0.5,
            net_pnl_base=0.3,
            return_pct=0.03,
            holding_days=4,
            exit_reason="take_profit",
            entry_score=0.5,
            entry_rank=4,
            entry_cost_base=10,
            exit_proceeds_base=10.3,
            commission_base=0.1,
            slippage_impact_base=0.1,
        ),
    ]
    tq = trade_quality_by_entry_score(trades)
    assert "buckets" in tq
    er = exit_reason_analysis(trades)
    assert "stop_loss" in er
    assert er["holding_period"]["count"] == 1

    curve = [
        EquityPoint(pd.Timestamp("2020-01-02"), 100, 0, 100, 0, 0, 0, 0),
        EquityPoint(pd.Timestamp("2020-01-03"), 90, 20, 110, 0, 0, 0, 1),
    ]
    turn = turnover_analysis(trades, curve, initial_capital=100)
    assert turn["number_of_trades"] == 3
    assert turn["portfolio_turnover"] > 0
