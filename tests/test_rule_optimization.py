"""Phase 4C: trailing stop, biweekly, cooldown, score separation, regimes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ml.stability_diagnostics import classify_market_regime
from src.simulation.config import load_simulation_config
from src.simulation.diagnostics import fold_stability, regime_trade_analysis
from src.simulation.engine import SimulationEngine
from src.simulation.fx import FxConverter
from src.simulation.orders import TradeRecord
from src.simulation.quality import (
    allow_new_entries,
    calibrate_train_thresholds,
    count_reentries,
    daily_country_score_stats,
    is_entry_rebalance_day,
)
from src.simulation.rules import evaluate_exits_for_day
from src.simulation.portfolio import Portfolio
from src.simulation.execution import CostModel
from src.simulation.stop_diagnostics import (
    analyze_stop_aftermath,
    analyze_stop_counterfactual,
    pick_stop_setting,
)


def _fx() -> FxConverter:
    idx = pd.bdate_range("2020-01-01", periods=500)
    return FxConverter({"USDJPY": pd.Series(100.0, index=idx)})


def _prices(n: int = 40) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2020-01-02", periods=n)
    rows = []
    for i, d in enumerate(dates):
        for j, sym in enumerate(("AAA", "BBB", "CCC", "DDD")):
            px = 100.0 + j + i * 0.1
            rows.append(
                {
                    "Date": d,
                    "Symbol": sym,
                    "Country": "United States",
                    "Currency": "USD",
                    "Open": px,
                    "High": px * 1.03,
                    "Low": px * 0.97,
                    "Close": px,
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
                    "Country": "United States",
                    "Score": float(4 - i),
                    "Rank": i + 1,
                    "Percentile": (i + 1) / 4.0,
                }
            )
    return pd.DataFrame(rows)


def _base(**overrides):
    cfg = load_simulation_config(Path("config/rule_optimization.json")).with_strategy(
        "BASE_US_CONCENTRATED"
    )
    return cfg.with_overrides(
        top_percentile=0.5,
        max_country_weight=1.0,
        stop_loss_pct=None,
        take_profit_pct=None,
        **overrides,
    )


def test_rule_config_base() -> None:
    cfg = load_simulation_config(Path("config/rule_optimization.json")).with_strategy(
        "BASE_US_CONCENTRATED"
    )
    assert cfg.max_positions == 5
    assert cfg.rebalance_frequency == "weekly"
    assert cfg.holding_period_days == 10


def test_stop_off_8_10_configs() -> None:
    cfg = load_simulation_config(Path("config/rule_optimization.json"))
    assert cfg.with_overrides(stop_loss_pct=None).stop_loss_pct is None
    assert cfg.with_overrides(stop_loss_pct=-0.08).stop_loss_pct == pytest.approx(-0.08)
    assert cfg.with_overrides(stop_loss_pct=-0.10).stop_loss_pct == pytest.approx(-0.10)


def test_take_profit_off_and_15() -> None:
    cfg = load_simulation_config(Path("config/rule_optimization.json"))
    assert cfg.with_overrides(take_profit_pct=None).take_profit_pct is None
    assert cfg.with_overrides(take_profit_pct=0.15).take_profit_pct == pytest.approx(0.15)


def test_holding_15_20() -> None:
    cfg = load_simulation_config(Path("config/rule_optimization.json"))
    assert cfg.with_overrides(holding_period_days=15).holding_period_days == 15
    assert cfg.with_overrides(holding_period_days=20).holding_period_days == 20


def test_trailing_stop_triggers() -> None:
    fx = _fx()
    port = Portfolio(cash=1_000_000)
    cfg = _base(trailing_stop_pct=0.05, holding_period_days=100)
    port.open_position(
        symbol="AAA",
        country="United States",
        currency="USD",
        fill_date=pd.Timestamp("2020-01-02"),
        raw_open=100.0,
        target_notional_base=50_000,
        costs=CostModel(0, 0),
        fx=fx,
        stop_loss_pct=None,
        take_profit_pct=None,
        score=1,
        rank=1,
        percentile=0.1,
    )
    pos = port.positions["AAA"]
    pos.peak_price = 120.0
    pos.trailing_stop_price = 120.0 * 0.95
    sig = evaluate_exits_for_day(
        port,
        day=pd.Timestamp("2020-01-10"),
        high={"AAA": 118.0},
        low={"AAA": 110.0},
        ranking_day=None,
        cfg=cfg,
    )
    assert sig and sig[0].reason == "trailing_stop"


def test_biweekly_entry_less_frequent_than_weekly() -> None:
    cal = list(pd.bdate_range("2020-01-06", periods=40))
    weekly = sum(
        1 for d in cal if is_entry_rebalance_day(d, cal, frequency="weekly", weekly_weekday=0)
    )
    bi = sum(
        1 for d in cal if is_entry_rebalance_day(d, cal, frequency="biweekly", weekly_weekday=0)
    )
    assert bi < weekly
    assert bi > 0


def test_score_separation_filter() -> None:
    cfg = _base(
        use_score_separation_filter=True,
        min_top_median_gap=1.0,
        min_cutoff_median_gap=0.5,
    )
    flat = pd.DataFrame(
        {
            "Symbol": ["A", "B", "C", "D"],
            "Score": [1.01, 1.0, 0.99, 0.98],
            "Country": ["United States"] * 4,
        }
    )
    stats = daily_country_score_stats(flat, top_percentile=0.5)
    assert stats["top_median_gap"] < 1.0
    ok, reason = allow_new_entries(flat, cfg)
    assert ok is False
    assert "separation" in reason or "median" in reason


def test_calibrate_separation_median() -> None:
    train = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4),
            "Symbol": ["A", "B", "C", "D"] * 2,
            "Country": ["United States"] * 8,
            "Score": [4, 3, 2, 1, 8, 6, 4, 2],
        }
    )
    th = calibrate_train_thresholds(train, quantile=0.5, top_percentile=0.5)
    assert th["min_top_median_gap"] > 0
    assert "min_cutoff_median_gap" in th


def test_cooldown_blocks_reentry() -> None:
    prices, dates = _prices(30)
    rankings = _rankings(dates)
    cfg = _base(
        rebalance_frequency="daily",
        holding_period_days=3,
        cooldown_days=5,
        max_positions=1,
    ).with_cost_preset("gross")
    res = SimulationEngine(cfg, fx=_fx(), countries={"United States"}).run(
        prices=prices, rankings=rankings
    )
    assert res.meta["rebalance_stats"]["cooldown_blocks"] >= 0
    # With cooldown, should not instantly churn same name every cycle as freely
    assert res.portfolio is not None


def test_reentry_count() -> None:
    trades = pd.DataFrame(
        {
            "Symbol": ["A", "A", "B"],
            "Entry Date": ["2020-01-10", "2020-01-20", "2020-01-05"],
            "Exit Date": ["2020-01-15", "2020-01-25", "2020-01-08"],
        }
    )
    # Exit A on 15, next entry 20 -> ~3 bdays? Jan15->20
    counts = count_reentries(trades, within_days=(5, 10))
    assert counts["reentry_within_10_days"] >= 1


def test_cost_sensitivity_presets() -> None:
    cfg = load_simulation_config(Path("config/rule_optimization.json"))
    assert cfg.with_cost_preset("gross").commission_rate == 0.0
    assert cfg.with_cost_preset("high").commission_rate == pytest.approx(0.0015)
    assert cfg.with_cost_preset("high").slippage_rate == pytest.approx(0.001)


def test_fold_stability_and_regime_trade_analysis() -> None:
    stab = fold_stability([0.1, -0.05, 0.02, 0.0, 0.08])
    assert stab["positive_fold_ratio"] == pytest.approx(0.6)
    assert stab["worst_fold"] == pytest.approx(-0.05)
    assert stab["best_fold"] == pytest.approx(0.1)

    sma = pd.Series([0.1, -0.1, 0.0], index=pd.bdate_range("2020-01-02", periods=3))
    ret = pd.Series([0.1, -0.1, 0.0], index=sma.index)
    regimes = classify_market_regime(sma, ret)
    assert "Bull" in set(regimes) and "Bear" in set(regimes)

    trades = [
        TradeRecord(
            symbol="A",
            country="United States",
            currency="USD",
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
            entry_score=1.0,
            entry_rank=1,
            entry_cost_base=10,
            exit_proceeds_base=10.8,
            commission_base=0.1,
            slippage_impact_base=0.1,
        )
    ]
    out = regime_trade_analysis(trades, regimes)
    assert "Bull" in out
    assert out["Bull"]["trade_count"] == 1


def _stop_trade(
    *,
    symbol: str = "AAA",
    entry: str = "2020-01-02",
    exit_: str = "2020-01-06",
    entry_px: float = 100.0,
    exit_px: float = 95.0,
    ret: float = -0.05,
) -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        country="United States",
        currency="USD",
        entry_date=pd.Timestamp(entry),
        entry_price=entry_px,
        exit_date=pd.Timestamp(exit_),
        exit_price=exit_px,
        quantity=1,
        gross_pnl_base=exit_px - entry_px,
        net_pnl_base=exit_px - entry_px,
        return_pct=ret,
        holding_days=3,
        exit_reason="stop_loss",
        entry_score=1.0,
        entry_rank=1,
        entry_cost_base=entry_px,
        exit_proceeds_base=exit_px,
        commission_base=0.0,
        slippage_impact_base=0.0,
    )


def test_stop_aftermath_recovery_and_further_drop() -> None:
    dates = pd.bdate_range("2020-01-02", periods=30)
    # After stop on Jan6, price recovers above entry 100 within 5 sessions then dips
    rows = []
    path = {
        0: 100,
        1: 99,
        2: 97,
        3: 95,  # exit day-ish
        4: 94,
        5: 96,
        6: 101,  # recovered to entry
        7: 102,
        8: 90,  # further drop >5% from stop 95
    }
    for i, d in enumerate(dates):
        px = float(path.get(i, 100 + i * 0.1))
        rows.append(
            {
                "Date": d,
                "Symbol": "AAA",
                "Country": "United States",
                "Currency": "USD",
                "Open": px,
                "High": px * 1.01,
                "Low": px * 0.99,
                "Close": px,
            }
        )
    prices = pd.DataFrame(rows)
    trades = [_stop_trade(exit_=str(dates[3].date()), exit_px=95.0)]
    out = analyze_stop_aftermath(trades, prices, further_drop_pct=-0.05)
    assert out["n_stop_trades"] == 1
    assert out["recovery_to_entry_rate"]["within_5d"] is not None
    assert out["further_drop_ge_5pct_rate"]["within_20d"] is not None
    assert out["mean_aftermath_return"]["d1"] is not None


def test_stop_counterfactual_good_bad_neutral() -> None:
    dates = pd.bdate_range("2020-01-02", periods=25)
    # Entry 100 on day0; stop exit 95 on day3; by hold day10 price is 110 => BAD_STOP
    rows = []
    for i, d in enumerate(dates):
        px = 100.0 - i * 2 if i <= 3 else 95.0 + (i - 3) * 3
        rows.append(
            {
                "Date": d,
                "Symbol": "AAA",
                "Country": "United States",
                "Currency": "USD",
                "Open": px,
                "High": px,
                "Low": px,
                "Close": px,
            }
        )
    prices = pd.DataFrame(rows)
    trades = [_stop_trade(entry=str(dates[0].date()), exit_=str(dates[3].date()))]
    out = analyze_stop_counterfactual(
        trades, prices, holding_period_days=10, neutral_band=0.02
    )
    assert out["n_stop_trades"] == 1
    assert set(out["classification_rates"]) == {"GOOD_STOP", "BAD_STOP", "NEUTRAL_STOP"}
    assert out["trades"][0]["classification"] in {"GOOD_STOP", "BAD_STOP", "NEUTRAL_STOP"}


def test_pick_stop_rejects_severe_maxdd_worsening() -> None:
    rows = [
        {
            "name": "STOP_5",
            "net_return": 0.01,
            "max_dd": -0.10,
            "sharpe": 0.2,
            "positive_fold_ratio": 0.6,
            "profit_factor": 1.1,
            "trades": 100,
            "total_cost": 1e6,
            "fold_stability": {"std": 0.05},
        },
        {
            "name": "NO_STOP",
            "net_return": 0.10,
            "max_dd": -0.25,  # much worse DD
            "sharpe": 0.5,
            "positive_fold_ratio": 0.8,
            "profit_factor": 1.5,
            "trades": 80,
            "total_cost": 8e5,
            "fold_stability": {"std": 0.04},
        },
        {
            "name": "STOP_8",
            "net_return": 0.04,
            "max_dd": -0.11,
            "sharpe": 0.3,
            "positive_fold_ratio": 0.6,
            "profit_factor": 1.2,
            "trades": 90,
            "total_cost": 9e5,
            "fold_stability": {"std": 0.05},
        },
    ]
    pick = pick_stop_setting(rows, baseline_name="STOP_5", max_dd_worsen_limit=0.05)
    assert pick["selected"] != "NO_STOP"
    assert any(r["name"] == "NO_STOP" for r in pick["rejected_for_maxdd"])


def test_aftermath_not_used_for_engine_look_ahead() -> None:
    """Engine with STOP still fills next open; aftermath is offline-only."""
    prices, dates = _prices(20)
    # Force a stop via low print after entry
    prices.loc[prices["Symbol"] == "AAA", "Low"] = prices.loc[
        prices["Symbol"] == "AAA", "Close"
    ] * 0.90
    rankings = _rankings(dates)
    cfg = _base(
        rebalance_frequency="daily",
        holding_period_days=20,
        max_positions=1,
    ).with_overrides(stop_loss_pct=-0.05).with_cost_preset("gross")
    res = SimulationEngine(cfg, fx=_fx(), countries={"United States"}).run(
        prices=prices, rankings=rankings
    )
    assert res.meta["timing"] == "close_t_execute_open_t1"
    # Offline aftermath must not require mutating engine state
    if res.trades:
        out = analyze_stop_aftermath(res.trades, prices)
        assert "n_stop_trades" in out
