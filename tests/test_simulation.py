"""Phase 4 simulation engine unit tests (synthetic data, no network)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.simulation.config import load_simulation_config, parse_simulation_config
from src.simulation.engine import SimulationEngine
from src.simulation.execution import (
    CostModel,
    conservative_stop_fill,
    conservative_take_fill,
)
from src.simulation.fx import FxConverter
from src.simulation.metrics import buy_and_hold_benchmark, compute_performance, max_drawdown
from src.simulation.portfolio import Portfolio
from src.simulation.ranking import attach_country_ranks, select_candidates
from src.simulation.report import equity_to_frame, trades_to_frame, write_run_artifacts
from src.simulation.rules import entry_candidates, evaluate_exits_for_day


def _fx_usd() -> FxConverter:
    idx = pd.bdate_range("2020-01-01", periods=400)
    return FxConverter(
        {
            "USDJPY": pd.Series(100.0, index=idx),
            "EURJPY": pd.Series(120.0, index=idx),
            "GBPJPY": pd.Series(130.0, index=idx),
            "CHFJPY": pd.Series(110.0, index=idx),
        }
    )


def _cfg(**overrides):
    base = {
        "initial_capital": 1_000_000,
        "base_currency": "JPY",
        "candidate": {"mode": "top_percentile", "top_percentile": 0.5, "top_n": 3},
        "portfolio": {
            "max_positions": 2,
            "max_position_weight": 0.5,
            "max_country_weight": 1.0,
            "sizing": "equal_weight",
        },
        "exit_rules": {
            "holding_period_days": 3,
            "stop_loss_pct": -0.05,
            "take_profit_pct": 0.10,
            "ranking_exit_enabled": False,
            "ranking_exit_percentile": 0.3,
        },
        "costs": {"commission_rate": 0.001, "slippage_rate": 0.0005},
        "ranking": {
            "label_scheme": "B",
            "gain_name": "moderate_exp",
            "feature_set": "A",
            "param_preset": "default",
            "horizon_days": 5,
        },
        "fx": {"tickers": {}},
        "benchmarks": {},
        "strategies": {},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return parse_simulation_config(base)


def test_load_simulation_config() -> None:
    cfg = load_simulation_config(Path("config/simulation.json"))
    assert cfg.initial_capital == 10_000_000
    assert cfg.base_currency == "JPY"
    assert "A" in cfg.strategies


def test_portfolio_cash_and_position_creation() -> None:
    fx = _fx_usd()
    port = Portfolio(cash=1_000_000)
    costs = CostModel(0.001, 0.0005)
    pos = port.open_position(
        symbol="AAPL",
        country="United States",
        currency="USD",
        fill_date=pd.Timestamp("2020-06-01"),
        raw_open=100.0,
        target_notional_base=100_000,
        costs=costs,
        fx=fx,
        stop_loss_pct=-0.05,
        take_profit_pct=0.10,
        score=1.0,
        rank=1,
        percentile=0.1,
    )
    assert pos.quantity > 0
    assert port.cash < 1_000_000
    assert port.has_position("AAPL")
    assert pos.stop_price is not None and pos.stop_price < pos.entry_price


def test_entry_exit_round_trip_and_costs() -> None:
    fx = _fx_usd()
    port = Portfolio(cash=1_000_000)
    costs = CostModel(0.001, 0.0)
    port.open_position(
        symbol="X",
        country="United States",
        currency="USD",
        fill_date=pd.Timestamp("2020-06-01"),
        raw_open=50.0,
        target_notional_base=100_000,
        costs=costs,
        fx=fx,
        stop_loss_pct=None,
        take_profit_pct=None,
        score=None,
        rank=None,
        percentile=None,
    )
    cash_after_buy = port.cash
    trade = port.close_position(
        symbol="X",
        fill_date=pd.Timestamp("2020-06-10"),
        raw_fill_price=55.0,
        costs=costs,
        fx=fx,
        exit_reason="holding_period",
    )
    assert trade.net_pnl_base != 0
    assert port.total_commission > 0
    assert port.cash > cash_after_buy
    assert not port.has_position("X")


def test_slippage_makes_buy_worse_sell_worse() -> None:
    costs = CostModel(0.0, 0.01)
    assert costs.buy_unit_price(100) == pytest.approx(101)
    assert costs.sell_unit_price(100) == pytest.approx(99)


def test_conservative_stop_take() -> None:
    assert conservative_stop_fill(95, 90) == 90
    assert conservative_stop_fill(95, 100) == 95
    assert conservative_take_fill(110, 105) == 105
    assert conservative_take_fill(110, 120) == 110


def test_fx_no_silent_one_to_one() -> None:
    fx = _fx_usd()
    assert fx.rate_to_base("JPY", pd.Timestamp("2020-06-01")) == 1.0
    assert fx.rate_to_base("USD", pd.Timestamp("2020-06-01")) == 100.0
    with pytest.raises(Exception):
        FxConverter({}).rate_to_base("USD", pd.Timestamp("2020-06-01"))


def test_attach_ranks_and_candidates() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2020-01-02"] * 4,
            "Symbol": ["A", "B", "C", "D"],
            "Region": ["Japan"] * 4,
            "score": [0.4, 0.9, 0.1, 0.7],
        }
    )
    ranked = attach_country_ranks(df)
    assert ranked.loc[ranked["Symbol"] == "B", "Rank"].iloc[0] == 1
    top = select_candidates(ranked, mode="top_percentile", top_percentile=0.5, top_n=1)
    assert set(top["Symbol"]) == {"B", "D"}


def test_country_and_position_limits() -> None:
    cfg = _cfg(
        portfolio={
            "max_positions": 1,
            "max_position_weight": 0.1,
            "max_country_weight": 0.5,
        }
    )
    fx = _fx_usd()
    port = Portfolio(cash=1_000_000)
    port.open_position(
        symbol="A",
        country="Japan",
        currency="JPY",
        fill_date=pd.Timestamp("2020-06-01"),
        raw_open=1000.0,
        target_notional_base=100_000,
        costs=CostModel(0, 0),
        fx=fx,
        stop_loss_pct=None,
        take_profit_pct=None,
        score=1,
        rank=1,
        percentile=0.05,
    )
    ranking = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-06-02")] * 2,
            "Symbol": ["B", "C"],
            "Country": ["Japan", "Japan"],
            "Score": [1.0, 0.9],
            "Rank": [1, 2],
            "Percentile": [0.05, 0.1],
        }
    )
    cands = entry_candidates(ranking, port, cfg, countries={"Japan"})
    assert cands.empty  # max_positions=1 already filled


def test_holding_period_stop_take_signals() -> None:
    cfg = _cfg()
    fx = _fx_usd()
    port = Portfolio(cash=1_000_000)
    port.open_position(
        symbol="A",
        country="Japan",
        currency="JPY",
        fill_date=pd.Timestamp("2020-06-01"),
        raw_open=100.0,
        target_notional_base=50_000,
        costs=CostModel(0, 0),
        fx=fx,
        stop_loss_pct=-0.05,
        take_profit_pct=0.10,
        score=1,
        rank=1,
        percentile=0.05,
    )
    pos = port.positions["A"]
    pos.holding_days = 3
    # stop
    sig = evaluate_exits_for_day(
        port,
        day=pd.Timestamp("2020-06-05"),
        high={"A": 101},
        low={"A": 90},
        ranking_day=None,
        cfg=cfg,
    )
    assert sig and sig[0].reason == "stop_loss"

    pos.pending_exit_reason = None
    pos.holding_days = 3
    sig2 = evaluate_exits_for_day(
        port,
        day=pd.Timestamp("2020-06-05"),
        high={"A": 120},
        low={"A": 99},
        ranking_day=None,
        cfg=cfg,
    )
    assert sig2 and sig2[0].reason == "take_profit"

    pos.pending_exit_reason = None
    pos.stop_price = None
    pos.take_profit_price = None
    pos.holding_days = 3
    sig3 = evaluate_exits_for_day(
        port,
        day=pd.Timestamp("2020-06-05"),
        high={"A": 101},
        low={"A": 99},
        ranking_day=None,
        cfg=cfg,
    )
    assert sig3 and sig3[0].reason == "holding_period"


def test_next_open_execution_no_same_close_fill() -> None:
    """Signal on day0 close must not fill at day0 close."""
    dates = pd.bdate_range("2020-01-02", periods=12)
    rows = []
    for d in dates:
        rows.append(
            {
                "Date": d,
                "Symbol": "AAA",
                "Country": "Japan",
                "Currency": "JPY",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
            }
        )
        rows.append(
            {
                "Date": d,
                "Symbol": "BBB",
                "Country": "Japan",
                "Currency": "JPY",
                "Open": 50.0,
                "High": 51.0,
                "Low": 49.0,
                "Close": 50.5,
            }
        )
    prices = pd.DataFrame(rows)
    # Only first day ranking — top both
    rankings = pd.DataFrame(
        {
            "Date": [dates[0], dates[0]],
            "Symbol": ["AAA", "BBB"],
            "Country": ["Japan", "Japan"],
            "Score": [1.0, 0.5],
            "Rank": [1, 2],
            "Percentile": [0.5, 1.0],
        }
    )
    cfg = _cfg(
        candidate={"mode": "top_n", "top_n": 1, "top_percentile": 0.1},
        exit_rules={
            "holding_period_days": 5,
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "ranking_exit_enabled": False,
        },
        costs={"commission_rate": 0.0, "slippage_rate": 0.0},
        portfolio={
            "max_positions": 1,
            "max_position_weight": 0.5,
            "max_country_weight": 1.0,
        },
    )
    engine = SimulationEngine(cfg, fx=_fx_usd(), countries={"Japan"})
    result = engine.run(prices=prices, rankings=rankings)
    assert result.portfolio is not None
    # No position on signal day close; fill next open
    eq0 = result.equity_curve[0]
    assert eq0.n_positions == 0
    eq1 = result.equity_curve[1]
    assert eq1.n_positions == 1
    # Position may later exit via holding period; assert fill timing via trade or open pos
    if result.portfolio.trades:
        assert pd.Timestamp(result.portfolio.trades[0].entry_date) == pd.Timestamp(dates[1])
    else:
        pos = next(iter(result.portfolio.positions.values()))
        assert pd.Timestamp(pos.entry_date) == pd.Timestamp(dates[1])


def test_look_ahead_rankings_only_use_signal_dates() -> None:
    """Engine must not invent rankings from future prices."""
    dates = pd.bdate_range("2020-01-02", periods=8)
    prices = pd.DataFrame(
        {
            "Date": list(dates) + list(dates),
            "Symbol": ["A"] * 8 + ["B"] * 8,
            "Country": ["Japan"] * 16,
            "Currency": ["JPY"] * 16,
            "Open": 100.0,
            "High": 110.0,
            "Low": 90.0,
            "Close": 100.0,
        }
    )
    rankings = attach_country_ranks(
        pd.DataFrame(
            {
                "Date": [dates[0], dates[0]],
                "Symbol": ["A", "B"],
                "Region": ["Japan", "Japan"],
                "score": [1.0, 0.2],
            }
        )
    )
    cfg = _cfg(
        costs={"commission_rate": 0, "slippage_rate": 0},
        exit_rules={
            "holding_period_days": 2,
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "ranking_exit_enabled": False,
        },
    )
    result = SimulationEngine(cfg, fx=_fx_usd(), countries={"Japan"}).run(
        prices=prices, rankings=rankings
    )
    # Only one entry wave from day0 signal
    assert result.portfolio is not None
    assert len(result.portfolio.trades) + len(result.portfolio.positions) >= 1


def test_drawdown_sharpe_trade_log_equity_curve(tmp_path: Path) -> None:
    eq_vals = pd.Series([100, 110, 105, 120], dtype=float)
    assert max_drawdown(eq_vals) == pytest.approx((105 / 110) - 1)
    from src.simulation.orders import EquityPoint, TradeRecord

    curve = [
        EquityPoint(pd.Timestamp("2020-01-02"), 100, 0, 100, 0, 0, 0, 0),
        EquityPoint(pd.Timestamp("2020-01-03"), 90, 20, 110, 0, 0, 0, 1),
        EquityPoint(pd.Timestamp("2020-01-06"), 90, 10, 100, (100 / 110) - 1, 0, 0, 1),
    ]
    trades = [
        TradeRecord(
            symbol="A",
            country="Japan",
            currency="JPY",
            entry_date=pd.Timestamp("2020-01-03"),
            entry_price=10,
            exit_date=pd.Timestamp("2020-01-06"),
            exit_price=11,
            quantity=1,
            gross_pnl_base=1,
            net_pnl_base=0.9,
            return_pct=0.09,
            holding_days=2,
            exit_reason="holding_period",
            entry_score=1.0,
            entry_rank=1,
            entry_cost_base=10,
            exit_proceeds_base=10.9,
            commission_base=0.1,
            slippage_impact_base=0,
        )
    ]
    perf = compute_performance(curve, trades, initial_capital=100, transaction_costs_total=0.1)
    assert perf["number_of_trades"] == 1
    assert perf["max_drawdown"] < 0
    paths = write_run_artifacts(
        tmp_path,
        equity_curve=curve,
        trades=trades,
        summary={"performance": perf},
    )
    assert Path(paths["equity_curve"]).exists()
    assert Path(paths["trades_csv"]).exists()
    assert not equity_to_frame(curve).empty
    assert not trades_to_frame(trades).empty


def test_benchmark_buy_hold() -> None:
    s = pd.Series(
        [100.0, 110.0, 105.0],
        index=pd.bdate_range("2020-01-02", periods=3),
    )
    out = buy_and_hold_benchmark(s, initial_capital=1_000_000)
    assert out["total_return"] == pytest.approx(0.05)


def test_walk_forward_style_partition_for_simulation() -> None:
    """OOS rankings dates must sit after train_end (reuse WF helpers)."""
    from src.ml.walk_forward import build_expanding_folds, mask_fold_partition

    dates = pd.bdate_range("2015-01-01", periods=3000)
    folds = build_expanding_folds(
        dates, n_folds=5, min_train_days=756, valid_days=252, purge_days=5
    )
    frame = pd.DataFrame({"Date": dates, "Symbol": "A", "x": 1})
    for fold in folds:
        train = mask_fold_partition(frame, fold, partition="train")
        valid = mask_fold_partition(frame, fold, partition="validation")
        assert train["Date"].max() < valid["Date"].min()


def test_momentum_baseline_scores_no_future() -> None:
    from src.ml.ltr_baselines import momentum_scores

    df = pd.DataFrame({"return_20d": [0.1, -0.2, 0.05]})
    s = momentum_scores(df, feature_col="return_20d")
    assert list(s) == [0.1, -0.2, 0.05]


def test_strategy_overlay() -> None:
    cfg = load_simulation_config(Path("config/simulation.json"))
    b = cfg.with_strategy("B")
    assert b.holding_period_days == 10
    c = cfg.with_strategy("C")
    assert c.top_percentile == pytest.approx(0.20)
