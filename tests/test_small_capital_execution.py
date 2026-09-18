"""SC1/SC2 Small Capital execution foundation tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import PROJECT_ROOT, Settings
from src.core.exceptions import TrainingError
from src.paper.small_capital import (
    EXPERIMENT_ID,
    apply_small_capital_overlays,
    assert_small_capital_state_dir,
    resolve_small_capital_state_dir,
)
from src.simulation.config import load_simulation_config, parse_simulation_config
from src.simulation.engine import SimulationEngine
from src.simulation.execution import CostModel
from src.simulation.execution_policy import (
    REASON_MIN_QUANTITY_NOT_MET,
    REASON_NOT_AFFORDABLE,
    ExecutionPolicy,
    buy_unit_cost_base,
    entry_debit_base,
    max_integer_shares_for_cash,
)
from src.simulation.fx import FxConverter
from src.simulation.portfolio import Portfolio
from src.simulation.rules import equal_weight_notional, plan_integer_affordable_entries


def _fx(rate: float = 150.0) -> FxConverter:
    idx = pd.bdate_range("2020-01-01", periods=40)
    return FxConverter(
        {
            "USDJPY": pd.Series(rate, index=idx),
            "EURJPY": pd.Series(160.0, index=idx),
            "GBPJPY": pd.Series(180.0, index=idx),
            "CHFJPY": pd.Series(140.0, index=idx),
        }
    )


def _sc_cfg(**overrides):
    base = {
        "initial_capital": 10_000,
        "base_currency": "JPY",
        "candidate": {"mode": "top_n", "top_percentile": 0.5, "top_n": 10},
        "portfolio": {
            "max_positions": 5,
            "max_position_weight": 1.0,
            "max_country_weight": 1.0,
        },
        "exit_rules": {
            "holding_period_days": 20,
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "ranking_exit_enabled": False,
            "ranking_exit_percentile": 0.3,
            "cooldown_days": 0,
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
        "rebalance": {"frequency": "daily"},
        "execution_policy": {
            "share_mode": "integer",
            "minimum_quantity": 1,
            "allow_fractional": False,
            "sizing_mode": "integer_affordable",
            "min_lot_overrides_weight": True,
        },
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return parse_simulation_config(base)


def _legacy_cfg(**overrides):
    base = {
        "initial_capital": 10_000_000,
        "base_currency": "JPY",
        "candidate": {"mode": "top_percentile", "top_percentile": 0.5, "top_n": 3},
        "portfolio": {
            "max_positions": 5,
            "max_position_weight": 0.20,
            "max_country_weight": 1.0,
        },
        "exit_rules": {
            "holding_period_days": 20,
            "stop_loss_pct": None,
            "take_profit_pct": 0.15,
            "ranking_exit_enabled": False,
            "cooldown_days": 0,
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
        "rebalance": {"frequency": "daily"},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return parse_simulation_config(base)


def test_A_stock_100_usd_not_affordable_at_10k() -> None:
    costs = CostModel(0.001, 0.0005)
    q, reason = max_integer_shares_for_cash(
        cash=10_000,
        raw_price=100.0,
        fx_rate=150.0,
        costs=costs,
        minimum_quantity=1,
    )
    assert q == 0
    assert reason in (REASON_NOT_AFFORDABLE, REASON_MIN_QUANTITY_NOT_MET)
    unit = buy_unit_cost_base(raw_price=100.0, fx_rate=150.0, costs=costs)
    assert unit > 10_000


def test_B_stock_50_usd_one_share_affordable() -> None:
    costs = CostModel(0.001, 0.0005)
    q, reason = max_integer_shares_for_cash(
        cash=10_000,
        raw_price=50.0,
        fx_rate=150.0,
        costs=costs,
        minimum_quantity=1,
    )
    assert reason is None
    assert q >= 1
    debit = entry_debit_base(quantity=1.0, raw_price=50.0, fx_rate=150.0, costs=costs)
    assert debit <= 10_000 + 1e-6


def test_C_rank1_skip_rank2_buy() -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    costs = CostModel(cfg.commission_rate, cfg.slippage_rate)
    port = Portfolio(cash=10_000)
    day = pd.Timestamp("2020-06-01")
    ranking = pd.DataFrame(
        {
            "Date": [day, day],
            "Symbol": ["EXPENSIVE", "CHEAP"],
            "Country": ["United States", "United States"],
            "Score": [0.99, 0.90],
            "Rank": [1, 2],
            "Percentile": [0.01, 0.05],
        }
    )
    meta = {
        "EXPENSIVE": {"Country": "United States", "Currency": "USD"},
        "CHEAP": {"Country": "United States", "Currency": "USD"},
    }
    plans, skips = plan_integer_affordable_entries(
        ranking,
        port,
        cfg,
        costs=costs,
        fx=fx,
        signal_day=day,
        close_prices={"EXPENSIVE": 100.0, "CHEAP": 50.0},
        meta_map=meta,
        pending_buy_symbols=set(),
        cooldown_until={},
        countries={"United States"},
    )
    assert any(s.symbol == "EXPENSIVE" for s in skips)
    assert len(plans) == 1
    assert plans[0].symbol == "CHEAP"
    assert plans[0].shares >= 1


def test_D_commission_overflow_blocks_buy() -> None:
    # 1 share gross exactly 10000; commission pushes over.
    costs = CostModel(commission_rate=0.001, slippage_rate=0.0)
    raw = 10000.0 / 150.0  # ~66.666...
    q, reason = max_integer_shares_for_cash(
        cash=10_000,
        raw_price=raw,
        fx_rate=150.0,
        costs=costs,
        minimum_quantity=1,
    )
    assert q == 0
    assert reason in (REASON_NOT_AFFORDABLE, REASON_MIN_QUANTITY_NOT_MET)
    debit = entry_debit_base(quantity=1.0, raw_price=raw, fx_rate=150.0, costs=costs)
    assert debit > 10_000


def test_E_slippage_overflow_blocks_buy() -> None:
    costs = CostModel(commission_rate=0.0, slippage_rate=0.01)
    raw = 10000.0 / 150.0
    q, reason = max_integer_shares_for_cash(
        cash=10_000,
        raw_price=raw,
        fx_rate=150.0,
        costs=costs,
        minimum_quantity=1,
    )
    assert q == 0
    assert reason in (REASON_NOT_AFFORDABLE, REASON_MIN_QUANTITY_NOT_MET)


def test_F_remaining_cash_buys_next_candidate() -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    costs = CostModel(0.0, 0.0)  # simplify
    port = Portfolio(cash=10_000)
    day = pd.Timestamp("2020-06-01")
    ranking = pd.DataFrame(
        {
            "Date": [day, day],
            "Symbol": ["A", "B"],
            "Country": ["United States", "United States"],
            "Score": [0.99, 0.90],
            "Rank": [1, 2],
            "Percentile": [0.01, 0.05],
        }
    )
    meta = {
        "A": {"Country": "United States", "Currency": "USD"},
        "B": {"Country": "United States", "Currency": "USD"},
    }
    # $20 * 150 = 3000 per share → can buy multiple names
    plans, _ = plan_integer_affordable_entries(
        ranking,
        port,
        cfg,
        costs=costs,
        fx=fx,
        signal_day=day,
        close_prices={"A": 20.0, "B": 20.0},
        meta_map=meta,
        pending_buy_symbols=set(),
        cooldown_until={},
    )
    assert len(plans) >= 2
    assert {p.symbol for p in plans} == {"A", "B"}
    assert sum(p.estimated_debit_base for p in plans) <= 10_000 + 1e-6


def test_G_all_unaffordable_keeps_cash() -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    costs = CostModel(cfg.commission_rate, cfg.slippage_rate)
    port = Portfolio(cash=10_000)
    day = pd.Timestamp("2020-06-01")
    ranking = pd.DataFrame(
        {
            "Date": [day, day],
            "Symbol": ["A", "B"],
            "Country": ["United States", "United States"],
            "Score": [0.99, 0.90],
            "Rank": [1, 2],
            "Percentile": [0.01, 0.05],
        }
    )
    meta = {
        "A": {"Country": "United States", "Currency": "USD"},
        "B": {"Country": "United States", "Currency": "USD"},
    }
    plans, skips = plan_integer_affordable_entries(
        ranking,
        port,
        cfg,
        costs=costs,
        fx=fx,
        signal_day=day,
        close_prices={"A": 100.0, "B": 90.0},
        meta_map=meta,
        pending_buy_symbols=set(),
        cooldown_until={},
    )
    assert plans == []
    assert len(skips) >= 2
    assert port.cash == 10_000
    assert port.positions == {}


def test_H_fractional_legacy_mode_unchanged() -> None:
    fx = _fx(100.0)
    costs = CostModel(0.001, 0.0005)
    port = Portfolio(cash=1_000_000)
    pos = port.open_position(
        symbol="AAPL",
        country="United States",
        currency="USD",
        fill_date=pd.Timestamp("2020-06-01"),
        raw_open=100.0,
        target_notional_base=100_000,
        costs=costs,
        fx=fx,
        stop_loss_pct=None,
        take_profit_pct=None,
        score=1.0,
        rank=1,
        percentile=0.1,
        execution_policy=ExecutionPolicy.legacy(),
    )
    buy_px = costs.buy_unit_price(100.0)
    # Legacy continuous qty from notional (commission may scale slightly).
    assert pos.quantity == pytest.approx(100_000 / (buy_px * 100.0), rel=1e-3)
    assert abs(pos.quantity - round(pos.quantity)) > 1e-9  # not forced integer


def test_I_existing_10m_equal_weight_sizing_matches() -> None:
    cfg = _legacy_cfg()
    assert cfg.execution_policy.sizing_mode == "legacy_equal_weight"
    assert cfg.execution_policy.share_mode == "fractional"
    port = Portfolio(cash=10_000_000)
    notional = equal_weight_notional(port, cfg, n_new=2)
    assert notional == pytest.approx(min(10_000_000 / 5, 10_000_000 * 0.20))


def test_J_small_capital_state_dir_guard(tmp_path: Path) -> None:
    marker = PROJECT_ROOT / "data" / "paper" / ".sc_guard_probe"
    existed = marker.exists()
    try:
        with pytest.raises(TrainingError):
            assert_small_capital_state_dir(PROJECT_ROOT / "data" / "paper")
        ok = resolve_small_capital_state_dir(
            explicit=tmp_path / "paper_experiments" / "small_capital_10k"
        )
        assert "small_capital_10k" in str(ok)
        assert not marker.exists() or existed
    finally:
        if marker.exists() and not existed:
            marker.unlink()


def test_K_fx_conversion_in_affordability() -> None:
    costs = CostModel(0.0, 0.0)
    # $50 @ 100 JPY = 5000 → affordable; @ 250 = 12500 → not
    q_ok, _ = max_integer_shares_for_cash(
        cash=10_000, raw_price=50.0, fx_rate=100.0, costs=costs, minimum_quantity=1
    )
    q_no, reason = max_integer_shares_for_cash(
        cash=10_000, raw_price=50.0, fx_rate=250.0, costs=costs, minimum_quantity=1
    )
    assert q_ok >= 1
    assert q_no == 0
    assert reason == REASON_NOT_AFFORDABLE


def test_L_quantity_respects_minimum_quantity() -> None:
    costs = CostModel(0.0, 0.0)
    q, reason = max_integer_shares_for_cash(
        cash=10_000,
        raw_price=50.0,
        fx_rate=150.0,
        costs=costs,
        minimum_quantity=2,
    )
    # 1 share costs 7500; 2 shares 15000 > cash
    assert q == 0
    assert reason in (REASON_MIN_QUANTITY_NOT_MET, REASON_NOT_AFFORDABLE)


def test_engine_integer_fill_and_fallback() -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    days = pd.bdate_range("2020-06-01", periods=5)
    rows = []
    for d in days:
        for sym, px in [("EXP", 100.0), ("CHP", 50.0)]:
            rows.append(
                {
                    "Date": d,
                    "Symbol": sym,
                    "Country": "United States",
                    "Currency": "USD",
                    "Open": px,
                    "High": px,
                    "Low": px,
                    "Close": px,
                }
            )
    prices = pd.DataFrame(rows)
    rankings = pd.DataFrame(
        [
            {
                "Date": days[0],
                "Symbol": "EXP",
                "Country": "United States",
                "Score": 0.99,
                "Rank": 1,
                "Percentile": 0.01,
            },
            {
                "Date": days[0],
                "Symbol": "CHP",
                "Country": "United States",
                "Score": 0.9,
                "Rank": 2,
                "Percentile": 0.05,
            },
        ]
    )
    eng = SimulationEngine(cfg, fx=fx, countries={"United States"})
    result = eng.run(
        prices=prices,
        rankings=rankings,
        start=days[0],
        end=days[-1],
        portfolio=Portfolio(cash=10_000),
    )
    assert "CHP" in result.portfolio.positions
    assert "EXP" not in result.portfolio.positions
    pos = result.portfolio.positions["CHP"]
    assert float(pos.quantity) == int(pos.quantity)
    assert pos.quantity >= 1
    assert result.portfolio.cash >= 0
    skips = result.meta["rebalance_stats"]["entry_skips"]
    assert any(s["symbol"] == "EXP" for s in skips)


def test_small_capital_config_file_overlays() -> None:
    raw = json.loads(
        Path("config/paper_trading_small_capital_10k.json").read_text(encoding="utf-8")
    )
    assert raw["experiment_id"] == EXPERIMENT_ID
    assert raw["initial_capital"] == 10000
    # Holdout strategy base then overlays
    from src.simulation.holdout_runner import load_true_holdout_bundle

    _, cfg = load_true_holdout_bundle(PROJECT_ROOT / "config" / "true_holdout.json")
    cfg2 = apply_small_capital_overlays(raw, cfg)
    assert cfg2.initial_capital == 10_000
    assert cfg2.execution_policy.is_integer_affordable
    assert cfg2.max_position_weight == 1.0
    # Strategy signals unchanged
    assert cfg2.max_positions == cfg.max_positions
    assert cfg2.top_percentile == cfg.top_percentile
    assert cfg2.holding_period_days == cfg.holding_period_days


def test_default_simulation_config_is_legacy() -> None:
    cfg = load_simulation_config(Path("config/simulation.json"))
    assert cfg.execution_policy.share_mode == "fractional"
    assert cfg.execution_policy.sizing_mode == "legacy_equal_weight"


def test_integer_open_rejects_cash_overflow() -> None:
    fx = _fx(150.0)
    costs = CostModel(0.001, 0.0005)
    port = Portfolio(cash=10_000)
    policy = ExecutionPolicy.from_dict(
        {
            "share_mode": "integer",
            "minimum_quantity": 1,
            "allow_fractional": False,
            "sizing_mode": "integer_affordable",
        }
    )
    with pytest.raises(TrainingError):
        port.open_position(
            symbol="EXP",
            country="United States",
            currency="USD",
            fill_date=pd.Timestamp("2020-06-01"),
            raw_open=100.0,
            target_notional_base=15_000,  # would be ~1 share if cheaper
            costs=costs,
            fx=fx,
            stop_loss_pct=None,
            take_profit_pct=None,
            score=1.0,
            rank=1,
            percentile=0.01,
            execution_policy=policy,
        )
    assert port.cash == 10_000


def test_sc_state_does_not_touch_canonical_paper(tmp_path: Path) -> None:
    canonical = PROJECT_ROOT / "data" / "paper"
    before = {p.name: p.stat().st_mtime_ns for p in canonical.iterdir() if p.is_file()}
    sc_dir = tmp_path / "small_capital_10k"
    sc_dir.mkdir()
    (sc_dir / "probe.txt").write_text("sc-only", encoding="utf-8")
    assert_small_capital_state_dir(sc_dir)
    after = {p.name: p.stat().st_mtime_ns for p in canonical.iterdir() if p.is_file()}
    assert before == after
