"""SC3: execution decision persistence + cash reservation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import PROJECT_ROOT
from src.paper.execution_audit import (
    AUDIT_COLUMNS,
    DECISION_FILLED,
    DECISION_QUEUED,
    DECISION_REJECTED,
    DECISION_SKIPPED,
    REASON_AFFORDABLE,
    REASON_FILL_PRICE_UNAFFORDABLE,
    REASON_INSUFFICIENT_CASH,
    REASON_NOT_AFFORDABLE,
    ExecutionAuditStore,
)
from src.paper.small_capital import EXPERIMENT_ID, assert_small_capital_state_dir
from src.simulation.config import parse_simulation_config
from src.simulation.engine import SimulationEngine
from src.simulation.execution import CostModel
from src.simulation.execution_policy import ExecutionPolicy
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
            "cooldown_days": 0,
        },
        "costs": {"commission_rate": 0.0, "slippage_rate": 0.0},
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


def _ranking_three(day: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": [day, day, day],
            "Symbol": ["R1", "R2", "R3"],
            "Country": ["United States"] * 3,
            "Score": [0.99, 0.90, 0.80],
            "Rank": [1, 2, 3],
            "Percentile": [0.01, 0.05, 0.10],
        }
    )


def _meta() -> dict:
    return {
        "R1": {"Country": "United States", "Currency": "USD"},
        "R2": {"Country": "United States", "Currency": "USD"},
        "R3": {"Country": "United States", "Currency": "USD"},
    }


def test_A_rank1_skipped_not_affordable_persisted(tmp_path: Path) -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    costs = CostModel(0.0, 0.0)
    day = pd.Timestamp("2020-06-01")
    plans, skips = plan_integer_affordable_entries(
        _ranking_three(day),
        Portfolio(cash=10_000),
        cfg,
        costs=costs,
        fx=fx,
        signal_day=day,
        close_prices={"R1": 100.0, "R2": 7000 / 150.0, "R3": 5000 / 150.0},
        meta_map=_meta(),
        pending_buy_symbols=set(),
        cooldown_until={},
    )
    sk = next(s for s in skips if s.symbol == "R1")
    assert sk.reason == REASON_NOT_AFFORDABLE
    store = ExecutionAuditStore(tmp_path)
    n = store.append_events(
        [
            {
                "Date": str(day.date()),
                "Symbol": sk.symbol,
                "Rank": sk.rank,
                "Score": sk.score,
                "Decision": DECISION_SKIPPED,
                "Reason": sk.reason,
                "Estimated Cost Base": sk.estimated_cost_base,
                "Cash Before": sk.cash_before,
                "Cash Reserved": sk.cash_reserved,
                "Cash Available": sk.cash_available,
                "experiment_id": EXPERIMENT_ID,
                "order_identity": f"{day.date()}|R1|SKIPPED|{sk.reason}|1",
            }
        ]
    )
    assert n == 1
    df = store.load_frame()
    assert len(df) == 1
    assert df.iloc[0]["Decision"] == DECISION_SKIPPED
    assert df.iloc[0]["Reason"] == REASON_NOT_AFFORDABLE


def test_B_C_queue_and_reservation_insufficient(tmp_path: Path) -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    costs = CostModel(0.0, 0.0)
    day = pd.Timestamp("2020-06-01")
    plans, skips = plan_integer_affordable_entries(
        _ranking_three(day),
        Portfolio(cash=10_000),
        cfg,
        costs=costs,
        fx=fx,
        signal_day=day,
        close_prices={"R1": 100.0, "R2": 7000 / 150.0, "R3": 5000 / 150.0},
        meta_map=_meta(),
        pending_buy_symbols=set(),
        cooldown_until={},
    )
    assert len(plans) == 1 and plans[0].symbol == "R2"
    assert plans[0].estimated_debit_base == pytest.approx(7000.0)
    sk3 = next(s for s in skips if s.symbol == "R3")
    assert sk3.reason == REASON_INSUFFICIENT_CASH
    assert sk3.cash_reserved == pytest.approx(7000.0)
    assert sk3.cash_available == pytest.approx(3000.0)

    store = ExecutionAuditStore(tmp_path)
    rows = []
    for sk in skips:
        rows.append(
            {
                "Date": str(day.date()),
                "Symbol": sk.symbol,
                "Rank": sk.rank,
                "Score": sk.score,
                "Decision": DECISION_SKIPPED,
                "Reason": sk.reason,
                "Estimated Cost Base": sk.estimated_cost_base,
                "Cash Before": sk.cash_before,
                "Cash Reserved": sk.cash_reserved,
                "Cash Available": sk.cash_available,
                "experiment_id": EXPERIMENT_ID,
                "order_identity": f"{day.date()}|{sk.symbol}|SKIPPED|{sk.reason}|{sk.rank}",
            }
        )
    for p in plans:
        rows.append(
            {
                "Date": str(day.date()),
                "Symbol": p.symbol,
                "Rank": p.rank,
                "Score": p.score,
                "Decision": DECISION_QUEUED,
                "Reason": REASON_AFFORDABLE,
                "Requested Notional Base": p.target_notional_base,
                "Tradable Quantity": p.shares,
                "Estimated Cost Base": p.estimated_debit_base,
                "Cash Before": p.cash_before,
                "Cash Reserved": p.cash_reserved + p.estimated_debit_base,
                "Cash Available": p.cash_available,
                "experiment_id": EXPERIMENT_ID,
                "order_identity": f"{day.date()}|{p.symbol}|buy|entry_rank",
            }
        )
    assert store.append_events(rows) == 3
    df = store.load_frame()
    assert set(df["Decision"]) == {DECISION_SKIPPED, DECISION_QUEUED}
    assert (df["Reason"] == REASON_INSUFFICIENT_CASH).any()
    assert (df["Decision"] == DECISION_QUEUED).sum() == 1


def test_D_idempotent_retry_no_duplicates(tmp_path: Path) -> None:
    store = ExecutionAuditStore(tmp_path)
    row = {
        "Date": "2020-06-01",
        "Symbol": "R2",
        "Rank": 2,
        "Decision": DECISION_QUEUED,
        "Reason": REASON_AFFORDABLE,
        "experiment_id": EXPERIMENT_ID,
        "order_identity": "2020-06-01|R2|buy|entry_rank",
    }
    assert store.append_events([row]) == 1
    assert store.append_events([row]) == 0
    assert len(store.load_frame()) == 1


def test_E_queued_to_filled_traceable(tmp_path: Path) -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    days = pd.bdate_range("2020-06-01", periods=4)
    px = 7000 / 150.0
    rows = []
    for d in days:
        rows.append(
            {
                "Date": d,
                "Symbol": "R2",
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
                "Symbol": "R2",
                "Country": "United States",
                "Score": 0.9,
                "Rank": 1,
                "Percentile": 0.05,
            }
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
    events = result.meta["execution_events"]
    decisions = [e["Decision"] for e in events if e["Symbol"] == "R2"]
    assert DECISION_QUEUED in decisions
    assert DECISION_FILLED in decisions
    store = ExecutionAuditStore(tmp_path)
    for e in events:
        e["experiment_id"] = EXPERIMENT_ID
    store.append_events(events)
    store.append_events(events)  # retry
    df = store.load_frame()
    assert set(df.loc[df["Symbol"] == "R2", "Decision"]) == {
        DECISION_QUEUED,
        DECISION_FILLED,
    }
    assert "R2" in result.portfolio.positions


def test_F_gap_up_rejected_releases_reservation() -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    days = pd.bdate_range("2020-06-01", periods=3)
    # Signal close cheap enough to queue; next open gap-up unaffordable.
    close_px = 60.0  # 9000 JPY
    open_px = 70.0  # 10500 JPY > 10000
    rows = []
    for i, d in enumerate(days):
        px_o = close_px if i == 0 else open_px
        px_c = close_px if i == 0 else open_px
        rows.append(
            {
                "Date": d,
                "Symbol": "GAP",
                "Country": "United States",
                "Currency": "USD",
                "Open": open_px if i > 0 else close_px,
                "High": px_c,
                "Low": px_c,
                "Close": close_px if i == 0 else open_px,
            }
        )
    # Day0 close=60 queue; Day1 open=70 reject
    prices = pd.DataFrame(
        [
            {
                "Date": days[0],
                "Symbol": "GAP",
                "Country": "United States",
                "Currency": "USD",
                "Open": 60.0,
                "High": 60.0,
                "Low": 60.0,
                "Close": 60.0,
            },
            {
                "Date": days[1],
                "Symbol": "GAP",
                "Country": "United States",
                "Currency": "USD",
                "Open": 70.0,
                "High": 70.0,
                "Low": 70.0,
                "Close": 70.0,
            },
            {
                "Date": days[2],
                "Symbol": "GAP",
                "Country": "United States",
                "Currency": "USD",
                "Open": 70.0,
                "High": 70.0,
                "Low": 70.0,
                "Close": 70.0,
            },
        ]
    )
    rankings = pd.DataFrame(
        [
            {
                "Date": days[0],
                "Symbol": "GAP",
                "Country": "United States",
                "Score": 0.9,
                "Rank": 1,
                "Percentile": 0.05,
            }
        ]
    )
    port = Portfolio(cash=10_000)
    eng = SimulationEngine(cfg, fx=fx, countries={"United States"})
    result = eng.run(
        prices=prices,
        rankings=rankings,
        start=days[0],
        end=days[-1],
        portfolio=port,
    )
    events = result.meta["execution_events"]
    assert any(e["Decision"] == DECISION_QUEUED for e in events)
    rej = [e for e in events if e["Decision"] == DECISION_REJECTED]
    assert rej and rej[0]["Reason"] == REASON_FILL_PRICE_UNAFFORDABLE
    assert result.portfolio.positions == {}
    assert result.portfolio.cash == pytest.approx(10_000.0)
    # No lingering pending buys
    pending = result.meta["pending_orders"]
    assert not any(o.side == "buy" for o in pending)


def test_G_multiple_pending_reserved_le_cash() -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    costs = CostModel(0.0, 0.0)
    day = pd.Timestamp("2020-06-01")
    # Cheap names so multiple can queue under reservation.
    ranking = pd.DataFrame(
        {
            "Date": [day, day],
            "Symbol": ["A", "B"],
            "Country": ["United States", "United States"],
            "Score": [0.99, 0.9],
            "Rank": [1, 2],
            "Percentile": [0.01, 0.05],
        }
    )
    meta = {
        "A": {"Country": "United States", "Currency": "USD"},
        "B": {"Country": "United States", "Currency": "USD"},
    }
    plans, _ = plan_integer_affordable_entries(
        ranking,
        Portfolio(cash=10_000),
        cfg,
        costs=costs,
        fx=fx,
        signal_day=day,
        close_prices={"A": 20.0, "B": 20.0},
        meta_map=meta,
        pending_buy_symbols=set(),
        cooldown_until={},
    )
    total = sum(p.estimated_debit_base for p in plans)
    assert total <= 10_000 + 1e-6
    assert len(plans) >= 1


def test_H_sell_does_not_reserve_buy_cash() -> None:
    cfg = _sc_cfg()
    fx = _fx(150.0)
    # Build portfolio with a position then ensure sell pending doesn't reduce buy available.
    from src.simulation.orders import Order

    costs = CostModel(0.0, 0.0)
    port = Portfolio(cash=5_000)
    # Simulate existing reserved buy only counts buy side
    pending = [
        Order(
            symbol="S",
            country="United States",
            currency="USD",
            side="sell",
            quantity=1.0,
            signal_date=pd.Timestamp("2020-06-01"),
            reason="holding_period",
        )
    ]
    from src.simulation.engine import _pending_buy_reserved_cost

    assert _pending_buy_reserved_cost(pending, costs) == 0.0
    day = pd.Timestamp("2020-06-02")
    ranking = pd.DataFrame(
        {
            "Date": [day],
            "Symbol": ["BUY"],
            "Country": ["United States"],
            "Score": [0.9],
            "Rank": [1],
            "Percentile": [0.05],
        }
    )
    plans, _ = plan_integer_affordable_entries(
        ranking,
        port,
        cfg,
        costs=costs,
        fx=fx,
        signal_day=day,
        close_prices={"BUY": 20.0},
        meta_map={"BUY": {"Country": "United States", "Currency": "USD"}},
        pending_buy_symbols=set(),
        cooldown_until={},
        remaining_cash=port.cash - _pending_buy_reserved_cost(pending, costs),
        cash_reserved_before=_pending_buy_reserved_cost(pending, costs),
    )
    assert plans and plans[0].symbol == "BUY"


def test_I_legacy_policy_unchanged() -> None:
    cfg = parse_simulation_config(
        {
            "initial_capital": 10_000_000,
            "candidate": {"mode": "top_percentile", "top_percentile": 0.5, "top_n": 3},
            "portfolio": {
                "max_positions": 5,
                "max_position_weight": 0.2,
                "max_country_weight": 1.0,
            },
            "exit_rules": {"holding_period_days": 20},
            "costs": {"commission_rate": 0.001, "slippage_rate": 0.0005},
            "ranking": {
                "label_scheme": "B",
                "gain_name": "moderate_exp",
                "feature_set": "A",
                "param_preset": "default",
                "horizon_days": 5,
            },
            "fx": {},
            "benchmarks": {},
            "strategies": {},
        }
    )
    assert cfg.execution_policy.sizing_mode == "legacy_equal_weight"
    port = Portfolio(cash=10_000_000)
    assert equal_weight_notional(port, cfg, 2) == pytest.approx(
        min(10_000_000 / 5, 10_000_000 * 0.2)
    )
    days = pd.bdate_range("2020-06-01", periods=3)
    prices = pd.DataFrame(
        [
            {
                "Date": d,
                "Symbol": "X",
                "Country": "United States",
                "Currency": "USD",
                "Open": 50.0,
                "High": 50.0,
                "Low": 50.0,
                "Close": 50.0,
            }
            for d in days
        ]
    )
    rankings = pd.DataFrame(
        [
            {
                "Date": days[0],
                "Symbol": "X",
                "Country": "United States",
                "Score": 1.0,
                "Rank": 1,
                "Percentile": 0.01,
            }
        ]
    )
    eng = SimulationEngine(cfg, fx=_fx(100.0), countries={"United States"})
    result = eng.run(
        prices=prices,
        rankings=rankings,
        start=days[0],
        end=days[-1],
        portfolio=Portfolio(cash=10_000_000),
    )
    assert result.meta.get("execution_events") == []


def test_J_canonical_10m_untouched() -> None:
    canonical = PROJECT_ROOT / "data" / "paper"
    before = {p.name: p.stat().st_mtime_ns for p in canonical.iterdir() if p.is_file()}
    with pytest.raises(Exception):
        assert_small_capital_state_dir(canonical)
    after = {p.name: p.stat().st_mtime_ns for p in canonical.iterdir() if p.is_file()}
    assert before == after


def test_K_corrupt_audit_history_safe(tmp_path: Path) -> None:
    store = ExecutionAuditStore(tmp_path)
    store.path.write_text("{not csv", encoding="utf-8")
    df = store.load_frame()
    assert list(df.columns) == AUDIT_COLUMNS
    assert df.empty
    n = store.append_events(
        [
            {
                "Date": "2020-06-01",
                "Symbol": "Z",
                "Decision": DECISION_SKIPPED,
                "Reason": REASON_NOT_AFFORDABLE,
                "experiment_id": EXPERIMENT_ID,
                "order_identity": "x",
            }
        ]
    )
    assert n == 1
    assert len(store.load_frame()) == 1


def test_L_model_ranking_outputs_unchanged_by_audit_module() -> None:
    # Audit store must not alter ranking helper outputs.
    from src.simulation.ranking import attach_country_ranks

    df = pd.DataFrame(
        {
            "Date": ["2020-01-02"] * 3,
            "Symbol": ["A", "B", "C"],
            "Region": ["Japan"] * 3,
            "score": [0.1, 0.9, 0.5],
        }
    )
    ranked = attach_country_ranks(df)
    assert list(ranked.sort_values("Rank")["Symbol"]) == ["B", "C", "A"]
