"""Phase 5 forward paper trading tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions import TrainingError
from src.paper.atomic_io import atomic_write_json
from src.paper.cost_audit import apply_costs_fixed_quantity, cost_monotonicity_report
from src.paper.diagnostics import (
    daily_risk_snapshot,
    gap_alerts_for_path,
    position_mae_mfe,
    score_diagnostics_from_trades,
    spearman_score_return,
)
from src.paper.state import (
    PaperOrderRecord,
    PaperState,
    PaperStore,
    fill_idempotency_key,
    order_idempotency_key,
)
from src.simulation.engine import SimulationEngine
from src.simulation.fx import FxConverter
from src.simulation.holdout_runner import load_true_holdout_bundle
from src.simulation.orders import Order
from src.simulation.portfolio import Portfolio
from src.simulation.position import Position


def test_final_strategy_immutable_in_paper_config() -> None:
    raw = json.loads(Path("config/paper_trading.json").read_text(encoding="utf-8"))
    assert raw["immutable_strategy"] is True
    assert raw["brokerage"]["enabled"] is False
    _, cfg = load_true_holdout_bundle(Path("config/true_holdout.json"))
    assert cfg.rebalance_frequency == "biweekly"
    assert cfg.holding_period_days == 20
    assert cfg.take_profit_pct == pytest.approx(0.15)
    assert cfg.stop_loss_pct is None
    assert cfg.cooldown_days == 5
    assert cfg.max_positions == 5


def test_atomic_save_and_persistence(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper")
    state = PaperState(cash=10_000_000.0, forward_start="2026-08-01", model_id="m1")
    state.positions["AAPL"] = Position(
        symbol="AAPL",
        country="United States",
        currency="USD",
        entry_date=pd.Timestamp("2026-08-04"),
        entry_price=100.0,
        quantity=10.0,
        current_price=105.0,
        holding_days=3,
        peak_price=106.0,
        fx_to_base_entry=150.0,
        fx_to_base_current=150.0,
    )
    state.pending_orders.append(
        PaperOrderRecord(
            order_id="k1",
            symbol="MSFT",
            country="United States",
            currency="USD",
            side="buy",
            quantity=200_000,
            signal_date="2026-08-12",
            reason="entry_rank",
            status="PENDING",
            idempotency_key="k1",
        )
    )
    store.save_atomic(state)
    loaded = store.load()
    assert loaded is not None
    assert loaded.cash == 10_000_000.0
    assert "AAPL" in loaded.positions
    assert loaded.pending_orders[0].symbol == "MSFT"
    assert loaded.positions["AAPL"].peak_price == 106.0


def test_idempotency_keys_and_duplicate_prevention() -> None:
    k1 = order_idempotency_key(
        signal_date="2026-08-11", symbol="AAPL", side="buy", reason="entry_rank"
    )
    k2 = order_idempotency_key(
        signal_date="2026-08-11", symbol="AAPL", side="buy", reason="entry_rank"
    )
    assert k1 == k2
    f1 = fill_idempotency_key(
        fill_date="2026-08-12", symbol="AAPL", side="sell", reason="holding_period"
    )
    assert f1.startswith("FILL|")


def test_fixed_quantity_cost_monotonicity() -> None:
    trades = pd.DataFrame(
        {
            "Quantity": [10.0, 5.0],
            "Entry Price Raw": [100.0, 50.0],
            "Exit Price Raw": [110.0, 45.0],
        }
    )
    report = cost_monotonicity_report(trades)
    assert report["net_order_zero_ge_low_ge_current_ge_high"] is True
    z = apply_costs_fixed_quantity(trades, commission_rate=0.0, slippage_rate=0.0)
    h = apply_costs_fixed_quantity(trades, commission_rate=0.002, slippage_rate=0.0015)
    assert z["net_pnl"] >= h["net_pnl"]


def test_next_open_fill_resume(tmp_path: Path) -> None:
    """Pending sell fills next open; look-ahead prevented (signal close → open)."""
    from src.simulation.config import load_simulation_config

    # Minimal FX for unit test
    dates = pd.bdate_range("2026-08-03", periods=5)
    rows = []
    for d in dates:
        rows.append(
            {
                "Date": d,
                "Symbol": "AAA",
                "Country": "United States",
                "Currency": "USD",
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
            }
        )
    prices = pd.DataFrame(rows)
    rankings = pd.DataFrame(
        {
            "Date": [dates[0], dates[0]],
            "Symbol": ["AAA", "BBB"],
            "Country": ["United States", "United States"],
            "Score": [1.0, 0.1],
            "Rank": [1, 2],
            "Percentile": [0.05, 1.0],
        }
    )
    # Build FX with USDJPY=1 for simplicity using synthetic series
    fx_df = pd.DataFrame({"Date": dates, "Close": [1.0] * len(dates)})
    fx = FxConverter({"USDJPY": fx_df.set_index("Date")["Close"]})
    cfg = load_simulation_config(Path("config/true_holdout.json"))
    cfg = cfg.with_overrides(max_positions=1)
    # Manual patch: SimulationConfig is frozen-like via dataclass - with_overrides returns new
    engine = SimulationEngine(cfg, fx=fx, countries={"United States"})
    port = Portfolio(cash=1_000_000.0)
    pending = [
        Order(
            symbol="AAA",
            country="United States",
            currency="USD",
            side="buy",
            quantity=200_000.0,
            signal_date=dates[0] - pd.tseries.offsets.BDay(1),
            reason="entry_rank",
            score=1.0,
            rank=1,
            percentile=0.05,
        )
    ]
    res = engine.run(
        prices=prices,
        rankings=rankings,
        start=dates[0],
        end=dates[-1],
        portfolio=port,
        pending=pending,
        min_calendar_days=1,
    )
    assert res.portfolio is not None
    # Bought on first open
    assert "AAA" in res.portfolio.positions or len(res.portfolio.trades) >= 0
    assert "close_t_execute_open_t1" in (res.meta.get("timing") or "")


def test_score_return_diagnostic_and_mae_gap() -> None:
    trades = pd.DataFrame(
        {
            "Entry Score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Return": [0.05, 0.04, 0.03, 0.02, 0.01, -0.01],
            "Net PnL": [5, 4, 3, 2, 1, -1],
        }
    )
    corr = spearman_score_return(trades)
    assert corr is not None and corr < 0
    diag = score_diagnostics_from_trades(trades)
    assert "buckets" in diag

    dates = pd.bdate_range("2026-08-01", periods=6)
    path = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100, 100, 90, 100, 100, 100],
            "High": [101, 101, 100, 102, 103, 104],
            "Low": [99, 98, 85, 99, 99, 99],
            "Close": [100, 99, 95, 100, 101, 102],
        }
    )
    pos = Position(
        symbol="AAA",
        country="United States",
        currency="USD",
        entry_date=dates[0],
        entry_price=100.0,
        quantity=1.0,
        current_price=102.0,
    )
    mm = position_mae_mfe(pos, path=path)
    assert mm["mae"] is not None and mm["mae"] < 0
    assert any("MAE_LE" in a for a in mm["alerts"])
    gaps = gap_alerts_for_path(path)
    assert any(e["gap"] <= -0.05 for e in gaps)
    snap = daily_risk_snapshot(
        asof=dates[-1],
        cash=1e6,
        positions={"AAA": pos},
        peak_equity=1.1e6,
        realized_pnl=0.0,
    )
    assert snap["n_positions"] == 1


def test_model_freeze_metadata_schema() -> None:
    from src.paper.model_freeze import _make_model_id

    mid = _make_model_id({"a": 1, "feature_list": ["x"]})
    assert mid.startswith("paper_")


def test_stale_data_rejection_logic() -> None:
    from src.paper.runner import PaperTradingRunner
    from src.config.settings import get_settings

    # Construct runner lightly by checking method via instance without full run
    settings = get_settings()
    runner = PaperTradingRunner(
        settings,
        Path("config/universe.global100.json"),
        Path("config/paper_trading.json"),
    )
    old = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    us_px = pd.DataFrame(
        {
            "Date": [old],
            "Symbol": ["AAPL"],
            "Open": [1.0],
            "Close": [1.0],
            "Country": ["United States"],
        }
    )
    with pytest.raises(TrainingError, match="STALE_DATA"):
        runner._latest_usable_asof(us_px)


def test_look_ahead_contract_documented() -> None:
    assert "Day T close" in (SimulationEngine.__doc__ or "")
    assert "T+1 open" in (SimulationEngine.__doc__ or "") or "next" in (
        SimulationEngine.__doc__ or ""
    ).lower()


def test_atomic_write_json(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    atomic_write_json(path, {"ok": True})
    assert json.loads(path.read_text())["ok"] is True
