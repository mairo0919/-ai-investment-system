"""Phase D1: Dashboard performance / equity metrics tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard.data import DashboardDataSource, resolve_initial_capital
from src.dashboard.metrics import (
    build_overview,
    build_performance,
    compute_dashboard_performance,
    get_equity_curve,
)
from src.simulation.metrics import compute_performance
from src.simulation.orders import EquityPoint


@pytest.fixture
def initial_capital() -> float:
    return resolve_initial_capital()


def _equity_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "Date": "2026-08-01",
                "Cash": 10_000_000.0,
                "Position Value": 0.0,
                "Total Equity": 10_000_000.0,
                "Drawdown": 0.0,
                "Realized PnL": 0.0,
                "Unrealized PnL": 0.0,
                "N Positions": 0,
                "Transaction Costs": 0.0,
            },
            {
                "Date": "2026-08-04",
                "Cash": 8_000_000.0,
                "Position Value": 2_200_000.0,
                "Total Equity": 10_200_000.0,
                "Drawdown": 0.0,
                "Realized PnL": 0.0,
                "Unrealized PnL": 200_000.0,
                "N Positions": 1,
                "Transaction Costs": 1000.0,
            },
            {
                "Date": "2026-08-05",
                "Cash": 8_000_000.0,
                "Position Value": 1_900_000.0,
                "Total Equity": 9_900_000.0,
                "Drawdown": 9_900_000.0 / 10_200_000.0 - 1.0,
                "Realized PnL": 0.0,
                "Unrealized PnL": -100_000.0,
                "N Positions": 1,
                "Transaction Costs": 1000.0,
            },
            {
                "Date": "2026-09-01",
                "Cash": 10_300_000.0,
                "Position Value": 0.0,
                "Total Equity": 10_300_000.0,
                "Drawdown": 0.0,
                "Realized PnL": 300_000.0,
                "Unrealized PnL": 0.0,
                "N Positions": 0,
                "Transaction Costs": 2000.0,
            },
        ]
    ).to_csv(path, index=False)


def _trades_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "Symbol": "AAPL",
                "Country": "United States",
                "Currency": "USD",
                "Entry Date": "2026-08-04",
                "Exit Date": "2026-09-01",
                "Entry Price": 100.0,
                "Exit Price": 110.0,
                "Quantity": 10.0,
                "Gross PnL": 320_000.0,
                "Net PnL": 300_000.0,
                "Return": 0.1,
                "Holding Days": 20,
                "Exit Reason": "holding_period",
                "Entry Score": 0.8,
                "Entry Rank": 2,
                "Entry Cost Base": 2_000_000.0,
                "Exit Proceeds Base": 2_320_000.0,
                "Commission Base": 15_000.0,
                "Slippage Impact Base": 5_000.0,
                "idempotency_key": "FILL|2026-09-01|AAPL|sell|holding_period",
                "model_id": "paper_x",
                "config_id": "FINAL_US_PHASE4C",
            },
            {
                "Symbol": "MSFT",
                "Country": "United States",
                "Currency": "USD",
                "Entry Date": "2026-08-04",
                "Exit Date": "2026-08-20",
                "Entry Price": 50.0,
                "Exit Price": 45.0,
                "Quantity": 5.0,
                "Gross PnL": -40_000.0,
                "Net PnL": -50_000.0,
                "Return": -0.1,
                "Holding Days": 10,
                "Exit Reason": "take_profit",
                "Entry Score": 0.7,
                "Entry Rank": 5,
                "Entry Cost Base": 500_000.0,
                "Exit Proceeds Base": 450_000.0,
                "Commission Base": 8_000.0,
                "Slippage Impact Base": 2_000.0,
                "idempotency_key": "FILL|2026-08-20|MSFT|sell|take_profit",
                "model_id": "paper_x",
                "config_id": "FINAL_US_PHASE4C",
            },
        ]
    ).to_csv(path, index=False)


def test_equity_curve_shape(tmp_path: Path, initial_capital: float) -> None:
    _equity_csv(tmp_path / "equity_history.csv")
    src = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    curve = get_equity_curve(src)
    assert len(curve) == 4
    assert set(curve[0].keys()) >= {
        "date",
        "total_equity",
        "cash",
        "position_value",
        "drawdown",
    }
    assert curve[-1]["date"] == "2026-09-01"
    assert curve[-1]["total_equity"] == pytest.approx(10_300_000.0)


def test_performance_reuses_compute_performance(
    tmp_path: Path, initial_capital: float
) -> None:
    _equity_csv(tmp_path / "equity_history.csv")
    _trades_csv(tmp_path / "trade_history.csv")
    (tmp_path / "portfolio.json").write_text(
        json.dumps(
            {
                "cash": 10_300_000.0,
                "base_currency": "JPY",
                "realized_pnl": 250_000.0,
                "unrealized_pnl": 0.0,
                "total_equity": 10_300_000.0,
                "position_value": 0.0,
                "total_commission": 2000.0,
                "total_slippage_impact": 0.0,
                "peak_equity": 10_300_000.0,
                "n_positions": 0,
                "model_id": "paper_x",
                "config_id": "FINAL_US_PHASE4C",
            }
        ),
        encoding="utf-8",
    )

    src = DashboardDataSource(tmp_path, experiment_id="perf", initial_capital=initial_capital)
    perf = build_performance(src)

    equity = pd.read_csv(tmp_path / "equity_history.csv")
    trades = pd.read_csv(tmp_path / "trade_history.csv")
    curve = [
        EquityPoint(
            date=pd.Timestamp(r["Date"]),
            cash=float(r["Cash"]),
            position_value=float(r["Position Value"]),
            total_equity=float(r["Total Equity"]),
            drawdown=float(r["Drawdown"]),
            realized_pnl=float(r["Realized PnL"]),
            unrealized_pnl=float(r["Unrealized PnL"]),
            n_positions=int(r["N Positions"]),
        )
        for _, r in equity.iterrows()
    ]
    core = compute_performance(
        curve,
        trades,
        initial_capital=initial_capital,
        transaction_costs_total=float(equity["Transaction Costs"].iloc[-1]),
    )

    assert perf["experiment_id"] == "perf"
    assert perf["total_return"] == pytest.approx(core["total_return"])
    assert perf["cumulative_return"] == pytest.approx(core["total_return"])
    assert perf["sharpe"] == core["sharpe"] or (
        perf["sharpe"] is not None
        and core["sharpe"] is not None
        and perf["sharpe"] == pytest.approx(core["sharpe"])
    )
    assert perf["max_drawdown"] == pytest.approx(core["max_drawdown"])
    assert perf["win_rate"] == pytest.approx(core["win_rate"])
    assert perf["profit_factor"] == pytest.approx(core["profit_factor"])
    assert perf["average_win"] == pytest.approx(core["average_profit"])
    assert perf["average_loss"] == pytest.approx(core["average_loss"])
    assert perf["completed_trades"] == 2
    assert perf["realized_pnl"] == pytest.approx(250_000.0)
    assert len(perf["daily_returns"]) >= 1
    assert len(perf["monthly_returns"]) >= 1
    assert perf["average_win"] is not None
    assert perf["average_loss"] is not None


def test_average_win_loss_via_dashboard_glue(initial_capital: float) -> None:
    equity = pd.DataFrame(
        [
            {
                "Date": "2026-08-01",
                "Cash": 10_000_000.0,
                "Position Value": 0.0,
                "Total Equity": 10_000_000.0,
                "Drawdown": 0.0,
                "Realized PnL": 0.0,
                "Unrealized PnL": 0.0,
                "N Positions": 0,
                "Transaction Costs": 0.0,
            },
            {
                "Date": "2026-08-02",
                "Cash": 10_100_000.0,
                "Position Value": 0.0,
                "Total Equity": 10_100_000.0,
                "Drawdown": 0.0,
                "Realized PnL": 100_000.0,
                "Unrealized PnL": 0.0,
                "N Positions": 0,
                "Transaction Costs": 0.0,
            },
        ]
    )
    trades = pd.DataFrame(
        {
            "Net PnL": [100.0, -40.0, 50.0],
            "Holding Days": [5, 3, 4],
            "Entry Cost Base": [1000.0, 1000.0, 1000.0],
        }
    )
    out = compute_dashboard_performance(equity, trades, initial_capital=initial_capital)
    assert out["average_win"] == pytest.approx(75.0)
    assert out["average_loss"] == pytest.approx(-40.0)
    assert out["completed_trades"] == 3


def test_overview_uses_same_return_definition(
    tmp_path: Path, initial_capital: float
) -> None:
    _equity_csv(tmp_path / "equity_history.csv")
    (tmp_path / "portfolio.json").write_text(
        json.dumps(
            {
                "cash": 10_300_000.0,
                "base_currency": "JPY",
                "realized_pnl": 300_000.0,
                "unrealized_pnl": 0.0,
                "total_equity": 10_300_000.0,
                "position_value": 0.0,
                "total_commission": 2000.0,
                "total_slippage_impact": 0.0,
                "peak_equity": 10_300_000.0,
                "n_positions": 0,
                "last_processed_date": "2026-09-01",
                "model_id": "paper_x",
                "config_id": "FINAL_US_PHASE4C",
            }
        ),
        encoding="utf-8",
    )
    _trades_csv(tmp_path / "trade_history.csv")
    src = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    ov = build_overview(src)
    perf = build_performance(src)
    assert ov["total_return"] == pytest.approx(perf["total_return"])
    assert ov["completed_trades"] == 2
    assert ov["initial_capital"] == initial_capital


def test_missing_equity_performance_nulls(tmp_path: Path, initial_capital: float) -> None:
    src = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    perf = build_performance(src)
    assert perf["total_return"] is None
    assert perf["sharpe"] is None
    assert perf["average_win"] is None
    assert perf["completed_trades"] == 0
    assert perf["daily_returns"] == []
    assert perf["monthly_returns"] == []
