"""Phase D1: Dashboard read-only data layer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard.data import DashboardDataSource, resolve_initial_capital
from src.dashboard.metrics import (
    build_current_portfolio,
    build_overview,
    build_system_status,
    build_trade_history,
    get_equity_curve,
    get_rankings,
    list_ranking_dates,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _portfolio(**overrides: object) -> dict:
    base = {
        "cash": 5_000_000.0,
        "base_currency": "JPY",
        "realized_pnl": 100_000.0,
        "unrealized_pnl": 50_000.0,
        "total_equity": 10_150_000.0,
        "position_value": 5_150_000.0,
        "total_commission": 1_000.0,
        "total_slippage_impact": 500.0,
        "peak_equity": 10_200_000.0,
        "last_processed_date": "2026-09-17",
        "forward_start": "2026-08-01",
        "model_id": "paper_4ab12cb5ade1681f",
        "config_id": "FINAL_US_PHASE4C",
        "n_positions": 1,
        "cooldown_until": {},
        "seen_order_keys": [],
        "seen_fill_keys": [],
        "brokerage_orders_submitted": 0,
        "brokerage_enabled": False,
    }
    base.update(overrides)
    return base


@pytest.fixture
def initial_capital() -> float:
    return resolve_initial_capital()


def test_resolve_initial_capital_from_config_ssot(initial_capital: float) -> None:
    assert initial_capital == pytest.approx(10_000_000.0)
    # Must not be a dashboard-local magic number path — same as simulation config.
    assert initial_capital > 0


def test_empty_state_directory(tmp_path: Path, initial_capital: float) -> None:
    src = DashboardDataSource(
        tmp_path, experiment_id="us_quant_paper", initial_capital=initial_capital
    )
    assert src.load_portfolio() is None
    assert src.load_positions() is None
    assert src.load_pending_orders() is None
    assert src.load_trade_history() is None
    assert src.load_equity_history() is None
    assert src.load_latest_validity() is None
    assert src.load_run_health() is None
    assert src.load_ranking_dates() == []
    assert src.load_rankings("2026-09-17") is None

    ov = build_overview(src)
    assert ov["experiment_id"] == "us_quant_paper"
    assert ov["initial_capital"] == initial_capital
    assert ov["current_equity"] is None
    assert ov["completed_trades"] is None
    assert ov["validity_status"] is None
    assert get_equity_curve(src) == []
    assert build_current_portfolio(src) == []
    assert build_trade_history(src) == []
    assert list(tmp_path.iterdir()) == []  # read-only: no mkdir / no writes


def test_portfolio_only(tmp_path: Path, initial_capital: float) -> None:
    _write_json(tmp_path / "portfolio.json", _portfolio())
    src = DashboardDataSource(tmp_path, experiment_id="exp_a", initial_capital=initial_capital)
    ov = build_overview(src)
    assert ov["current_equity"] == pytest.approx(10_150_000.0)
    assert ov["total_pnl"] == pytest.approx(150_000.0)
    assert ov["total_return"] == pytest.approx(0.015)
    assert ov["cash"] == pytest.approx(5_000_000.0)
    assert ov["model_id"] == "paper_4ab12cb5ade1681f"
    assert ov["config_id"] == "FINAL_US_PHASE4C"
    assert ov["experiment_id"] == "exp_a"
    assert ov["completed_trades"] is None
    assert ov["latest_asof"] == "2026-09-17"


def test_corrupt_json_and_empty_csv(tmp_path: Path, initial_capital: float) -> None:
    (tmp_path / "portfolio.json").write_text("{not-json", encoding="utf-8")
    (tmp_path / "positions.json").write_text("[]", encoding="utf-8")
    (tmp_path / "equity_history.csv").write_text("", encoding="utf-8")
    (tmp_path / "trade_history.csv").write_text(
        "Symbol,Entry Date,Exit Date,Net PnL,Return\n", encoding="utf-8"
    )
    src = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    assert src.load_portfolio() is None
    assert src.load_positions() == []
    # empty / header-only: may be empty DataFrame or None depending on parser
    eq = src.load_equity_history()
    assert eq is None or eq.empty
    tr = src.load_trade_history()
    assert tr is not None
    assert tr.empty
    ov = build_overview(src)
    assert ov["current_equity"] is None
    assert ov["completed_trades"] == 0


def test_read_only_does_not_mutate_files(tmp_path: Path, initial_capital: float) -> None:
    port_path = tmp_path / "portfolio.json"
    _write_json(port_path, _portfolio())
    before = port_path.read_text(encoding="utf-8")
    mtime = port_path.stat().st_mtime_ns

    src = DashboardDataSource(tmp_path, experiment_id="ro", initial_capital=initial_capital)
    build_overview(src)
    build_performance = __import__(
        "src.dashboard.metrics", fromlist=["build_performance"]
    ).build_performance
    build_performance(src)
    build_system_status(src)
    get_equity_curve(src)

    assert port_path.read_text(encoding="utf-8") == before
    assert port_path.stat().st_mtime_ns == mtime
    assert not (tmp_path / "rankings").exists()


def test_rankings_list_and_get(tmp_path: Path, initial_capital: float) -> None:
    rankings = tmp_path / "rankings"
    rankings.mkdir()
    pd.DataFrame(
        [
            {
                "Date": "2026-09-16",
                "Symbol": "AAPL",
                "Score": 0.9,
                "Rank": 1,
                "Percentile": 0.05,
                "Selected": True,
                "model_id": "paper_x",
                "Country": "United States",
                "Close": 200.0,
            },
            {
                "Date": "2026-09-16",
                "Symbol": "7203.T",
                "Score": 0.1,
                "Rank": 50,
                "Percentile": 0.9,
                "Selected": False,
                "model_id": "paper_x",
                "Country": "Japan",
                "Close": 3000.0,
            },
        ]
    ).to_csv(rankings / "2026-09-16.csv", index=False)
    pd.DataFrame(
        [
            {
                "Date": "2026-09-17",
                "Symbol": "MSFT",
                "Score": 0.8,
                "Rank": 2,
                "Percentile": 0.08,
                "Selected": True,
                "model_id": "paper_x",
                "Country": "United States",
                "Close": 400.0,
            }
        ]
    ).to_csv(rankings / "2026-09-17.csv", index=False)

    src = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    dates = list_ranking_dates(src)
    assert dates == ["2026-09-16", "2026-09-17"]
    rows = get_rankings(src, "2026-09-16")
    assert len(rows) == 2
    countries = {r["country"] for r in rows}
    assert "Japan" in countries and "United States" in countries
    assert rows[0]["symbol"] in ("AAPL", "7203.T")


def test_portfolio_join_entry_vs_current_rank(tmp_path: Path, initial_capital: float) -> None:
    _write_json(
        tmp_path / "portfolio.json",
        _portfolio(last_processed_date="2026-09-17", n_positions=1),
    )
    _write_json(
        tmp_path / "positions.json",
        [
            {
                "symbol": "AAPL",
                "country": "United States",
                "currency": "USD",
                "entry_date": "2026-09-01",
                "entry_price": 100.0,
                "quantity": 10.0,
                "current_price": 110.0,
                "market_value": 165000.0,
                "unrealized_pnl": 15000.0,
                "holding_days": 12,
                "entry_score": 0.5,
                "entry_rank": 7,
                "entry_percentile": 0.2,
                "pending_exit_reason": None,
            }
        ],
    )
    _write_json(
        tmp_path / "run_health.json",
        {
            "status": "SUCCESS",
            "last_asof": "2026-09-17",
            "last_success_utc": "2026-09-17T12:00:00+00:00",
            "model_id": "paper_4ab12cb5ade1681f",
            "processed_sessions": 1,
        },
    )
    rankings = tmp_path / "rankings"
    rankings.mkdir()
    pd.DataFrame(
        [
            {
                "Date": "2026-09-17",
                "Symbol": "AAPL",
                "Score": 0.95,
                "Rank": 3,
                "Percentile": 0.04,
                "Selected": True,
                "model_id": "paper_4ab12cb5ade1681f",
                "Country": "United States",
                "Close": 110.0,
            }
        ]
    ).to_csv(rankings / "2026-09-17.csv", index=False)

    src = DashboardDataSource(tmp_path, experiment_id="us", initial_capital=initial_capital)
    rows = build_current_portfolio(src)
    assert len(rows) == 1
    row = rows[0]
    assert row["entry_rank"] == 7
    assert row["current_rank"] == 3
    assert row["current_score"] == pytest.approx(0.95)
    assert row["selected"] is True


def test_portfolio_without_ranking_null_current_rank(
    tmp_path: Path, initial_capital: float
) -> None:
    _write_json(
        tmp_path / "positions.json",
        [
            {
                "symbol": "MSFT",
                "country": "United States",
                "currency": "USD",
                "entry_date": "2026-09-01",
                "entry_price": 50.0,
                "quantity": 1.0,
                "current_price": 55.0,
                "market_value": 55.0,
                "unrealized_pnl": 5.0,
                "holding_days": 2,
                "entry_rank": 1,
            }
        ],
    )
    src = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    rows = build_current_portfolio(src)
    assert rows[0]["entry_rank"] == 1
    assert rows[0]["current_rank"] is None
    assert rows[0]["current_score"] is None


def test_trade_history_ordering(tmp_path: Path, initial_capital: float) -> None:
    pd.DataFrame(
        [
            {
                "Symbol": "A",
                "Country": "United States",
                "Currency": "USD",
                "Entry Date": "2026-08-01",
                "Exit Date": "2026-08-10",
                "Entry Price": 10.0,
                "Exit Price": 11.0,
                "Quantity": 1.0,
                "Gross PnL": 100.0,
                "Net PnL": 90.0,
                "Return": 0.1,
                "Holding Days": 9,
                "Exit Reason": "time",
                "Entry Rank": 1,
                "model_id": "m1",
                "config_id": "c1",
            },
            {
                "Symbol": "B",
                "Country": "United States",
                "Currency": "USD",
                "Entry Date": "2026-08-05",
                "Exit Date": "2026-09-01",
                "Entry Price": 20.0,
                "Exit Price": 18.0,
                "Quantity": 1.0,
                "Gross PnL": -50.0,
                "Net PnL": -60.0,
                "Return": -0.1,
                "Holding Days": 20,
                "Exit Reason": "stop",
                "Entry Rank": 2,
                "model_id": "m1",
                "config_id": "c1",
            },
        ]
    ).to_csv(tmp_path / "trade_history.csv", index=False)

    src = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    rows = build_trade_history(src)
    assert [r["symbol"] for r in rows] == ["B", "A"]
    assert rows[0]["exit_reason"] == "stop"
    assert rows[0]["net_pnl"] == pytest.approx(-60.0)


def test_system_status_and_no_secrets(tmp_path: Path, initial_capital: float) -> None:
    _write_json(
        tmp_path / "run_health.json",
        {
            "status": "ERROR",
            "last_run_started_utc": "2026-09-17T01:00:00+00:00",
            "last_run_finished_utc": "2026-09-17T01:05:00+00:00",
            "last_success_utc": "2026-09-16T01:00:00+00:00",
            "last_asof": "2026-09-16",
            "error_type": "OSError",
            "error_message": "disk full",
            "processed_sessions": 0,
            "model_id": "paper_x",
        },
    )
    _write_json(
        tmp_path / "validity_gate_latest.json",
        {
            "status": "HOLD_INSUFFICIENT_DATA",
            "review_reasons": ["insufficient_completed_trades"],
            "sample": {"completed_trades": 2, "trading_days": 5},
            "model_id": "paper_x",
        },
    )
    _write_json(tmp_path / "portfolio.json", _portfolio(config_id="FINAL_US_PHASE4C"))

    src = DashboardDataSource(tmp_path, experiment_id="sys", initial_capital=initial_capital)
    st = build_system_status(src)
    assert st["experiment_id"] == "sys"
    assert st["run_status"] == "ERROR"
    assert st["error_type"] == "OSError"
    assert st["validity_status"] == "HOLD_INSUFFICIENT_DATA"
    assert st["validity_review_reasons"] == ["insufficient_completed_trades"]
    assert st["validity_sample"]["completed_trades"] == 2
    assert st["config_id"] == "FINAL_US_PHASE4C"
    blob = json.dumps(st)
    assert "DISCORD" not in blob.upper()
    assert "WEBHOOK" not in blob.upper()
    assert "TOKEN" not in blob.upper()
    assert "API_KEY" not in blob.upper()


def test_experiment_id_propagation(tmp_path: Path, initial_capital: float) -> None:
    src = DashboardDataSource(
        tmp_path, experiment_id="global_quant_paper", initial_capital=initial_capital
    )
    assert build_overview(src)["experiment_id"] == "global_quant_paper"
    assert build_system_status(src)["experiment_id"] == "global_quant_paper"
