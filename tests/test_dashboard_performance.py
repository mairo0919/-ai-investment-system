"""Phase D3: Performance page + equity charts tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard.charts import render_equity_chart, render_line_chart
from src.dashboard.data import DashboardDataSource, resolve_initial_capital
from src.dashboard.metrics import (
    build_daily_performance_rows,
    build_monthly_performance_rows,
    build_performance,
    build_trade_summary,
    get_equity_curve,
)
from src.dashboard.server import DashboardApp


@pytest.fixture
def initial_capital() -> float:
    return resolve_initial_capital()


def _write_equity(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _base_equity_rows() -> list[dict]:
    return [
        {
            "Date": "2026-08-03",
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
            "Cash": 2_000_000.0,
            "Position Value": 8_200_000.0,
            "Total Equity": 10_200_000.0,
            "Drawdown": 0.0,
            "Realized PnL": 0.0,
            "Unrealized PnL": 200_000.0,
            "N Positions": 2,
            "Transaction Costs": 1000.0,
        },
        {
            "Date": "2026-08-05",
            "Cash": 2_000_000.0,
            "Position Value": 7_800_000.0,
            "Total Equity": 9_800_000.0,
            "Drawdown": 9_800_000.0 / 10_200_000.0 - 1.0,
            "Realized PnL": 0.0,
            "Unrealized PnL": -200_000.0,
            "N Positions": 2,
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


def _seed(tmp_path: Path, *, trades: bool = True) -> None:
    _write_equity(tmp_path / "equity_history.csv", _base_equity_rows())
    (tmp_path / "portfolio.json").write_text(
        json.dumps(
            {
                "cash": 10_300_000.0,
                "base_currency": "JPY",
                "realized_pnl": 300_000.0,
                "unrealized_pnl": 0.0,
                "total_equity": 10_300_000.0,
                "position_value": 0.0,
                "n_positions": 0,
                "model_id": "paper_x",
                "config_id": "FINAL_US_PHASE4C",
            }
        ),
        encoding="utf-8",
    )
    if trades:
        pd.DataFrame(
            [
                {
                    "Symbol": "AAPL",
                    "Net PnL": 300_000.0,
                    "Return": 0.1,
                    "Holding Days": 20,
                    "Entry Cost Base": 1_000_000.0,
                },
                {
                    "Symbol": "MSFT",
                    "Net PnL": -50_000.0,
                    "Return": -0.05,
                    "Holding Days": 10,
                    "Entry Cost Base": 500_000.0,
                },
            ]
        ).to_csv(tmp_path / "trade_history.csv", index=False)


def test_performance_route_200_japanese(tmp_path: Path, initial_capital: float) -> None:
    _seed(tmp_path)
    app = DashboardApp(
        state_dir=tmp_path,
        experiment_id="us_quant_paper",
        source=DashboardDataSource(
            tmp_path, experiment_id="us_quant_paper", initial_capital=initial_capital
        ),
    )
    status, _, body = app.handle("GET", "/performance")
    assert status == 200
    html = body.decode()
    assert "運用成績" in html
    assert "総資産の推移" in html
    assert "損益推移" in html
    assert "ドローダウン推移" in html
    assert "日次成績" in html
    assert "月次成績" in html
    assert "トレード要約" in html
    assert "us_quant_paper" in html
    assert "<svg" in html
    assert "現在の総資産" in html
    assert "Sharpe Ratio" in html
    assert 'href="/"' in html  # overview nav


def test_overview_has_mini_equity_and_nav(tmp_path: Path, initial_capital: float) -> None:
    _seed(tmp_path)
    before = (tmp_path / "equity_history.csv").read_bytes()
    app = DashboardApp(
        state_dir=tmp_path,
        experiment_id="exp",
        source=DashboardDataSource(tmp_path, experiment_id="exp", initial_capital=initial_capital),
    )
    status, _, body = app.handle("GET", "/")
    assert status == 200
    html = body.decode()
    assert 'href="/performance"' in html
    assert "詳細を見る" in html
    assert "総資産の推移" in html
    assert "<svg" in html
    assert "初期資金" in html
    assert (tmp_path / "equity_history.csv").read_bytes() == before


def test_empty_and_single_and_flat_equity(tmp_path: Path, initial_capital: float) -> None:
    src = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    assert get_equity_curve(src) == []
    assert "データがありません" in render_line_chart([], y_key="total_equity") or True
    assert "総資産の履歴がありません" in render_equity_chart([], initial_capital=initial_capital)

    _write_equity(
        tmp_path / "equity_history.csv",
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
            }
        ],
    )
    src1 = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    svg1 = render_equity_chart(get_equity_curve(src1), initial_capital=initial_capital)
    assert "<circle" in svg1
    assert "NaN" not in svg1

    _write_equity(
        tmp_path / "equity_history.csv",
        [
            {
                "Date": "2026-08-01",
                "Cash": 10_000_000.0,
                "Position Value": 0.0,
                "Total Equity": 10_000_000.0,
                "Drawdown": 0.0,
                "N Positions": 0,
                "Transaction Costs": 0.0,
            },
            {
                "Date": "2026-08-02",
                "Cash": 10_000_000.0,
                "Position Value": 0.0,
                "Total Equity": 10_000_000.0,
                "Drawdown": 0.0,
                "N Positions": 0,
                "Transaction Costs": 0.0,
            },
        ],
    )
    src2 = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    svg2 = render_equity_chart(get_equity_curve(src2), initial_capital=initial_capital)
    assert "<polyline" in svg2
    assert "inf" not in svg2.lower()


def test_negative_performance_and_daily(tmp_path: Path, initial_capital: float) -> None:
    _write_equity(
        tmp_path / "equity_history.csv",
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
                "Cash": 9_000_000.0,
                "Position Value": 0.0,
                "Total Equity": 9_000_000.0,
                "Drawdown": -0.1,
                "Realized PnL": -1_000_000.0,
                "Unrealized PnL": 0.0,
                "N Positions": 0,
                "Transaction Costs": 0.0,
            },
        ],
    )
    src = DashboardDataSource(tmp_path, experiment_id="neg", initial_capital=initial_capital)
    daily = build_daily_performance_rows(src, limit=10)
    assert daily[0]["date"] == "2026-08-02"
    assert daily[0]["daily_pnl"] == pytest.approx(-1_000_000.0)
    assert daily[0]["daily_return"] == pytest.approx(-0.1)
    assert daily[1]["daily_pnl"] is None  # first row incomparable (after reverse: oldest last)

    # chronological first day is last in reversed list
    oldest = [r for r in daily if r["date"] == "2026-08-01"][0]
    assert oldest["daily_pnl"] is None
    assert oldest["daily_return"] is None

    perf = build_performance(src)
    assert perf["total_pnl"] == pytest.approx(-1_000_000.0)
    assert perf["total_return"] == pytest.approx(-0.1)


def test_monthly_and_trade_summary(tmp_path: Path, initial_capital: float) -> None:
    _seed(tmp_path)
    src = DashboardDataSource(tmp_path, initial_capital=initial_capital)
    months = build_monthly_performance_rows(src)
    assert len(months) >= 1
    assert "month" in months[0]
    assert "strategy_return" in months[0]
    assert "month_end_equity" in months[0]

    summary = build_trade_summary(src)
    assert summary["completed_trades"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["best_trade"] == pytest.approx(300_000.0)
    assert summary["worst_trade"] == pytest.approx(-50_000.0)


def test_svg_escaping_and_corrupt_state(tmp_path: Path, initial_capital: float) -> None:
    svg = render_line_chart(
        [{"date": '<script>x</script>', "total_equity": 1.0}],
        y_key="total_equity",
    )
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg

    (tmp_path / "equity_history.csv").write_text("not,a,csv\n{{{", encoding="utf-8")
    app = DashboardApp(
        state_dir=tmp_path,
        experiment_id="bad",
        source=DashboardDataSource(tmp_path, experiment_id="bad", initial_capital=initial_capital),
    )
    status, _, body = app.handle("GET", "/performance")
    assert status == 200
    html = body.decode()
    assert "運用成績" in html
    assert "—" in html
    assert "Traceback" not in html
    assert str(tmp_path) not in html


def test_performance_read_only(tmp_path: Path, initial_capital: float) -> None:
    _seed(tmp_path)
    before = {
        p.name: p.read_bytes()
        for p in tmp_path.iterdir()
        if p.is_file()
    }
    app = DashboardApp(
        state_dir=tmp_path,
        source=DashboardDataSource(tmp_path, initial_capital=initial_capital),
    )
    app.handle("GET", "/performance")
    after = {
        p.name: p.read_bytes()
        for p in tmp_path.iterdir()
        if p.is_file()
    }
    assert after == before
