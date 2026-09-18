"""Phase D2: Dashboard HTTP shell + Overview tests (stdlib only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.dashboard.data import DashboardDataSource, resolve_initial_capital
from src.dashboard.formatters import esc, format_money, format_percent
from src.dashboard.server import DashboardApp
from src.dashboard.templates import render_overview


@pytest.fixture
def initial_capital() -> float:
    return resolve_initial_capital()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_state(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "portfolio.json",
        {
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
            "model_id": "paper_4ab12cb5ade1681f",
            "config_id": "FINAL_US_PHASE4C",
            "n_positions": 1,
        },
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
                "entry_rank": 7,
                "pending_exit_reason": None,
            }
        ],
    )
    _write_json(
        tmp_path / "run_health.json",
        {
            "status": "SUCCESS",
            "last_success_utc": "2026-09-17T12:00:00+00:00",
            "last_asof": "2026-09-17",
            "model_id": "paper_4ab12cb5ade1681f",
            "processed_sessions": 1,
        },
    )
    _write_json(
        tmp_path / "validity_gate_latest.json",
        {
            "status": "CONTINUE",
            "review_reasons": [],
            "sample": {"completed_trades": 40},
            "model_id": "paper_4ab12cb5ade1681f",
        },
    )
    pd.DataFrame(
        [
            {
                "Date": "2026-09-16",
                "Cash": 5_000_000.0,
                "Position Value": 5_000_000.0,
                "Total Equity": 10_000_000.0,
                "Drawdown": 0.0,
                "Realized PnL": 0.0,
                "Unrealized PnL": 0.0,
                "N Positions": 1,
                "Transaction Costs": 0.0,
            },
            {
                "Date": "2026-09-17",
                "Cash": 5_000_000.0,
                "Position Value": 5_150_000.0,
                "Total Equity": 10_150_000.0,
                "Drawdown": 0.0,
                "Realized PnL": 100_000.0,
                "Unrealized PnL": 50_000.0,
                "N Positions": 1,
                "Transaction Costs": 1000.0,
            },
        ]
    ).to_csv(tmp_path / "equity_history.csv", index=False)
    rankings = tmp_path / "rankings"
    rankings.mkdir()
    pd.DataFrame(
        [
            {
                "Date": "2026-09-17",
                "Symbol": "AAPL",
                "Score": 0.9,
                "Rank": 3,
                "Percentile": 0.05,
                "Selected": True,
                "model_id": "paper_4ab12cb5ade1681f",
                "Country": "United States",
                "Close": 110.0,
            }
        ]
    ).to_csv(rankings / "2026-09-17.csv", index=False)


def test_health_ok(tmp_path: Path, initial_capital: float) -> None:
    app = DashboardApp(
        state_dir=tmp_path,
        experiment_id="exp_health",
        source=DashboardDataSource(tmp_path, experiment_id="exp_health", initial_capital=initial_capital),
    )
    status, headers, body = app.handle("GET", "/health")
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    assert json.loads(body.decode()) == {"status": "ok", "service": "paper-dashboard"}


def test_health_ok_even_if_state_empty(tmp_path: Path) -> None:
    app = DashboardApp(state_dir=tmp_path, experiment_id=None)
    status, _, body = app.handle("GET", "/health")
    assert status == 200
    assert json.loads(body.decode())["status"] == "ok"


def test_overview_empty_state(tmp_path: Path, initial_capital: float) -> None:
    app = DashboardApp(
        state_dir=tmp_path,
        experiment_id="empty_exp",
        source=DashboardDataSource(tmp_path, experiment_id="empty_exp", initial_capital=initial_capital),
    )
    status, headers, body = app.handle("GET", "/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    html = body.decode()
    assert "概要" in html
    assert "empty_exp" in html
    assert "保有銘柄はありません" in html
    assert "資産推移データはまだありません" in html
    assert "—" in html
    assert 'lang="ja"' in html
    assert list(tmp_path.iterdir()) == []


def test_overview_renders_values_and_positions(
    tmp_path: Path, initial_capital: float
) -> None:
    _seed_state(tmp_path)
    before = {
        p.name: p.read_bytes() if p.is_file() else None
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    app = DashboardApp(
        state_dir=tmp_path,
        experiment_id="us_quant_paper",
        source=DashboardDataSource(
            tmp_path, experiment_id="us_quant_paper", initial_capital=initial_capital
        ),
    )
    status, _, body = app.handle("GET", "/")
    assert status == 200
    html = body.decode()
    assert "¥10,150,000" in html or "¥10,150,000" in html.replace(",", ",")
    assert "AAPL" in html
    assert "United States" in html
    assert "us_quant_paper" in html
    assert "正常" in html
    assert "検証継続" in html
    assert 'title="SUCCESS"' in html
    assert 'title="CONTINUE"' in html
    assert "paper_4ab12cb5ade1681f" in html
    assert "現在順位" in html
    assert ">3<" in html or ">3</td>" in html
    assert "2026-09-17" in html
    assert "現在の資産評価額" in html
    assert "ペーパートレード状況" in html
    assert "準備中" in html
    assert "AI投資システム" in html
    assert 'lang="ja"' in html
    # state unchanged
    after = {
        p.name: p.read_bytes() if p.is_file() else None
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_null_and_signed_formatting() -> None:
    assert format_money(None, currency="JPY") == "—"
    assert format_percent(None) == "—"
    assert "+¥1,000" in format_money(1000, currency="JPY", signed=True)
    assert "-¥1,000" in format_money(-1000, currency="JPY", signed=True)
    assert format_percent(0.1234) == "+12.34%"
    assert format_percent(-0.0456) == "-4.56%"


def test_html_escaping_in_overview() -> None:
    overview = {
        "experiment_id": '<script>alert(1)</script>',
        "base_currency": "JPY",
        "current_equity": 1.0,
        "total_pnl": None,
        "total_return": None,
        "cash": None,
        "position_value": None,
        "unrealized_pnl": None,
        "model_id": 'x" onmouseover="alert(1)',
        "run_status": None,
        "validity_status": None,
        "latest_asof": None,
        "last_success_utc": None,
    }
    system = {
        "experiment_id": overview["experiment_id"],
        "run_status": "SUCCESS",
        "last_success_utc": None,
        "last_asof": None,
        "validity_status": None,
        "model_id": overview["model_id"],
    }
    positions = [
        {
            "symbol": "<img src=x onerror=alert(1)>",
            "country": "JP&US",
            "quantity": 1,
            "entry_price": 1,
            "current_price": 1,
            "market_value": 1,
            "unrealized_pnl": 0,
            "holding_days": 1,
            "current_rank": 1,
            "currency": "USD",
        }
    ]
    html = render_overview(
        overview=overview,
        system=system,
        positions=positions,
        equity_curve=[],
    )
    assert "<script>" not in html
    assert esc("<script>alert(1)</script>") in html
    assert "<img src=x" not in html
    assert "&lt;img" in html
    assert "JP&amp;US" in html


def test_unknown_route_404(tmp_path: Path) -> None:
    app = DashboardApp(state_dir=tmp_path, experiment_id="e")
    status, headers, body = app.handle("GET", "/secret")
    assert status == 404
    assert "text/html" in headers["Content-Type"]
    assert "ページが見つかりません".encode("utf-8") in body
    assert b"/Users/" not in body
    assert str(tmp_path).encode() not in body


def test_post_405(tmp_path: Path) -> None:
    app = DashboardApp(state_dir=tmp_path)
    status, headers, body = app.handle("POST", "/")
    assert status == 405
    assert "メソッドが許可されていません".encode("utf-8") in body
    assert headers.get("Allow") == "GET, HEAD"


def test_exception_does_not_leak_path_or_traceback(
    tmp_path: Path, initial_capital: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BoomSource(DashboardDataSource):
        def load_portfolio(self) -> dict[str, Any] | None:  # type: ignore[override]
            raise RuntimeError(f"failed reading {tmp_path}/portfolio.json secret_token_xyz")

    src = BoomSource(tmp_path, experiment_id="boom", initial_capital=initial_capital)
    app = DashboardApp(state_dir=tmp_path, experiment_id="boom", source=src)
    status, _, body = app.handle("GET", "/")
    assert status == 500
    text = body.decode()
    assert "ダッシュボードを表示できません" in text
    assert "Traceback" not in text
    assert "secret_token_xyz" not in text
    assert str(tmp_path) not in text
    assert "portfolio.json" not in text


def test_no_directory_listing_or_file_route(tmp_path: Path) -> None:
    (tmp_path / "portfolio.json").write_text(
        json.dumps({"cash": 123, "total_equity": 123}), encoding="utf-8"
    )
    app = DashboardApp(state_dir=tmp_path)
    for path in ("/portfolio.json", "/../etc/passwd", "/rankings/", "/static/"):
        status, _, body = app.handle("GET", path)
        assert status == 404
        assert "ページが見つかりません".encode("utf-8") in body
        assert b"123" not in body
