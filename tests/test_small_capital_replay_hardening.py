"""SC4C: audit provenance, session-limited validity as-of, gitignore."""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import PROJECT_ROOT, Settings
from src.paper import runner as runner_mod
from src.paper.small_capital import EXPERIMENT_ID, SmallCapitalPaperRunner
from src.paper.state import PaperState
from src.paper.validity_gate import evaluate_validity_gate
from src.simulation.config import parse_simulation_config
from src.simulation.engine import SimulationEngine, SimulationResult
from src.simulation.fx import FxConverter
from src.simulation.portfolio import Portfolio
from src.simulation.rules import plan_integer_affordable_entries


DAY = pd.Timestamp("2026-08-03")
NEXT = pd.Timestamp("2026-08-04")
MARKET = pd.Timestamp("2026-09-25")


def _fx() -> FxConverter:
    idx = pd.bdate_range("2026-08-01", periods=40)
    return FxConverter({"USDJPY": pd.Series(150.0, index=idx)})


def _cfg(*, top_n: int = 2, mode: str = "top_n", top_percentile: float = 0.10):
    return parse_simulation_config(
        {
            "initial_capital": 10_000,
            "base_currency": "JPY",
            "candidate": {
                "mode": mode,
                "top_percentile": top_percentile,
                "top_n": top_n,
            },
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
    )


def _prices(symbols: dict[str, float]) -> pd.DataFrame:
    rows = []
    for day in (DAY, NEXT):
        for sym, px in symbols.items():
            rows.append(
                {
                    "Date": day,
                    "Symbol": sym,
                    "Country": "United States",
                    "Currency": "USD",
                    "Open": px,
                    "High": px,
                    "Low": px,
                    "Close": px,
                }
            )
    return pd.DataFrame(rows)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        paper_state_dir=tmp_path / "sc",
        reports_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
    )


def _runner(tmp_path: Path, *, max_sessions: int | None) -> SmallCapitalPaperRunner:
    return SmallCapitalPaperRunner(
        _settings(tmp_path),
        Path("config/universe.global100.json"),
        Path("config/paper_trading_small_capital_10k.json"),
        max_sessions=max_sessions,
    )


def test_A_paper_experiments_runtime_state_is_gitignored() -> None:
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/paper_experiments/*" in ignore
    assert "!data/paper_experiments/.gitkeep" in ignore
    assert (PROJECT_ROOT / "data" / "paper_experiments" / ".gitkeep").exists()
    probe = "data/paper_experiments/small_capital_10k/portfolio.json"
    proc = subprocess.run(
        ["git", "check-ignore", "-v", probe],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "data/paper_experiments/*" in proc.stdout
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    listed = [
        line.split(maxsplit=1)[-1]
        for line in status.stdout.splitlines()
        if "paper_experiments" in line
    ]
    assert all(
        path == "data/paper_experiments/.gitkeep" or path.endswith(".gitkeep")
        for path in listed
    )
    others = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "data/paper_experiments"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "small_capital_10k" not in others.stdout
    tracked = subprocess.run(
        ["git", "ls-files", "data/paper_experiments/.gitkeep"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if tracked.stdout.strip():
        assert "?? data/paper_experiments/" not in status.stdout.splitlines()


def test_B_execution_audit_matches_ranking_row_used() -> None:
    rankings = pd.DataFrame(
        [
            {
                "Date": DAY,
                "Symbol": "QCOM",
                "Country": "United States",
                "Score": 0.203184,
                "Rank": 1,
                "Percentile": 0.025,
            },
            {
                "Date": DAY,
                "Symbol": "AAPL",
                "Country": "United States",
                "Score": 0.115739,
                "Rank": 2,
                "Percentile": 0.05,
            },
            {
                "Date": DAY,
                "Symbol": "MSFT",
                "Country": "United States",
                "Score": 0.032312,
                "Rank": 10,
                "Percentile": 0.25,
            },
        ]
    )
    engine = SimulationEngine(
        _cfg(mode="top_percentile", top_percentile=0.10, top_n=3),
        fx=_fx(),
        countries={"United States"},
    )
    result = engine.run(
        prices=_prices({"QCOM": 200.0, "AAPL": 200.0, "MSFT": 200.0}),
        rankings=rankings,
        start=DAY,
        end=NEXT,
        portfolio=Portfolio(cash=10_000),
        min_calendar_days=1,
    )
    events = result.meta["execution_events"]
    assert {e["Symbol"] for e in events} == {"QCOM", "AAPL"}
    by_symbol = {e["Symbol"]: e for e in events}
    for sym in ("QCOM", "AAPL"):
        row = rankings.loc[rankings["Symbol"] == sym].iloc[0]
        assert by_symbol[sym]["Score"] == pytest.approx(float(row["Score"]))
        assert int(by_symbol[sym]["Rank"]) == int(row["Rank"])
        assert by_symbol[sym]["Date"] == str(DAY.date())
    assert "MSFT" not in by_symbol


def test_C_fallback_keeps_candidate_ranking_provenance() -> None:
    day = DAY
    ranking = pd.DataFrame(
        [
            {
                "Date": day,
                "Symbol": "EXPENSIVE",
                "Country": "United States",
                "Score": 0.9,
                "Rank": 1,
                "Percentile": 0.05,
            },
            {
                "Date": day,
                "Symbol": "CHEAP",
                "Country": "United States",
                "Score": 0.4,
                "Rank": 2,
                "Percentile": 0.10,
            },
        ]
    )
    cfg = _cfg(top_n=2)
    plans, skips = plan_integer_affordable_entries(
        ranking,
        Portfolio(cash=10_000),
        cfg,
        costs=_costs(),
        fx=_fx(),
        signal_day=day,
        close_prices={"EXPENSIVE": 200.0, "CHEAP": 10.0},
        meta_map={
            "EXPENSIVE": {"Country": "United States", "Currency": "USD"},
            "CHEAP": {"Country": "United States", "Currency": "USD"},
        },
        pending_buy_symbols=set(),
        cooldown_until={},
    )
    assert [s.symbol for s in skips] == ["EXPENSIVE"]
    assert skips[0].rank == 1
    assert skips[0].score == pytest.approx(0.9)
    assert len(plans) == 1
    assert plans[0].symbol == "CHEAP"
    assert plans[0].rank == 2
    assert plans[0].score == pytest.approx(0.4)


def _costs():
    from src.simulation.execution import CostModel

    return CostModel(commission_rate=0.0, slippage_rate=0.0)


def test_D_duplicate_row_cannot_masquerade_as_another_rank(tmp_path: Path) -> None:
    rankings = pd.DataFrame(
        [
            {
                "Date": DAY,
                "Symbol": "DUP",
                "Country": "United States",
                "Score": 0.90,
                "Rank": 1,
                "Percentile": 0.025,
            },
            {
                "Date": DAY,
                "Symbol": "DUP",
                "Country": "United States",
                "Score": 0.10,
                "Rank": 4,
                "Percentile": 0.10,
            },
        ]
    )
    engine = SimulationEngine(
        _cfg(mode="top_percentile", top_percentile=0.10, top_n=3),
        fx=_fx(),
        countries={"United States"},
    )
    result = engine.run(
        prices=_prices({"DUP": 200.0}),
        rankings=rankings,
        start=DAY,
        end=NEXT,
        portfolio=Portfolio(cash=10_000),
        min_calendar_days=1,
    )
    events = [e for e in result.meta["execution_events"] if e["Symbol"] == "DUP"]
    pairs = {(round(float(e["Score"]), 2), int(e["Rank"])) for e in events}
    assert pairs == {(0.90, 1), (0.10, 4)}
    assert all(int(e["Rank"]) != 1 or float(e["Score"]) > 0.5 for e in events)
    runner = _runner(tmp_path, max_sessions=1)
    written = runner.persist_execution_events(
        SimulationResult(meta={"execution_events": events})
    )
    assert written == 2
    frame = runner.execution_audit.load_frame()
    assert set(zip(frame["Score"].astype(float).round(2), frame["Rank"].astype(int))) == pairs


def test_E_F_session_limited_validity_asof_excludes_future_rows(tmp_path: Path) -> None:
    runner = _runner(tmp_path, max_sessions=1)
    runner.store.save_atomic(
        PaperState(
            cash=10_000,
            peak_equity=10_000,
            last_processed_date=str(DAY.date()),
            forward_start="2026-08-01",
        )
    )
    equity = pd.DataFrame(
        [
            {
                "Date": str(DAY.date()),
                "Cash": 10_000,
                "Position Value": 0,
                "Total Equity": 10_000,
                "Drawdown": 0,
                "Realized PnL": 0,
                "Unrealized PnL": 0,
                "N Positions": 0,
                "Transaction Costs": 0,
            },
            {
                "Date": str(MARKET.date()),
                "Cash": 50_000,
                "Position Value": 0,
                "Total Equity": 50_000,
                "Drawdown": 0,
                "Realized PnL": 40_000,
                "Unrealized PnL": 0,
                "N Positions": 0,
                "Transaction Costs": 0,
            },
        ]
    )
    equity.to_csv(runner.store.equity_history_path, index=False)
    pd.DataFrame(
        [
            {
                "Symbol": "FUTURE",
                "Exit Date": str(MARKET.date()),
                "Net PnL": 999.0,
                "Return": 0.5,
            }
        ]
    ).to_csv(runner.store.trade_history_path, index=False)
    runner._emit_validity_gate(asof=MARKET, model_id="paper_4ab12cb5ade1681f")
    payload = json.loads(
        (runner.store.root / "validity_gate_latest.json").read_text(encoding="utf-8")
    )
    assert payload["asof"] == str(DAY.date())
    assert payload["sample"]["equity_rows"] == 1
    assert payload["sample"]["trading_days"] == 0
    assert payload["sample"]["completed_trades"] == 0
    assert payload["metrics"] == {}
    kept = pd.read_csv(runner.store.equity_history_path)
    assert set(kept["Date"].astype(str)) == {str(DAY.date()), str(MARKET.date())}


def test_G_legacy_validity_keeps_full_history_and_caller_asof(tmp_path: Path) -> None:
    source = inspect.getsource(runner_mod.PaperTradingRunner._emit_validity_gate)
    assert "history_end" not in source
    assert "max_sessions" not in source
    from src.paper.state import PaperStore

    store = PaperStore(tmp_path / "paper")
    pd.DataFrame(
        {
            "Date": [str(DAY.date()), str(MARKET.date())],
            "Total Equity": [10_000_000.0, 11_000_000.0],
        }
    ).to_csv(store.equity_history_path, index=False)
    out = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id="paper_4ab12cb5ade1681f",
        asof=str(MARKET.date()),
    )
    assert out["asof"] == str(MARKET.date())
    assert out["sample"]["equity_rows"] == 2
    runner = _runner(tmp_path / "sc", max_sessions=None)
    assert runner._session_limited_asof(MARKET) == MARKET.normalize()


def test_benchmark_track_does_not_audit_and_caps_asof(tmp_path: Path, monkeypatch) -> None:
    runner = _runner(tmp_path, max_sessions=1)
    pd.DataFrame(
        {
            "Date": [str(DAY.date())],
            "Total Equity": [10_000.0],
        }
    ).to_csv(runner.store.equity_history_path, index=False)
    seen: dict = {}

    def _fake(self, **kwargs):  # noqa: ANN001
        seen["suppressed"] = runner._suppress_execution_audit
        seen["asof"] = pd.Timestamp(kwargs["asof"]).normalize()

    monkeypatch.setattr(runner_mod.PaperTradingRunner, "_update_benchmark_tracks", _fake)
    runner._update_benchmark_tracks(
        us_px=pd.DataFrame(),
        mom_rankings=pd.DataFrame(),
        fx=_fx(),
        asof=MARKET,
        state_model_id="paper_4ab12cb5ade1681f",
    )
    assert seen["suppressed"] is True
    assert seen["asof"] == DAY.normalize()
    assert runner._suppress_execution_audit is False
    event = {
        "Date": str(DAY.date()),
        "Symbol": "MSFT",
        "Rank": 1,
        "Score": 0.260925,
        "Decision": "SKIPPED",
        "Reason": "NOT_AFFORDABLE",
        "order_identity": "momentum",
    }
    runner._suppress_execution_audit = True
    assert (
        runner.persist_execution_events(SimulationResult(meta={"execution_events": [event]}))
        == 0
    )
    assert runner.execution_audit.load_frame().empty
    runner._suppress_execution_audit = False
    assert (
        runner.persist_execution_events(SimulationResult(meta={"execution_events": [event]}))
        == 1
    )


def test_H_real_small_capital_state_not_deleted() -> None:
    root = PROJECT_ROOT / "data" / "paper_experiments" / "small_capital_10k"
    portfolio = root / "portfolio.json"
    audit = root / "execution_decisions.csv"
    assert portfolio.exists()
    assert audit.exists()
    port = json.loads(portfolio.read_text(encoding="utf-8"))
    assert port["last_processed_date"] == "2026-08-03"
    assert float(port["cash"]) == pytest.approx(10_000)
    decisions = pd.read_csv(audit)
    assert len(decisions) == 7
    assert set(decisions["Date"].astype(str)) == {"2026-08-03"}
