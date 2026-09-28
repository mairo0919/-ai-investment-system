"""SC4A: Small Capital invocation session limit (no live paper state)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import PROJECT_ROOT, Settings
from src.paper import runner as runner_mod
from src.paper.small_capital import (
    SmallCapitalPaperRunner,
    clamp_processed_session_days,
    limit_unprocessed_sessions,
    main,
    parse_max_sessions,
    unprocessed_market_sessions,
)
from src.paper.state import PaperState, PaperStore
from src.simulation.config import parse_simulation_config
from src.simulation.engine import SimulationEngine
from src.simulation.fx import FxConverter
from src.simulation.portfolio import Portfolio


FORWARD = pd.Timestamp("2026-08-01")
SESSIONS = list(pd.bdate_range("2026-08-03", periods=5))


def _unprocessed(last: str | None, asof: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    return unprocessed_market_sessions(
        SESSIONS,
        forward_start=FORWARD,
        last_processed=last,
        asof=asof or SESSIONS[-1],
    )


def test_A_empty_state_max_sessions_1_is_first_only() -> None:
    limited = limit_unprocessed_sessions(_unprocessed(None), 1)
    assert limited == [SESSIONS[0]]


def test_B_second_invocation_is_second_session_only() -> None:
    first = limit_unprocessed_sessions(_unprocessed(None), 1)[-1]
    limited = limit_unprocessed_sessions(_unprocessed(str(first.date())), 1)
    assert limited == [SESSIONS[1]]


def test_C_after_two_sessions_unlimited_is_remaining_three() -> None:
    done = SESSIONS[1]
    remaining = limit_unprocessed_sessions(_unprocessed(str(done.date())), None)
    assert remaining == SESSIONS[2:]
    assert len(remaining) == 3


def test_D_max_sessions_2_is_first_two() -> None:
    assert limit_unprocessed_sessions(_unprocessed(None), 2) == SESSIONS[:2]


def test_E_max_sessions_larger_than_remaining_keeps_all() -> None:
    remaining = _unprocessed(str(SESSIONS[1].date()))
    assert limit_unprocessed_sessions(remaining, 99) == remaining
    assert len(remaining) == 3


def test_F_omitted_limit_is_full_catch_up() -> None:
    assert limit_unprocessed_sessions(_unprocessed(None), None) == SESSIONS


def test_G_H_I_reject_non_positive_and_invalid() -> None:
    for bad in ("0", "-1", "abc", "1.5", ""):
        with pytest.raises(Exception):
            parse_max_sessions(bad)
    with pytest.raises(SystemExit):
        main(["--max-sessions", "0"])
    with pytest.raises(SystemExit):
        main(["--max-sessions", "-1"])
    with pytest.raises(SystemExit):
        main(["--max-sessions", "nope"])


def _fx() -> FxConverter:
    idx = pd.bdate_range("2026-08-01", periods=20)
    return FxConverter({"USDJPY": pd.Series(150.0, index=idx)})


def _cfg():
    return parse_simulation_config(
        {
            "initial_capital": 10_000,
            "base_currency": "JPY",
            "candidate": {"mode": "top_n", "top_percentile": 1.0, "top_n": 5},
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


def _prices(days: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for d in days:
        rows.append(
            {
                "Date": d,
                "Symbol": "CHEAP",
                "Country": "United States",
                "Currency": "USD",
                "Open": 10.0,
                "High": 10.0,
                "Low": 10.0,
                "Close": 10.0,
            }
        )
    return pd.DataFrame(rows)


def test_J_K_pending_survives_session_limit_then_fills_next() -> None:
    days = SESSIONS[:3]
    cfg = _cfg()
    engine = SimulationEngine(cfg, fx=_fx(), countries={"United States"})
    engine.process_session_limit = 1
    rankings = pd.DataFrame(
        [
            {
                "Date": days[0],
                "Symbol": "CHEAP",
                "Country": "United States",
                "Score": 0.9,
                "Rank": 1,
                "Percentile": 0.05,
            }
        ]
    )
    first = engine.run(
        prices=_prices(days),
        rankings=rankings,
        start=days[0],
        end=days[-1],
        portfolio=Portfolio(cash=10_000),
        min_calendar_days=1,
    )
    assert len(first.equity_curve) == 1
    assert first.equity_curve[0].date.normalize() == days[0]
    assert first.portfolio.positions == {}
    pending = [o for o in first.meta["pending_orders"] if o.side == "buy"]
    assert len(pending) == 1
    assert pending[0].symbol == "CHEAP"

    engine.process_session_limit = 1
    second = engine.run(
        prices=_prices(days),
        rankings=rankings,
        start=days[1],
        end=days[-1],
        portfolio=first.portfolio,
        pending=pending,
        min_calendar_days=1,
    )
    assert "CHEAP" in second.portfolio.positions
    assert float(second.portfolio.positions["CHEAP"].quantity) >= 1
    assert float(second.portfolio.positions["CHEAP"].quantity) == int(
        second.portfolio.positions["CHEAP"].quantity
    )
    assert len(second.equity_curve) == 1
    assert second.equity_curve[0].date.normalize() == days[1]


def test_L_M_last_processed_and_processed_sessions(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        paper_state_dir=tmp_path / "sc",
        reports_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
    )
    runner = SmallCapitalPaperRunner(
        settings,
        Path("config/universe.global100.json"),
        Path("config/paper_trading_small_capital_10k.json"),
        max_sessions=1,
    )
    PaperStore(settings.paper_state_dir).save_atomic(
        PaperState(
            cash=10_000,
            peak_equity=10_000,
            last_processed_date=str(SESSIONS[-1].date()),
            forward_start="2026-08-01",
        )
    )
    captured: dict = {}

    def _fake(self, **kwargs):  # noqa: ANN001
        captured.update(kwargs)

    monkeypatch.setattr(runner_mod.PaperTradingRunner, "_persist_observations", _fake)
    buffer = SESSIONS[-1] + pd.tseries.offsets.BDay(1)
    runner._persist_observations(
        run_health_ctx={},
        status="SUCCESS",
        asof=SESSIONS[-1],
        ai_rankings=pd.DataFrame(),
        price_panel=pd.DataFrame(),
        session_days=[*SESSIONS, pd.Timestamp(buffer).normalize()],
        processed_sessions=len(SESSIONS) + 1,
        skip_reason=None,
    )
    assert captured["processed_sessions"] == 1
    assert pd.Timestamp(captured["asof"]).normalize() == SESSIONS[0]
    assert len(captured["session_days"]) == 1
    loaded = runner.store.load()
    assert loaded is not None
    assert loaded.last_processed_date == str(SESSIONS[0].date())

    # Omitted limit does not clamp.
    runner.max_sessions = None
    captured.clear()
    runner.store.save_atomic(
        PaperState(
            cash=10_000,
            peak_equity=10_000,
            last_processed_date=str(SESSIONS[-1].date()),
            forward_start="2026-08-01",
        )
    )
    runner._persist_observations(
        run_health_ctx={},
        status="SUCCESS",
        asof=SESSIONS[-1],
        ai_rankings=pd.DataFrame(),
        price_panel=pd.DataFrame(),
        session_days=list(SESSIONS),
        processed_sessions=len(SESSIONS),
        skip_reason=None,
    )
    assert captured["processed_sessions"] == len(SESSIONS)
    assert runner.store.load().last_processed_date == str(SESSIONS[-1].date())


def test_clamp_drops_next_open_buffer_day() -> None:
    buffer = SESSIONS[-1] + pd.tseries.offsets.BDay(1)
    kept = clamp_processed_session_days(
        [*SESSIONS, pd.Timestamp(buffer).normalize()],
        asof=SESSIONS[-1],
        max_sessions=1,
    )
    assert kept == [SESSIONS[0]]


def test_N_existing_data_paper_untouched(tmp_path: Path) -> None:
    canonical = PROJECT_ROOT / "data" / "paper"
    before = {p.name: p.stat().st_mtime_ns for p in canonical.iterdir() if p.is_file()}
    _ = limit_unprocessed_sessions(_unprocessed(None), 1)
    _ = tmp_path / "unused"
    after = {p.name: p.stat().st_mtime_ns for p in canonical.iterdir() if p.is_file()}
    assert before == after
    assert not (PROJECT_ROOT / "data" / "paper_experiments" / "small_capital_10k").exists()


def test_O_legacy_paper_runner_has_no_session_limit() -> None:
    assert "max-sessions" not in inspect.getsource(runner_mod.main)
    assert "max_sessions" not in inspect.getsource(runner_mod.PaperTradingRunner.run)
    assert "process_session_limit" not in inspect.getsource(runner_mod.PaperTradingRunner)
    days = SESSIONS[:3]
    engine = SimulationEngine(_cfg(), fx=_fx(), countries={"United States"})
    assert getattr(engine, "process_session_limit", None) is None
    result = engine.run(
        prices=_prices(days),
        rankings=pd.DataFrame(
            [
                {
                    "Date": days[0],
                    "Symbol": "CHEAP",
                    "Country": "United States",
                    "Score": 0.9,
                    "Rank": 1,
                    "Percentile": 0.05,
                }
            ]
        ),
        start=days[0],
        end=days[-1],
        portfolio=Portfolio(cash=10_000),
        min_calendar_days=1,
    )
    assert len(result.equity_curve) == 3


def test_skipped_limit_does_not_invent_sessions(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        paper_state_dir=tmp_path / "sc",
        reports_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
    )
    runner = SmallCapitalPaperRunner(
        settings,
        Path("config/universe.global100.json"),
        Path("config/paper_trading_small_capital_10k.json"),
        max_sessions=1,
    )
    runner.store.save_atomic(
        PaperState(cash=10_000, peak_equity=10_000, last_processed_date="2026-08-07")
    )
    captured: dict = {}

    def _fake(self, **kwargs):  # noqa: ANN001
        captured.update(kwargs)

    monkeypatch.setattr(runner_mod.PaperTradingRunner, "_persist_observations", _fake)
    runner._persist_observations(
        run_health_ctx={},
        status="SKIPPED",
        asof=SESSIONS[-1],
        ai_rankings=pd.DataFrame(),
        price_panel=pd.DataFrame(),
        session_days=[SESSIONS[-1]],
        processed_sessions=0,
        skip_reason="no_new_trading_days",
    )
    assert captured["processed_sessions"] == 0
    assert runner.store.load().last_processed_date == "2026-08-07"
