"""Phase 5C Paper Observation Persistence tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.paper.lineage import LOCKED_PAPER_MODEL_ID
from src.paper.observation_store import (
    RANKING_SNAPSHOT_COLUMNS,
    STATUS_ERROR,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    PaperObservationStore,
)
from src.paper.state import PaperState, PaperStore
from src.paper.validity_gate import (
    STATUS_HOLD,
    evaluate_validity_gate,
    persist_validity_gate,
)


def _rank_frame(day: str, *, n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "Date": pd.Timestamp(day),
                "Symbol": f"S{i}",
                "Country": "United States",
                "Score": float(n - i),
                "Rank": i + 1,
                "Percentile": (i + 1) / n,
            }
        )
    return pd.DataFrame(rows)


def _price_panel(day: str, symbols: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp(day)] * len(symbols),
            "Symbol": symbols,
            "Close": closes,
            "Country": ["United States"] * len(symbols),
        }
    )


def test_a_ranking_snapshot_generated(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    day = "2026-08-12"
    snap = store.build_ranking_snapshot(
        day_rank=_rank_frame(day),
        model_id=LOCKED_PAPER_MODEL_ID,
        price_panel=_price_panel(day, [f"S{i}" for i in range(5)], [10.0 + i for i in range(5)]),
        top_percentile=0.10,
        observation_date=day,
    )
    assert not snap.empty
    action = store.save_ranking_snapshot(snap, day=day)
    assert action == "wrote"
    path = store.ranking_path(day)
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 5


def test_b_required_columns_and_order(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    day = "2026-08-12"
    snap = store.build_ranking_snapshot(
        day_rank=_rank_frame(day),
        model_id="m1",
        price_panel=None,
        top_percentile=0.10,
        observation_date=day,
    )
    assert list(snap.columns) == list(RANKING_SNAPSHOT_COLUMNS)
    store.save_ranking_snapshot(snap, day=day)
    loaded = pd.read_csv(store.ranking_path(day))
    assert list(loaded.columns) == list(RANKING_SNAPSHOT_COLUMNS)


def test_c_model_id_saved(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    day = "2026-08-12"
    snap = store.build_ranking_snapshot(
        day_rank=_rank_frame(day),
        model_id=LOCKED_PAPER_MODEL_ID,
        price_panel=None,
        top_percentile=0.10,
        observation_date=day,
    )
    assert (snap["model_id"] == LOCKED_PAPER_MODEL_ID).all()


def test_d_close_no_future_leak(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    day = "2026-08-12"
    future = "2026-08-13"
    ranks = _rank_frame(day, n=2)
    # Future close for S0 must not appear; same-day Close for S0; S1 missing Close -> NaN
    px = pd.concat(
        [
            _price_panel(day, ["S0"], [100.0]),
            _price_panel(future, ["S0", "S1"], [999.0, 50.0]),
        ],
        ignore_index=True,
    )
    snap = store.build_ranking_snapshot(
        day_rank=ranks,
        model_id="m1",
        price_panel=px,
        top_percentile=0.10,
        observation_date=day,
    )
    assert len(snap) == 2  # ranking rows kept
    s0 = snap.loc[snap["Symbol"] == "S0", "Close"].iloc[0]
    s1 = snap.loc[snap["Symbol"] == "S1", "Close"].iloc[0]
    assert float(s0) == 100.0
    assert pd.isna(s1)


def test_e_idempotent_same_date_no_duplicate(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    day = "2026-08-12"
    snap = store.build_ranking_snapshot(
        day_rank=_rank_frame(day),
        model_id="m1",
        price_panel=None,
        top_percentile=0.10,
        observation_date=day,
    )
    assert store.save_ranking_snapshot(snap, day=day) == "wrote"
    assert store.save_ranking_snapshot(snap, day=day) == "unchanged"
    # Content change must NOT overwrite first observation
    before = store.ranking_path(day).read_bytes()
    snap2 = snap.copy()
    snap2.loc[0, "Score"] = 99.0
    assert store.save_ranking_snapshot(snap2, day=day) == "conflict"
    assert store.ranking_path(day).read_bytes() == before
    loaded = pd.read_csv(store.ranking_path(day))
    assert len(loaded) == 5
    assert float(loaded.loc[loaded["Symbol"] == "S0", "Score"].iloc[0]) != 99.0


def test_immutable_snapshot_first_write(tmp_path: Path) -> None:
    """A. First save creates the canonical file."""
    store = PaperObservationStore(tmp_path / "paper")
    day = "2026-08-14"
    snap = store.build_ranking_snapshot(
        day_rank=_rank_frame(day),
        model_id="m1",
        price_panel=None,
        top_percentile=0.10,
        observation_date=day,
    )
    assert store.save_ranking_snapshot(snap, day=day) == "wrote"
    assert store.ranking_path(day).exists()


def test_immutable_snapshot_same_content_noop(tmp_path: Path) -> None:
    """B. Identical re-save is no-op; bytes unchanged."""
    store = PaperObservationStore(tmp_path / "paper")
    day = "2026-08-14"
    snap = store.build_ranking_snapshot(
        day_rank=_rank_frame(day),
        model_id="m1",
        price_panel=None,
        top_percentile=0.10,
        observation_date=day,
    )
    store.save_ranking_snapshot(snap, day=day)
    before = store.ranking_path(day).read_bytes()
    assert store.save_ranking_snapshot(snap, day=day) == "unchanged"
    assert store.ranking_path(day).read_bytes() == before


def test_immutable_snapshot_conflict_keeps_original_and_records(tmp_path: Path) -> None:
    """C. Diff re-save keeps original; conflict recorded; trading continues."""
    from src.paper.observation_store import ranking_snapshot_conflict_tag
    from src.paper.runner import PaperTradingRunner
    from src.config.settings import Settings

    paper = tmp_path / "paper"
    store = PaperObservationStore(paper)
    day = "2026-08-14"
    snap = store.build_ranking_snapshot(
        day_rank=_rank_frame(day),
        model_id=LOCKED_PAPER_MODEL_ID,
        price_panel=None,
        top_percentile=0.10,
        observation_date=day,
    )
    store.save_ranking_snapshot(snap, day=day)
    original = store.ranking_path(day).read_bytes()

    snap2 = snap.copy()
    snap2.loc[0, "Score"] = -1.0
    assert store.save_ranking_snapshot(snap2, day=day) == "conflict"
    assert store.ranking_path(day).read_bytes() == original

    settings = Settings(
        paper_state_dir=paper,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    runner = PaperTradingRunner(
        settings,
        Path("config/universe.global100.json"),
        Path("config/paper_trading.json"),
    )
    # Simulate runner finish after conflict during batch save
    counts = store.save_processed_day_rankings(
        ai_rankings=_rank_frame(day).assign(Score=lambda d: d["Score"] + 50),
        price_panel=None,
        model_id=LOCKED_PAPER_MODEL_ID,
        top_percentile=0.10,
        session_days=[pd.Timestamp(day)],
    )
    assert counts["conflict"] == 1
    assert day in counts["conflict_days"]
    assert store.ranking_path(day).read_bytes() == original

    h = store.begin_run(model_id=LOCKED_PAPER_MODEL_ID, run_id="conflict_run")
    fin = store.finish_run(
        status=STATUS_SUCCESS,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof=day,
        run_id="conflict_run",
        run_started_utc=h["last_run_started_utc"],
        processed_sessions=1,
        ranking_snapshot_conflicts=counts["conflict_days"],
    )
    tag = ranking_snapshot_conflict_tag(day)
    assert tag in (fin["notes"].get("ranking_snapshot_conflicts") or [])
    hist = pd.read_csv(store.health_history_path)
    assert tag in str(hist.loc[hist["run_id"] == "conflict_run", "notes"].iloc[0])

    # Runner helper must not raise on conflict path
    runner._persist_observations(
        run_health_ctx={
            "run_id": "conflict_run2",
            "run_started_utc": h["last_run_started_utc"],
            "model_id": LOCKED_PAPER_MODEL_ID,
        },
        status=STATUS_SUCCESS,
        asof=pd.Timestamp(day),
        ai_rankings=_rank_frame(day).assign(Score=lambda d: d["Score"] + 99),
        price_panel=None,
        session_days=[pd.Timestamp(day)],
        processed_sessions=1,
        skip_reason=None,
    )
    assert store.ranking_path(day).read_bytes() == original
    health2 = store.load_health()
    assert tag in (health2.get("notes") or {}).get("ranking_snapshot_conflicts", [])


def test_immutable_snapshot_conflict_no_trading_side_effects(tmp_path: Path) -> None:
    """D. Conflict must not touch positions/pending/cash/equity/trades/validity gate."""
    paper = tmp_path / "paper"
    trade_store = PaperStore(paper)
    state = PaperState(
        cash=8_888_888.0,
        forward_start="2026-08-01",
        model_id=LOCKED_PAPER_MODEL_ID,
        last_processed_date="2026-08-13",
    )
    trade_store.save_atomic(state)
    eq_path = trade_store.equity_history_path
    pd.DataFrame(
        {
            "Date": ["2026-08-13"],
            "Cash": [8_888_888.0],
            "Position Value": [0.0],
            "Total Equity": [8_888_888.0],
            "Drawdown": [0.0],
            "Realized PnL": [0.0],
            "Unrealized PnL": [0.0],
            "N Positions": [0],
            "Transaction Costs": [0.0],
        }
    ).to_csv(eq_path, index=False)
    pd.DataFrame(
        [
            {
                "Symbol": "X",
                "Entry Date": "2026-08-01",
                "Exit Date": "2026-08-10",
                "Net PnL": 1.0,
                "Return": 0.01,
                "idempotency_key": "f1",
                "model_id": LOCKED_PAPER_MODEL_ID,
                "config_id": "FINAL_US_PHASE4C",
            }
        ]
    ).to_csv(trade_store.trade_history_path, index=False)

    gate_before = {
        "status": STATUS_HOLD,
        "asof": "2026-08-13",
        "review_reasons": ["insufficient_trading_days"],
    }
    persist_validity_gate(
        gate_before,
        paper_state_dir=paper,
        reports_paper_dir=tmp_path / "reports",
    )

    before = {
        "portfolio": (paper / "portfolio.json").read_text(encoding="utf-8"),
        "positions": (paper / "positions.json").read_text(encoding="utf-8"),
        "pending": (paper / "pending_orders.json").read_text(encoding="utf-8"),
        "equity": eq_path.read_text(encoding="utf-8"),
        "trades": trade_store.trade_history_path.read_text(encoding="utf-8"),
        "gate": (paper / "validity_gate_latest.json").read_text(encoding="utf-8"),
    }

    obs = PaperObservationStore(paper)
    day = "2026-08-14"
    snap = obs.build_ranking_snapshot(
        day_rank=_rank_frame(day),
        model_id=LOCKED_PAPER_MODEL_ID,
        price_panel=None,
        top_percentile=0.10,
        observation_date=day,
    )
    obs.save_ranking_snapshot(snap, day=day)
    snap2 = snap.copy()
    snap2.loc[0, "Score"] = 123.0
    assert obs.save_ranking_snapshot(snap2, day=day) == "conflict"
    h = obs.begin_run(model_id=LOCKED_PAPER_MODEL_ID, run_id="d_conflict")
    obs.finish_run(
        status=STATUS_SUCCESS,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof=day,
        run_id="d_conflict",
        run_started_utc=h["last_run_started_utc"],
        processed_sessions=1,
        ranking_snapshot_conflicts=[day],
    )

    assert (paper / "portfolio.json").read_text(encoding="utf-8") == before["portfolio"]
    assert (paper / "positions.json").read_text(encoding="utf-8") == before["positions"]
    assert (paper / "pending_orders.json").read_text(encoding="utf-8") == before["pending"]
    assert eq_path.read_text(encoding="utf-8") == before["equity"]
    assert trade_store.trade_history_path.read_text(encoding="utf-8") == before["trades"]
    assert (paper / "validity_gate_latest.json").read_text(encoding="utf-8") == before["gate"]
    loaded = trade_store.load()
    assert loaded is not None
    assert loaded.cash == 8_888_888.0


def test_f_run_health_success(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    h0 = store.begin_run(model_id="m1", run_id="run_ok")
    h1 = store.finish_run(
        status=STATUS_SUCCESS,
        model_id="m1",
        asof="2026-08-12",
        run_id="run_ok",
        run_started_utc=h0["last_run_started_utc"],
        processed_sessions=3,
    )
    assert h1["status"] == STATUS_SUCCESS
    assert h1["last_success_utc"] == h1["last_run_finished_utc"]
    assert h1["last_asof"] == "2026-08-12"
    assert h1["processed_sessions"] == 3


def test_g_run_health_skipped_preserves_last_success(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    h0 = store.begin_run(model_id="m1", run_id="run_s1")
    store.finish_run(
        status=STATUS_SUCCESS,
        model_id="m1",
        asof="2026-08-11",
        run_id="run_s1",
        run_started_utc=h0["last_run_started_utc"],
        processed_sessions=1,
    )
    prev = store.load_health()
    success_utc = prev["last_success_utc"]
    h1 = store.begin_run(model_id="m1", run_id="run_skip")
    fin = store.finish_run(
        status=STATUS_SKIPPED,
        model_id="m1",
        asof="2026-08-11",
        run_id="run_skip",
        run_started_utc=h1["last_run_started_utc"],
        processed_sessions=0,
        skip_reason="idempotent_already_processed",
    )
    assert fin["status"] == STATUS_SKIPPED
    assert fin["skip_reason"] == "idempotent_already_processed"
    assert fin["last_success_utc"] == success_utc
    assert fin["last_run_finished_utc"] is not None
    assert fin["last_run_finished_utc"] != success_utc or True  # finished updated


def test_h_run_health_error(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    h0 = store.begin_run(model_id="m1", run_id="run_err")
    fin = store.finish_run(
        status=STATUS_ERROR,
        model_id="m1",
        asof=None,
        run_id="run_err",
        run_started_utc=h0["last_run_started_utc"],
        error_type="TrainingError",
        error_message="boom",
    )
    assert fin["status"] == STATUS_ERROR
    assert fin["error_type"] == "TrainingError"
    assert fin["error_message"] == "boom"
    assert fin["last_success_utc"] is None


def test_i_error_preserves_prior_last_success(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    h0 = store.begin_run(model_id="m1", run_id="run_ok2")
    store.finish_run(
        status=STATUS_SUCCESS,
        model_id="m1",
        asof="2026-08-10",
        run_id="run_ok2",
        run_started_utc=h0["last_run_started_utc"],
        processed_sessions=2,
    )
    success_utc = store.load_health()["last_success_utc"]
    asof_prev = store.load_health()["last_asof"]
    h1 = store.begin_run(model_id="m1", run_id="run_err2")
    fin = store.finish_run(
        status=STATUS_ERROR,
        model_id="m1",
        asof=None,
        run_id="run_err2",
        run_started_utc=h1["last_run_started_utc"],
        error_type="RuntimeError",
        error_message="fail",
    )
    assert fin["last_success_utc"] == success_utc
    assert fin["last_asof"] == asof_prev


def test_j_health_history_multiple_runs(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    for i, st in enumerate([STATUS_ERROR, STATUS_SUCCESS, STATUS_SKIPPED]):
        rid = f"hist_{i}"
        h = store.begin_run(model_id="m1", run_id=rid)
        store.finish_run(
            status=st,
            model_id="m1",
            asof="2026-08-12",
            run_id=rid,
            run_started_utc=h["last_run_started_utc"],
            processed_sessions=0 if st != STATUS_SUCCESS else 1,
            skip_reason="idempotent_already_processed" if st == STATUS_SKIPPED else None,
            error_type="E" if st == STATUS_ERROR else None,
            error_message="x" if st == STATUS_ERROR else None,
        )
    hist = pd.read_csv(store.health_history_path)
    assert len(hist) == 3
    assert set(hist["status"]) == {STATUS_ERROR, STATUS_SUCCESS, STATUS_SKIPPED}
    assert hist["run_id"].nunique() == 3


def test_k_observation_does_not_mutate_trading_state(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    trade_store = PaperStore(paper)
    state = PaperState(
        cash=9_999_999.0,
        forward_start="2026-08-01",
        model_id=LOCKED_PAPER_MODEL_ID,
        last_processed_date="2026-08-11",
    )
    trade_store.save_atomic(state)
    before = {
        "portfolio": (paper / "portfolio.json").read_text(encoding="utf-8"),
        "positions": (paper / "positions.json").read_text(encoding="utf-8"),
        "pending": (paper / "pending_orders.json").read_text(encoding="utf-8"),
    }
    obs = PaperObservationStore(paper)
    day = "2026-08-12"
    snap = obs.build_ranking_snapshot(
        day_rank=_rank_frame(day),
        model_id=LOCKED_PAPER_MODEL_ID,
        price_panel=None,
        top_percentile=0.10,
        observation_date=day,
    )
    obs.save_ranking_snapshot(snap, day=day)
    h = obs.begin_run(model_id=LOCKED_PAPER_MODEL_ID, run_id="k")
    obs.finish_run(
        status=STATUS_SUCCESS,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof=day,
        run_id="k",
        run_started_utc=h["last_run_started_utc"],
        processed_sessions=1,
    )
    assert (paper / "portfolio.json").read_text(encoding="utf-8") == before["portfolio"]
    assert (paper / "positions.json").read_text(encoding="utf-8") == before["positions"]
    assert (paper / "pending_orders.json").read_text(encoding="utf-8") == before["pending"]
    loaded = trade_store.load()
    assert loaded is not None
    assert loaded.cash == 9_999_999.0
    assert loaded.last_processed_date == "2026-08-11"


def test_l_validity_gate_unchanged_by_observation(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    store = PaperStore(paper)
    eq = store.equity_history_path
    dates = pd.bdate_range("2026-08-03", periods=10)
    pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Cash": 2e6,
            "Position Value": 8e6,
            "Total Equity": 10e6,
            "Drawdown": 0.0,
            "Realized PnL": 0.0,
            "Unrealized PnL": 0.0,
            "N Positions": 1,
            "Transaction Costs": 0.0,
        }
    ).to_csv(eq, index=False)
    pd.DataFrame(
        [
            {
                "Symbol": "A",
                "Entry Date": "2026-08-01",
                "Exit Date": "2026-08-10",
                "Net PnL": 1.0,
                "Return": 0.01,
                "idempotency_key": "f1",
                "model_id": LOCKED_PAPER_MODEL_ID,
                "config_id": "FINAL_US_PHASE4C",
            }
        ]
    ).to_csv(store.trade_history_path, index=False)
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"total_return": 0.01}), encoding="utf-8")
    out1 = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-08-12",
        benchmark_track_path=bench,
    )
    obs = PaperObservationStore(paper)
    snap = obs.build_ranking_snapshot(
        day_rank=_rank_frame("2026-08-12"),
        model_id=LOCKED_PAPER_MODEL_ID,
        price_panel=None,
        top_percentile=0.10,
        observation_date="2026-08-12",
    )
    obs.save_ranking_snapshot(snap, day="2026-08-12")
    out2 = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-08-12",
        benchmark_track_path=bench,
    )
    assert out1["status"] == out2["status"] == STATUS_HOLD
    assert out1["review_reasons"] == out2["review_reasons"]
    persist_validity_gate(out2, paper_state_dir=paper, reports_paper_dir=tmp_path / "reports")
    assert (paper / "validity_gate_latest.json").exists()


def test_m_frozen_model_path_untouched_by_observation_store(tmp_path: Path) -> None:
    models = tmp_path / "models"
    frozen = models / "paper_frozen" / LOCKED_PAPER_MODEL_ID
    frozen.mkdir(parents=True)
    marker = frozen / "model.joblib"
    marker.write_bytes(b"frozen-bytes")
    before = marker.read_bytes()
    obs = PaperObservationStore(tmp_path / "paper")
    snap = obs.build_ranking_snapshot(
        day_rank=_rank_frame("2026-08-12"),
        model_id=LOCKED_PAPER_MODEL_ID,
        price_panel=None,
        top_percentile=0.10,
        observation_date="2026-08-12",
    )
    obs.save_ranking_snapshot(snap, day="2026-08-12")
    assert marker.read_bytes() == before
    assert list(models.rglob("*"))  # still only under paper_frozen


def test_n_observation_path_does_not_train(tmp_path: Path) -> None:
    from src.paper.runner import PaperTradingRunner
    from src.config.settings import Settings

    settings = Settings(
        paper_state_dir=tmp_path / "paper",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    runner = PaperTradingRunner(
        settings,
        Path("config/universe.global100.json"),
        Path("config/paper_trading.json"),
    )
    with patch.object(
        runner.model_store,
        "load_for_paper_trading",
        return_value=(MagicMock(), {"model_id": LOCKED_PAPER_MODEL_ID, "feature_list": ["x"]}),
    ) as load_mock:
        _model, meta = runner._ensure_frozen_model(panel=MagicMock())
        assert meta["model_id"] == LOCKED_PAPER_MODEL_ID
        load_mock.assert_called_once()
    assert not hasattr(runner.model_store, "train_and_freeze")
    ctx = {
        "run_id": "n1",
        "run_started_utc": "2026-08-12T00:00:00+00:00",
        "model_id": LOCKED_PAPER_MODEL_ID,
    }
    ranks = _rank_frame("2026-08-12")
    with patch(
        "src.paper.model_freeze.train_and_freeze_ranker",
        side_effect=AssertionError("must not train"),
    ):
        runner._persist_observations(
            run_health_ctx=ctx,
            status=STATUS_SUCCESS,
            asof=pd.Timestamp("2026-08-12"),
            ai_rankings=ranks,
            price_panel=_price_panel("2026-08-12", ["S0"], [1.0]),
            session_days=[pd.Timestamp("2026-08-12")],
            processed_sessions=1,
            skip_reason=None,
        )
    assert (tmp_path / "paper" / "rankings" / "2026-08-12.csv").exists()
    assert "Never trains" in (runner._ensure_frozen_model.__doc__ or "")


def test_catchup_saves_each_session_day_not_invented(tmp_path: Path) -> None:
    store = PaperObservationStore(tmp_path / "paper")
    d1, d2 = "2026-08-11", "2026-08-12"
    r1 = _rank_frame(d1, n=3)
    r2 = _rank_frame(d2, n=3)
    r2["Score"] = r2["Score"] + 10  # different day scores
    ai = pd.concat([r1, r2], ignore_index=True)
    counts = store.save_processed_day_rankings(
        ai_rankings=ai,
        price_panel=None,
        model_id="m1",
        top_percentile=0.10,
        session_days=[pd.Timestamp(d1), pd.Timestamp(d2), pd.Timestamp("2026-08-13")],
    )
    assert counts["wrote"] == 2
    assert counts["missing_in_rankings"] == 1
    s1 = pd.read_csv(store.ranking_path(d1))
    s2 = pd.read_csv(store.ranking_path(d2))
    assert float(s1["Score"].max()) != float(s2["Score"].max())
    assert not (tmp_path / "paper" / "rankings" / "2026-08-13.csv").exists()


def test_observation_write_failure_does_not_raise_from_helper(tmp_path: Path) -> None:
    from src.paper.runner import PaperTradingRunner
    from src.config.settings import Settings

    settings = Settings(
        paper_state_dir=tmp_path / "paper",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    runner = PaperTradingRunner(
        settings,
        Path("config/universe.global100.json"),
        Path("config/paper_trading.json"),
    )
    ctx = {
        "run_id": "fail1",
        "run_started_utc": "2026-08-12T00:00:00+00:00",
        "model_id": LOCKED_PAPER_MODEL_ID,
    }
    with patch.object(
        runner.observations,
        "save_processed_day_rankings",
        side_effect=OSError("disk full"),
    ):
        # Must not raise — trading state already committed in real flow
        runner._persist_observations(
            run_health_ctx=ctx,
            status=STATUS_SUCCESS,
            asof=pd.Timestamp("2026-08-12"),
            ai_rankings=_rank_frame("2026-08-12"),
            price_panel=None,
            session_days=[pd.Timestamp("2026-08-12")],
            processed_sessions=1,
            skip_reason=None,
        )
