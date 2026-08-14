"""Phase 5B Paper Validity Gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.paper.lineage import LOCKED_PAPER_MODEL_ID
from src.paper.state import PaperStore
from src.paper.validity_gate import (
    STATUS_CONTINUE,
    STATUS_HOLD,
    STATUS_REVIEW,
    evaluate_validity_gate,
    persist_validity_gate,
)


def _write_equity(path: Path, n_days: int, *, end_equity: float = 11_000_000.0) -> None:
    dates = pd.bdate_range("2026-08-03", periods=n_days)
    start = 10_000_000.0
    eq = np.linspace(start, end_equity, n_days)
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Cash": start * 0.2,
            "Position Value": eq - start * 0.2,
            "Total Equity": eq,
            "Drawdown": 0.0,
            "Realized PnL": 0.0,
            "Unrealized PnL": eq - start,
            "N Positions": 3,
            "Transaction Costs": 1000.0,
        }
    )
    df.to_csv(path, index=False)


def _write_trades(path: Path, n: int, *, profitable: bool = True) -> None:
    rows = []
    for i in range(n):
        ret = 0.02 if profitable else -0.02
        pnl = 1000.0 if profitable else -1000.0
        rows.append(
            {
                "Symbol": f"S{i % 5}",
                "Entry Date": "2026-08-01",
                "Exit Date": "2026-08-20",
                "Net PnL": pnl,
                "Return": ret,
                "idempotency_key": f"fill_{i}",
                "model_id": LOCKED_PAPER_MODEL_ID,
                "config_id": "FINAL_US_PHASE4C",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _bench(path: Path, total_return: float) -> None:
    path.write_text(json.dumps({"total_return": total_return, "ticker": "^GSPC"}), encoding="utf-8")


def test_insufficient_sample_hold(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper")
    _write_equity(store.equity_history_path, 10)
    _write_trades(store.trade_history_path, 5)
    bench = tmp_path / "benchmark_track.json"
    _bench(bench, 0.01)
    out = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-08-12",
        benchmark_track_path=bench,
    )
    assert out["status"] == STATUS_HOLD
    assert "insufficient_trading_days" in out["review_reasons"]
    assert "insufficient_completed_trades" in out["review_reasons"]
    assert out["auto_retrain"] is False


def test_continue_when_sample_ok_and_healthy(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper")
    _write_equity(store.equity_history_path, 65, end_equity=11_500_000.0)
    _write_trades(store.trade_history_path, 35, profitable=True)
    bench = tmp_path / "benchmark_track.json"
    _bench(bench, 0.02)  # strategy ~15% total → excess positive
    out = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-10-01",
        benchmark_track_path=bench,
    )
    assert out["status"] == STATUS_CONTINUE
    assert out["review_reasons"] == []
    assert out["model_id"] == LOCKED_PAPER_MODEL_ID
    assert out["locked_paper_model_id"] == LOCKED_PAPER_MODEL_ID
    assert out["auto_retrain"] is False


def test_review_on_major_symbol_concentration(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper")
    _write_equity(store.equity_history_path, 65, end_equity=10_500_000.0)
    # One symbol dominates PnL (>=50%)
    rows = [{"Symbol": "AAPL", "Net PnL": 9000.0, "Return": 0.1, "idempotency_key": "a"}]
    for i in range(34):
        rows.append(
            {
                "Symbol": f"X{i}",
                "Net PnL": 10.0,
                "Return": 0.001,
                "idempotency_key": f"x{i}",
            }
        )
    pd.DataFrame(rows).to_csv(store.trade_history_path, index=False)
    bench = tmp_path / "benchmark_track.json"
    _bench(bench, 0.01)
    out = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-10-01",
        benchmark_track_path=bench,
    )
    assert out["status"] == STATUS_REVIEW
    assert "F_single_symbol_dependency" in out["review_reasons"]


def test_hold_reasons_distinguish_days_vs_trades(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper")
    # Enough days, not enough trades
    _write_equity(store.equity_history_path, 65)
    _write_trades(store.trade_history_path, 5)
    bench = tmp_path / "benchmark_track.json"
    _bench(bench, 0.01)
    out = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-10-01",
        benchmark_track_path=bench,
    )
    assert out["status"] == STATUS_HOLD
    assert out["review_reasons"] == ["insufficient_completed_trades"]

    store2 = PaperStore(tmp_path / "paper2")
    _write_equity(store2.equity_history_path, 10)
    _write_trades(store2.trade_history_path, 40)
    out2 = evaluate_validity_gate(
        store=store2,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-10-01",
        benchmark_track_path=bench,
    )
    assert out2["status"] == STATUS_HOLD
    assert out2["review_reasons"] == ["insufficient_trading_days"]


def test_benchmark_unavailable_reason_machine_readable(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper")
    _write_equity(store.equity_history_path, 65, end_equity=12_000_000.0)
    _write_trades(store.trade_history_path, 40, profitable=True)
    out = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-10-01",
        benchmark_track_path=None,
    )
    assert out["status"] == STATUS_HOLD
    assert "benchmark_unavailable" in out["review_reasons"]
    ids = {c["id"] for c in out["checks"]}
    assert "benchmark_required_for_decision" in ids


def test_deterministic_same_input(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper")
    _write_equity(store.equity_history_path, 65)
    _write_trades(store.trade_history_path, 35)
    bench = tmp_path / "benchmark_track.json"
    _bench(bench, 0.01)
    a = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-10-01",
        benchmark_track_path=bench,
    )
    b = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-10-01",
        benchmark_track_path=bench,
    )
    a.pop("generated_at_utc")
    b.pop("generated_at_utc")
    assert a == b


def test_history_dedupes_same_asof(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper")
    _write_equity(store.equity_history_path, 10)
    result = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-08-12",
        benchmark_track_path=None,
    )
    persist_validity_gate(result, paper_state_dir=store.root, reports_paper_dir=tmp_path / "reports")
    persist_validity_gate(result, paper_state_dir=store.root, reports_paper_dir=tmp_path / "reports")
    hist = pd.read_csv(store.root / "validity_gate_history.csv")
    assert len(hist) == 1
    assert (tmp_path / "reports" / "validity_gate.json").exists()


def test_weak_aggregation_triggers_review(tmp_path: Path) -> None:
    """Losing book + non-positive sharpe path + bench excess fail + PF<=1 => weak>=3."""
    store = PaperStore(tmp_path / "paper")
    # Declining equity → negative return & weak sharpe
    dates = pd.bdate_range("2026-08-03", periods=65)
    eq = np.linspace(10_000_000.0, 9_000_000.0, 65)
    pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Cash": 1_000_000.0,
            "Position Value": eq - 1_000_000.0,
            "Total Equity": eq,
            "Drawdown": eq / eq.max() - 1.0,
            "Realized PnL": 0.0,
            "Unrealized PnL": eq - 10_000_000.0,
            "N Positions": 2,
            "Transaction Costs": 1000.0,
        }
    ).to_csv(store.equity_history_path, index=False)
    _write_trades(store.trade_history_path, 35, profitable=False)
    bench = tmp_path / "benchmark_track.json"
    _bench(bench, 0.05)  # strategy negative ⇒ excess negative
    out = evaluate_validity_gate(
        store=store,
        initial_capital=10_000_000.0,
        model_id=LOCKED_PAPER_MODEL_ID,
        asof="2026-10-01",
        benchmark_track_path=bench,
    )
    assert out["status"] == STATUS_REVIEW
    assert any(r.startswith("weak_fail_count_") for r in out["review_reasons"])


def test_review_does_not_imply_retrain_or_lock_change() -> None:
    assert LOCKED_PAPER_MODEL_ID == "paper_4ab12cb5ade1681f"
    # Gate module constants
    from src.paper import validity_gate as vg

    src = Path(vg.__file__).read_text(encoding="utf-8")
    assert "auto_retrain" in src
    assert "train_and_freeze" not in src
