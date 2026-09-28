"""SC4E: execution audit appends must not reserialize existing rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.paper.execution_audit import (
    AUDIT_COLUMNS,
    DECISION_SKIPPED,
    REASON_NOT_AFFORDABLE,
    AuditLogCorruptError,
    ExecutionAuditStore,
)


def _row(symbol: str, *, score: str | float, rank: int, identity: str) -> dict:
    return {
        "Date": "2026-08-03",
        "Symbol": symbol,
        "Rank": rank,
        "Score": score,
        "Decision": DECISION_SKIPPED,
        "Reason": REASON_NOT_AFFORDABLE,
        "experiment_id": "small_capital_10k",
        "order_identity": identity,
    }


def test_A_first_write_creates_header_and_row(tmp_path: Path) -> None:
    store = ExecutionAuditStore(tmp_path)
    n = store.append_events([_row("QCOM", score=0.2, rank=1, identity="q")])
    assert n == 1
    text = store.path.read_text(encoding="utf-8")
    header, body = text.splitlines()
    assert header.split(",") == AUDIT_COLUMNS
    assert "QCOM" in body
    frame = store.load_frame()
    assert len(frame) == 1
    assert frame.iloc[0]["Symbol"] == "QCOM"


def test_B_C_D_old_bytes_are_exact_prefix(tmp_path: Path) -> None:
    store = ExecutionAuditStore(tmp_path)
    cells = [""] * len(AUDIT_COLUMNS)
    cells[AUDIT_COLUMNS.index("Date")] = "2026-08-03"
    cells[AUDIT_COLUMNS.index("Symbol")] = "CVX"
    cells[AUDIT_COLUMNS.index("Rank")] = "4"
    cells[AUDIT_COLUMNS.index("Score")] = "0.14919682130660616"
    cells[AUDIT_COLUMNS.index("Decision")] = "SKIPPED"
    cells[AUDIT_COLUMNS.index("Reason")] = "NOT_AFFORDABLE"
    cells[AUDIT_COLUMNS.index("Country")] = "United States"
    cells[AUDIT_COLUMNS.index("Currency")] = "USD"
    cells[AUDIT_COLUMNS.index("model_id")] = "paper_4ab12cb5ade1681f"
    cells[AUDIT_COLUMNS.index("config_id")] = "FINAL_US_PHASE4C"
    cells[AUDIT_COLUMNS.index("experiment_id")] = "small_capital_10k"
    cells[AUDIT_COLUMNS.index("idempotency_key")] = "legacy-cvx"
    legacy = (",".join(AUDIT_COLUMNS) + "\n" + ",".join(cells) + "\n").encode("utf-8")
    assert b"0.14919682130660616" in legacy
    store.path.write_bytes(legacy)
    n = store.append_events([_row("QCOM", score=0.12781, rank=1, identity="new-qcom")])
    assert n == 1
    after = store.path.read_bytes()
    assert after.startswith(legacy)
    assert b"0.14919682130660616" in after
    assert after.splitlines()[1] == legacy.splitlines()[1]


def test_E_duplicate_key_is_byte_for_byte_noop(tmp_path: Path) -> None:
    store = ExecutionAuditStore(tmp_path)
    row = _row("QCOM", score=0.2, rank=1, identity="same")
    assert store.append_events([row, row]) == 1
    before = store.path.read_bytes()
    assert store.append_events([row]) == 0
    assert store.path.read_bytes() == before
    assert len(store.load_frame()) == 1


def test_F_each_append_preserves_prior_prefix(tmp_path: Path) -> None:
    store = ExecutionAuditStore(tmp_path)
    assert store.append_events([_row("A", score=0.1, rank=1, identity="a")]) == 1
    bytes_a = store.path.read_bytes()
    assert store.append_events([_row("B", score=0.2, rank=2, identity="b")]) == 1
    bytes_b = store.path.read_bytes()
    assert bytes_b.startswith(bytes_a)
    assert store.append_events([_row("C", score=0.3, rank=3, identity="c")]) == 1
    bytes_c = store.path.read_bytes()
    assert bytes_c.startswith(bytes_b)
    assert len(store.load_frame()) == 3


def test_G_same_symbol_rank_different_score_stays_separate(tmp_path: Path) -> None:
    store = ExecutionAuditStore(tmp_path)
    first = _row("DUP", score=0.14919682130660616, rank=4, identity="score-high")
    assert store.append_events([first]) == 1
    before = store.path.read_bytes()
    second = _row("DUP", score=0.01, rank=4, identity="score-low")
    assert store.append_events([second]) == 1
    after = store.path.read_bytes()
    assert after.startswith(before)
    frame = store.load_frame()
    assert len(frame) == 2
    assert set(frame["Rank"].astype(int)) == {4}
    assert set(frame["idempotency_key"].astype(str)) == {
        "small_capital_10k|2026-08-03|DUP|SKIPPED|score-high",
        "small_capital_10k|2026-08-03|DUP|SKIPPED|score-low",
    }


def test_H_schema_columns_unchanged() -> None:
    assert AUDIT_COLUMNS == [
        "Date",
        "Symbol",
        "Rank",
        "Score",
        "Decision",
        "Reason",
        "Requested Notional Base",
        "Tradable Quantity",
        "Estimated Price Local",
        "FX To Base",
        "Estimated Cost Base",
        "Cash Before",
        "Cash Reserved",
        "Cash Available",
        "Country",
        "Currency",
        "model_id",
        "config_id",
        "experiment_id",
        "idempotency_key",
    ]


def test_I_corrupt_ledger_is_not_overwritten(tmp_path: Path) -> None:
    store = ExecutionAuditStore(tmp_path)
    store.path.write_bytes(b"{not csv")
    before = store.path.read_bytes()
    with pytest.raises(AuditLogCorruptError):
        store.append_events([_row("Z", score=0.1, rank=1, identity="z")])
    assert store.path.read_bytes() == before
