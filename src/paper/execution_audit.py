"""Small Capital execution decision audit (CSV event history, idempotent append)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.paper.atomic_io import atomic_write_csv
from src.simulation.execution_policy import (
    DECISION_FILLED,
    DECISION_QUEUED,
    DECISION_REJECTED,
    DECISION_SKIPPED,
    REASON_AFFORDABLE,
    REASON_ALREADY_HELD,
    REASON_COOLDOWN,
    REASON_COUNTRY_LIMIT,
    REASON_DUPLICATE,
    REASON_FILL_PRICE_UNAFFORDABLE,
    REASON_INSUFFICIENT_CASH,
    REASON_INVALID_FX,
    REASON_INVALID_PRICE,
    REASON_MIN_QUANTITY_NOT_MET,
    REASON_NOT_AFFORDABLE,
    REASON_PENDING_BUY,
    REASON_POSITION_LIMIT,
)

logger = logging.getLogger(__name__)

REASON_CANCELLED = "CANCELLED"
REASON_EXPIRED = "EXPIRED"

# Re-export decision/reason names for callers.
__all__ = [
    "DECISION_QUEUED",
    "DECISION_SKIPPED",
    "DECISION_REJECTED",
    "DECISION_FILLED",
    "REASON_AFFORDABLE",
    "REASON_INSUFFICIENT_CASH",
    "REASON_NOT_AFFORDABLE",
    "REASON_MIN_QUANTITY_NOT_MET",
    "REASON_INVALID_PRICE",
    "REASON_INVALID_FX",
    "REASON_POSITION_LIMIT",
    "REASON_COUNTRY_LIMIT",
    "REASON_DUPLICATE",
    "REASON_FILL_PRICE_UNAFFORDABLE",
    "REASON_COOLDOWN",
    "REASON_ALREADY_HELD",
    "REASON_PENDING_BUY",
    "REASON_CANCELLED",
    "REASON_EXPIRED",
    "AUDIT_COLUMNS",
    "ExecutionAuditStore",
    "execution_decision_idempotency_key",
    "normalize_audit_row",
]

AUDIT_COLUMNS: list[str] = [
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


def execution_decision_idempotency_key(
    *,
    experiment_id: str,
    date: str,
    symbol: str,
    decision: str,
    order_identity: str,
) -> str:
    """Stable key: experiment|date|symbol|decision|order_identity."""
    return "|".join(
        [
            str(experiment_id),
            str(date),
            str(symbol),
            str(decision),
            str(order_identity),
        ]
    )


def normalize_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {c: row.get(c) for c in AUDIT_COLUMNS}
    key = out.get("idempotency_key")
    if not key:
        out["idempotency_key"] = execution_decision_idempotency_key(
            experiment_id=str(out.get("experiment_id") or ""),
            date=str(out.get("Date") or ""),
            symbol=str(out.get("Symbol") or ""),
            decision=str(out.get("Decision") or ""),
            order_identity=str(
                row.get("order_identity")
                or f"{out.get('Reason')}|{out.get('Rank')}|{out.get('Tradable Quantity')}"
            ),
        )
    return out


class ExecutionAuditStore:
    """Append-only event history under ``{paper_state_dir}/execution_decisions.csv``."""

    def __init__(self, paper_state_dir: Path) -> None:
        self.root = Path(paper_state_dir)
        self.path = self.root / "execution_decisions.csv"

    def ensure_file(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            atomic_write_csv(self.path, pd.DataFrame(columns=AUDIT_COLUMNS))

    def load_frame(self) -> pd.DataFrame:
        """Load audit CSV; corrupt/empty files yield an empty framed schema."""
        if not self.path.exists():
            return pd.DataFrame(columns=AUDIT_COLUMNS)
        try:
            df = pd.read_csv(self.path)
        except Exception:  # noqa: BLE001 — corrupt history must not break paper
            logger.exception("Corrupt execution_decisions.csv; starting empty frame")
            return pd.DataFrame(columns=AUDIT_COLUMNS)
        if df is None or df.empty:
            return pd.DataFrame(columns=AUDIT_COLUMNS)
        for col in AUDIT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[AUDIT_COLUMNS].copy()

    def existing_keys(self) -> set[str]:
        df = self.load_frame()
        if df.empty or "idempotency_key" not in df.columns:
            return set()
        return {str(k) for k in df["idempotency_key"].dropna().astype(str).tolist() if str(k)}

    def append_events(self, rows: Iterable[dict[str, Any]]) -> int:
        """Append new events only (idempotent). Returns number of rows newly written."""
        prepared = [normalize_audit_row(dict(r)) for r in rows]
        if not prepared:
            return 0
        self.ensure_file()
        existing = self.existing_keys()
        new_rows = [r for r in prepared if str(r["idempotency_key"]) not in existing]
        if not new_rows:
            return 0
        prev = self.load_frame()
        merged = pd.concat([prev, pd.DataFrame(new_rows)], ignore_index=True)
        # Safety: drop duplicate keys keeping first event
        if "idempotency_key" in merged.columns:
            merged = merged.drop_duplicates(subset=["idempotency_key"], keep="first")
        merged = merged.reindex(columns=AUDIT_COLUMNS)
        atomic_write_csv(self.path, merged)
        return len(new_rows)
