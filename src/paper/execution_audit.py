"""Small Capital execution decision audit (CSV event history, idempotent append)."""

from __future__ import annotations

import csv
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
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


class AuditLogCorruptError(RuntimeError):
    """Existing audit file cannot be appended without rewriting its bytes."""

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
    "AuditLogCorruptError",
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


def _format_cell(value: Any) -> str:
    """Text for a newly appended cell. Never applied to rows already on disk."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return str(value)
    return str(value)


def _encode_new_rows(rows: list[dict[str, Any]], *, include_header: bool) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    if include_header:
        writer.writerow(AUDIT_COLUMNS)
    for row in rows:
        writer.writerow([_format_cell(row.get(col)) for col in AUDIT_COLUMNS])
    return buffer.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _idempotency_keys(raw: bytes) -> set[str]:
    """Read keys from raw CSV text. Raises if the ledger cannot be appended safely."""
    if not raw.strip():
        return set()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditLogCorruptError("execution audit is not valid UTF-8") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        return set()
    except csv.Error as exc:
        raise AuditLogCorruptError("execution audit CSV could not be parsed") from exc
    if header != AUDIT_COLUMNS:
        raise AuditLogCorruptError(
            "execution audit header does not match the execution decision schema"
        )
    key_at = AUDIT_COLUMNS.index("idempotency_key")
    keys: set[str] = set()
    try:
        for record in reader:
            if not record or all(not cell.strip() for cell in record):
                continue
            if len(record) != len(AUDIT_COLUMNS):
                raise AuditLogCorruptError(
                    "execution audit row does not match the execution decision schema"
                )
            key = record[key_at]
            if key:
                keys.add(key)
    except csv.Error as exc:
        raise AuditLogCorruptError("execution audit CSV could not be parsed") from exc
    return keys


def _dedupe_new_rows(
    prepared: list[dict[str, Any]], existing: set[str]
) -> list[dict[str, Any]]:
    """Keep the first row for each new idempotency key, including within one call."""
    seen = set(existing)
    fresh: list[dict[str, Any]] = []
    for row in prepared:
        key = str(row.get("idempotency_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        fresh.append(row)
    return fresh


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
            _atomic_write_bytes(self.path, _encode_new_rows([], include_header=True))

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
        if not self.path.exists():
            return set()
        try:
            return _idempotency_keys(self.path.read_bytes())
        except AuditLogCorruptError:
            logger.exception("Corrupt execution_decisions.csv; no idempotency keys loaded")
            return set()

    def append_events(self, rows: Iterable[dict[str, Any]]) -> int:
        """Append new events only. Existing file bytes are never re-serialized.

        Returns the number of rows newly written. A duplicate idempotency key
        is a no-op. A corrupt existing ledger raises ``AuditLogCorruptError``
        and is left untouched.
        """
        prepared = [normalize_audit_row(dict(r)) for r in rows]
        if not prepared:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        prior = self.path.read_bytes() if self.path.exists() else b""
        if not prior.strip():
            fresh = _dedupe_new_rows(prepared, set())
            if not fresh:
                return 0
            _atomic_write_bytes(
                self.path, _encode_new_rows(fresh, include_header=True)
            )
            return len(fresh)
        try:
            existing = _idempotency_keys(prior)
        except AuditLogCorruptError:
            logger.exception(
                "Refusing to append to corrupt execution_decisions.csv; file unchanged"
            )
            raise
        fresh = _dedupe_new_rows(prepared, existing)
        if not fresh:
            return 0
        addition = _encode_new_rows(fresh, include_header=False)
        if prior.endswith(b"\n"):
            updated = prior + addition
        else:
            updated = prior + b"\n" + addition
        _atomic_write_bytes(self.path, updated)
        return len(fresh)
