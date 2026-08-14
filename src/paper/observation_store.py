"""Phase 5C: persist paper ranking observations + run health (Volume-safe).

Does not mutate trading state. Does not retrain. Does not change Validity Gate.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.paper.atomic_io import atomic_write_csv, atomic_write_json

logger = logging.getLogger(__name__)

# Fixed column order for ranking snapshots (audit / future IC join keys).
RANKING_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "Date",
    "Symbol",
    "Score",
    "Rank",
    "Percentile",
    "Selected",
    "model_id",
    "Country",
    "Close",
)

RUN_HEALTH_HISTORY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "run_started_utc",
    "run_finished_utc",
    "asof",
    "status",
    "skip_reason",
    "error_type",
    "error_message",
    "model_id",
    "processed_sessions",
    "notes",
)


def ranking_snapshot_conflict_tag(day: pd.Timestamp | str) -> str:
    """Machine-readable audit marker for immutable-snapshot conflicts."""
    d = pd.Timestamp(day).normalize().date().isoformat()
    return f"ranking_snapshot_conflict:{d}"

STATUS_SUCCESS = "SUCCESS"
STATUS_SKIPPED = "SKIPPED"
STATUS_ERROR = "ERROR"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PaperObservationStore:
    """Filesystem store under ``paper_state_dir`` for observations (not portfolio state)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rankings_dir = root / "rankings"
        self.health_path = root / "run_health.json"
        self.health_history_path = root / "run_health_history.csv"
        self.root.mkdir(parents=True, exist_ok=True)
        self.rankings_dir.mkdir(parents=True, exist_ok=True)

    def ranking_path(self, day: pd.Timestamp | str) -> Path:
        d = pd.Timestamp(day).normalize().date().isoformat()
        return self.rankings_dir / f"{d}.csv"

    def load_health(self) -> dict[str, Any]:
        if not self.health_path.exists():
            return {}
        try:
            return json.loads(self.health_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read run_health.json: %s", exc)
            return {}

    def begin_run(self, *, model_id: str, run_id: str | None = None) -> dict[str, Any]:
        """Start a Cron/manual run; preserves prior last_success_utc."""
        prev = self.load_health()
        started = _utc_now()
        rid = run_id or f"{started}_{uuid.uuid4().hex[:8]}"
        health = {
            "run_id": rid,
            "model_id": model_id,
            "last_run_started_utc": started,
            "last_run_finished_utc": None,
            "last_success_utc": prev.get("last_success_utc"),
            "last_asof": prev.get("last_asof"),
            "status": "RUNNING",
            "skip_reason": None,
            "error_type": None,
            "error_message": None,
            "processed_sessions": 0,
            "generated_at_utc": started,
        }
        try:
            atomic_write_json(self.health_path, health)
        except OSError:
            logger.exception("Failed to write run_health begin (non-fatal)")
        return health

    def build_ranking_snapshot(
        self,
        *,
        day_rank: pd.DataFrame,
        model_id: str,
        price_panel: pd.DataFrame | None,
        top_percentile: float,
        observation_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Build one-day snapshot from country rankings (+ optional Close as-of that day)."""
        if day_rank is None or day_rank.empty:
            return pd.DataFrame(columns=list(RANKING_SNAPSHOT_COLUMNS))

        frame = day_rank.copy()
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
        day = pd.Timestamp(observation_date).normalize()
        frame = frame.loc[frame["Date"] == day].copy()
        if frame.empty:
            return pd.DataFrame(columns=list(RANKING_SNAPSHOT_COLUMNS))

        if "Selected" not in frame.columns:
            frame["Selected"] = frame["Percentile"].astype(float) <= float(top_percentile)
        frame["model_id"] = str(model_id)
        if "Country" not in frame.columns:
            frame["Country"] = pd.NA

        closes: dict[str, float] = {}
        if price_panel is not None and not price_panel.empty:
            px = price_panel.copy()
            px["Date"] = pd.to_datetime(px["Date"]).dt.normalize()
            day_px = px.loc[px["Date"] == day]
            if not day_px.empty and "Symbol" in day_px.columns and "Close" in day_px.columns:
                for sym, g in day_px.groupby("Symbol", sort=False):
                    val = g["Close"].iloc[-1]
                    closes[str(sym)] = float(val) if pd.notna(val) else float("nan")

        frame["Close"] = frame["Symbol"].map(lambda s: closes.get(str(s), float("nan")))
        frame["Date"] = frame["Date"].map(lambda d: str(pd.Timestamp(d).date()))
        out = frame.loc[:, [c for c in RANKING_SNAPSHOT_COLUMNS if c in frame.columns]].copy()
        for col in RANKING_SNAPSHOT_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out.loc[:, list(RANKING_SNAPSHOT_COLUMNS)].reset_index(drop=True)

    def save_ranking_snapshot(self, snapshot: pd.DataFrame, *, day: pd.Timestamp | str) -> str:
        """Save first observation only. Returns wrote|unchanged|conflict|empty.

        Existing ``rankings/YYYY-MM-DD.csv`` is immutable: never overwrite on diff.
        """
        if snapshot is None or snapshot.empty:
            return "empty"
        path = self.ranking_path(day)
        ordered = snapshot.loc[:, list(RANKING_SNAPSHOT_COLUMNS)].copy()
        if not path.exists():
            atomic_write_csv(path, ordered)
            return "wrote"
        try:
            prev = pd.read_csv(path)
            prev_cmp = prev.reindex(columns=list(RANKING_SNAPSHOT_COLUMNS))
            new_cmp = ordered.copy()
            for col in ("Score", "Percentile", "Close", "Rank"):
                if col in prev_cmp.columns:
                    prev_cmp[col] = pd.to_numeric(prev_cmp[col], errors="coerce")
                new_cmp[col] = pd.to_numeric(new_cmp[col], errors="coerce")
            if "Selected" in prev_cmp.columns:
                prev_cmp["Selected"] = prev_cmp["Selected"].astype(str)
                new_cmp["Selected"] = new_cmp["Selected"].astype(str)
            prev_cmp = prev_cmp.sort_values(["Symbol"]).reset_index(drop=True)
            new_cmp = new_cmp.sort_values(["Symbol"]).reset_index(drop=True)
            if prev_cmp.equals(new_cmp):
                return "unchanged"
            tag = ranking_snapshot_conflict_tag(day)
            logger.error(
                "Ranking snapshot CONFLICT %s — keeping first observation immutable "
                "(path=%s). Recomputed ranking differs; treat as audit signal "
                "(data revision / cache / dependency). Original file NOT overwritten.",
                tag,
                path,
            )
            return "conflict"
        except Exception as exc:  # noqa: BLE001
            tag = ranking_snapshot_conflict_tag(day)
            logger.error(
                "Ranking snapshot CONFLICT %s — existing file unreadable; "
                "refusing overwrite to preserve first observation. path=%s err=%s",
                tag,
                path,
                exc,
            )
            return "conflict"

    def finish_run(
        self,
        *,
        status: str,
        model_id: str,
        asof: pd.Timestamp | str | None,
        run_id: str,
        run_started_utc: str,
        processed_sessions: int = 0,
        skip_reason: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        ranking_snapshot_conflicts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Write run_health.json + append run_health_history.csv (run-level history)."""
        prev = self.load_health()
        finished = _utc_now()
        last_success = prev.get("last_success_utc")
        if status == STATUS_SUCCESS:
            last_success = finished

        asof_s = str(pd.Timestamp(asof).date()) if asof is not None else prev.get("last_asof")
        if status in (STATUS_SUCCESS, STATUS_SKIPPED):
            last_asof = asof_s
        else:
            last_asof = prev.get("last_asof")

        conflict_tags = [
            ranking_snapshot_conflict_tag(d) if not str(d).startswith("ranking_snapshot_conflict:") else str(d)
            for d in (ranking_snapshot_conflicts or [])
        ]
        # Deduplicate while preserving order
        seen: set[str] = set()
        conflict_tags_unique: list[str] = []
        for t in conflict_tags:
            if t not in seen:
                seen.add(t)
                conflict_tags_unique.append(t)

        notes: dict[str, Any] = {
            "last_success_utc": "Updated only on SUCCESS (new session processing), not SKIPPED",
            "ranking_catchup": (
                "For each Date in the processed engine calendar that exists in "
                "ai_rankings, a snapshot is saved. Dates absent from ai_rankings "
                "are not invented. Never backfilled with later-run scores."
            ),
            "ranking_snapshot_policy": (
                "rankings/YYYY-MM-DD.csv is the first durable observation for that Date; "
                "later recomputes never overwrite it."
            ),
        }
        if conflict_tags_unique:
            notes["ranking_snapshot_conflicts"] = conflict_tags_unique

        health = {
            "run_id": run_id,
            "model_id": model_id,
            "last_run_started_utc": run_started_utc,
            "last_run_finished_utc": finished,
            "last_success_utc": last_success,
            "last_asof": last_asof,
            "status": status,
            "skip_reason": skip_reason,
            "error_type": error_type,
            "error_message": (error_message[:500] if error_message else None),
            "processed_sessions": int(processed_sessions),
            "generated_at_utc": finished,
            "notes": notes,
        }
        try:
            atomic_write_json(self.health_path, health)
        except OSError:
            logger.exception("Failed to write run_health.json (non-fatal)")

        notes_cell = ";".join(conflict_tags_unique) if conflict_tags_unique else None
        row = {
            "run_id": run_id,
            "run_started_utc": run_started_utc,
            "run_finished_utc": finished,
            "asof": asof_s,
            "status": status,
            "skip_reason": skip_reason,
            "error_type": error_type,
            "error_message": (error_message[:200] if error_message else None),
            "model_id": model_id,
            "processed_sessions": int(processed_sessions),
            "notes": notes_cell,
        }
        try:
            self._append_health_history(row)
        except OSError:
            logger.exception("Failed to append run_health_history.csv (non-fatal)")
        return health

    def _append_health_history(self, row: dict[str, Any]) -> None:
        df = pd.DataFrame([row], columns=list(RUN_HEALTH_HISTORY_COLUMNS))
        if self.health_history_path.exists():
            prev = pd.read_csv(self.health_history_path)
            for col in RUN_HEALTH_HISTORY_COLUMNS:
                if col not in prev.columns:
                    prev[col] = pd.NA
            if "run_id" in prev.columns:
                prev = prev.loc[prev["run_id"].astype(str) != str(row["run_id"])]
            prev = prev.reindex(columns=list(RUN_HEALTH_HISTORY_COLUMNS))
            df = pd.concat([prev, df], ignore_index=True)
        atomic_write_csv(self.health_history_path, df)

    def save_processed_day_rankings(
        self,
        *,
        ai_rankings: pd.DataFrame,
        price_panel: pd.DataFrame | None,
        model_id: str,
        top_percentile: float,
        session_days: list[pd.Timestamp],
    ) -> dict[str, Any]:
        """Persist snapshots for session days that exist in ai_rankings (no invention).

        Returns counts plus ``conflict_days`` (ISO dates whose first snapshot was kept).
        """
        counts: dict[str, Any] = {
            "wrote": 0,
            "unchanged": 0,
            "conflict": 0,
            "empty": 0,
            "missing_in_rankings": 0,
            "conflict_days": [],
        }
        if ai_rankings is None or ai_rankings.empty:
            return counts
        ranked_dates = {
            pd.Timestamp(x).normalize().date()
            for x in pd.to_datetime(ai_rankings["Date"]).dt.normalize().unique()
        }
        for day in session_days:
            d = pd.Timestamp(day).normalize()
            if d.date() not in ranked_dates:
                counts["missing_in_rankings"] += 1
                continue
            snap = self.build_ranking_snapshot(
                day_rank=ai_rankings,
                model_id=model_id,
                price_panel=price_panel,
                top_percentile=top_percentile,
                observation_date=d,
            )
            action = self.save_ranking_snapshot(snap, day=d)
            counts[action] = counts.get(action, 0) + 1
            if action == "conflict":
                counts["conflict_days"].append(str(d.date()))
        return counts
