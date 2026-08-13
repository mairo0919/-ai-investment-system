"""Score-resolution / dispersion / rebalance entry filters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ml.score_resolution import score_resolution_by_group
from src.simulation.config import SimulationConfig


def daily_country_score_stats(
    ranking_day: pd.DataFrame,
    *,
    top_percentile: float = 0.10,
) -> dict[str, Any]:
    """Compute resolution + score-separation stats for one Date×Country slice."""
    empty = {
        "n_names": 0,
        "score_std": 0.0,
        "score_range": 0.0,
        "tie_ratio": 1.0,
        "unique_score_ratio": 0.0,
        "top_median_gap": 0.0,
        "cutoff_median_gap": 0.0,
    }
    if ranking_day.empty or "Score" not in ranking_day.columns:
        return empty
    s = ranking_day["Score"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(s))
    if n == 0:
        return empty
    counts = s.value_counts()
    tied_rows = int(counts[counts > 1].sum())
    median = float(s.median())
    top = float(s.max())
    k = max(1, int(np.ceil(n * float(top_percentile))))
    cutoff = float(s.nlargest(k).min())
    return {
        "n_names": n,
        "score_std": float(s.std(ddof=0)) if n > 1 else 0.0,
        "score_range": float(top - s.min()),
        "tie_ratio": float(tied_rows / n),
        "unique_score_ratio": float(s.nunique() / n),
        "top_median_gap": float(top - median),
        "cutoff_median_gap": float(cutoff - median),
    }


def calibrate_train_thresholds(
    train_rankings: pd.DataFrame,
    *,
    quantile: float = 0.25,
    country_col: str = "Country",
    top_percentile: float = 0.10,
) -> dict[str, float]:
    """Derive min dispersion / separation thresholds from train rankings only."""
    if train_rankings.empty:
        return {
            "min_score_std": 0.0,
            "min_top_median_gap": 0.0,
            "min_unique_score_ratio": 0.0,
            "min_cutoff_median_gap": 0.0,
        }

    frame = train_rankings.copy()
    if "Score" not in frame.columns and "score" in frame.columns:
        frame = frame.rename(columns={"score": "Score"})
    tmp = frame.rename(columns={"Country": "Region", "Score": "score"})
    daily = score_resolution_by_group(tmp, score_col="score", region_col="Region")
    top_gaps: list[float] = []
    cut_gaps: list[float] = []
    for _, g in frame.groupby(["Date", country_col], sort=False):
        stats = daily_country_score_stats(g, top_percentile=top_percentile)
        top_gaps.append(float(stats["top_median_gap"]))
        cut_gaps.append(float(stats["cutoff_median_gap"]))
    gap_s = pd.Series(top_gaps, dtype=float)
    cut_s = pd.Series(cut_gaps, dtype=float)
    q = float(quantile)
    return {
        "min_score_std": float(daily["score_std"].quantile(q)) if not daily.empty else 0.0,
        "min_top_median_gap": float(gap_s.quantile(q)) if len(gap_s) else 0.0,
        "min_unique_score_ratio": (
            float(daily["unique_score_ratio"].quantile(q)) if not daily.empty else 0.0
        ),
        "min_cutoff_median_gap": float(cut_s.quantile(q)) if len(cut_s) else 0.0,
    }


def allow_new_entries(
    ranking_day: pd.DataFrame,
    cfg: SimulationConfig,
) -> tuple[bool, str]:
    """Return whether new entries are allowed for this country-day ranking."""
    stats = daily_country_score_stats(ranking_day, top_percentile=cfg.top_percentile)
    if cfg.block_low_resolution_entries:
        if stats["tie_ratio"] >= cfg.low_resolution_tie_threshold:
            return False, "low_resolution_tie"
        if (
            cfg.min_unique_score_ratio is not None
            and stats["unique_score_ratio"] < cfg.min_unique_score_ratio
        ):
            return False, "low_unique_score_ratio"
    if cfg.min_score_std is not None and stats["score_std"] < cfg.min_score_std:
        return False, "low_score_std"
    if cfg.min_top_median_gap is not None and stats["top_median_gap"] < cfg.min_top_median_gap:
        return False, "low_top_median_gap"
    if cfg.use_score_separation_filter:
        if (
            cfg.min_top_median_gap is not None
            and stats["top_median_gap"] < cfg.min_top_median_gap
        ):
            return False, "score_separation_top_median"
        if (
            cfg.min_cutoff_median_gap is not None
            and stats["cutoff_median_gap"] < cfg.min_cutoff_median_gap
        ):
            return False, "score_separation_cutoff_median"
    elif (
        cfg.min_cutoff_median_gap is not None
        and stats["cutoff_median_gap"] < cfg.min_cutoff_median_gap
    ):
        return False, "low_cutoff_median_gap"
    return True, "ok"


def _weekly_target(
    day: pd.Timestamp,
    calendar: list[pd.Timestamp],
    weekly_weekday: int,
) -> pd.Timestamp | None:
    year, week, _ = day.isocalendar()
    week_days = [
        d
        for d in calendar
        if d.isocalendar().year == year and d.isocalendar().week == week
    ]
    if not week_days:
        return None
    preferred = [d for d in week_days if int(d.weekday()) == int(weekly_weekday)]
    return preferred[0] if preferred else week_days[0]


def is_entry_rebalance_day(
    day: pd.Timestamp,
    calendar: list[pd.Timestamp],
    *,
    frequency: str,
    weekly_weekday: int = 0,
) -> bool:
    """Daily / weekly / biweekly entry permission (exits remain daily)."""
    if frequency == "daily":
        return True
    day = pd.Timestamp(day).normalize()
    target = _weekly_target(day, calendar, weekly_weekday)
    if target is None or day != target:
        return False
    if frequency == "weekly":
        return True
    if frequency == "biweekly":
        weekly_days = []
        seen: set[tuple[int, int]] = set()
        for d in calendar:
            key = (d.isocalendar().year, d.isocalendar().week)
            if key in seen:
                continue
            t = _weekly_target(d, calendar, weekly_weekday)
            if t is not None:
                weekly_days.append(t)
                seen.add(key)
        bi = set(weekly_days[::2])
        return day in bi
    return False


def count_reentries(
    trades: pd.DataFrame,
    *,
    within_days: tuple[int, ...] = (5, 10),
) -> dict[str, int]:
    """Count sell→buy re-entries of the same symbol within N trading days."""
    out = {f"reentry_within_{n}_days": 0 for n in within_days}
    if trades.empty:
        return out
    tr = trades.copy()
    tr["Exit Date"] = pd.to_datetime(tr["Exit Date"])
    tr["Entry Date"] = pd.to_datetime(tr["Entry Date"])
    for symbol, g in tr.groupby("Symbol"):
        g = g.sort_values("Exit Date")
        exits = list(g["Exit Date"])
        entries = list(g["Entry Date"])
        # Pair each exit with later entries of same symbol
        for i, ex in enumerate(exits):
            for en in entries:
                if en <= ex:
                    continue
                delta = len(pd.bdate_range(ex, en)) - 1
                for n in within_days:
                    if 0 < delta <= n:
                        out[f"reentry_within_{n}_days"] += 1
                break  # nearest subsequent entry only
    return out
