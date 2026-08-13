"""Score resolution and Top10 membership diagnostics for LTR."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ml.stability_diagnostics import daily_score_dispersion, summarize_dispersion


def score_resolution_by_group(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    date_col: str = "Date",
    region_col: str = "Region",
) -> pd.DataFrame:
    """Per Date×Country unique-score / tie / dispersion stats."""
    rows: list[dict[str, Any]] = []
    keys = [date_col, region_col]
    for (dt, region), group in frame.groupby(keys, sort=True):
        s = group[score_col].replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(s))
        if n == 0:
            continue
        n_unique = int(s.nunique())
        counts = s.value_counts()
        tied_rows = int(counts[counts > 1].sum())
        rows.append(
            {
                "Date": pd.Timestamp(dt),
                "Region": str(region),
                "n_names": n,
                "n_unique_scores": n_unique,
                "unique_score_ratio": float(n_unique / n),
                "tie_ratio": float(tied_rows / n),
                "score_std": float(s.std(ddof=0)) if n > 1 else 0.0,
                "score_min": float(s.min()),
                "score_max": float(s.max()),
                "score_range": float(s.max() - s.min()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Date", "Region"]).reset_index(drop=True)


def summarize_score_resolution(
    daily: pd.DataFrame,
    *,
    low_resolution_tie_threshold: float = 0.5,
) -> dict[str, Any]:
    if daily.empty:
        return {"n_groups": 0}
    tie = daily["tie_ratio"]
    uniq = daily["unique_score_ratio"]
    return {
        "n_groups": int(len(daily)),
        "median_tie_ratio": float(tie.median()),
        "mean_tie_ratio": float(tie.mean()),
        "p90_tie_ratio": float(tie.quantile(0.90)),
        "median_unique_score_ratio": float(uniq.median()),
        "mean_unique_score_ratio": float(uniq.mean()),
        "median_score_std": float(daily["score_std"].median()),
        "median_score_range": float(daily["score_range"].median()),
        "low_resolution_day_ratio": float((tie >= low_resolution_tie_threshold).mean()),
    }


def top_k_membership_diagnostics(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    symbol_col: str = "Symbol",
    date_col: str = "Date",
    region_col: str = "Region",
    k: int = 10,
) -> dict[str, Any]:
    """Top-k cutoff ambiguity and day-to-day membership turnover by country."""
    by_region: dict[str, Any] = {}
    for region, reg_frame in frame.groupby(region_col, sort=True):
        daily_sets: list[set[str]] = []
        cutoffs: list[float] = []
        ambiguity: list[float] = []
        for dt, group in reg_frame.groupby(date_col, sort=True):
            if len(group) < max(3, k // 2):
                continue
            ranked = group.sort_values(score_col, ascending=False, kind="mergesort")
            k_eff = min(k, len(ranked))
            top = ranked.head(k_eff)
            cutoff = float(top[score_col].iloc[-1])
            n_ge = int((group[score_col] >= cutoff - 1e-15).sum())
            # Ambiguity: extras tied at/above cutoff beyond k
            ambiguity.append(max(0.0, float(n_ge - k_eff) / max(k_eff, 1)))
            cutoffs.append(cutoff)
            daily_sets.append(set(top[symbol_col].astype(str)))

        turnovers: list[float] = []
        for i in range(1, len(daily_sets)):
            a, b = daily_sets[i - 1], daily_sets[i]
            if not a and not b:
                continue
            jacc = len(a & b) / len(a | b)
            turnovers.append(1.0 - jacc)

        by_region[str(region)] = {
            "n_days": len(daily_sets),
            "mean_top_cutoff": float(np.mean(cutoffs)) if cutoffs else None,
            "std_top_cutoff": float(np.std(cutoffs)) if cutoffs else None,
            "mean_cutoff_ambiguity": float(np.mean(ambiguity)) if ambiguity else None,
            "median_membership_turnover": float(np.median(turnovers)) if turnovers else None,
            "mean_membership_turnover": float(np.mean(turnovers)) if turnovers else None,
        }
    return {"k": k, "by_region": by_region}


def fold_score_diagnostics(
    scored: pd.DataFrame,
    *,
    low_resolution_tie_threshold: float = 0.5,
) -> dict[str, Any]:
    """Bundle resolution + membership diagnostics for one validation fold."""
    by_group = score_resolution_by_group(scored)
    # Also keep Date-level dispersion summary for compatibility.
    date_disp = daily_score_dispersion(scored)
    return {
        "resolution": summarize_score_resolution(
            by_group, low_resolution_tie_threshold=low_resolution_tie_threshold
        ),
        "date_dispersion_compat": summarize_dispersion(date_disp),
        "top10_membership": top_k_membership_diagnostics(scored, k=10),
    }
