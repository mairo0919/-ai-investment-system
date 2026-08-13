"""Ranking adapter: model-agnostic score table for the simulation engine."""

from __future__ import annotations

import pandas as pd


RANK_COLUMNS = ("Date", "Symbol", "Country", "Score", "Rank", "Percentile")


def attach_country_ranks(
    scored: pd.DataFrame,
    *,
    score_col: str = "score",
    date_col: str = "Date",
    symbol_col: str = "Symbol",
    country_col: str = "Region",
) -> pd.DataFrame:
    """Build Date/Symbol/Country/Score/Rank/Percentile within each Country×Date.

    Rank 1 = best (highest score). Percentile in (0, 1], top = near 0.
    ``Percentile`` = rank / n  (so Top 10% <=> Percentile <= 0.10).
    """
    if scored.empty:
        return pd.DataFrame(columns=list(RANK_COLUMNS))

    frame = scored[[date_col, symbol_col, country_col, score_col]].copy()
    frame = frame.rename(
        columns={
            date_col: "Date",
            symbol_col: "Symbol",
            country_col: "Country",
            score_col: "Score",
        }
    )
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Score"] = frame["Score"].astype(float)

    def _rank_group(g: pd.DataFrame) -> pd.DataFrame:
        out = g.copy()
        # Dense rank: highest score -> 1
        out["Rank"] = out["Score"].rank(method="first", ascending=False).astype(int)
        n = max(len(out), 1)
        out["Percentile"] = out["Rank"] / float(n)
        return out

    parts: list[pd.DataFrame] = []
    for _, g in frame.groupby(["Date", "Country"], sort=False):
        parts.append(_rank_group(g))
    ranked = pd.concat(parts, ignore_index=True) if parts else frame
    return ranked.loc[:, list(RANK_COLUMNS)]


def select_candidates(
    ranking_day: pd.DataFrame,
    *,
    mode: str,
    top_percentile: float,
    top_n: int,
) -> pd.DataFrame:
    """Filter one day's country ranking to entry candidates."""
    if ranking_day.empty:
        return ranking_day.copy()
    if mode == "top_n":
        return ranking_day.loc[ranking_day["Rank"] <= int(top_n)].copy()
    return ranking_day.loc[ranking_day["Percentile"] <= float(top_percentile)].copy()
