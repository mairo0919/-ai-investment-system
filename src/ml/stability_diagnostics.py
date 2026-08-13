"""Phase 3E ranking-edge stability diagnostics (evaluation only, no trading)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.ml.cross_section import cross_sectional_metrics_by_date, summarize_daily_metrics
from src.ml.ranking import information_coefficient, top_k_average_return

REGIONS = ("Japan", "United States", "Europe")

# Days with tiny cross-sectional score dispersion are flagged (not dropped).
LOW_DISPERSION_STD = 1e-4
LOW_UNIQUE_RATIO = 0.25
HIGH_TIE_RATIO = 0.5


def _clean_pair(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
) -> pd.DataFrame:
    required = {score_col, return_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, return_col])
    return out


def country_normalize_scores(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    date_col: str = "Date",
    region_col: str = "Region",
    method: str = "zscore",
) -> pd.DataFrame:
    """Add country-normalized score columns (post-hoc; no model retrain).

    method:
        - ``zscore``: within Date × Region z-score
        - ``percentile``: within Date × Region percentile rank in [0, 1]
    """
    if method not in {"zscore", "percentile"}:
        raise ValueError("method must be 'zscore' or 'percentile'")
    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    group_keys = [date_col, region_col]

    if method == "zscore":
        means = out.groupby(group_keys, sort=False)[score_col].transform("mean")
        stds = out.groupby(group_keys, sort=False)[score_col].transform("std")
        # ddof default of groupby std is 1; constant groups → NaN → 0
        z = (out[score_col] - means) / stds.replace(0.0, np.nan)
        out["score_country_z"] = z.fillna(0.0)
    else:
        out["score_country_pct"] = out.groupby(group_keys, sort=False)[score_col].rank(
            method="average", pct=True
        )
    return out


def score_distribution_summary(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
) -> dict[str, float | int | None]:
    """Aggregate probability score distribution diagnostics."""
    s = frame[score_col].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "n_unique": 0,
        }
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "p10": float(q.loc[0.10]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.90]),
        "n_unique": int(s.nunique()),
    }


def daily_score_dispersion(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    date_col: str = "Date",
) -> pd.DataFrame:
    """Per-Date cross-sectional score dispersion / ties / diagnostic flags."""
    rows: list[dict[str, Any]] = []
    for dt, group in frame.groupby(pd.to_datetime(frame[date_col]), sort=True):
        s = group[score_col].replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(s))
        if n == 0:
            continue
        n_unique = int(s.nunique())
        # Fraction of rows whose value appears more than once.
        counts = s.value_counts()
        tied_rows = int(counts[counts > 1].sum())
        tie_ratio = float(tied_rows / n)
        std = float(s.std(ddof=0)) if n > 1 else 0.0
        unique_ratio = float(n_unique / n)
        low_dispersion = bool(
            std < LOW_DISPERSION_STD
            or unique_ratio < LOW_UNIQUE_RATIO
            or tie_ratio > HIGH_TIE_RATIO
        )
        rows.append(
            {
                "Date": pd.Timestamp(dt),
                "n_names": n,
                "score_std": std,
                "n_unique_scores": n_unique,
                "unique_ratio": unique_ratio,
                "tie_ratio": tie_ratio,
                "low_dispersion_flag": low_dispersion,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def summarize_dispersion(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return {"n_days": 0}
    return {
        "n_days": int(len(daily)),
        "mean_score_std": float(daily["score_std"].mean()),
        "median_score_std": float(daily["score_std"].median()),
        "mean_unique_ratio": float(daily["unique_ratio"].mean()),
        "mean_tie_ratio": float(daily["tie_ratio"].mean()),
        "low_dispersion_day_ratio": float(daily["low_dispersion_flag"].mean()),
        "low_dispersion_days": int(daily["low_dispersion_flag"].sum()),
    }


def one_day_ic_diagnostics(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
    region_col: str = "Region",
    symbol_col: str = "Symbol",
) -> dict[str, Any]:
    """Investigate suspicious 1d IC vs near-random ROC-AUC."""
    clean = _clean_pair(frame, score_col=score_col, return_col=return_col)
    score_stats = score_distribution_summary(clean, score_col=score_col)
    ret = clean[return_col]
    ret_stats = {
        "mean": float(ret.mean()),
        "std": float(ret.std(ddof=0)),
        "min": float(ret.min()),
        "max": float(ret.max()),
        "p01": float(ret.quantile(0.01)),
        "p99": float(ret.quantile(0.99)),
    }
    daily_disp = daily_score_dispersion(clean, score_col=score_col)
    daily_ic = cross_sectional_metrics_by_date(
        clean, score_col=score_col, return_col=return_col, top_fractions=(0.10,)
    )
    # IC vs dispersion relationship
    merged = pd.DataFrame()
    if not daily_ic.empty and not daily_disp.empty:
        merged = daily_ic.merge(daily_disp, on="Date", how="inner")
    ic_low = (
        float(merged.loc[merged["low_dispersion_flag"], "ic"].mean())
        if not merged.empty and merged["low_dispersion_flag"].any()
        else None
    )
    ic_ok = (
        float(merged.loc[~merged["low_dispersion_flag"], "ic"].mean())
        if not merged.empty and (~merged["low_dispersion_flag"]).any()
        else None
    )

    # Country contribution to pooled IC days
    country_daily: dict[str, Any] = {}
    for region, group in clean.groupby(region_col):
        d = cross_sectional_metrics_by_date(
            group,
            score_col=score_col,
            return_col=return_col,
            top_fractions=(0.10,),
            min_names=3,
        )
        country_daily[str(region)] = summarize_daily_metrics(d)

    # Symbol dominance: correlation of leave-one-symbol-out IC shift is heavy;
    # instead report top symbols by |score - median| * |return| contribution proxy.
    clean = clean.copy()
    clean["_abs_score_dev"] = (clean[score_col] - clean[score_col].median()).abs()
    clean["_contrib_proxy"] = clean["_abs_score_dev"] * clean[return_col].abs()
    top_symbols = (
        clean.groupby(symbol_col)["_contrib_proxy"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # Pool Spearman sample size / ties
    n = len(clean)
    n_unique_score = int(clean[score_col].nunique())
    n_unique_ret = int(clean[return_col].nunique())
    pooled_ic = information_coefficient(clean, score_col=score_col, return_col=return_col)

    # Rank ties: average number of rows per distinct score
    artifact_flags: list[str] = []
    if score_stats["std"] is not None and score_stats["std"] < LOW_DISPERSION_STD:
        artifact_flags.append("near_constant_global_score")
    if n > 0 and n_unique_score / n < LOW_UNIQUE_RATIO:
        artifact_flags.append("low_unique_score_ratio")
    disp_summary = summarize_dispersion(daily_disp)
    if disp_summary.get("low_dispersion_day_ratio", 0) and (
        disp_summary["low_dispersion_day_ratio"] > 0.3
    ):
        artifact_flags.append("frequent_low_dispersion_days")
    if ic_low is not None and ic_ok is not None and abs(ic_low) > abs(ic_ok) * 1.5:
        artifact_flags.append("ic_driven_by_low_dispersion_days")

    return {
        "n_rows": n,
        "pooled_spearman_ic": pooled_ic,
        "score_distribution": score_stats,
        "future_return_1d_distribution": ret_stats,
        "n_unique_scores": n_unique_score,
        "n_unique_returns": n_unique_ret,
        "unique_score_ratio": float(n_unique_score / n) if n else None,
        "dispersion_summary": disp_summary,
        "mean_ic_on_low_dispersion_days": ic_low,
        "mean_ic_on_normal_dispersion_days": ic_ok,
        "country_daily_ic": country_daily,
        "top_symbols_by_contrib_proxy": {
            str(k): float(v) for k, v in top_symbols.items()
        },
        "artifact_flags": artifact_flags,
        "interpretation_notes": [
            "Daily Mean IC averages cross-sectional Spearman; ROC-AUC is pooled classification.",
            "High Mean IC with ROC≈0.5 can arise from weak but consistent cross-sectional ordering,",
            "or from low-dispersion / tied scores making Spearman unstable on small N.",
        ],
    }


def classify_market_regime(sma_60_ratio: pd.Series, return_20d: pd.Series) -> pd.Series:
    """Simple Bull / Bear / Neutral from benchmark features (no ML regimes)."""
    above = sma_60_ratio > 0
    below = sma_60_ratio < 0
    pos = return_20d > 0
    neg = return_20d < 0
    regime = pd.Series(np.full(len(sma_60_ratio), "Neutral", dtype=object), index=sma_60_ratio.index)
    regime = regime.mask(above & pos, "Bull")
    regime = regime.mask(below & neg, "Bear")
    return regime


def assign_volatility_regime(
    train_vol: pd.Series,
    valid_vol: pd.Series,
) -> tuple[pd.Series, dict[str, float]]:
    """Tercile volatility regimes; thresholds from train only (no look-ahead)."""
    clean = train_vol.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        thresholds = {"q33": float("nan"), "q66": float("nan")}
        labels = pd.Series(["Medium"] * len(valid_vol), index=valid_vol.index, dtype=object)
        return labels, thresholds
    q33, q66 = float(clean.quantile(1 / 3)), float(clean.quantile(2 / 3))
    thresholds = {"q33": q33, "q66": q66}
    v = valid_vol
    labels = pd.Series(np.full(len(v), "Medium", dtype=object), index=v.index)
    labels = labels.mask(v <= q33, "Low")
    labels = labels.mask(v > q66, "High")
    # NaN vol → Neutral-like Medium already; leave as Medium
    labels = labels.where(v.notna(), other="Medium")
    return labels, thresholds


def regime_performance(
    frame: pd.DataFrame,
    *,
    regime_col: str,
    score_col: str = "score",
    return_col: str = "future_return",
) -> dict[str, Any]:
    """Top10% excess / Mean IC by regime label."""
    out: dict[str, Any] = {}
    for label, group in frame.groupby(regime_col, dropna=False):
        key = "null" if pd.isna(label) else str(label)
        daily = cross_sectional_metrics_by_date(
            group,
            score_col=score_col,
            return_col=return_col,
            top_fractions=(0.10,),
            min_names=3,
        )
        summary = summarize_daily_metrics(daily)
        out[key] = {
            "sample_count": int(len(group)),
            "n_days": summary.get("n_days", 0),
            "top10_excess_return": summary.get("top_10pct_excess_return"),
            "ic": summary.get("ic"),
        }
    return out


def country_ranking_report(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
    region_col: str = "Region",
) -> dict[str, Any]:
    """Independent cross-sectional ranking within each country/region."""
    global_daily = cross_sectional_metrics_by_date(
        frame, score_col=score_col, return_col=return_col, top_fractions=(0.10,)
    )
    report: dict[str, Any] = {
        "global": summarize_daily_metrics(global_daily),
        "by_country": {},
    }
    for region in REGIONS:
        group = frame.loc[frame[region_col] == region]
        if group.empty:
            report["by_country"][region] = {"n_rows": 0}
            continue
        daily = cross_sectional_metrics_by_date(
            group,
            score_col=score_col,
            return_col=return_col,
            top_fractions=(0.10,),
            min_names=3,
        )
        report["by_country"][region] = {
            "n_rows": int(len(group)),
            **summarize_daily_metrics(daily),
        }
    return report


def compare_raw_vs_normalized(
    frame: pd.DataFrame,
    *,
    return_col: str = "future_return",
) -> dict[str, Any]:
    """Compare raw probability ranking vs country-normalized ranking (post-hoc)."""
    z_frame = country_normalize_scores(frame, method="zscore")
    p_frame = country_normalize_scores(frame, method="percentile")
    raw = country_ranking_report(frame, score_col="score", return_col=return_col)
    z_rep = country_ranking_report(
        z_frame, score_col="score_country_z", return_col=return_col
    )
    p_rep = country_ranking_report(
        p_frame, score_col="score_country_pct", return_col=return_col
    )
    return {
        "A_raw_probability": raw,
        "B_country_zscore": z_rep,
        "B_country_percentile": p_rep,
    }


def top_bucket_contribution(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
    fraction: float = 0.10,
    date_col: str = "Date",
    region_col: str = "Region",
    sector_col: str = "Sector",
    symbol_col: str = "Symbol",
) -> dict[str, Any]:
    """Composition of daily Top fraction selections (pooled across dates)."""
    base = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, return_col])
    selected_rows: list[pd.DataFrame] = []
    for _, group in base.groupby(pd.to_datetime(base[date_col]), sort=True):
        n = max(1, int(round(len(group) * fraction)))
        selected_rows.append(group.nlargest(n, score_col))
    if not selected_rows:
        return {"n_selected": 0}
    top = pd.concat(selected_rows, ignore_index=True)
    country_counts = Counter(top[region_col].fillna("null").astype(str))
    sector_counts = Counter(top[sector_col].fillna("null").astype(str))
    symbol_counts = Counter(top[symbol_col].astype(str))
    n = len(top)
    return {
        "n_selected": n,
        "country_share": {k: v / n for k, v in sorted(country_counts.items())},
        "country_counts": dict(sorted(country_counts.items())),
        "sector_share": {k: v / n for k, v in sorted(sector_counts.items())},
        "sector_counts": dict(sorted(sector_counts.items())),
        "top_symbols": [
            {"symbol": s, "count": c, "share": c / n}
            for s, c in symbol_counts.most_common(15)
        ],
    }


def leave_one_group_out(
    frame: pd.DataFrame,
    *,
    group_col: str,
    score_col: str = "score",
    return_col: str = "future_return",
) -> dict[str, Any]:
    """Evaluation-only leave-one-group-out (no retrain)."""
    baseline_daily = cross_sectional_metrics_by_date(
        frame, score_col=score_col, return_col=return_col, top_fractions=(0.10,)
    )
    baseline = summarize_daily_metrics(baseline_daily)
    results: dict[str, Any] = {"baseline": baseline, "excluded": {}}
    groups = sorted(
        {str(x) for x in frame[group_col].fillna("null").unique()},
        key=str,
    )
    for g in groups:
        keep = frame.loc[frame[group_col].fillna("null").astype(str) != g]
        if len(keep) < 20:
            results["excluded"][g] = {"error": "insufficient_rows"}
            continue
        daily = cross_sectional_metrics_by_date(
            keep, score_col=score_col, return_col=return_col, top_fractions=(0.10,)
        )
        summary = summarize_daily_metrics(daily)
        base_ic = (baseline.get("ic") or {}).get("mean")
        base_t10 = (baseline.get("top_10pct_excess_return") or {}).get("mean")
        new_ic = (summary.get("ic") or {}).get("mean")
        new_t10 = (summary.get("top_10pct_excess_return") or {}).get("mean")
        results["excluded"][g] = {
            "metrics": summary,
            "delta_mean_ic": None
            if base_ic is None or new_ic is None
            else float(new_ic - base_ic),
            "delta_top10_excess": None
            if base_t10 is None or new_t10 is None
            else float(new_t10 - base_t10),
        }
    return results


def quintile_bucket_analysis(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
    date_col: str = "Date",
) -> dict[str, Any]:
    """Daily score quintiles → mean future return / excess; check monotonicity."""
    base = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, return_col])
    records: list[dict[str, Any]] = []
    for dt, group in base.groupby(pd.to_datetime(base[date_col]), sort=True):
        if len(group) < 10:
            continue
        # qcut may fail on ties; rank then cut.
        ranks = group[score_col].rank(method="first")
        try:
            q = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            continue
        uni_mean = float(group[return_col].mean())
        for qi in range(1, 6):
            sub = group.loc[q == qi, return_col]
            if sub.empty:
                continue
            records.append(
                {
                    "Date": pd.Timestamp(dt),
                    "quintile": qi,
                    "mean_future_return": float(sub.mean()),
                    "excess_return": float(sub.mean() - uni_mean),
                    "n": int(len(sub)),
                }
            )
    if not records:
        return {"n_days": 0, "quintiles": {}}
    qdf = pd.DataFrame(records)
    quintiles: dict[str, Any] = {}
    means = []
    for qi in range(1, 6):
        sub = qdf.loc[qdf["quintile"] == qi]
        m = float(sub["mean_future_return"].mean()) if not sub.empty else None
        e = float(sub["excess_return"].mean()) if not sub.empty else None
        means.append(m)
        quintiles[f"Q{qi}"] = {
            "mean_future_return": m,
            "mean_excess_return": e,
            "n_day_buckets": int(len(sub)),
        }
    mono = None
    if all(m is not None for m in means):
        mono = all(means[i] <= means[i + 1] + 1e-12 for i in range(4))
    # Soft score: fraction of adjacent pairs that increase
    soft = None
    if all(m is not None for m in means):
        soft = float(sum(1 for i in range(4) if means[i] < means[i + 1]) / 4)
    return {
        "n_days": int(qdf["Date"].nunique()),
        "quintiles": quintiles,
        "strictly_nondecreasing": mono,
        "adjacent_increase_ratio": soft,
    }


def top_bottom_spread(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
    date_col: str = "Date",
    fraction: float = 0.20,
) -> dict[str, Any]:
    """Diagnostic Top20% − Bottom20% future return spread (not a strategy)."""
    base = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, return_col])
    spreads: list[float] = []
    for _, group in base.groupby(pd.to_datetime(base[date_col]), sort=True):
        n = max(1, int(round(len(group) * fraction)))
        if len(group) < n * 2:
            continue
        top = float(group.nlargest(n, score_col)[return_col].mean())
        bottom = float(group.nsmallest(n, score_col)[return_col].mean())
        spreads.append(top - bottom)
    if not spreads:
        return {"n_days": 0, "mean_spread": None, "median_spread": None, "positive_ratio": None}
    s = pd.Series(spreads, dtype=float)
    return {
        "n_days": int(len(s)),
        "mean_spread": float(s.mean()),
        "median_spread": float(s.median()),
        "std_spread": float(s.std(ddof=0)),
        "positive_ratio": float((s > 0).mean()),
    }


def ranking_metrics_bundle(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "future_return",
) -> dict[str, Any]:
    """Compact Top10 + IC summary used across diagnostics."""
    daily = cross_sectional_metrics_by_date(
        frame, score_col=score_col, return_col=return_col, top_fractions=(0.10,)
    )
    return summarize_daily_metrics(daily)
