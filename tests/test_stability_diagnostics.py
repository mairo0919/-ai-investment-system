"""Tests for Phase 3E ranking stability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.stability_diagnostics import (
    assign_volatility_regime,
    classify_market_regime,
    compare_raw_vs_normalized,
    country_normalize_scores,
    country_ranking_report,
    daily_score_dispersion,
    leave_one_group_out,
    one_day_ic_diagnostics,
    quintile_bucket_analysis,
    score_distribution_summary,
    top_bottom_spread,
    top_bucket_contribution,
)


def _synthetic_panel(n_dates: int = 30, n_per_region: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    regions = ["Japan", "United States", "Europe"]
    sectors = ["Technology", "Healthcare", "Financial Services", "Industrials"]
    for i, dt in enumerate(pd.bdate_range("2022-01-03", periods=n_dates)):
        for region in regions:
            for j in range(n_per_region):
                # Region-specific score bias + signal
                base = {"Japan": 0.4, "United States": 0.55, "Europe": 0.48}[region]
                signal = (j / n_per_region) * 0.2
                score = base + signal + rng.normal(0, 0.01)
                fut = signal * 0.05 + rng.normal(0, 0.01)
                rows.append(
                    {
                        "Date": dt,
                        "Symbol": f"{region[:2]}{j}",
                        "Region": region,
                        "Sector": sectors[j % len(sectors)],
                        "score": score,
                        "future_return": fut,
                        "mkt_sma_60_ratio": 0.02 if i % 3 else -0.02,
                        "mkt_return_20d": 0.01 if i % 3 == 0 else (-0.01 if i % 3 == 1 else 0.0),
                        "mkt_volatility_20": 0.1 + 0.05 * (i % 5),
                    }
                )
    return pd.DataFrame(rows)


def test_country_normalize_zscore_and_percentile() -> None:
    frame = _synthetic_panel(n_dates=5, n_per_region=10)
    z = country_normalize_scores(frame, method="zscore")
    assert "score_country_z" in z.columns
    # Within a date×region, mean z ≈ 0
    g = z.groupby(["Date", "Region"])["score_country_z"].mean().abs()
    assert (g < 1e-8).all()
    p = country_normalize_scores(frame, method="percentile")
    assert p["score_country_pct"].between(0, 1).all()


def test_country_internal_ranking() -> None:
    frame = _synthetic_panel()
    report = country_ranking_report(frame)
    assert "global" in report
    assert "Japan" in report["by_country"]
    assert report["by_country"]["Japan"]["ic"]["mean"] is not None


def test_raw_vs_normalized_comparison() -> None:
    frame = _synthetic_panel()
    cmp = compare_raw_vs_normalized(frame)
    assert "A_raw_probability" in cmp
    assert "B_country_zscore" in cmp
    assert "B_country_percentile" in cmp


def test_score_dispersion_and_tie_ratio() -> None:
    # Constant scores → high tie / low dispersion flag
    dates = pd.bdate_range("2022-01-03", periods=3)
    rows = []
    for dt in dates:
        for i in range(10):
            rows.append(
                {
                    "Date": dt,
                    "score": 0.5,
                    "future_return": float(i) * 0.01,
                    "Symbol": f"S{i}",
                    "Region": "US",
                }
            )
    daily = daily_score_dispersion(pd.DataFrame(rows))
    assert daily["tie_ratio"].iloc[0] == 1.0
    assert bool(daily["low_dispersion_flag"].iloc[0]) is True


def test_score_distribution_summary() -> None:
    s = score_distribution_summary(pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]}))
    assert s["n"] == 4
    assert s["p50"] == pytest.approx(0.25)
    assert s["min"] == 0.1


def test_one_day_ic_diagnostics() -> None:
    frame = _synthetic_panel(n_dates=15)
    diag = one_day_ic_diagnostics(frame)
    assert "artifact_flags" in diag
    assert diag["n_rows"] > 0
    assert "score_distribution" in diag


def test_market_regime_classification() -> None:
    sma = pd.Series([0.1, -0.1, 0.1, -0.05])
    ret = pd.Series([0.02, -0.02, -0.01, 0.01])
    regime = classify_market_regime(sma, ret)
    assert list(regime) == ["Bull", "Bear", "Neutral", "Neutral"]


def test_volatility_regime_thresholds_from_train_only() -> None:
    train = pd.Series(np.linspace(0.05, 0.30, 300))
    valid = pd.Series([0.06, 0.15, 0.28])
    labels, thr = assign_volatility_regime(train, valid)
    assert thr["q33"] < thr["q66"]
    assert list(labels) == ["Low", "Medium", "High"]


def test_quintile_and_spread() -> None:
    frame = _synthetic_panel(n_dates=20, n_per_region=25)
    q = quintile_bucket_analysis(frame)
    assert set(q["quintiles"]) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert q["adjacent_increase_ratio"] is not None
    spread = top_bottom_spread(frame, fraction=0.20)
    assert spread["n_days"] > 0
    assert spread["mean_spread"] is not None


def test_top_contribution_and_leave_one_out() -> None:
    frame = _synthetic_panel()
    contrib = top_bucket_contribution(frame, fraction=0.10)
    assert contrib["n_selected"] > 0
    assert "United States" in contrib["country_share"] or "Japan" in contrib["country_share"]
    assert len(contrib["top_symbols"]) > 0
    loco = leave_one_group_out(frame, group_col="Region")
    assert "baseline" in loco
    assert "Japan" in loco["excluded"]
    assert "delta_mean_ic" in loco["excluded"]["Japan"]
    loso = leave_one_group_out(frame, group_col="Sector")
    assert "Technology" in loso["excluded"]
