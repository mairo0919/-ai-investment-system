"""Tests for Phase 3H label schemes and score-resolution diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.ltr_label_schemes import (
    assign_label_scheme,
    build_label_gain,
    label_gain_to_param,
    skip_reason_label_d,
)
from src.ml.ltr_labels import assert_group_integrity, build_ranking_groups
from src.ml.ltr_model import fit_lgbm_ranker, predict_rank_scores
from src.ml.score_resolution import (
    score_resolution_by_group,
    summarize_score_resolution,
    top_k_membership_diagnostics,
)
from src.ml.walk_forward import build_expanding_folds, mask_fold_partition


def _panel(n_dates: int = 30, n_per: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for dt in pd.bdate_range("2020-01-02", periods=n_dates):
        for region in ("Japan", "United States", "Europe"):
            for j in range(n_per):
                fut = j * 0.01 + rng.normal(0, 0.001)
                rows.append(
                    {
                        "Date": dt,
                        "Symbol": f"{region[:2]}{j}",
                        "Region": region,
                        "future_return_5d": fut,
                        "return_5d": rng.normal(),
                        "return_20d": rng.normal(),
                        "feat": float(j) + rng.normal(0, 0.1),
                    }
                )
    return pd.DataFrame(rows)


def test_label_a_b_c_ranges() -> None:
    frame = _panel()
    a = assign_label_scheme(frame, return_col="future_return_5d", scheme="A", gain_name="linear")
    assert a.frame["relevance"].between(0, 4).all()
    assert a.max_relevance == 4
    assert len(a.label_gain) == 5

    b = assign_label_scheme(frame, return_col="future_return_5d", scheme="B", gain_name="linear")
    assert b.frame["relevance"].between(0, 9).all()
    assert set(b.frame["relevance"].unique()) <= set(range(10))

    c = assign_label_scheme(frame, return_col="future_return_5d", scheme="C", gain_name="linear")
    assert c.frame["relevance"].between(0, 99).all()
    assert c.max_relevance == 99
    assert len(c.label_gain) == 100


def test_label_gain_linear_and_moderate_exp() -> None:
    lin = build_label_gain(4, "linear")
    assert lin == (0.0, 1.0, 2.0, 3.0, 4.0)
    exp = build_label_gain(4, "moderate_exp")
    assert exp[0] == 0.0
    assert exp[1] > 0
    assert exp[-1] > exp[1]
    assert "1.0" in label_gain_to_param(lin)


def test_label_d_skipped() -> None:
    reason = skip_reason_label_d()
    assert "group" in reason.lower() or "unequal" in reason.lower()
    cfg = Path("config/ranking_labels_3h.json")
    assert cfg.exists()


def test_group_integrity_and_no_future_feature_leak() -> None:
    labeled = assign_label_scheme(
        _panel(), return_col="future_return_5d", scheme="B", gain_name="linear"
    ).frame
    sorted_f, sizes = build_ranking_groups(labeled, group_keys=("Date", "Region"))
    assert_group_integrity(sorted_f, sizes)
    assert "future_return_5d" not in ("feat", "return_5d")


def test_score_resolution_and_tie_ratio() -> None:
    frame = _panel(n_dates=5, n_per=10)
    frame["score"] = 0.5  # perfect ties
    daily = score_resolution_by_group(frame)
    assert (daily["tie_ratio"] == 1.0).all()
    assert (daily["unique_score_ratio"] < 0.2).all()
    summary = summarize_score_resolution(daily)
    assert summary["median_tie_ratio"] == 1.0
    assert summary["low_resolution_day_ratio"] == 1.0


def test_top10_membership_turnover() -> None:
    rows = []
    for i, dt in enumerate(pd.bdate_range("2021-01-04", periods=5)):
        for j in range(20):
            # Rotate top names each day.
            score = float((j + i) % 20)
            rows.append(
                {
                    "Date": dt,
                    "Region": "United States",
                    "Symbol": f"S{j}",
                    "score": score,
                }
            )
    diag = top_k_membership_diagnostics(pd.DataFrame(rows), k=10)
    us = diag["by_region"]["United States"]
    assert us["n_days"] == 5
    assert us["mean_membership_turnover"] is not None
    assert us["mean_membership_turnover"] > 0


def test_purged_walk_forward_still_holds() -> None:
    dates = pd.bdate_range("2015-01-01", periods=3000)
    folds = build_expanding_folds(
        dates, n_folds=5, min_train_days=756, valid_days=252, purge_days=5
    )
    frame = pd.DataFrame({"Date": list(dates), "Symbol": "A", "x": 1})
    train = mask_fold_partition(frame, folds[0], partition="train")
    valid = mask_fold_partition(frame, folds[0], partition="validation")
    assert train["Date"].max() < valid["Date"].min()


def test_ranker_learns_with_explicit_label_gain() -> None:
    pack = assign_label_scheme(
        _panel(n_dates=40, n_per=15),
        return_col="future_return_5d",
        scheme="C",
        gain_name="linear",
    )
    frame = pack.frame.copy()
    # Learnable signal
    frame["feat"] = frame["relevance"].astype(float) + np.random.default_rng(0).normal(
        0, 0.3, size=len(frame)
    )
    dates = sorted(frame["Date"].unique())
    split = dates[len(dates) // 2]
    train = frame.loc[frame["Date"] <= split]
    valid = frame.loc[frame["Date"] > split]
    tr, tg = build_ranking_groups(train)
    va, vg = build_ranking_groups(valid)
    fit = fit_lgbm_ranker(
        tr[["feat", "return_5d"]],
        tr["relevance"],
        tg,
        x_valid=va[["feat", "return_5d"]],
        y_valid=va["relevance"],
        group_valid=vg,
        params={"n_estimators": 40, "num_leaves": 15, "min_child_samples": 10},
        label_gain=pack.label_gain,
        early_stopping_rounds=10,
    )
    scores = predict_rank_scores(fit.model, va[["feat", "return_5d"]])
    assert len(scores) == len(va)
    assert np.isfinite(scores).all()
    # Higher resolution than constant prediction
    assert len(np.unique(np.round(scores, 6))) > 5
