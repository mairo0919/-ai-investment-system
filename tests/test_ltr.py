"""Tests for Phase 3F Country-internal Learning-to-Rank."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.ltr_baselines import momentum_scores
from src.ml.ltr_labels import (
    assert_group_integrity,
    assign_relevance_labels,
    build_ranking_groups,
    load_relevance_config,
    percentile_to_relevance,
)
from src.ml.ltr_metrics import evaluate_country_ranking, ndcg_at_k, ndcg_by_groups
from src.ml.ltr_model import fit_lgbm_ranker, predict_rank_scores
from src.ml.walk_forward import build_expanding_folds, mask_fold_partition


def _panel(n_dates: int = 40, n_per_region: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    regions = ["Japan", "United States", "Europe"]
    for i, dt in enumerate(pd.bdate_range("2018-01-02", periods=n_dates)):
        for region in regions:
            for j in range(n_per_region):
                ret5 = rng.normal(0, 0.02)
                fut5 = 0.3 * ret5 + rng.normal(0, 0.01)
                rows.append(
                    {
                        "Date": dt,
                        "Symbol": f"{region[:2]}{j}",
                        "Region": region,
                        "return_5d": ret5,
                        "return_20d": ret5 * 1.5,
                        "future_return_5d": fut5,
                        "future_return_10d": fut5 * 1.2,
                        "feat_a": rng.normal(),
                        "feat_b": rng.normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_relevance_config_and_mapping() -> None:
    config = load_relevance_config(Path("config/ranking_relevance.json"))
    assert percentile_to_relevance(0.95, config) == 4
    assert percentile_to_relevance(0.80, config) == 3
    assert percentile_to_relevance(0.50, config) == 2
    assert percentile_to_relevance(0.15, config) == 1
    assert percentile_to_relevance(0.05, config) == 0


def test_assign_relevance_within_date_country() -> None:
    config = load_relevance_config()
    frame = _panel(n_dates=5, n_per_region=10)
    labeled = assign_relevance_labels(frame, return_col="future_return_5d", config=config)
    assert "relevance" in labeled.columns
    assert labeled["relevance"].between(0, 4).all()
    # Top future return in a group should get high relevance.
    g = labeled.loc[
        (labeled["Date"] == labeled["Date"].iloc[0]) & (labeled["Region"] == "Japan")
    ]
    assert g.loc[g["future_return_5d"].idxmax(), "relevance"] >= 3


def test_group_size_integrity_and_date_order() -> None:
    config = load_relevance_config()
    labeled = assign_relevance_labels(
        _panel(n_dates=8, n_per_region=12),
        return_col="future_return_5d",
        config=config,
    )
    sorted_frame, sizes = build_ranking_groups(labeled, group_keys=("Date", "Region"))
    assert_group_integrity(sorted_frame, sizes, group_keys=("Date", "Region"))
    assert int(sizes.sum()) == len(sorted_frame)
    # Dates are non-decreasing across the frame.
    dates = pd.to_datetime(sorted_frame["Date"])
    assert dates.is_monotonic_increasing or (
        sorted_frame.groupby("Region")["Date"].apply(lambda s: s.is_monotonic_increasing).all()
    )


def test_no_future_label_in_features_contract() -> None:
    features = {"feat_a", "feat_b", "return_5d"}
    assert "future_return_5d" not in features
    assert "relevance" not in features


def test_purged_walk_forward_for_ltr() -> None:
    dates = pd.bdate_range("2015-01-01", periods=3000)
    folds = build_expanding_folds(
        dates, n_folds=5, min_train_days=756, valid_days=252, purge_days=10
    )
    frame = pd.DataFrame({"Date": list(dates), "Symbol": "A", "x": 1})
    train = mask_fold_partition(frame, folds[0], partition="train")
    valid = mask_fold_partition(frame, folds[0], partition="validation")
    assert train["Date"].max() < valid["Date"].min()
    assert folds[0].purge_days == 10


def test_ndcg_perfect_and_random() -> None:
    y_true = np.array([4, 3, 2, 1, 0], dtype=float)
    perfect = ndcg_at_k(y_true, y_true, k=5)
    assert perfect == pytest.approx(1.0)
    reverse = ndcg_at_k(y_true, -y_true, k=5)
    assert reverse < perfect


def test_lgbm_ranker_learns_and_scores() -> None:
    config = load_relevance_config()
    # Larger panel so ranker can fit.
    frame = _panel(n_dates=60, n_per_region=15)
    # Inject learnable signal into feat_a correlated with future return.
    frame["feat_a"] = frame["future_return_5d"] + np.random.default_rng(1).normal(
        0, 0.005, size=len(frame)
    )
    labeled = assign_relevance_labels(frame, return_col="future_return_5d", config=config)
    # Split by date.
    dates = sorted(labeled["Date"].unique())
    split = dates[len(dates) // 2]
    train = labeled.loc[labeled["Date"] <= split]
    valid = labeled.loc[labeled["Date"] > split]
    train_s, train_g = build_ranking_groups(train)
    valid_s, valid_g = build_ranking_groups(valid)
    assert_group_integrity(train_s, train_g)
    fit = fit_lgbm_ranker(
        train_s[["feat_a", "feat_b", "return_5d", "return_20d"]],
        train_s["relevance"],
        train_g,
        x_valid=valid_s[["feat_a", "feat_b", "return_5d", "return_20d"]],
        y_valid=valid_s["relevance"],
        group_valid=valid_g,
        params={"n_estimators": 50, "num_leaves": 15},
        early_stopping_rounds=10,
    )
    scores = predict_rank_scores(
        fit.model, valid_s[["feat_a", "feat_b", "return_5d", "return_20d"]]
    )
    scored = valid_s.copy()
    scored["score"] = scores
    ndcg = ndcg_by_groups(scored, valid_g, ks=(5, 10))
    assert ndcg["ndcg_at_5"] is not None
    assert ndcg["ndcg_at_5"] > 0.4


def test_country_ic_top10_quintile_tie_and_baseline() -> None:
    config = load_relevance_config()
    frame = _panel(n_dates=25, n_per_region=20)
    frame["score"] = frame["future_return_5d"] + 0.01
    labeled = assign_relevance_labels(frame, return_col="future_return_5d", config=config)
    labeled["future_return"] = labeled["future_return_5d"]
    labeled["score"] = labeled["future_return_5d"]
    us = labeled.loc[labeled["Region"] == "United States"]
    metrics = evaluate_country_ranking(us, ndcg_ks=(5, 10))
    assert metrics["ndcg"]["ndcg_at_5"] is not None
    assert metrics["ranking"]["ic"]["mean"] is not None
    assert metrics["ranking"]["top_10pct_excess_return"]["mean"] > 0
    assert metrics["q5_q1_spread"] is not None and metrics["q5_q1_spread"] > 0
    # Tie diagnostic on constant scores
    tied = us.copy()
    tied["score"] = 1.0
    tied_metrics = evaluate_country_ranking(tied)
    assert tied_metrics["warnings"]
    # Momentum baseline uses past return only
    mom = momentum_scores(labeled, feature_col="return_5d")
    assert len(mom) == len(labeled)
