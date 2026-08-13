"""Tests for probability decile / top-probability analysis."""

from __future__ import annotations

import numpy as np

from src.ml.probability_analysis import probability_decile_analysis, top_probability_analysis


def test_probability_decile_and_top_fraction_stats() -> None:
    rng = np.random.default_rng(0)
    prob = np.linspace(0.05, 0.95, 200)
    # Higher probability tends to higher future return.
    future = (prob - 0.5) * 0.1 + rng.normal(0, 0.001, size=200)

    deciles = probability_decile_analysis(prob, future, n_deciles=10)
    assert len(deciles) >= 2
    assert deciles[0]["count"] > 0
    assert "mean_future_return" in deciles[0]

    top = top_probability_analysis(prob, future)
    assert top["overall"]["count"] == 200
    assert top["top_fractions"]["top_10pct"]["count"] == 20
    assert top["top_fractions"]["top_20pct"]["count"] == 40
    assert top["top_fractions"]["top_30pct"]["count"] == 60
    assert (
        top["top_fractions"]["top_10pct"]["mean_future_return"]
        > top["overall"]["mean_future_return"]
    )
