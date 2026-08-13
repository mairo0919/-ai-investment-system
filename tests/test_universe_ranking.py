"""Tests for universe loading and ranking evaluation foundation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.universe import load_universe
from src.ml.ranking import evaluate_ranking, information_coefficient, top_k_average_return


def test_load_sample_universe() -> None:
    path = Path("config/universe.sample.json")
    universe = load_universe(path)
    assert universe.name == "pilot_global_8"
    assert len(universe.symbols) == 8
    assert "7203.T" in universe.symbols
    assert "AAPL" in universe.symbols
    assert "SAP.DE" in universe.symbols
    jp = universe.by_country("JP")
    assert len(jp) == 2


def test_ranking_metrics_detect_positive_signal() -> None:
    rng = np.random.default_rng(0)
    score = np.linspace(-1, 1, 100)
    future = score * 0.05 + rng.normal(0, 0.001, size=100)
    frame = pd.DataFrame(
        {
            "score": score,
            "future_return": future,
            "country": ["US"] * 50 + ["JP"] * 50,
        }
    )
    ic = information_coefficient(frame)
    assert ic > 0.8
    top = top_k_average_return(frame, fraction=0.1)
    assert top["excess_return"] > 0
    report = evaluate_ranking(frame, group_col="country")
    assert report["scope"] == "global"
    assert "US" in report["by_group"]["groups"]
    assert "JP" in report["by_group"]["groups"]


def test_ranking_requires_columns() -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        information_coefficient(pd.DataFrame({"a": [1.0]}))
