"""Phase-1 paper memory opts: result-identical, lower DataFrame copies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config.settings import Settings
from src.ml.ltr_baselines import momentum_scores
from src.paper.model_freeze import score_panel_with_frozen
from src.simulation.engine import SimulationEngine
from src.simulation.fx import FxConverter
from src.simulation.holdout_runner import load_true_holdout_bundle
from src.simulation.ranking import attach_country_ranks
from src.simulation.runner import SimulationRunner


def _legacy_price_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Pre-optimization _price_panel behavior (full panel.copy)."""
    cols = ["Date", "Symbol", "Country", "Currency", "Open", "High", "Low", "Close"]
    frame = panel.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    if "Region" in frame.columns:
        frame["Country"] = frame["Region"]
    return frame.loc[:, cols].drop_duplicates(["Date", "Symbol"])


def _legacy_score_panel_with_frozen(
    *,
    model: Any,
    panel: pd.DataFrame,
    feature_cols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Pre-optimization score_panel_with_frozen (full panel.copy + double hold.copy)."""
    from src.ml.ltr_model import predict_rank_scores

    frame = panel.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    mask = frame["Date"] >= start
    if end is not None:
        mask &= frame["Date"] <= end
    hold = frame.loc[mask].dropna(subset=[c for c in feature_cols if c in frame.columns]).copy()
    if hold.empty:
        return hold
    x = hold.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    hold = hold.copy()
    hold["score"] = predict_rank_scores(model, x)
    return hold


def _legacy_momentum_rankings(
    panel: pd.DataFrame,
    *,
    forward_start: pd.Timestamp,
    asof: pd.Timestamp,
    country: str,
) -> pd.DataFrame:
    mom = panel.copy()
    mom["Date"] = pd.to_datetime(mom["Date"]).dt.normalize()
    region_col = "Region" if "Region" in mom.columns else "Country"
    mom = mom.loc[
        (mom["Date"] >= forward_start)
        & (mom["Date"] <= asof)
        & (mom[region_col] == country)
    ].dropna(subset=["return_20d"]).copy()
    mom["score"] = momentum_scores(mom, feature_col="return_20d").to_numpy()
    return attach_country_ranks(mom, country_col=region_col)


class _SumRanker:
    """Deterministic stand-in: score = sum of feature columns (float64)."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x.to_numpy(dtype=float).sum(axis=1)


def _wide_panel(n_symbols: int = 8, n_days: int = 40) -> pd.DataFrame:
    """Fixed fixture: wide unused columns + Feature-A-like fields."""
    dates = pd.bdate_range("2026-06-01", periods=n_days)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    for i, sym in enumerate([f"S{i}" for i in range(n_symbols)]):
        region = "United States" if i % 2 == 0 else "Japan"
        for d in dates:
            base = float(rng.normal(100 + i, 1.0))
            rows.append(
                {
                    "Date": d,
                    "Symbol": sym,
                    "Region": region,
                    "Country": "US" if region == "United States" else "JP",
                    "Currency": "USD" if region == "United States" else "JPY",
                    "Open": base,
                    "High": base + 1,
                    "Low": base - 1,
                    "Close": base + 0.1,
                    "return_1d": float(rng.normal(0, 0.01)),
                    "return_5d": float(rng.normal(0, 0.02)),
                    "return_20d": float(rng.normal(0, 0.05)),
                    "mkt_return_1d": float(rng.normal(0, 0.01)),
                    # Wide unused ballast (CS/macro-like) to stress copy memory.
                    **{f"ballast_{k}": float(rng.normal()) for k in range(40)},
                }
            )
    return pd.DataFrame(rows)


FEATURE_COLS = ["return_1d", "return_5d", "return_20d", "mkt_return_1d"]


def test_price_panel_matches_legacy_and_uses_less_memory() -> None:
    panel = _wide_panel()
    before_mem = panel.memory_usage(deep=True).sum()
    legacy = _legacy_price_panel(panel)
    settings = Settings()
    runner = SimulationRunner(
        settings,
        Path("config/universe.global100.json"),
        load_true_holdout_bundle(Path("config/true_holdout.json"))[1],
    )
    optimized = runner._price_panel(panel)
    pd.testing.assert_frame_equal(
        optimized.reset_index(drop=True),
        legacy.reset_index(drop=True),
        check_exact=True,
    )
    # Optimized path never materializes a full-width working copy.
    slim_src_cols = ["Date", "Symbol", "Currency", "Open", "High", "Low", "Close", "Region"]
    slim_mem = panel.loc[:, slim_src_cols].memory_usage(deep=True).sum()
    assert slim_mem < before_mem
    assert optimized.memory_usage(deep=True).sum() < before_mem


def test_score_panel_feature_matrix_and_scores_match_legacy() -> None:
    panel = _wide_panel()
    # Mutate-guard: scoring must not alter the caller's panel.
    panel_before = panel.copy(deep=True)
    start = pd.Timestamp("2026-07-01")
    end = pd.Timestamp("2026-07-20")
    model = _SumRanker()

    legacy = _legacy_score_panel_with_frozen(
        model=model, panel=panel, feature_cols=FEATURE_COLS, start=start, end=end
    )
    optimized = score_panel_with_frozen(
        model=model, panel=panel, feature_cols=FEATURE_COLS, start=start, end=end
    )
    pd.testing.assert_frame_equal(panel, panel_before, check_exact=True)

    x_legacy = legacy.loc[:, FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    x_opt = optimized.loc[:, FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    pd.testing.assert_frame_equal(x_opt, x_legacy, check_exact=True)
    np.testing.assert_array_equal(
        optimized["score"].to_numpy(),
        legacy["score"].to_numpy(),
    )
    # Row identity / order for ranking keys
    pd.testing.assert_frame_equal(
        optimized[["Date", "Symbol", "Region"]].reset_index(drop=True),
        legacy[["Date", "Symbol", "Region"]].reset_index(drop=True),
        check_exact=True,
    )
    assert optimized.memory_usage(deep=True).sum() < legacy.memory_usage(deep=True).sum()


def test_rank_percentile_selected_and_orders_match_legacy() -> None:
    panel = _wide_panel()
    start = pd.Timestamp("2026-07-01")
    end = pd.Timestamp("2026-07-20")
    country = "United States"
    model = _SumRanker()

    legacy_scored = _legacy_score_panel_with_frozen(
        model=model, panel=panel, feature_cols=FEATURE_COLS, start=start, end=end
    )
    opt_scored = score_panel_with_frozen(
        model=model, panel=panel, feature_cols=FEATURE_COLS, start=start, end=end
    )
    legacy_ai = attach_country_ranks(legacy_scored, country_col="Region")
    legacy_ai = legacy_ai.loc[legacy_ai["Country"] == country].copy()
    opt_ai = attach_country_ranks(opt_scored, country_col="Region")
    opt_ai = opt_ai.loc[opt_ai["Country"] == country].copy()

    pd.testing.assert_frame_equal(
        opt_ai.reset_index(drop=True),
        legacy_ai.reset_index(drop=True),
        check_exact=True,
    )
    top_pct = 0.25
    legacy_sel = legacy_ai.assign(Selected=legacy_ai["Percentile"] <= top_pct)
    opt_sel = opt_ai.assign(Selected=opt_ai["Percentile"] <= top_pct)
    pd.testing.assert_series_equal(
        opt_sel["Selected"].reset_index(drop=True),
        legacy_sel["Selected"].reset_index(drop=True),
        check_names=False,
    )

    legacy_mom = _legacy_momentum_rankings(
        panel, forward_start=start, asof=end, country=country
    )
    region_col = "Region"
    mom = panel.loc[:, ["Date", "Symbol", region_col, "return_20d"]].copy()
    mom["Date"] = pd.to_datetime(mom["Date"]).dt.normalize()
    mom = mom.loc[
        (mom["Date"] >= start) & (mom["Date"] <= end) & (mom[region_col] == country)
    ].dropna(subset=["return_20d"]).copy()
    mom["score"] = momentum_scores(mom, feature_col="return_20d").to_numpy()
    opt_mom = attach_country_ranks(mom, country_col=region_col)
    pd.testing.assert_frame_equal(
        opt_mom.reset_index(drop=True),
        legacy_mom.reset_index(drop=True),
        check_exact=True,
    )

    # Order / portfolio state: same rankings + prices => identical engine result.
    _, cfg = load_true_holdout_bundle(Path("config/true_holdout.json"))
    idx = pd.DatetimeIndex(pd.bdate_range("2026-05-01", periods=80))
    fx = FxConverter({"USDJPY": pd.Series(150.0, index=idx)})
    settings = Settings()
    sim = SimulationRunner(settings, Path("config/universe.global100.json"), cfg)
    prices = sim._price_panel(panel)
    prices = prices.loc[prices["Country"] == country].copy()

    def _run(rankings: pd.DataFrame):
        engine = SimulationEngine(cfg, fx=fx, countries={country})
        return engine.run(
            prices=prices,
            rankings=rankings,
            start=start,
            end=end,
            min_calendar_days=1,
        )

    legacy_res = _run(legacy_ai)
    opt_res = _run(opt_ai)
    assert legacy_res.portfolio is not None and opt_res.portfolio is not None
    assert legacy_res.portfolio.cash == opt_res.portfolio.cash
    assert set(legacy_res.portfolio.positions) == set(opt_res.portfolio.positions)
    for sym, pos in legacy_res.portfolio.positions.items():
        other = opt_res.portfolio.positions[sym]
        assert pos.quantity == other.quantity
        assert pos.entry_price == other.entry_price
        assert pd.Timestamp(pos.entry_date) == pd.Timestamp(other.entry_date)

    legacy_pending = list(legacy_res.meta.get("pending_orders") or [])
    opt_pending = list(opt_res.meta.get("pending_orders") or [])
    assert len(legacy_pending) == len(opt_pending)
    for a, b in zip(legacy_pending, opt_pending, strict=True):
        assert a.symbol == b.symbol
        assert a.side == b.side
        assert a.quantity == b.quantity
        assert pd.Timestamp(a.signal_date) == pd.Timestamp(b.signal_date)

    legacy_eq = [(ep.date, ep.total_equity, ep.cash) for ep in legacy_res.equity_curve]
    opt_eq = [(ep.date, ep.total_equity, ep.cash) for ep in opt_res.equity_curve]
    assert legacy_eq == opt_eq


def test_score_panel_memory_usage_diagnostic() -> None:
    """Document deep memory of full vs slim score working sets."""
    panel = _wide_panel(n_symbols=12, n_days=60)
    full_copy_mb = panel.copy().memory_usage(deep=True).sum() / (1024 * 1024)
    keep = ["Date", "Symbol", "Region", *FEATURE_COLS]
    slim_mb = panel.loc[:, keep].copy().memory_usage(deep=True).sum() / (1024 * 1024)
    assert slim_mb < full_copy_mb
    # Ballast columns make slim clearly smaller.
    assert slim_mb / full_copy_mb < 0.5
