"""Paper Feature Set A lean panel: skip unused CS/macro; results identical."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.cross_section_features import (
    BREADTH_COLUMNS,
    CS_FEATURE_COLUMNS,
    DISPERSION_COLUMNS,
    RELATIVE_MARKET_COLUMNS,
    RELATIVE_SECTOR_COLUMNS,
    SECTOR_PCT_COLUMNS,
    add_cross_section_feature_block,
)
from src.features.feature_sets import base_feature_columns_a
from src.features.indicators import FEATURE_COLUMNS
from src.features.macro_features import (
    attach_mkt_return_60d,
    compute_mkt_return_60d,
    join_macro_and_currency_features,
    load_macro_config,
    macro_feature_column_names,
)
from src.features.market_features import MARKET_FEATURE_SUFFIXES
from src.ml.ltr_baselines import momentum_scores
from src.ml.walk_forward_runner import GENERIC_MARKET_PREFIX
from src.paper.lineage import LOCKED_PAPER_MODEL_ID
from src.paper.model_freeze import score_panel_with_frozen
from src.simulation.engine import SimulationEngine
from src.simulation.fx import FxConverter
from src.simulation.holdout_runner import load_true_holdout_bundle
from src.simulation.ranking import attach_country_ranks
from src.simulation.runner import SimulationRunner
from src.config.settings import Settings


META_PATH = Path("models/paper_frozen/model_meta.json")

# Paper Trading downstream columns (code-traced; not guesses).
PAPER_PRICE_COLS = ("Date", "Symbol", "Country", "Currency", "Open", "High", "Low", "Close")
PAPER_MOMENTUM_COLS = ("Date", "Symbol", "Region", "return_20d")
PAPER_RANK_KEYS = ("Date", "Symbol", "Region")


def _load_locked_feature_list() -> list[str]:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert meta["model_id"] == LOCKED_PAPER_MODEL_ID
    assert meta["feature_set"] == "A"
    return list(meta["feature_list"])


def test_locked_feature_list_matches_feature_set_a_exactly() -> None:
    feature_list = _load_locked_feature_list()
    expected = list(base_feature_columns_a())
    assert feature_list == expected
    assert feature_list == [
        *FEATURE_COLUMNS,
        *[f"{GENERIC_MARKET_PREFIX}_{s}" for s in MARKET_FEATURE_SUFFIXES],
    ]


def test_cs_macro_mkt60_not_in_locked_feature_list() -> None:
    feature_list = set(_load_locked_feature_list())
    forbidden = set(CS_FEATURE_COLUMNS) | set(SECTOR_PCT_COLUMNS) | set(
        RELATIVE_MARKET_COLUMNS
    ) | set(RELATIVE_SECTOR_COLUMNS) | set(BREADTH_COLUMNS) | set(DISPERSION_COLUMNS)
    forbidden.add("mkt_return_60d")
    specs = load_macro_config()
    forbidden |= set(macro_feature_column_names(specs))
    overlap = feature_list & forbidden
    assert not overlap, f"Locked model unexpectedly requires enrichment cols: {overlap}"


def test_paper_downstream_columns_covered_without_cs_macro() -> None:
    """signal / price / orders need only panel_build outputs + feature_list."""
    feature_list = _load_locked_feature_list()
    # Columns produced by WalkForwardRunner._build_panel (stock feats + mkt join + meta).
    panel_build_cols = set(FEATURE_COLUMNS) | {
        f"{GENERIC_MARKET_PREFIX}_{s}" for s in MARKET_FEATURE_SUFFIXES
    } | {
        "Date",
        "Symbol",
        "Country",
        "Market",
        "Currency",
        "Sector",
        "Industry",
        "Region",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "future_return_1d",
        "future_return_5d",
        "future_return_10d",
    }
    for col in feature_list:
        assert col in panel_build_cols
    for col in PAPER_PRICE_COLS:
        # Country may be filled from Region in _price_panel; Region must exist.
        if col == "Country":
            assert "Region" in panel_build_cols or "Country" in panel_build_cols
        else:
            assert col in panel_build_cols
    for col in PAPER_MOMENTUM_COLS:
        assert col in panel_build_cols
    for col in PAPER_RANK_KEYS:
        assert col in panel_build_cols


class _SumRanker:
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x.to_numpy(dtype=float).sum(axis=1)


def _base_panel_like_build(n_symbols: int = 6, n_days: int = 50) -> pd.DataFrame:
    """Synthetic panel with Feature Set A columns (post panel_build shape)."""
    dates = pd.bdate_range("2026-05-01", periods=n_days)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(7)
    feat_a = list(base_feature_columns_a())
    for i in range(n_symbols):
        sym = f"T{i}"
        region = "United States" if i % 2 == 0 else "Japan"
        for d in dates:
            row: dict[str, Any] = {
                "Date": d,
                "Symbol": sym,
                "Region": region,
                "Country": "US" if region == "United States" else "JP",
                "Market": "US" if region == "United States" else "JP",
                "Currency": "USD" if region == "United States" else "JPY",
                "Sector": "Tech",
                "Industry": "Software",
                "Open": 100.0 + i,
                "High": 101.0 + i,
                "Low": 99.0 + i,
                "Close": 100.5 + i,
                "Adj Close": 100.5 + i,
                "Volume": 1_000_000,
            }
            for c in feat_a:
                row[c] = float(rng.normal(0, 1))
            rows.append(row)
    return pd.DataFrame(rows)


def _apply_full_enrichment(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply the same enrichment blocks paper lean skips (fixture-local)."""
    # Synthetic per-region mkt 60d series for attach.
    mkt_60: dict[str, pd.DataFrame] = {}
    for region, g in panel.groupby("Region", sort=False):
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(g["Date"]).unique()))
        mkt_60[str(region)] = pd.DataFrame(
            {"Date": dates, "mkt_return_60d": np.linspace(-0.1, 0.1, len(dates))}
        )
    out = attach_mkt_return_60d(panel, mkt_60)
    out = add_cross_section_feature_block(out, min_sector_group_size=2)
    # Minimal macro frame: one global series so join path runs.
    specs = load_macro_config()
    global_specs = [s for s in specs if s.attach == "global"]
    assert global_specs
    spec = global_specs[0]
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["Date"]).unique()))
    macro_cols = {
        "Date": dates,
        f"macro_{spec.key}_return_1d": np.zeros(len(dates)),
        f"macro_{spec.key}_return_5d": np.zeros(len(dates)),
        f"macro_{spec.key}_return_20d": np.zeros(len(dates)),
        f"macro_{spec.key}_sma_20_ratio": np.ones(len(dates)),
        f"macro_{spec.key}_volatility_20": np.full(len(dates), 0.1),
    }
    if spec.style == "level":
        macro_cols = {
            "Date": dates,
            f"macro_{spec.key}_level": np.ones(len(dates)),
            f"macro_{spec.key}_change_5d": np.zeros(len(dates)),
            f"macro_{spec.key}_sma_20_ratio": np.ones(len(dates)),
            f"macro_{spec.key}_return_1d": np.zeros(len(dates)),
            f"macro_{spec.key}_return_5d": np.zeros(len(dates)),
        }
    feats = pd.DataFrame(macro_cols)
    frames = {spec.key: (spec, feats)}
    out, _ = join_macro_and_currency_features(out, frames)
    return out


def _aligned_feature_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    # Enrichment joins may cast Date to datetime64[ns]; normalize unit for exact compare.
    out["Date"] = pd.to_datetime(out["Date"]).astype("datetime64[ns]").dt.normalize()
    out = out.sort_values(["Date", "Symbol"], kind="mergesort").reset_index(drop=True)
    return out.loc[:, ["Date", "Symbol", *feature_cols]]


def test_lean_vs_full_feature_matrix_scores_ranks_orders_exact() -> None:
    feature_cols = _load_locked_feature_list()
    lean = _base_panel_like_build()
    full = _apply_full_enrichment(lean)

    # Memory / shape diagnostic
    lean_mb = lean.memory_usage(deep=True).sum() / (1024 * 1024)
    full_mb = full.memory_usage(deep=True).sum() / (1024 * 1024)
    assert full.shape[1] > lean.shape[1]
    assert full_mb > lean_mb
    print(
        f"[paper_lean_mem] lean_shape={lean.shape} lean_MB={lean_mb:.3f} "
        f"full_shape={full.shape} full_MB={full_mb:.3f}"
    )

    x_lean = _aligned_feature_matrix(lean, feature_cols)
    x_full = _aligned_feature_matrix(full, feature_cols)
    pd.testing.assert_frame_equal(x_lean, x_full, check_exact=True)

    model = _SumRanker()
    start = pd.Timestamp("2026-06-01")
    end = pd.Timestamp("2026-06-20")
    country = "United States"

    scored_lean = score_panel_with_frozen(
        model=model, panel=lean, feature_cols=feature_cols, start=start, end=end
    )
    scored_full = score_panel_with_frozen(
        model=model, panel=full, feature_cols=feature_cols, start=start, end=end
    )
    # Compare on keys (enrichment may reorder rows).
    def _score_table(scored: pd.DataFrame) -> pd.DataFrame:
        t = scored[["Date", "Symbol", "score", *feature_cols]].copy()
        t["Date"] = pd.to_datetime(t["Date"]).astype("datetime64[ns]").dt.normalize()
        return t.sort_values(["Date", "Symbol"], kind="mergesort").reset_index(drop=True)

    pd.testing.assert_frame_equal(_score_table(scored_lean), _score_table(scored_full), check_exact=True)

    ai_lean = attach_country_ranks(scored_lean, country_col="Region")
    ai_full = attach_country_ranks(scored_full, country_col="Region")

    def _norm_rank(df: pd.DataFrame) -> pd.DataFrame:
        out = df.loc[df["Country"] == country].copy()
        out["Date"] = pd.to_datetime(out["Date"]).astype("datetime64[ns]").dt.normalize()
        return out.sort_values(["Date", "Symbol"], kind="mergesort").reset_index(drop=True)

    ai_lean = _norm_rank(ai_lean)
    ai_full = _norm_rank(ai_full)
    pd.testing.assert_frame_equal(ai_lean, ai_full, check_exact=True)

    top_pct = 0.25
    pd.testing.assert_series_equal(
        (ai_lean["Percentile"] <= top_pct).reset_index(drop=True),
        (ai_full["Percentile"] <= top_pct).reset_index(drop=True),
        check_names=False,
    )

    # Momentum signal (uses return_20d only)
    def _mom(panel: pd.DataFrame) -> pd.DataFrame:
        region_col = "Region"
        mom = panel.loc[:, ["Date", "Symbol", region_col, "return_20d"]].copy()
        mom["Date"] = pd.to_datetime(mom["Date"]).dt.normalize()
        mom = mom.loc[
            (mom["Date"] >= start) & (mom["Date"] <= end) & (mom[region_col] == country)
        ].dropna(subset=["return_20d"]).copy()
        mom["score"] = momentum_scores(mom, feature_col="return_20d").to_numpy()
        return (
            attach_country_ranks(mom, country_col=region_col)
            .assign(Date=lambda d: pd.to_datetime(d["Date"]).astype("datetime64[ns]").dt.normalize())
            .sort_values(["Date", "Symbol"], kind="mergesort")
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(_mom(lean), _mom(full), check_exact=True)

    _, cfg = load_true_holdout_bundle(Path("config/true_holdout.json"))
    idx = pd.DatetimeIndex(pd.bdate_range("2026-04-01", periods=90))
    fx = FxConverter({"USDJPY": pd.Series(150.0, index=idx), "EURJPY": pd.Series(160.0, index=idx)})
    sim = SimulationRunner(Settings(), Path("config/universe.global100.json"), cfg)
    prices_lean = sim._price_panel(lean)
    prices_full = sim._price_panel(full)

    def _norm_px(df: pd.DataFrame) -> pd.DataFrame:
        out = df.loc[df["Country"] == country].copy()
        out["Date"] = pd.to_datetime(out["Date"]).astype("datetime64[ns]").dt.normalize()
        return out.sort_values(["Date", "Symbol"], kind="mergesort").reset_index(drop=True)

    prices_lean = _norm_px(prices_lean)
    prices_full = _norm_px(prices_full)
    pd.testing.assert_frame_equal(prices_lean, prices_full, check_exact=True)

    def _engine(rankings: pd.DataFrame, prices: pd.DataFrame):
        return SimulationEngine(cfg, fx=fx, countries={country}).run(
            prices=prices,
            rankings=rankings,
            start=start,
            end=end,
            min_calendar_days=1,
        )

    res_l = _engine(ai_lean, prices_lean)
    res_f = _engine(ai_full, prices_full)
    assert res_l.portfolio is not None and res_f.portfolio is not None
    assert res_l.portfolio.cash == res_f.portfolio.cash
    assert set(res_l.portfolio.positions) == set(res_f.portfolio.positions)
    for sym, pos in res_l.portfolio.positions.items():
        other = res_f.portfolio.positions[sym]
        assert pos.quantity == other.quantity
        assert pos.entry_price == other.entry_price
        assert pd.Timestamp(pos.entry_date) == pd.Timestamp(other.entry_date)
    pending_l = list(res_l.meta.get("pending_orders") or [])
    pending_f = list(res_f.meta.get("pending_orders") or [])
    assert len(pending_l) == len(pending_f)
    for a, b in zip(pending_l, pending_f, strict=True):
        assert (a.symbol, a.side, a.quantity, pd.Timestamp(a.signal_date)) == (
            b.symbol,
            b.side,
            b.quantity,
            pd.Timestamp(b.signal_date),
        )
    assert [(ep.date, ep.total_equity, ep.cash) for ep in res_l.equity_curve] == [
        (ep.date, ep.total_equity, ep.cash) for ep in res_f.equity_curve
    ]


def test_paper_runner_uses_lean_builder_for_feature_set_a() -> None:
    source = Path("src/paper/runner.py").read_text(encoding="utf-8")
    assert "_build_paper_feature_a_panel" in source
    assert 'str(self.cfg.feature_set) == "A"' in source


@pytest.mark.skipif(
    not Path("data/raw").exists() or len(list(Path("data/raw").glob("*.csv"))) < 50,
    reason="local OHLCV cache required for live panel memory probe",
)
def test_live_cache_lean_panel_memory_vs_full_enrichment_steps() -> None:
    """If cache present: measure panel_build vs CS/macro-expanded deep memory."""
    from src.config.settings import get_settings
    from src.data.universe import load_universe, region_of
    from src.data.cache import OhlcvCache
    from src.features.indicators import add_features
    from src.features.market_features import join_market_features, build_market_features
    from src.ml.walk_forward_runner import REGION_BENCHMARKS
    from src.ml.dataset import add_future_returns

    settings = get_settings()
    universe = load_universe(Path("config/universe.global100.json"))
    cache = OhlcvCache(settings.raw_data_dir)
    meta_by = {i.symbol: i for i in universe.instruments}
    market_frames = {}
    mkt_ohlcv = {}
    for region, (ticker, prefix) in REGION_BENCHMARKS.items():
        raw = cache.load(ticker)
        if raw is None:
            pytest.skip(f"missing benchmark cache {ticker}")
        mkt_ohlcv[region] = raw
        market_frames[region] = build_market_features(raw, prefix=prefix)

    parts = []
    for inst in universe.instruments:
        raw = cache.load(inst.symbol)
        if raw is None or len(raw) < 60:
            continue
        featured = add_features(raw).dropna(subset=list(FEATURE_COLUMNS)).copy()
        featured["Symbol"] = inst.symbol
        featured["Country"] = inst.country
        featured["Market"] = inst.market
        featured["Currency"] = inst.currency
        featured["Sector"] = inst.sector
        featured["Industry"] = inst.industry
        featured["Region"] = region_of(inst.country)
        region = featured["Region"].iloc[0]
        if region not in market_frames:
            continue
        prefix = REGION_BENCHMARKS[region][1]
        renamed = market_frames[region].rename(
            columns={
                f"{prefix}_{s}": f"{GENERIC_MARKET_PREFIX}_{s}"
                for s in MARKET_FEATURE_SUFFIXES
            }
        )
        parts.append(join_market_features(featured, renamed))
    panel_build = add_future_returns(pd.concat(parts, ignore_index=True), horizons=(1, 5, 10))
    after_mkt60 = attach_mkt_return_60d(panel_build, compute_mkt_return_60d(mkt_ohlcv))
    after_cs = add_cross_section_feature_block(after_mkt60)

    def _mb(df: pd.DataFrame) -> float:
        return df.memory_usage(deep=True).sum() / (1024 * 1024)

    print(
        f"[live_panel_mem] panel_build={panel_build.shape} MB={_mb(panel_build):.1f} ; "
        f"after_cs={after_cs.shape} MB={_mb(after_cs):.1f}"
    )
    assert _mb(panel_build) < _mb(after_cs)
    # Feature A columns identical after CS attach (no mutation of base feats).
    feat = list(base_feature_columns_a())
    left = _aligned_feature_matrix(panel_build, feat)
    right = _aligned_feature_matrix(after_cs, feat)
    pd.testing.assert_frame_equal(left, right, check_exact=True)
