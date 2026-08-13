"""Phase 3G Feature Set A/B/C definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.features.cross_section_features import (
    BREADTH_COLUMNS,
    CS_FEATURE_COLUMNS,
    DISPERSION_COLUMNS,
    RELATIVE_MARKET_COLUMNS,
    RELATIVE_SECTOR_COLUMNS,
    SECTOR_PCT_COLUMNS,
)
from src.features.indicators import FEATURE_COLUMNS
from src.features.macro_features import MacroSeriesSpec, macro_feature_column_names
from src.features.market_features import MARKET_FEATURE_SUFFIXES
from src.ml.walk_forward_runner import GENERIC_MARKET_PREFIX

DEFAULT_FEATURE_SET_CONFIG = Path("config/feature_sets_3g.json")


def base_feature_columns_a() -> tuple[str, ...]:
    """Phase 3F features: stock technical + country market context."""
    mkt = tuple(f"{GENERIC_MARKET_PREFIX}_{s}" for s in MARKET_FEATURE_SUFFIXES)
    return tuple(list(FEATURE_COLUMNS) + list(mkt))


def feature_columns_b() -> tuple[str, ...]:
    return tuple(
        list(base_feature_columns_a())
        + list(CS_FEATURE_COLUMNS)
        + list(SECTOR_PCT_COLUMNS)
        + list(RELATIVE_MARKET_COLUMNS)
        + list(RELATIVE_SECTOR_COLUMNS)
    )


def feature_columns_c(macro_specs: tuple[MacroSeriesSpec, ...]) -> tuple[str, ...]:
    return tuple(
        list(feature_columns_b())
        + list(macro_feature_column_names(macro_specs))
        + list(BREADTH_COLUMNS)
        + list(DISPERSION_COLUMNS)
    )


def load_feature_set_config(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_FEATURE_SET_CONFIG
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_feature_sets(
    macro_specs: tuple[MacroSeriesSpec, ...],
) -> dict[str, tuple[str, ...]]:
    return {
        "A": base_feature_columns_a(),
        "B": feature_columns_b(),
        "C": feature_columns_c(macro_specs),
    }


# Ablation category removals on top of Feature Set C.
ABLATION_CATEGORIES: dict[str, tuple[str, ...]] = {
    "without_cross_sectional": CS_FEATURE_COLUMNS + SECTOR_PCT_COLUMNS,
    "without_relative_strength": RELATIVE_MARKET_COLUMNS + RELATIVE_SECTOR_COLUMNS,
    "without_breadth_dispersion": BREADTH_COLUMNS + DISPERSION_COLUMNS,
}


def ablation_feature_set(
    full_c: tuple[str, ...],
    *,
    drop_columns: tuple[str, ...],
) -> tuple[str, ...]:
    drop = set(drop_columns)
    return tuple(c for c in full_c if c not in drop)


def without_macro_columns(
    full_c: tuple[str, ...],
    macro_specs: tuple[MacroSeriesSpec, ...],
) -> tuple[str, ...]:
    macro_cols = set(macro_feature_column_names(macro_specs))
    return tuple(c for c in full_c if c not in macro_cols)
