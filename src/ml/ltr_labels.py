"""Learning-to-Rank relevance labels and Date×Country groups."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import TrainingError

logger = logging.getLogger(__name__)

DEFAULT_RELEVANCE_CONFIG = Path("config/ranking_relevance.json")


@dataclass(frozen=True)
class RelevanceBucket:
    relevance: int
    min_percentile: float
    max_percentile: float
    label: str


@dataclass(frozen=True)
class RelevanceConfig:
    name: str
    group_keys: tuple[str, ...]
    buckets: tuple[RelevanceBucket, ...]
    min_group_size: int
    ndcg_ks: tuple[int, ...]
    tie_warning_ratio: float
    description: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RelevanceConfig:
        buckets = tuple(
            RelevanceBucket(
                relevance=int(item["relevance"]),
                min_percentile=float(item["min_percentile"]),
                max_percentile=float(item["max_percentile"]),
                label=str(item.get("label", item["relevance"])),
            )
            for item in payload["relevance_levels"]
        )
        # Highest percentile first for mapping.
        buckets = tuple(sorted(buckets, key=lambda b: b.min_percentile, reverse=True))
        return cls(
            name=str(payload.get("name", "relevance")),
            group_keys=tuple(payload.get("group_keys", ["Date", "Region"])),
            buckets=buckets,
            min_group_size=int(payload.get("min_group_size", 5)),
            ndcg_ks=tuple(int(k) for k in payload.get("ndcg_ks", [5, 10])),
            tie_warning_ratio=float(payload.get("tie_warning_ratio", 0.5)),
            description=payload.get("description"),
        )


def load_relevance_config(path: Path | None = None) -> RelevanceConfig:
    path = path or DEFAULT_RELEVANCE_CONFIG
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RelevanceConfig.from_dict(payload)


def percentile_to_relevance(pct: float, config: RelevanceConfig) -> int:
    """Map a within-group percentile rank in [0, 1] to a relevance integer."""
    if pct < 0 or pct > 1 or np.isnan(pct):
        raise ValueError(f"percentile must be in [0, 1], got {pct}")
    for bucket in config.buckets:
        # Upper bucket includes 1.0; lower edge is exclusive except bottom.
        if pct >= bucket.min_percentile:
            return bucket.relevance
    return config.buckets[-1].relevance


def assign_relevance_labels(
    frame: pd.DataFrame,
    *,
    return_col: str,
    config: RelevanceConfig,
    region_col: str = "Region",
) -> pd.DataFrame:
    """Assign relevance from within Date×Country future-return ranks.

    Future returns are used only for labels. Rows in groups smaller than
    ``min_group_size`` are dropped.
    """
    if return_col not in frame.columns:
        raise TrainingError(f"Missing return column for relevance: {return_col}")
    for key in config.group_keys:
        if key not in frame.columns and not (key == "Region" and region_col in frame.columns):
            raise TrainingError(f"Missing group key '{key}' for relevance labeling")

    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    if "Region" not in out.columns and region_col in out.columns:
        out["Region"] = out[region_col]

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[return_col, "Date", "Region"])
    # Percentile rank within Date × Region (higher future return → higher pct).
    out["_ret_pct"] = out.groupby(["Date", "Region"], sort=False)[return_col].rank(
        method="average", pct=True
    )
    out["relevance"] = out["_ret_pct"].map(lambda p: percentile_to_relevance(float(p), config))
    out = out.drop(columns=["_ret_pct"])

    sizes = out.groupby(["Date", "Region"], sort=False)["relevance"].transform("size")
    before = len(out)
    out = out.loc[sizes >= config.min_group_size].copy()
    dropped = before - len(out)
    if dropped:
        logger.info(
            "Dropped %d rows in ranking groups smaller than %d",
            dropped,
            config.min_group_size,
        )
    return out


def build_ranking_groups(
    frame: pd.DataFrame,
    *,
    group_keys: tuple[str, ...] = ("Date", "Region"),
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sort by group keys and return contiguous group sizes for LGBMRanker.

    Returns:
        (sorted_frame, group_sizes) where ``sum(group_sizes) == len(sorted_frame)``.
    """
    missing = [k for k in group_keys if k not in frame.columns]
    if missing:
        raise TrainingError(f"Missing group keys: {missing}")

    sorted_frame = frame.sort_values(
        list(group_keys) + (["Symbol"] if "Symbol" in frame.columns else []),
        kind="mergesort",
    ).reset_index(drop=True)

    # Preserve chronological Date order within Region ordering of keys.
    sizes = (
        sorted_frame.groupby(list(group_keys), sort=False, dropna=False)
        .size()
        .to_numpy(dtype=np.int32)
    )
    if int(sizes.sum()) != len(sorted_frame):
        raise TrainingError(
            f"Group size mismatch: sum(groups)={int(sizes.sum())} rows={len(sorted_frame)}"
        )
    if (sizes <= 0).any():
        raise TrainingError("Found empty ranking groups")
    return sorted_frame, sizes


def assert_group_integrity(
    frame: pd.DataFrame,
    group_sizes: np.ndarray,
    *,
    group_keys: tuple[str, ...] = ("Date", "Region"),
) -> None:
    """Validate that group sizes match contiguous blocks in ``frame``."""
    if int(group_sizes.sum()) != len(frame):
        raise AssertionError(
            f"sum(group_sizes)={int(group_sizes.sum())} != rows={len(frame)}"
        )
    start = 0
    for size in group_sizes:
        block = frame.iloc[start : start + int(size)]
        for key in group_keys:
            if block[key].nunique(dropna=False) != 1:
                raise AssertionError(
                    f"Group boundary broken at rows[{start}:{start + int(size)}] key={key}"
                )
        start += int(size)
    if start != len(frame):
        raise AssertionError("Group sizes did not cover the full frame")
