"""Phase 3H ranking label schemes and explicit label_gain builders."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.core.exceptions import TrainingError
from src.ml.ltr_labels import RelevanceConfig, assign_relevance_labels, load_relevance_config

logger = logging.getLogger(__name__)

DEFAULT_LABEL_SCHEME_CONFIG = Path("config/ranking_labels_3h.json")

GainName = Literal["linear", "moderate_exp"]
SchemeName = Literal["A", "B", "C"]


@dataclass(frozen=True)
class LabelSchemeResult:
    name: str
    max_relevance: int
    description: str
    gain_name: GainName
    label_gain: tuple[float, ...]
    frame: pd.DataFrame


def load_label_scheme_config(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_LABEL_SCHEME_CONFIG
    return json.loads(path.read_text(encoding="utf-8"))


def build_label_gain(max_relevance: int, gain_name: GainName) -> tuple[float, ...]:
    """Build label_gain of length max_relevance+1 (index = relevance)."""
    if max_relevance < 0:
        raise ValueError("max_relevance must be >= 0")
    n = max_relevance + 1
    if gain_name == "linear":
        gains = [float(i) for i in range(n)]
    elif gain_name == "moderate_exp":
        # Milder than 2^i - 1 to avoid overflow / extreme top focus for large label spaces.
        gains = [float((1.25**i) - 1.0) for i in range(n)]
    else:
        raise ValueError(f"Unknown gain_name: {gain_name}")
    if len(gains) != n:
        raise TrainingError("label_gain length mismatch")
    if max(gains) > 1e15:
        raise TrainingError(f"label_gain values too large for scheme max={max_relevance}")
    return tuple(gains)


def label_gain_to_param(gains: tuple[float, ...]) -> str:
    """Serialize label_gain for LightGBM (comma-separated)."""
    return ",".join(str(g) for g in gains)


def assign_decile_relevance(frame: pd.DataFrame, *, return_col: str) -> pd.Series:
    """0..9 from within Date×Country future-return percentile."""
    pct = frame.groupby(["Date", "Region"], sort=False)[return_col].rank(
        method="average", pct=True
    )
    return np.minimum(9, np.floor(pct.to_numpy(dtype=float) * 10.0)).astype(int)


def assign_percentile100_relevance(frame: pd.DataFrame, *, return_col: str) -> pd.Series:
    """0..99 from within Date×Country future-return percentile."""
    pct = frame.groupby(["Date", "Region"], sort=False)[return_col].rank(
        method="average", pct=True
    )
    return np.minimum(99, np.floor(pct.to_numpy(dtype=float) * 100.0)).astype(int)


def assign_label_scheme(
    frame: pd.DataFrame,
    *,
    return_col: str,
    scheme: SchemeName,
    gain_name: GainName,
    bucket_config: RelevanceConfig | None = None,
    min_group_size: int = 5,
) -> LabelSchemeResult:
    """Assign relevance labels for scheme A/B/C and attach matching label_gain."""
    cfg = load_label_scheme_config()
    meta = cfg["schemes"][scheme]
    if meta.get("implemented") is False:
        raise TrainingError(f"Label scheme {scheme} is not implemented: {meta.get('reason')}")

    max_rel = int(meta["max_relevance"])
    description = str(meta.get("description", scheme))
    gains = build_label_gain(max_rel, gain_name)

    if scheme == "A":
        relevance_config = bucket_config or load_relevance_config()
        # Ensure bucket scheme max matches expected 0..4
        out = assign_relevance_labels(
            frame, return_col=return_col, config=relevance_config
        )
    else:
        out = frame.copy()
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[return_col, "Date", "Region"]
        )
        if scheme == "B":
            out["relevance"] = assign_decile_relevance(out, return_col=return_col)
        elif scheme == "C":
            out["relevance"] = assign_percentile100_relevance(out, return_col=return_col)
        else:
            raise TrainingError(f"Unsupported scheme {scheme}")
        sizes = out.groupby(["Date", "Region"], sort=False)["relevance"].transform("size")
        out = out.loc[sizes >= min_group_size].copy()

    if out["relevance"].min() < 0 or out["relevance"].max() > max_rel:
        raise TrainingError(
            f"Label scheme {scheme} produced relevance outside 0..{max_rel}: "
            f"[{out['relevance'].min()}, {out['relevance'].max()}]"
        )
    if out["relevance"].max() >= len(gains):
        raise TrainingError(
            f"label_gain length {len(gains)} insufficient for max relevance "
            f"{out['relevance'].max()}"
        )

    logger.info(
        "Assigned label scheme=%s gain=%s rows=%d max_rel=%d unique_labels=%d",
        scheme,
        gain_name,
        len(out),
        int(out["relevance"].max()),
        int(out["relevance"].nunique()),
    )
    return LabelSchemeResult(
        name=scheme,
        max_relevance=max_rel,
        description=description,
        gain_name=gain_name,
        label_gain=gains,
        frame=out,
    )


def skip_reason_label_d() -> str:
    cfg = load_label_scheme_config()
    return str(cfg["schemes"]["D"].get("reason", "not implemented"))
