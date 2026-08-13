"""Target configuration for Phase 3C multi-horizon experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetConfig:
    """Declarative definition of a prediction target.

    Future prices are used only when materializing labels from Adj Close.
    """

    name: str
    horizon_days: int
    future_return_column: str
    target_column: str
    threshold: float | None = None
    drop_neutral: bool = False
    purge_days: int = 0

    @property
    def uses_threshold(self) -> bool:
        return self.threshold is not None


def default_threshold(settings_threshold: float = 0.02) -> float:
    """Return the configured absolute-return threshold (e.g. 0.02 = 2%)."""
    return settings_threshold


TARGET_UP_1D = TargetConfig(
    name="target_up_1d",
    horizon_days=1,
    future_return_column="future_return_1d",
    target_column="target_up_1d",
    purge_days=1,
)

TARGET_UP_5D = TargetConfig(
    name="target_up_5d",
    horizon_days=5,
    future_return_column="future_return_5d",
    target_column="target_up_5d",
    purge_days=5,
)

TARGET_UP_10D = TargetConfig(
    name="target_up_10d",
    horizon_days=10,
    future_return_column="future_return_10d",
    target_column="target_up_10d",
    purge_days=10,
)


def target_5d_threshold(threshold: float = 0.02) -> TargetConfig:
    """5-day direction target with a neutral band around zero."""
    return TargetConfig(
        name="target_5d_threshold",
        horizon_days=5,
        future_return_column="future_return_5d",
        target_column="target_5d_threshold",
        threshold=threshold,
        drop_neutral=True,
        purge_days=5,
    )


def all_target_configs(threshold: float = 0.02) -> tuple[TargetConfig, ...]:
    """Return the Phase 3C target suite."""
    return (
        TARGET_UP_1D,
        TARGET_UP_5D,
        TARGET_UP_10D,
        target_5d_threshold(threshold),
    )


# Backward-compatible alias used by Phase 3A/3B reports.
LEGACY_TARGET_COLUMN = "target_up"
