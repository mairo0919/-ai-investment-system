"""Execution policy: share lots / affordability (opt-in; legacy default unchanged)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from src.simulation.execution import CostModel

ShareMode = Literal["fractional", "integer"]
SizingMode = Literal["legacy_equal_weight", "integer_affordable"]

# Decision event types (execution audit).
DECISION_QUEUED = "QUEUED"
DECISION_SKIPPED = "SKIPPED"
DECISION_REJECTED = "REJECTED"
DECISION_FILLED = "FILLED"

# Internal skip / reject reasons (SSOT; also mirrored in execution_audit).
REASON_INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
REASON_MIN_QUANTITY_NOT_MET = "MIN_QUANTITY_NOT_MET"
REASON_NOT_AFFORDABLE = "NOT_AFFORDABLE"
REASON_INVALID_PRICE = "INVALID_PRICE"
REASON_INVALID_FX = "INVALID_FX"
REASON_POSITION_LIMIT = "POSITION_LIMIT"
REASON_COUNTRY_LIMIT = "COUNTRY_LIMIT"
REASON_COOLDOWN = "COOLDOWN"
REASON_ALREADY_HELD = "ALREADY_HELD"
REASON_PENDING_BUY = "PENDING_BUY"
REASON_AFFORDABLE = "AFFORDABLE"
REASON_DUPLICATE = "DUPLICATE"
REASON_FILL_PRICE_UNAFFORDABLE = "FILL_PRICE_UNAFFORDABLE"


@dataclass(frozen=True)
class ExecutionPolicy:
    """Portfolio constraints only — does not alter ranking / model signals."""

    share_mode: ShareMode = "fractional"
    minimum_quantity: float = 1.0
    allow_fractional: bool = True
    sizing_mode: SizingMode = "legacy_equal_weight"
    # When weight budget < cost(1 share) but cash can buy 1, allow exactly 1 share.
    min_lot_overrides_weight: bool = False

    @property
    def is_integer_affordable(self) -> bool:
        return self.sizing_mode == "integer_affordable" and self.share_mode == "integer"

    @classmethod
    def legacy(cls) -> ExecutionPolicy:
        return cls()

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ExecutionPolicy:
        if not raw:
            return cls.legacy()
        share_mode = str(raw.get("share_mode", "fractional")).lower()
        if share_mode not in ("fractional", "integer"):
            share_mode = "fractional"
        sizing = str(raw.get("sizing_mode", "legacy_equal_weight")).lower()
        if sizing not in ("legacy_equal_weight", "integer_affordable"):
            sizing = "legacy_equal_weight"
        allow_frac = bool(raw.get("allow_fractional", share_mode == "fractional"))
        if share_mode == "integer":
            allow_frac = False
        return cls(
            share_mode=share_mode,  # type: ignore[arg-type]
            minimum_quantity=float(raw.get("minimum_quantity", 1.0)),
            allow_fractional=allow_frac,
            sizing_mode=sizing,  # type: ignore[arg-type]
            min_lot_overrides_weight=bool(raw.get("min_lot_overrides_weight", False)),
        )


def buy_unit_cost_base(
    *,
    raw_price: float,
    fx_rate: float,
    costs: CostModel,
) -> float:
    """All-in base cost for exactly 1 share (slippage + proportional commission)."""
    buy_px = costs.buy_unit_price(raw_price)
    gross = buy_px * fx_rate
    return float(gross + costs.commission(gross))


def entry_debit_base(
    *,
    quantity: float,
    raw_price: float,
    fx_rate: float,
    costs: CostModel,
) -> float:
    buy_px = costs.buy_unit_price(raw_price)
    gross = float(quantity) * buy_px * fx_rate
    return float(gross + costs.commission(gross))


def max_integer_shares_for_cash(
    *,
    cash: float,
    raw_price: float,
    fx_rate: float,
    costs: CostModel,
    minimum_quantity: float = 1.0,
    budget_cap: float | None = None,
) -> tuple[int, str | None]:
    """Largest integer share count affordable under cash (and optional budget cap)."""
    if raw_price is None or not math.isfinite(float(raw_price)) or float(raw_price) <= 0:
        return 0, REASON_INVALID_PRICE
    if fx_rate is None or not math.isfinite(float(fx_rate)) or float(fx_rate) <= 0:
        return 0, REASON_INVALID_FX
    spendable = float(cash)
    if budget_cap is not None:
        spendable = min(spendable, float(budget_cap))
    if spendable <= 0:
        return 0, REASON_INSUFFICIENT_CASH

    unit = buy_unit_cost_base(raw_price=raw_price, fx_rate=fx_rate, costs=costs)
    if unit <= 0 or not math.isfinite(unit):
        return 0, REASON_INVALID_PRICE
    if unit > spendable + 1e-6:
        return 0, REASON_NOT_AFFORDABLE

    # Proportional commission => closed form, then verify downward.
    q = int(math.floor(spendable / unit + 1e-12))
    min_q = max(1, int(math.floor(minimum_quantity + 1e-12)))
    while q >= min_q:
        debit = entry_debit_base(
            quantity=float(q), raw_price=raw_price, fx_rate=fx_rate, costs=costs
        )
        if debit <= spendable + 1e-6 and debit <= float(cash) + 1e-6:
            return q, None
        q -= 1
    if unit <= float(cash) + 1e-6:
        return 0, REASON_MIN_QUANTITY_NOT_MET
    return 0, REASON_NOT_AFFORDABLE


def target_notional_for_shares(
    *,
    shares: int,
    raw_price: float,
    fx_rate: float,
    costs: CostModel,
) -> float:
    """JPY notional stored on Order.quantity (legacy semantics: pre-commission gross)."""
    buy_px = costs.buy_unit_price(raw_price)
    return float(shares) * buy_px * float(fx_rate)


def apply_integer_floor_to_notional(
    *,
    target_notional_base: float,
    raw_open: float,
    fx_rate: float,
    costs: CostModel,
    cash: float,
    minimum_quantity: float,
) -> tuple[float, str | None]:
    """Convert legacy notional fill into integer shares under cash (fill-time).

    Never increases shares beyond floor(notional / (buy_px * fx)).
    """
    buy_px = costs.buy_unit_price(raw_open)
    if buy_px <= 0 or fx_rate <= 0:
        return 0.0, REASON_INVALID_PRICE
    qty_float = float(target_notional_base) / (buy_px * fx_rate)
    q = int(math.floor(qty_float + 1e-12))
    min_q = max(1, int(math.floor(minimum_quantity + 1e-12)))
    while q >= min_q:
        debit = entry_debit_base(
            quantity=float(q), raw_price=raw_open, fx_rate=fx_rate, costs=costs
        )
        if debit <= float(cash) + 1e-6:
            return float(q), None
        q -= 1
    return 0.0, REASON_MIN_QUANTITY_NOT_MET
