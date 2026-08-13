"""Fill price helpers: next-open execution, costs, conservative stop/take."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005

    def buy_unit_price(self, raw_open: float) -> float:
        """Buyer pays higher price after slippage."""
        return float(raw_open * (1.0 + self.slippage_rate))

    def sell_unit_price(self, raw_open: float) -> float:
        """Seller receives lower price after slippage."""
        return float(raw_open * (1.0 - self.slippage_rate))

    def commission(self, notional_base: float) -> float:
        return float(abs(notional_base) * self.commission_rate)


def conservative_stop_fill(stop_price: float, next_open: float) -> float:
    """Long stop: fill at the worse (lower) of stop and next open."""
    return float(min(stop_price, next_open))


def conservative_take_fill(take_price: float, next_open: float) -> float:
    """Long take-profit: fill at the worse (lower) of take and next open."""
    return float(min(take_price, next_open))
