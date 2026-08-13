"""Open position state for paper trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Position:
    """Long-only paper position tracked in local currency and base (JPY)."""

    symbol: str
    country: str
    currency: str
    entry_date: pd.Timestamp
    entry_price: float  # local currency, post-slippage effective unit price
    quantity: float
    entry_score: float | None = None
    entry_rank: int | None = None
    entry_percentile: float | None = None
    current_price: float = 0.0
    holding_days: int = 0
    fx_to_base_entry: float = 1.0
    fx_to_base_current: float = 1.0
    stop_price: float | None = None
    take_profit_price: float | None = None
    peak_price: float = 0.0  # local high-water mark for trailing stop
    trailing_stop_price: float | None = None
    pending_exit_reason: str | None = None

    @property
    def market_value_local(self) -> float:
        return float(self.quantity * self.current_price)

    @property
    def market_value_base(self) -> float:
        return float(self.market_value_local * self.fx_to_base_current)

    @property
    def cost_basis_base(self) -> float:
        return float(self.quantity * self.entry_price * self.fx_to_base_entry)

    @property
    def unrealized_pnl_base(self) -> float:
        return float(self.market_value_base - self.cost_basis_base)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "country": self.country,
            "currency": self.currency,
            "entry_date": str(pd.Timestamp(self.entry_date).date()),
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "current_price": self.current_price,
            "market_value": self.market_value_base,
            "unrealized_pnl": self.unrealized_pnl_base,
            "holding_days": self.holding_days,
            "entry_score": self.entry_score,
            "entry_rank": self.entry_rank,
            "entry_percentile": self.entry_percentile,
            "pending_exit_reason": self.pending_exit_reason,
            "highest_price": self.peak_price,
            "peak_price": self.peak_price,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "trailing_stop_price": self.trailing_stop_price,
            "fx_to_base_entry": self.fx_to_base_entry,
            "fx_to_base_current": self.fx_to_base_current,
        }
