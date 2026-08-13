"""Order and trade log records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

Side = Literal["buy", "sell"]
OrderType = Literal["market_open", "stop", "take_profit", "trailing_stop"]


@dataclass
class Order:
    """Virtual order queued for next-session open (or conservative stop/take)."""

    symbol: str
    country: str
    currency: str
    side: Side
    quantity: float
    signal_date: pd.Timestamp
    reason: str
    order_type: OrderType = "market_open"
    limit_or_stop_price: float | None = None
    score: float | None = None
    rank: int | None = None
    percentile: float | None = None
    # For sells: link to entry metadata
    entry_date: pd.Timestamp | None = None
    entry_price: float | None = None
    entry_score: float | None = None
    entry_rank: int | None = None


@dataclass
class TradeRecord:
    """Closed round-trip trade."""

    symbol: str
    country: str
    currency: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    quantity: float
    gross_pnl_base: float
    net_pnl_base: float
    return_pct: float
    holding_days: int
    exit_reason: str
    entry_score: float | None
    entry_rank: int | None
    entry_cost_base: float
    exit_proceeds_base: float
    commission_base: float
    slippage_impact_base: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "Symbol": self.symbol,
            "Country": self.country,
            "Currency": self.currency,
            "Entry Date": str(pd.Timestamp(self.entry_date).date()),
            "Entry Price": self.entry_price,
            "Exit Date": str(pd.Timestamp(self.exit_date).date()),
            "Exit Price": self.exit_price,
            "Quantity": self.quantity,
            "Gross PnL": self.gross_pnl_base,
            "Net PnL": self.net_pnl_base,
            "Return": self.return_pct,
            "Holding Days": self.holding_days,
            "Exit Reason": self.exit_reason,
            "Entry Score": self.entry_score,
            "Entry Rank": self.entry_rank,
            "Entry Cost Base": self.entry_cost_base,
            "Exit Proceeds Base": self.exit_proceeds_base,
            "Commission Base": self.commission_base,
            "Slippage Impact Base": self.slippage_impact_base,
        }


@dataclass
class EquityPoint:
    date: pd.Timestamp
    cash: float
    position_value: float
    total_equity: float
    drawdown: float
    realized_pnl: float
    unrealized_pnl: float
    n_positions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "Date": str(pd.Timestamp(self.date).date()),
            "Cash": self.cash,
            "Position Value": self.position_value,
            "Total Equity": self.total_equity,
            "Drawdown": self.drawdown,
            "Realized PnL": self.realized_pnl,
            "Unrealized PnL": self.unrealized_pnl,
            "N Positions": self.n_positions,
        }
