"""Cash + positions portfolio in base currency (JPY)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.core.exceptions import TrainingError
from src.simulation.execution import CostModel
from src.simulation.fx import FxConverter
from src.simulation.orders import TradeRecord
from src.simulation.position import Position


@dataclass
class Portfolio:
    cash: float
    base_currency: str = "JPY"
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage_impact: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    peak_equity: float = 0.0

    def __post_init__(self) -> None:
        self.peak_equity = float(self.cash)

    @property
    def position_value(self) -> float:
        return float(sum(p.market_value_base for p in self.positions.values()))

    @property
    def unrealized_pnl(self) -> float:
        return float(sum(p.unrealized_pnl_base for p in self.positions.values()))

    @property
    def total_equity(self) -> float:
        return float(self.cash + self.position_value)

    @property
    def total_value(self) -> float:
        return self.total_equity

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def country_weight(self, country: str) -> float:
        eq = self.total_equity
        if eq <= 0:
            return 0.0
        mv = sum(p.market_value_base for p in self.positions.values() if p.country == country)
        return float(mv / eq)

    def mark_to_market(
        self,
        *,
        prices_local: dict[str, float],
        asof: pd.Timestamp,
        fx: FxConverter,
    ) -> None:
        for symbol, pos in self.positions.items():
            if symbol not in prices_local:
                continue
            pos.current_price = float(prices_local[symbol])
            pos.fx_to_base_current = fx.rate_to_base(pos.currency, asof)

    def open_position(
        self,
        *,
        symbol: str,
        country: str,
        currency: str,
        fill_date: pd.Timestamp,
        raw_open: float,
        target_notional_base: float,
        costs: CostModel,
        fx: FxConverter,
        stop_loss_pct: float | None,
        take_profit_pct: float | None,
        score: float | None,
        rank: int | None,
        percentile: float | None,
    ) -> Position:
        if symbol in self.positions:
            raise TrainingError(f"Duplicate position not allowed: {symbol}")
        if target_notional_base <= 0:
            raise TrainingError("target_notional_base must be positive")

        fx_rate = fx.rate_to_base(currency, fill_date)
        buy_px = costs.buy_unit_price(raw_open)
        # shares from target notional after slippage unit price
        qty = target_notional_base / (buy_px * fx_rate)
        gross_base = qty * buy_px * fx_rate
        commission = costs.commission(gross_base)
        total_debit = gross_base + commission
        if total_debit > self.cash + 1e-6:
            # Scale down to available cash
            scale = self.cash / (gross_base + commission) if (gross_base + commission) > 0 else 0.0
            if scale <= 1e-12:
                raise TrainingError("Insufficient cash for entry")
            qty *= scale
            gross_base = qty * buy_px * fx_rate
            commission = costs.commission(gross_base)
            total_debit = gross_base + commission

        slip_impact = qty * (buy_px - raw_open) * fx_rate
        self.cash -= total_debit
        self.total_commission += commission
        self.total_slippage_impact += abs(slip_impact)

        stop_price = None
        take_price = None
        if stop_loss_pct is not None:
            stop_price = float(buy_px * (1.0 + stop_loss_pct))
        if take_profit_pct is not None:
            take_price = float(buy_px * (1.0 + take_profit_pct))

        pos = Position(
            symbol=symbol,
            country=country,
            currency=currency,
            entry_date=pd.Timestamp(fill_date),
            entry_price=buy_px,
            quantity=float(qty),
            entry_score=score,
            entry_rank=rank,
            entry_percentile=percentile,
            current_price=buy_px,
            holding_days=0,
            fx_to_base_entry=fx_rate,
            fx_to_base_current=fx_rate,
            stop_price=stop_price,
            take_profit_price=take_price,
            peak_price=buy_px,
            trailing_stop_price=None,
        )
        self.positions[symbol] = pos
        return pos

    def close_position(
        self,
        *,
        symbol: str,
        fill_date: pd.Timestamp,
        raw_fill_price: float,
        costs: CostModel,
        fx: FxConverter,
        exit_reason: str,
        apply_sell_slippage: bool = True,
    ) -> TradeRecord:
        if symbol not in self.positions:
            raise TrainingError(f"No position to close: {symbol}")
        pos = self.positions[symbol]
        fx_rate = fx.rate_to_base(pos.currency, fill_date)
        sell_px = costs.sell_unit_price(raw_fill_price) if apply_sell_slippage else float(raw_fill_price)
        # If raw_fill already embeds conservative stop/take vs open, still apply slippage on that fill.
        proceeds_base = pos.quantity * sell_px * fx_rate
        commission = costs.commission(proceeds_base)
        net_proceeds = proceeds_base - commission
        cost_basis = pos.cost_basis_base
        gross_pnl = proceeds_base - cost_basis
        net_pnl = net_proceeds - cost_basis
        slip_impact = pos.quantity * (raw_fill_price - sell_px) * fx_rate

        self.cash += net_proceeds
        self.realized_pnl += net_pnl
        self.total_commission += commission
        self.total_slippage_impact += abs(slip_impact)

        ret = net_pnl / cost_basis if cost_basis > 0 else 0.0
        trade = TradeRecord(
            symbol=pos.symbol,
            country=pos.country,
            currency=pos.currency,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=pd.Timestamp(fill_date),
            exit_price=sell_px,
            quantity=pos.quantity,
            gross_pnl_base=float(gross_pnl),
            net_pnl_base=float(net_pnl),
            return_pct=float(ret),
            holding_days=int(pos.holding_days),
            exit_reason=exit_reason,
            entry_score=pos.entry_score,
            entry_rank=pos.entry_rank,
            entry_cost_base=float(cost_basis),
            exit_proceeds_base=float(net_proceeds),
            commission_base=float(commission),
            slippage_impact_base=float(abs(slip_impact)),
        )
        del self.positions[symbol]
        self.trades.append(trade)
        return trade
