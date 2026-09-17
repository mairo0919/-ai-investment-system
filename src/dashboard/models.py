"""Typed shapes for Dashboard read models (display / API glue; not persisted)."""

from __future__ import annotations

from typing import Any, TypedDict


class EquityCurvePoint(TypedDict):
    date: str
    total_equity: float
    cash: float
    position_value: float
    drawdown: float


class DailyReturnPoint(TypedDict):
    date: str
    daily_return: float


class MonthlyReturnPoint(TypedDict):
    month: str
    strategy_return: float


class RankingRow(TypedDict, total=False):
    date: str | None
    symbol: str | None
    score: float | None
    rank: int | None
    percentile: float | None
    selected: bool | None
    model_id: str | None
    country: str | None
    close: float | None


class PortfolioRow(TypedDict, total=False):
    symbol: str | None
    country: str | None
    currency: str | None
    quantity: float | None
    entry_date: str | None
    entry_price: float | None
    current_price: float | None
    market_value: float | None
    unrealized_pnl: float | None
    holding_days: int | None
    entry_score: float | None
    entry_rank: int | None
    entry_percentile: float | None
    current_score: float | None
    current_rank: int | None
    current_percentile: float | None
    selected: bool | None
    pending_exit_reason: str | None


class TradeRow(TypedDict, total=False):
    symbol: str | None
    country: str | None
    currency: str | None
    entry_date: str | None
    exit_date: str | None
    entry_price: float | None
    exit_price: float | None
    quantity: float | None
    gross_pnl: float | None
    net_pnl: float | None
    return_: float | None  # mapped from CSV "Return"; serialized as "return"
    holding_days: int | None
    exit_reason: str | None
    entry_rank: int | None
    model_id: str | None
    config_id: str | None


OverviewDict = dict[str, Any]
PerformanceDict = dict[str, Any]
SystemStatusDict = dict[str, Any]
