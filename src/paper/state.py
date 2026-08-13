"""Persistent paper portfolio state with idempotency keys."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.paper.atomic_io import atomic_write_csv, atomic_write_json
from src.simulation.orders import Order
from src.simulation.portfolio import Portfolio
from src.simulation.position import Position


def order_idempotency_key(
    *,
    signal_date: pd.Timestamp | str,
    symbol: str,
    side: str,
    reason: str,
) -> str:
    d = str(pd.Timestamp(signal_date).date())
    return f"{d}|{symbol}|{side}|{reason}"


def fill_idempotency_key(
    *,
    fill_date: pd.Timestamp | str,
    symbol: str,
    side: str,
    reason: str,
) -> str:
    d = str(pd.Timestamp(fill_date).date())
    return f"FILL|{d}|{symbol}|{side}|{reason}"


@dataclass
class PaperOrderRecord:
    order_id: str
    symbol: str
    country: str
    currency: str
    side: str
    quantity: float
    signal_date: str
    reason: str
    status: str  # PENDING / FILLED / CANCELLED / SKIPPED
    order_type: str = "market_open"
    limit_or_stop_price: float | None = None
    score: float | None = None
    rank: int | None = None
    percentile: float | None = None
    entry_date: str | None = None
    entry_price: float | None = None
    entry_score: float | None = None
    entry_rank: int | None = None
    fill_date: str | None = None
    fill_price: float | None = None
    model_id: str | None = None
    config_id: str | None = None
    idempotency_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "country": self.country,
            "currency": self.currency,
            "side": self.side,
            "quantity": self.quantity,
            "signal_date": self.signal_date,
            "reason": self.reason,
            "status": self.status,
            "order_type": self.order_type,
            "limit_or_stop_price": self.limit_or_stop_price,
            "score": self.score,
            "rank": self.rank,
            "percentile": self.percentile,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "entry_score": self.entry_score,
            "entry_rank": self.entry_rank,
            "fill_date": self.fill_date,
            "fill_price": self.fill_price,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PaperOrderRecord:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]

    def to_engine_order(self) -> Order:
        return Order(
            symbol=self.symbol,
            country=self.country,
            currency=self.currency,
            side=self.side,  # type: ignore[arg-type]
            quantity=float(self.quantity),
            signal_date=pd.Timestamp(self.signal_date),
            reason=self.reason,
            order_type=self.order_type,  # type: ignore[arg-type]
            limit_or_stop_price=self.limit_or_stop_price,
            score=self.score,
            rank=self.rank,
            percentile=self.percentile,
            entry_date=pd.Timestamp(self.entry_date) if self.entry_date else None,
            entry_price=self.entry_price,
            entry_score=self.entry_score,
            entry_rank=self.entry_rank,
        )


@dataclass
class PaperState:
    cash: float
    base_currency: str = "JPY"
    positions: dict[str, Position] = field(default_factory=dict)
    pending_orders: list[PaperOrderRecord] = field(default_factory=list)
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage_impact: float = 0.0
    peak_equity: float = 0.0
    last_processed_date: str | None = None
    forward_start: str | None = None
    model_id: str | None = None
    config_id: str = "FINAL_US_PHASE4C"
    cooldown_until: dict[str, str] = field(default_factory=dict)
    seen_order_keys: set[str] = field(default_factory=set)
    seen_fill_keys: set[str] = field(default_factory=set)
    brokerage_orders_submitted: int = 0

    @property
    def position_value(self) -> float:
        return float(sum(p.market_value_base for p in self.positions.values()))

    @property
    def unrealized_pnl(self) -> float:
        return float(sum(p.unrealized_pnl_base for p in self.positions.values()))

    @property
    def total_equity(self) -> float:
        return float(self.cash + self.position_value)

    def to_portfolio(self) -> Portfolio:
        port = Portfolio(
            cash=float(self.cash),
            base_currency=self.base_currency,
            positions=dict(self.positions),
            realized_pnl=float(self.realized_pnl),
            total_commission=float(self.total_commission),
            total_slippage_impact=float(self.total_slippage_impact),
            trades=[],
            peak_equity=float(self.peak_equity or self.cash),
        )
        return port

    def sync_from_portfolio(self, portfolio: Portfolio) -> None:
        self.cash = float(portfolio.cash)
        self.positions = dict(portfolio.positions)
        self.realized_pnl = float(portfolio.realized_pnl)
        self.total_commission = float(portfolio.total_commission)
        self.total_slippage_impact = float(portfolio.total_slippage_impact)
        self.peak_equity = float(portfolio.peak_equity)

    def portfolio_json(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "base_currency": self.base_currency,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_equity": self.total_equity,
            "position_value": self.position_value,
            "total_commission": self.total_commission,
            "total_slippage_impact": self.total_slippage_impact,
            "peak_equity": self.peak_equity,
            "last_processed_date": self.last_processed_date,
            "forward_start": self.forward_start,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "n_positions": len(self.positions),
            "cooldown_until": self.cooldown_until,
            "seen_order_keys": sorted(self.seen_order_keys),
            "seen_fill_keys": sorted(self.seen_fill_keys),
            "brokerage_orders_submitted": self.brokerage_orders_submitted,
            "brokerage_enabled": False,
        }

    def positions_json(self) -> list[dict[str, Any]]:
        rows = []
        for p in self.positions.values():
            d = p.to_dict()
            d["highest_price"] = p.peak_price
            rows.append(d)
        return rows


class PaperStore:
    """Filesystem store under data/paper/."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def portfolio_path(self) -> Path:
        return self.root / "portfolio.json"

    @property
    def positions_path(self) -> Path:
        return self.root / "positions.json"

    @property
    def pending_path(self) -> Path:
        return self.root / "pending_orders.json"

    @property
    def trade_history_path(self) -> Path:
        return self.root / "trade_history.csv"

    @property
    def equity_history_path(self) -> Path:
        return self.root / "equity_history.csv"

    @property
    def audit_log_path(self) -> Path:
        return self.root / "audit_log.csv"

    def exists(self) -> bool:
        return self.portfolio_path.exists()

    def load(self) -> PaperState | None:
        if not self.exists():
            return None
        port = json.loads(self.portfolio_path.read_text(encoding="utf-8"))
        positions_raw = json.loads(self.positions_path.read_text(encoding="utf-8"))
        pending_raw = json.loads(self.pending_path.read_text(encoding="utf-8"))
        positions: dict[str, Position] = {}
        for row in positions_raw:
            positions[str(row["symbol"])] = Position(
                symbol=str(row["symbol"]),
                country=str(row["country"]),
                currency=str(row["currency"]),
                entry_date=pd.Timestamp(row["entry_date"]),
                entry_price=float(row["entry_price"]),
                quantity=float(row["quantity"]),
                entry_score=row.get("entry_score"),
                entry_rank=row.get("entry_rank"),
                entry_percentile=row.get("entry_percentile"),
                current_price=float(row.get("current_price") or 0),
                holding_days=int(row.get("holding_days") or 0),
                fx_to_base_entry=float(row.get("fx_to_base_entry") or 1.0),
                fx_to_base_current=float(row.get("fx_to_base_current") or 1.0),
                stop_price=row.get("stop_price"),
                take_profit_price=row.get("take_profit_price"),
                peak_price=float(row.get("highest_price") or row.get("peak_price") or row.get("current_price") or 0),
                trailing_stop_price=row.get("trailing_stop_price"),
                pending_exit_reason=row.get("pending_exit_reason"),
            )
        state = PaperState(
            cash=float(port["cash"]),
            base_currency=str(port.get("base_currency", "JPY")),
            positions=positions,
            pending_orders=[PaperOrderRecord.from_dict(x) for x in pending_raw],
            realized_pnl=float(port.get("realized_pnl") or 0),
            total_commission=float(port.get("total_commission") or 0),
            total_slippage_impact=float(port.get("total_slippage_impact") or 0),
            peak_equity=float(port.get("peak_equity") or port.get("cash") or 0),
            last_processed_date=port.get("last_processed_date"),
            forward_start=port.get("forward_start"),
            model_id=port.get("model_id"),
            config_id=str(port.get("config_id") or "FINAL_US_PHASE4C"),
            cooldown_until=dict(port.get("cooldown_until") or {}),
            seen_order_keys=set(port.get("seen_order_keys") or []),
            seen_fill_keys=set(port.get("seen_fill_keys") or []),
            brokerage_orders_submitted=int(port.get("brokerage_orders_submitted") or 0),
        )
        return state

    def save_atomic(self, state: PaperState) -> None:
        """Write all state files atomically (temp + replace). Partial writes avoided."""
        atomic_write_json(self.portfolio_path, state.portfolio_json())
        atomic_write_json(self.positions_path, state.positions_json())
        atomic_write_json(
            self.pending_path,
            [o.to_dict() for o in state.pending_orders if o.status == "PENDING"],
        )

    def ensure_history_files(self) -> None:
        if not self.trade_history_path.exists():
            atomic_write_csv(
                self.trade_history_path,
                pd.DataFrame(
                    columns=[
                        "Symbol",
                        "Entry Date",
                        "Exit Date",
                        "Net PnL",
                        "Return",
                        "idempotency_key",
                        "model_id",
                        "config_id",
                    ]
                ),
            )
        if not self.equity_history_path.exists():
            atomic_write_csv(
                self.equity_history_path,
                pd.DataFrame(
                    columns=[
                        "Date",
                        "Cash",
                        "Position Value",
                        "Total Equity",
                        "Drawdown",
                        "Realized PnL",
                        "Unrealized PnL",
                        "N Positions",
                        "Transaction Costs",
                    ]
                ),
            )

    def append_equity(self, row: dict[str, Any]) -> None:
        path = self.equity_history_path
        df = pd.DataFrame([row])
        if path.exists():
            prev = pd.read_csv(path)
            # Idempotent: replace same date
            if "Date" in prev.columns:
                prev = prev.loc[prev["Date"].astype(str) != str(row["Date"])]
            df = pd.concat([prev, df], ignore_index=True)
        atomic_write_csv(path, df)

    def append_trades(self, trades: list[dict[str, Any]]) -> None:
        if not trades:
            return
        path = self.trade_history_path
        df = pd.DataFrame(trades)
        if path.exists():
            prev = pd.read_csv(path)
            df = pd.concat([prev, df], ignore_index=True)
            if "idempotency_key" in df.columns:
                df = df.drop_duplicates(subset=["idempotency_key"], keep="first")
        atomic_write_csv(path, df)

    def append_audit(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.audit_log_path
        df = pd.DataFrame(rows)
        if path.exists():
            prev = pd.read_csv(path)
            df = pd.concat([prev, df], ignore_index=True)
            if "idempotency_key" in df.columns:
                df = df.drop_duplicates(subset=["idempotency_key"], keep="first")
        atomic_write_csv(path, df)
