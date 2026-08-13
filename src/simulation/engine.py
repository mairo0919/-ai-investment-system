"""Day-by-day historical simulation engine (signal close T → fill open T+1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import TrainingError
from src.simulation.config import SimulationConfig
from src.simulation.execution import (
    CostModel,
    conservative_stop_fill,
    conservative_take_fill,
)
from src.simulation.fx import FxConverter
from src.simulation.orders import EquityPoint, Order
from src.simulation.portfolio import Portfolio
from src.simulation.quality import allow_new_entries, is_entry_rebalance_day
from src.simulation.rules import entry_candidates, equal_weight_notional, evaluate_exits_for_day


@dataclass
class SimulationResult:
    equity_curve: list[EquityPoint] = field(default_factory=list)
    portfolio: Portfolio | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def trades(self):
        return [] if self.portfolio is None else self.portfolio.trades


class SimulationEngine:
    """Historical paper trading with no look-ahead fills.

    Timing contract
    ---------------
    - Day T close: ranking / rules use data available through T close.
    - Day T+1 open: entries and exits fill (stop/take use conservative prices).
    - Equity marks use day-T close after processing T's open fills and T close signals.
    """

    def __init__(
        self,
        cfg: SimulationConfig,
        *,
        fx: FxConverter,
        countries: set[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.fx = fx
        self.countries = countries
        self.costs = CostModel(
            commission_rate=cfg.commission_rate,
            slippage_rate=cfg.slippage_rate,
        )

    def run(
        self,
        *,
        prices: pd.DataFrame,
        rankings: pd.DataFrame,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        portfolio: Portfolio | None = None,
        pending: list[Order] | None = None,
        cooldown_until: dict[str, pd.Timestamp] | None = None,
        min_calendar_days: int = 2,
    ) -> SimulationResult:
        """Run simulation.

        Parameters
        ----------
        prices:
            Columns: Date, Symbol, Country, Currency, Open, High, Low, Close
        rankings:
            Columns: Date, Symbol, Country, Score, Rank, Percentile
            Dates are signal dates (close of T).
        portfolio / pending / cooldown_until:
            Optional resume state for forward paper trading.
        min_calendar_days:
            Historical runs require >=2; single-day catch-up may use 1.
        """
        px = prices.copy()
        px["Date"] = pd.to_datetime(px["Date"]).dt.normalize()
        rk = rankings.copy()
        if not rk.empty:
            rk["Date"] = pd.to_datetime(rk["Date"]).dt.normalize()

        if self.countries is not None:
            px = px.loc[px["Country"].isin(self.countries)].copy()
            if not rk.empty:
                rk = rk.loc[rk["Country"].isin(self.countries)].copy()

        # Drop symbols whose currency cannot be converted
        convertible = []
        for ccy in sorted(px["Currency"].dropna().unique()):
            if self.fx.can_convert(str(ccy)):
                convertible.append(str(ccy))
        px = px.loc[px["Currency"].isin(convertible)].copy()
        if px.empty:
            raise TrainingError("No convertible-currency symbols for simulation")

        calendar = sorted(pd.to_datetime(px["Date"].unique()))
        if start is not None:
            start = pd.Timestamp(start).normalize()
            calendar = [d for d in calendar if d >= start]
        if end is not None:
            end = pd.Timestamp(end).normalize()
            calendar = [d for d in calendar if d <= end]
        if len(calendar) < min_calendar_days:
            raise TrainingError("Simulation calendar too short")

        # Index helpers
        open_map = _price_maps(px, "Open")
        high_map = _price_maps(px, "High")
        low_map = _price_maps(px, "Low")
        close_map = _price_maps(px, "Close")
        meta_map = (
            px.sort_values("Date")
            .drop_duplicates(["Symbol"], keep="last")
            .set_index("Symbol")[["Country", "Currency"]]
            .to_dict(orient="index")
        )

        ranking_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        if not rk.empty:
            for dt, g in rk.groupby("Date"):
                ranking_by_date[pd.Timestamp(dt).normalize()] = g.copy()

        if portfolio is None:
            portfolio = Portfolio(
                cash=float(self.cfg.initial_capital),
                base_currency=self.cfg.base_currency,
            )
        pending = list(pending or [])
        equity_curve: list[EquityPoint] = []
        cooldown_until = dict(cooldown_until or {})
        rebalance_stats = {
            "entry_signal_days": 0,
            "blocked_rebalance_days": 0,
            "blocked_quality_days": 0,
            "entries_queued": 0,
            "exits_queued": 0,
            "cooldown_blocks": 0,
            "entries_on_rebalance_days": [],
            "exits_on_rebalance_days": [],
        }

        for i, day in enumerate(calendar):
            next_day = calendar[i + 1] if i + 1 < len(calendar) else None
            day_entries = 0
            day_exits = 0

            # 1) Fill pending orders at today's open
            still_pending: list[Order] = []
            for order in pending:
                if day not in open_map or order.symbol not in open_map[day]:
                    # Keep one more session if possible
                    if next_day is not None:
                        still_pending.append(order)
                    continue
                raw_open = float(open_map[day][order.symbol])
                if order.side == "buy":
                    try:
                        portfolio.open_position(
                            symbol=order.symbol,
                            country=order.country,
                            currency=order.currency,
                            fill_date=day,
                            raw_open=raw_open,
                            target_notional_base=order.quantity,  # stores notional
                            costs=self.costs,
                            fx=self.fx,
                            stop_loss_pct=self.cfg.stop_loss_pct,
                            take_profit_pct=self.cfg.take_profit_pct,
                            score=order.score,
                            rank=order.rank,
                            percentile=order.percentile,
                        )
                    except TrainingError:
                        continue
                else:
                    raw_fill = raw_open
                    if (
                        order.order_type in ("stop", "trailing_stop")
                        and order.limit_or_stop_price is not None
                    ):
                        raw_fill = conservative_stop_fill(order.limit_or_stop_price, raw_open)
                    elif order.order_type == "take_profit" and order.limit_or_stop_price is not None:
                        raw_fill = conservative_take_fill(order.limit_or_stop_price, raw_open)
                    if order.symbol in portfolio.positions:
                        portfolio.close_position(
                            symbol=order.symbol,
                            fill_date=day,
                            raw_fill_price=raw_fill,
                            costs=self.costs,
                            fx=self.fx,
                            exit_reason=order.reason,
                        )
                        if self.cfg.cooldown_days and self.cfg.cooldown_days > 0:
                            # Block re-entry for cooldown_days trading sessions after exit fill
                            cool_i = min(i + int(self.cfg.cooldown_days), len(calendar) - 1)
                            cooldown_until[order.symbol] = calendar[cool_i]
            pending = still_pending

            # 2) Mark to market on close
            if day in close_map:
                portfolio.mark_to_market(prices_local=close_map[day], asof=day, fx=self.fx)

            # 3) Bump holding days for open positions (after open fills)
            for pos in portfolio.positions.values():
                pos.holding_days += 1

            # 4) Exit / entry signals at close → queue for next open
            if next_day is not None:
                ranking_day = ranking_by_date.get(day)
                high = high_map.get(day, {})
                low = low_map.get(day, {})
                exits = evaluate_exits_for_day(
                    portfolio,
                    day=day,
                    high=high,
                    low=low,
                    ranking_day=ranking_day,
                    cfg=self.cfg,
                )
                exiting = {e.symbol for e in exits}
                for ex in exits:
                    pos = portfolio.positions.get(ex.symbol)
                    if pos is None or pos.pending_exit_reason:
                        continue
                    pos.pending_exit_reason = ex.reason
                    otype = "market_open"
                    if ex.reason in ("stop_loss",):
                        otype = "stop"
                    elif ex.reason == "trailing_stop":
                        otype = "trailing_stop"
                    elif ex.reason == "take_profit":
                        otype = "take_profit"
                    pending.append(
                        Order(
                            symbol=ex.symbol,
                            country=pos.country,
                            currency=pos.currency,
                            side="sell",
                            quantity=pos.quantity,
                            signal_date=day,
                            reason=ex.reason,
                            order_type=otype,  # type: ignore[arg-type]
                            limit_or_stop_price=ex.stop_or_take_price,
                            entry_date=pos.entry_date,
                            entry_price=pos.entry_price,
                            entry_score=pos.entry_score,
                            entry_rank=pos.entry_rank,
                        )
                    )
                    day_exits += 1
                    rebalance_stats["exits_queued"] += 1

                allow_entries = is_entry_rebalance_day(
                    day,
                    calendar,
                    frequency=self.cfg.rebalance_frequency,
                    weekly_weekday=self.cfg.weekly_weekday,
                )
                if not allow_entries:
                    rebalance_stats["blocked_rebalance_days"] += 1
                elif ranking_day is not None and not ranking_day.empty:
                    # Quality / no-trade filters apply per country slice
                    countries_today = sorted(ranking_day["Country"].unique())
                    entry_frames: list[pd.DataFrame] = []
                    quality_blocked = False
                    for cty in countries_today:
                        slice_df = ranking_day.loc[ranking_day["Country"] == cty]
                        ok, reason = allow_new_entries(slice_df, self.cfg)
                        if not ok:
                            quality_blocked = True
                            continue
                        entry_frames.append(slice_df)
                    if quality_blocked and not entry_frames:
                        rebalance_stats["blocked_quality_days"] += 1
                    elif entry_frames:
                        filtered_day = pd.concat(entry_frames, ignore_index=True)
                        rebalance_stats["entry_signal_days"] += 1
                        cands = entry_candidates(
                            filtered_day,
                            portfolio,
                            self.cfg,
                            countries=self.countries,
                        )
                        if not cands.empty:
                            cands = cands.loc[~cands["Symbol"].isin(exiting)].copy()
                        pending_buy_symbols = {o.symbol for o in pending if o.side == "buy"}
                        if not cands.empty:
                            cands = cands.loc[~cands["Symbol"].isin(pending_buy_symbols)].copy()
                        # Cooldown filter
                        if self.cfg.cooldown_days and self.cfg.cooldown_days > 0 and not cands.empty:
                            before = len(cands)

                            def _cooled(sym: str) -> bool:
                                until = cooldown_until.get(str(sym))
                                return until is not None and day <= until

                            cands = cands.loc[~cands["Symbol"].map(_cooled)].copy()
                            blocked = before - len(cands)
                            rebalance_stats["cooldown_blocks"] += int(blocked)

                        n_new = len(cands)
                        notional = equal_weight_notional(portfolio, self.cfg, n_new)
                        for _, row in cands.iterrows():
                            symbol = str(row["Symbol"])
                            meta = meta_map.get(symbol)
                            if meta is None:
                                continue
                            equity = portfolio.total_equity
                            country = str(meta["Country"])
                            cur_w = portfolio.country_weight(country)
                            add_w = notional / equity if equity > 0 else 1.0
                            if cur_w + add_w > self.cfg.max_country_weight + 1e-9:
                                continue
                            if len(portfolio.positions) + sum(
                                1 for o in pending if o.side == "buy"
                            ) >= self.cfg.max_positions:
                                break
                            pending.append(
                                Order(
                                    symbol=symbol,
                                    country=country,
                                    currency=str(meta["Currency"]),
                                    side="buy",
                                    quantity=notional,
                                    signal_date=day,
                                    reason="entry_rank",
                                    score=float(row["Score"]),
                                    rank=int(row["Rank"]),
                                    percentile=float(row["Percentile"]),
                                )
                            )
                            day_entries += 1
                            rebalance_stats["entries_queued"] += 1

                if day_entries or day_exits:
                    rebalance_stats["entries_on_rebalance_days"].append(day_entries)
                    rebalance_stats["exits_on_rebalance_days"].append(day_exits)

            # 5) Equity point
            eq = portfolio.total_equity
            portfolio.peak_equity = max(portfolio.peak_equity, eq)
            dd = eq / portfolio.peak_equity - 1.0 if portfolio.peak_equity > 0 else 0.0
            equity_curve.append(
                EquityPoint(
                    date=day,
                    cash=float(portfolio.cash),
                    position_value=float(portfolio.position_value),
                    total_equity=float(eq),
                    drawdown=float(dd),
                    realized_pnl=float(portfolio.realized_pnl),
                    unrealized_pnl=float(portfolio.unrealized_pnl),
                    n_positions=len(portfolio.positions),
                )
            )

        # Force flatten remaining positions on last available open after last day if needed:
        # mark final with last close already done; leave open positions marked (reported in equity).
        return SimulationResult(
            equity_curve=equity_curve,
            portfolio=portfolio,
            config_snapshot=self.cfg.to_dict(),
            meta={
                "start": str(calendar[0].date()) if calendar else None,
                "end": str(calendar[-1].date()) if calendar else None,
                "n_days": len(calendar),
                "countries": sorted(self.countries) if self.countries else "ALL",
                "timing": "close_t_execute_open_t1",
                "convertible_currencies": convertible,
                "rebalance_frequency": self.cfg.rebalance_frequency,
                "pending_orders": pending,
                "cooldown_until": {
                    k: str(pd.Timestamp(v).date()) for k, v in cooldown_until.items()
                },
                "rebalance_stats": {
                    "entry_signal_days": rebalance_stats["entry_signal_days"],
                    "blocked_rebalance_days": rebalance_stats["blocked_rebalance_days"],
                    "blocked_quality_days": rebalance_stats["blocked_quality_days"],
                    "entries_queued": rebalance_stats["entries_queued"],
                    "exits_queued": rebalance_stats["exits_queued"],
                    "cooldown_blocks": rebalance_stats["cooldown_blocks"],
                    "avg_entries_per_rebalance": (
                        float(np.mean(rebalance_stats["entries_on_rebalance_days"]))
                        if rebalance_stats["entries_on_rebalance_days"]
                        else 0.0
                    ),
                    "avg_exits_per_rebalance": (
                        float(np.mean(rebalance_stats["exits_on_rebalance_days"]))
                        if rebalance_stats["exits_on_rebalance_days"]
                        else 0.0
                    ),
                },
            },
        )


def _price_maps(px: pd.DataFrame, col: str) -> dict[pd.Timestamp, dict[str, float]]:
    out: dict[pd.Timestamp, dict[str, float]] = {}
    for dt, g in px.groupby("Date"):
        day = pd.Timestamp(dt).normalize()
        out[day] = {
            str(r.Symbol): float(getattr(r, col))
            for r in g.itertuples(index=False)
            if pd.notna(getattr(r, col))
        }
    return out
