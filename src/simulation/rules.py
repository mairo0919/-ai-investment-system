"""Entry / exit rule engine for paper trading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.simulation.config import SimulationConfig
from src.simulation.execution import CostModel
from src.simulation.execution_policy import (
    REASON_ALREADY_HELD,
    REASON_COUNTRY_LIMIT,
    REASON_COOLDOWN,
    REASON_INSUFFICIENT_CASH,
    REASON_INVALID_FX,
    REASON_INVALID_PRICE,
    REASON_MIN_QUANTITY_NOT_MET,
    REASON_NOT_AFFORDABLE,
    REASON_PENDING_BUY,
    REASON_POSITION_LIMIT,
    buy_unit_cost_base,
    entry_debit_base,
    max_integer_shares_for_cash,
    target_notional_for_shares,
)
from src.simulation.fx import FxConverter
from src.simulation.portfolio import Portfolio
from src.simulation.ranking import select_candidates


@dataclass(frozen=True)
class ExitSignal:
    symbol: str
    reason: str
    stop_or_take_price: float | None = None


@dataclass(frozen=True)
class AffordableEntryPlan:
    symbol: str
    country: str
    currency: str
    shares: int
    target_notional_base: float
    estimated_debit_base: float
    score: float
    rank: int
    percentile: float
    estimated_price_local: float
    fx_to_base: float
    cash_before: float
    cash_reserved: float
    cash_available: float


@dataclass(frozen=True)
class EntrySkip:
    symbol: str
    reason: str
    rank: int | None = None
    score: float | None = None
    country: str | None = None
    currency: str | None = None
    estimated_price_local: float | None = None
    fx_to_base: float | None = None
    estimated_cost_base: float | None = None
    requested_notional_base: float | None = None
    tradable_quantity: float | None = None
    cash_before: float | None = None
    cash_reserved: float | None = None
    cash_available: float | None = None


def entry_candidates(
    ranking_day: pd.DataFrame,
    portfolio: Portfolio,
    cfg: SimulationConfig,
    *,
    countries: set[str] | None = None,
) -> pd.DataFrame:
    """Select new entries: top candidates, not held, respecting portfolio caps."""
    cands = select_candidates(
        ranking_day,
        mode=cfg.candidate_mode,
        top_percentile=cfg.top_percentile,
        top_n=cfg.top_n,
    )
    if countries is not None:
        cands = cands.loc[cands["Country"].isin(countries)].copy()
    if cands.empty:
        return cands

    # Exclude already held / pending handled by caller via portfolio.has_position
    cands = cands.loc[~cands["Symbol"].map(portfolio.has_position)].copy()
    if cands.empty:
        return cands

    # Score descending for priority
    cands = cands.sort_values(["Score", "Rank"], ascending=[False, True])

    free_slots = cfg.max_positions - len(portfolio.positions)
    if free_slots <= 0:
        return cands.iloc[0:0].copy()

    equity = portfolio.total_equity
    selected_rows: list[pd.Series] = []
    # Tentative country exposure using current MTM
    country_mv = {
        c: sum(p.market_value_base for p in portfolio.positions.values() if p.country == c)
        for c in cands["Country"].unique()
    }

    for _, row in cands.iterrows():
        if len(selected_rows) >= free_slots:
            break
        country = str(row["Country"])
        # Equal-weight target notional per new name (and remaining slots)
        remaining = free_slots - len(selected_rows)
        target = min(
            equity * cfg.max_position_weight,
            equity / max(cfg.max_positions, 1),
        )
        # Also don't exceed equal split of free cash-ish capacity
        target = min(target, equity * cfg.max_position_weight)
        projected_country = country_mv.get(country, 0.0) + target
        if equity > 0 and projected_country / equity > cfg.max_country_weight + 1e-12:
            continue
        selected_rows.append(row)
        country_mv[country] = projected_country

    if not selected_rows:
        return cands.iloc[0:0].copy()
    return pd.DataFrame(selected_rows)


def ranked_entry_universe(
    ranking_day: pd.DataFrame,
    portfolio: Portfolio,
    cfg: SimulationConfig,
    *,
    countries: set[str] | None = None,
) -> pd.DataFrame:
    """Ranked candidates for affordability walk (no equal-weight pre-cut)."""
    cands = select_candidates(
        ranking_day,
        mode=cfg.candidate_mode,
        top_percentile=cfg.top_percentile,
        top_n=cfg.top_n,
    )
    if countries is not None and not cands.empty:
        cands = cands.loc[cands["Country"].isin(countries)].copy()
    if cands.empty:
        return cands
    cands = cands.loc[~cands["Symbol"].map(portfolio.has_position)].copy()
    if cands.empty:
        return cands
    return cands.sort_values(["Score", "Rank"], ascending=[False, True])


def plan_integer_affordable_entries(
    ranking_day: pd.DataFrame,
    portfolio: Portfolio,
    cfg: SimulationConfig,
    *,
    costs: CostModel,
    fx: FxConverter,
    signal_day: pd.Timestamp,
    close_prices: dict[str, float],
    meta_map: dict[str, dict[str, Any]],
    pending_buy_symbols: set[str],
    cooldown_until: dict[str, pd.Timestamp],
    countries: set[str] | None = None,
    remaining_cash: float | None = None,
    cash_reserved_before: float = 0.0,
) -> tuple[list[AffordableEntryPlan], list[EntrySkip]]:
    """Walk ranking order; allocate integer lots under cash / slots / limits.

    ``remaining_cash`` is available cash after prior BUY reservations
    (``portfolio.cash - cash_reserved_before``). Same-day QUEUED plans further
    reserve by decrementing the local available balance.
    """
    plans: list[AffordableEntryPlan] = []
    skips: list[EntrySkip] = []
    policy = cfg.execution_policy
    equity = portfolio.total_equity
    cash_before_day = float(portfolio.cash)
    reserved_base = float(cash_reserved_before)
    cash_left = float(
        portfolio.cash - reserved_base if remaining_cash is None else remaining_cash
    )
    free_slots = cfg.max_positions - len(portfolio.positions) - len(pending_buy_symbols)
    if free_slots <= 0:
        return plans, skips

    country_mv = {
        c: sum(p.market_value_base for p in portfolio.positions.values() if p.country == c)
        for c in {p.country for p in portfolio.positions.values()}
    }

    universe = ranked_entry_universe(
        ranking_day, portfolio, cfg, countries=countries
    )
    if universe.empty:
        return plans, skips

    def _score() -> float | None:
        try:
            return float(row["Score"])
        except Exception:  # noqa: BLE001
            return None

    def _skip(
        reason: str,
        *,
        country: str | None = None,
        currency: str | None = None,
        price: float | None = None,
        fx_rate: float | None = None,
        cost: float | None = None,
        notional: float | None = None,
        qty: float | None = None,
    ) -> None:
        reserved_now = reserved_base + sum(p.estimated_debit_base for p in plans)
        skips.append(
            EntrySkip(
                symbol=symbol,
                reason=reason,
                rank=rank,
                score=_score(),
                country=country,
                currency=currency,
                estimated_price_local=price,
                fx_to_base=fx_rate,
                estimated_cost_base=cost,
                requested_notional_base=notional,
                tradable_quantity=qty,
                cash_before=cash_before_day,
                cash_reserved=reserved_now,
                cash_available=cash_left,
            )
        )

    for _, row in universe.iterrows():
        if len(plans) >= free_slots:
            break
        symbol = str(row["Symbol"])
        rank = int(row["Rank"]) if "Rank" in row and pd.notna(row["Rank"]) else None
        if symbol in pending_buy_symbols:
            _skip(REASON_PENDING_BUY)
            continue
        if portfolio.has_position(symbol):
            _skip(REASON_ALREADY_HELD)
            continue
        until = cooldown_until.get(symbol)
        if until is not None and signal_day <= until:
            _skip(REASON_COOLDOWN)
            continue
        meta = meta_map.get(symbol)
        if meta is None:
            _skip(REASON_INVALID_PRICE)
            continue
        country = str(meta["Country"])
        currency = str(meta["Currency"])
        raw_px = close_prices.get(symbol)
        if raw_px is None or not pd.notna(raw_px) or float(raw_px) <= 0:
            _skip(REASON_INVALID_PRICE, country=country, currency=currency)
            continue
        try:
            fx_rate = fx.rate_to_base(currency, signal_day)
        except Exception:  # noqa: BLE001 — treat as invalid FX for this name
            _skip(REASON_INVALID_FX, country=country, currency=currency, price=float(raw_px))
            continue
        if fx_rate <= 0:
            _skip(REASON_INVALID_FX, country=country, currency=currency, price=float(raw_px))
            continue

        unit = buy_unit_cost_base(
            raw_price=float(raw_px), fx_rate=fx_rate, costs=costs
        )
        weight_budget = equity * cfg.max_position_weight if equity > 0 else 0.0
        remaining_slots = free_slots - len(plans)
        # Soft diversification: do not dump all cash into rank-1 by default.
        slot_budget = cash_left / max(remaining_slots, 1)
        if weight_budget > 0:
            budget_cap = min(weight_budget, slot_budget)
        else:
            budget_cap = slot_budget

        shares, reason = max_integer_shares_for_cash(
            cash=cash_left,
            raw_price=float(raw_px),
            fx_rate=fx_rate,
            costs=costs,
            minimum_quantity=policy.minimum_quantity,
            budget_cap=budget_cap,
        )
        if shares <= 0 and policy.min_lot_overrides_weight:
            shares2, reason2 = max_integer_shares_for_cash(
                cash=cash_left,
                raw_price=float(raw_px),
                fx_rate=fx_rate,
                costs=costs,
                minimum_quantity=policy.minimum_quantity,
                budget_cap=None,
            )
            min_q = max(1, int(policy.minimum_quantity))
            if shares2 >= min_q:
                shares = min_q
                reason = None
            else:
                reason = reason2 or reason or REASON_NOT_AFFORDABLE
                shares = 0

        if shares <= 0:
            # Distinguish pure unaffordable vs reservation-blocked.
            if unit > cash_left + 1e-6 and unit <= cash_before_day + 1e-6:
                reason = REASON_INSUFFICIENT_CASH
            elif reason is None:
                reason = REASON_MIN_QUANTITY_NOT_MET
            _skip(
                reason,
                country=country,
                currency=currency,
                price=float(raw_px),
                fx_rate=float(fx_rate),
                cost=float(unit),
                qty=0.0,
            )
            continue

        notional = target_notional_for_shares(
            shares=shares, raw_price=float(raw_px), fx_rate=fx_rate, costs=costs
        )
        debit = entry_debit_base(
            quantity=float(shares),
            raw_price=float(raw_px),
            fx_rate=fx_rate,
            costs=costs,
        )
        if debit > cash_left + 1e-6:
            reason = (
                REASON_INSUFFICIENT_CASH
                if debit <= cash_before_day + 1e-6
                else REASON_NOT_AFFORDABLE
            )
            _skip(
                reason,
                country=country,
                currency=currency,
                price=float(raw_px),
                fx_rate=float(fx_rate),
                cost=float(debit),
                notional=float(notional),
                qty=float(shares),
            )
            continue

        projected_country = country_mv.get(country, 0.0) + notional
        if equity > 0 and projected_country / equity > cfg.max_country_weight + 1e-12:
            _skip(
                REASON_COUNTRY_LIMIT,
                country=country,
                currency=currency,
                price=float(raw_px),
                fx_rate=float(fx_rate),
                cost=float(debit),
                notional=float(notional),
                qty=float(shares),
            )
            continue

        if len(portfolio.positions) + len(pending_buy_symbols) + len(plans) >= cfg.max_positions:
            _skip(
                REASON_POSITION_LIMIT,
                country=country,
                currency=currency,
                price=float(raw_px),
                fx_rate=float(fx_rate),
                cost=float(debit),
                notional=float(notional),
                qty=float(shares),
            )
            break

        reserved_now = reserved_base + sum(p.estimated_debit_base for p in plans)
        plans.append(
            AffordableEntryPlan(
                symbol=symbol,
                country=country,
                currency=currency,
                shares=int(shares),
                target_notional_base=float(notional),
                estimated_debit_base=float(debit),
                score=float(row["Score"]),
                rank=int(row["Rank"]),
                percentile=float(row["Percentile"]),
                estimated_price_local=float(raw_px),
                fx_to_base=float(fx_rate),
                cash_before=cash_before_day,
                cash_reserved=float(reserved_now),
                cash_available=float(cash_left),
            )
        )
        cash_left -= debit
        country_mv[country] = projected_country

    return plans, skips



def evaluate_exits_for_day(
    portfolio: Portfolio,
    *,
    day: pd.Timestamp,
    high: dict[str, float],
    low: dict[str, float],
    ranking_day: pd.DataFrame | None,
    cfg: SimulationConfig,
) -> list[ExitSignal]:
    """Decide exits at day-T close (fills happen next open)."""
    signals: list[ExitSignal] = []
    rank_map: dict[str, float] = {}
    if ranking_day is not None and not ranking_day.empty:
        rank_map = {
            str(r.Symbol): float(r.Percentile)
            for r in ranking_day.itertuples(index=False)
        }

    for symbol, pos in list(portfolio.positions.items()):
        if pos.pending_exit_reason:
            continue

        # Update trailing peak using today's High (close-time info only; no future)
        if symbol in high:
            hi = float(high[symbol])
            if hi > float(pos.peak_price or 0.0):
                pos.peak_price = hi
            if cfg.trailing_stop_pct is not None and cfg.trailing_stop_pct > 0:
                pos.trailing_stop_price = float(pos.peak_price * (1.0 - cfg.trailing_stop_pct))

        # Fixed stop / trailing / take using day's High/Low (conservative fills later)
        if (
            cfg.stop_loss_pct is not None
            and pos.stop_price is not None
            and symbol in low
            and float(low[symbol]) <= float(pos.stop_price)
        ):
            signals.append(ExitSignal(symbol, "stop_loss", pos.stop_price))
            continue
        if (
            cfg.trailing_stop_pct is not None
            and pos.trailing_stop_price is not None
            and symbol in low
            and float(low[symbol]) <= float(pos.trailing_stop_price)
        ):
            signals.append(ExitSignal(symbol, "trailing_stop", pos.trailing_stop_price))
            continue
        if (
            cfg.take_profit_pct is not None
            and pos.take_profit_price is not None
            and symbol in high
            and float(high[symbol]) >= float(pos.take_profit_price)
        ):
            signals.append(ExitSignal(symbol, "take_profit", pos.take_profit_price))
            continue

        # Holding period: exit when held >= N trading days (entry day counts as 1 after bump)
        if pos.holding_days >= cfg.holding_period_days:
            signals.append(ExitSignal(symbol, "holding_period", None))
            continue

        if cfg.ranking_exit_enabled and symbol in rank_map:
            if rank_map[symbol] > cfg.ranking_exit_percentile:
                signals.append(ExitSignal(symbol, "ranking_exit", None))
                continue

    return signals


def equal_weight_notional(portfolio: Portfolio, cfg: SimulationConfig, n_new: int) -> float:
    """Target base-currency notional for each new equal-weight entry."""
    if n_new <= 0:
        return 0.0
    equity = portfolio.total_equity
    per = equity / max(cfg.max_positions, 1)
    per = min(per, equity * cfg.max_position_weight)
    return float(max(per, 0.0))
