"""Entry / exit rule engine for paper trading."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.simulation.config import SimulationConfig
from src.simulation.portfolio import Portfolio
from src.simulation.ranking import select_candidates


@dataclass(frozen=True)
class ExitSignal:
    symbol: str
    reason: str
    stop_or_take_price: float | None = None


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
