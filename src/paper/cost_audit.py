"""Cost accounting audit: path dependence vs fixed-quantity cost isolation."""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_costs_fixed_quantity(
    trades: pd.DataFrame,
    *,
    commission_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    """Recompute net PnL with fixed quantities/prices — isolates cost effects.

    Same trade paths must satisfy ZERO >= LOW >= CURRENT >= HIGH net PnL
    when only commission/slippage rates increase (monotonicity).
    """
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "commission": 0.0,
            "slippage_impact": 0.0,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
        }
    tr = trades.copy()
    qty = tr["Quantity"].astype(float)
    if "Entry Price Raw" in tr.columns:
        entry_raw = tr["Entry Price Raw"].astype(float)
    else:
        entry_raw = tr["Entry Price"].astype(float)
    if "Exit Price Raw" in tr.columns:
        exit_raw = tr["Exit Price Raw"].astype(float)
    else:
        exit_raw = tr["Exit Price"].astype(float)
    fx_e = tr["Fx Entry"].astype(float) if "Fx Entry" in tr.columns else 1.0
    fx_x = tr["Fx Exit"].astype(float) if "Fx Exit" in tr.columns else 1.0

    buy_px = entry_raw * (1.0 + slippage_rate)
    sell_px = exit_raw * (1.0 - slippage_rate)
    entry_notional = qty * buy_px * fx_e
    exit_notional = qty * sell_px * fx_x
    entry_mid = qty * entry_raw * fx_e
    exit_mid = qty * exit_raw * fx_x
    commission = commission_rate * (entry_notional.abs() + exit_notional.abs())
    slip = (entry_notional - entry_mid) + (exit_mid - exit_notional)
    gross = exit_mid - entry_mid
    net = exit_notional - entry_notional - commission
    return {
        "n_trades": int(len(tr)),
        "gross_pnl": float(gross.sum()),
        "net_pnl": float(net.sum()),
        "commission": float(commission.sum()),
        "slippage_impact": float(slip.sum()),
        "commission_rate": commission_rate,
        "slippage_rate": slippage_rate,
    }


def cost_monotonicity_report(trades: pd.DataFrame) -> dict[str, Any]:
    """Compare fixed-qty nets across cost ladders; assert ordering when possible."""
    ladders = {
        "zero": {"commission_rate": 0.0, "slippage_rate": 0.0},
        "low": {"commission_rate": 0.0005, "slippage_rate": 0.00025},
        "current": {"commission_rate": 0.001, "slippage_rate": 0.0005},
        "high": {"commission_rate": 0.0015, "slippage_rate": 0.001},
    }
    results = {name: apply_costs_fixed_quantity(trades, **rates) for name, rates in ladders.items()}
    nets = [results[k]["net_pnl"] for k in ("zero", "low", "current", "high")]
    mono = all(nets[i] + 1e-9 >= nets[i + 1] for i in range(len(nets) - 1))
    return {
        "policy": (
            "Fixed quantities/prices isolate cost effects. "
            "Full simulation path-dependence (cost → size → later trades) "
            "can break return ordering; this audit does not change strategy."
        ),
        "ladders": results,
        "net_order_zero_ge_low_ge_current_ge_high": mono,
        "nets": {
            "zero": nets[0],
            "low": nets[1],
            "current": nets[2],
            "high": nets[3],
        },
    }


def path_dependence_note() -> dict[str, str]:
    return {
        "mechanism": (
            "Higher costs reduce cash after each fill → equal-weight notional shrinks → "
            "subsequent trade quantities change → equity path diverges. "
            "Therefore HIGH cost simulations can occasionally finish with higher "
            "total return than CURRENT even though per-share costs are worse."
        ),
        "diagnostic": (
            "Use apply_costs_fixed_quantity / cost_monotonicity_report for pure cost comparison."
        ),
    }
