"""Phase 5B: Paper Validity & Monitoring Gate (observe only; no trading / retrain).

Decision states:
  HOLD_INSUFFICIENT_DATA — sample or required benchmark inputs insufficient
  CONTINUE — sample OK and no review triggers (not a proof of model quality)
  MODEL_REVIEW_REQUIRED — human review requested; never auto-retrains

Thresholds reuse Phase 4D ``success_checklist`` and Paper ``performance_summary``
sample rules. Combination rule for weak failures is documented as new policy.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.paper.diagnostics import performance_summary
from src.paper.lineage import LOCKED_PAPER_MODEL_ID
from src.paper.state import PaperStore
from src.simulation.holdout_diagnostics import (
    monthly_performance,
    success_checklist,
    symbol_contribution,
)
from src.simulation.quality import is_entry_rebalance_day

STATUS_HOLD = "HOLD_INSUFFICIENT_DATA"
STATUS_CONTINUE = "CONTINUE"
STATUS_REVIEW = "MODEL_REVIEW_REQUIRED"

# --- Existing sample gates (src/paper/diagnostics.py:performance_summary) ---
MIN_COMPLETED_TRADES = 30
MIN_RETURN_DAYS = 60  # len(equity pct_change)

# --- Existing success_checklist thresholds (src/simulation/holdout_diagnostics.py) ---
# Used inverted for review triggers (same numeric cutoffs as holdout).

# NEW policy (not in holdout): how many weak inverted-checklist fails → REVIEW
WEAK_FAIL_COUNT_FOR_REVIEW = 3

DEFAULT_GATE_CONFIG = Path("config/paper_validity_gate.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_gate_config(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_GATE_CONFIG
    if not path.exists():
        return {
            "min_completed_trades": MIN_COMPLETED_TRADES,
            "min_return_days": MIN_RETURN_DAYS,
            "weak_fail_count_for_review": WEAK_FAIL_COUNT_FOR_REVIEW,
            "require_benchmark": True,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    # Normalize common column aliases for contribution helpers
    rename = {}
    if "Symbol" not in df.columns and "symbol" in df.columns:
        rename["symbol"] = "Symbol"
    if "Net PnL" not in df.columns and "net_pnl" in df.columns:
        rename["net_pnl"] = "Net PnL"
    if "Return" not in df.columns and "return" in df.columns:
        rename["return"] = "Return"
    return df.rename(columns=rename) if rename else df


def _count_rebalances(equity: pd.DataFrame, *, frequency: str) -> int:
    if equity is None or equity.empty or "Date" not in equity.columns:
        return 0
    calendar = sorted(pd.to_datetime(equity["Date"]).dt.normalize().unique())
    cal_list = [pd.Timestamp(d) for d in calendar]
    n = 0
    for d in cal_list:
        if is_entry_rebalance_day(d, cal_list, frequency=frequency):
            n += 1
    return int(n)


def _read_optional_total_return(path: Path) -> tuple[float | None, str]:
    """Return (total_return, status) where status is ok|missing|invalid."""
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "invalid"
    val = payload.get("total_return")
    if val is None:
        return None, "missing"
    try:
        return float(val), "ok"
    except (TypeError, ValueError):
        return None, "invalid"


def evaluate_validity_gate(
    *,
    store: PaperStore,
    initial_capital: float,
    model_id: str,
    asof: pd.Timestamp | str | None,
    rebalance_frequency: str = "biweekly",
    benchmark_track_path: Path | None = None,
    momentum_state_path: Path | None = None,
    gate_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic gate from PaperStore artifacts (+ optional bench JSON paths)."""
    cfg = gate_config or load_gate_config()
    min_trades = int(cfg.get("min_completed_trades", MIN_COMPLETED_TRADES))
    min_days = int(cfg.get("min_return_days", MIN_RETURN_DAYS))
    weak_need = int(cfg.get("weak_fail_count_for_review", WEAK_FAIL_COUNT_FOR_REVIEW))
    require_benchmark = bool(cfg.get("require_benchmark", True))

    equity = (
        pd.read_csv(store.equity_history_path)
        if store.equity_history_path.exists()
        else pd.DataFrame()
    )
    trades = _load_trades(store.trade_history_path)
    port = store.load()
    asof_s = (
        str(pd.Timestamp(asof).date())
        if asof is not None
        else (port.last_processed_date if port else None)
    )

    n_trades = int(len(trades)) if trades is not None and not trades.empty else 0
    rets = (
        equity["Total Equity"].astype(float).pct_change().dropna()
        if equity is not None and not equity.empty and "Total Equity" in equity.columns
        else pd.Series(dtype=float)
    )
    n_return_days = int(len(rets))
    n_equity_rows = int(len(equity)) if equity is not None else 0
    rebalance_count = _count_rebalances(equity, frequency=rebalance_frequency)

    sample = {
        "trading_days": n_return_days,
        "equity_rows": n_equity_rows,
        "completed_trades": n_trades,
        "rebalance_count": rebalance_count,
        "min_completed_trades": min_trades,
        "min_return_days": min_days,
        "source": "performance_summary.statistically_insufficient",
    }

    checks: list[dict[str, Any]] = []
    review_reasons: list[str] = []

    sample_ok = n_trades >= min_trades and n_return_days >= min_days
    checks.append(
        {
            "id": "sample_sufficient",
            "passed": sample_ok,
            "detail": {
                "completed_trades": n_trades,
                "return_days": n_return_days,
                "required_trades": min_trades,
                "required_return_days": min_days,
            },
            "source": "src/paper/diagnostics.py:performance_summary",
        }
    )
    if not sample_ok:
        hold_reasons: list[str] = []
        if n_return_days < min_days:
            hold_reasons.append("insufficient_trading_days")
        if n_trades < min_trades:
            hold_reasons.append("insufficient_completed_trades")
        return _finalize(
            status=STATUS_HOLD,
            model_id=model_id,
            asof=asof_s,
            sample=sample,
            metrics={},
            benchmark={"status": "not_evaluated"},
            checks=checks,
            review_reasons=hold_reasons,
            checklist=None,
        )

    bench_path = benchmark_track_path
    mom_path = momentum_state_path
    if bench_path is None:
        bench_ret, bench_status = None, "missing"
    else:
        bench_ret, bench_status = _read_optional_total_return(bench_path)
    if mom_path is None:
        mom_ret, mom_status = None, "missing"
    else:
        mom_ret, mom_status = _read_optional_total_return(mom_path)

    benchmark_block = {
        "spx_status": bench_status,
        "spx_total_return": bench_ret,
        "momentum_status": mom_status,
        "momentum_total_return": mom_ret,
    }
    checks.append(
        {
            "id": "benchmark_spx_available",
            "passed": bench_status == "ok",
            "detail": {"status": bench_status},
            "source": "reports/paper/benchmark_track.json (existing paper runner output)",
        }
    )

    if require_benchmark and bench_status != "ok":
        checks.append(
            {
                "id": "benchmark_required_for_decision",
                "passed": False,
                "detail": {
                    "reason": "SPX benchmark total_return unavailable; refuse CONTINUE",
                },
                "source": "Phase 5B policy (no silent 0% / no ignore)",
            }
        )
        return _finalize(
            status=STATUS_HOLD,
            model_id=model_id,
            asof=asof_s,
            sample=sample,
            metrics={},
            benchmark=benchmark_block,
            checks=checks,
            review_reasons=["benchmark_unavailable"],
            checklist=None,
        )

    perf = performance_summary(
        equity,
        trades,
        initial_capital=initial_capital,
        benchmark_return=bench_ret,
        momentum_return=mom_ret if mom_status == "ok" else None,
    )
    # Strip insufficient flag for gate metrics copy (we already gated sample)
    metrics = {
        k: v
        for k, v in perf.items()
        if k
        not in (
            "statistically_insufficient",
            "note",
        )
    }

    top1_share = None
    if n_trades > 0 and "Net PnL" in trades.columns and "Symbol" in trades.columns:
        sym = symbol_contribution(trades, initial_capital=initial_capital)
        top1_share = (sym.get("top_dependency") or {}).get("top1", {}).get("pnl_share")

    pos_month = None
    if n_equity_rows > 0:
        monthly = monthly_performance(equity)
        if not monthly.empty and "strategy_return" in monthly.columns:
            pos_month = float((monthly["strategy_return"] > 0).mean())

    # Sector share: not available from paper trade CSV alone (no sector column).
    top_sector_share = None
    worst_mae = None  # requires path OHLC; not in paper trade CSV — leave unevaluated

    checklist = success_checklist(
        net_return=metrics.get("total_return"),
        sharpe=metrics.get("sharpe"),
        benchmark_excess=metrics.get("benchmark_excess"),
        momentum_excess=metrics.get("momentum_excess"),
        profit_factor=metrics.get("profit_factor"),
        top1_share=top1_share,
        top_sector_share=top_sector_share,
        positive_month_ratio=pos_month,
        worst_mae=worst_mae,
    )

    # Major triggers (EXISTING checklist F / G / I) — single fail → REVIEW
    major: list[tuple[str, bool | None, str]] = [
        (
            "F_single_symbol_dependency",
            (top1_share is not None and top1_share >= 0.50),
            "success_checklist F inverted (top1 pnl share < 50%)",
        ),
        (
            "G_single_sector_dependency",
            (top_sector_share is not None and top_sector_share >= 0.60),
            "success_checklist G inverted (top sector share < 60%)",
        ),
        (
            "I_extreme_tail_mae",
            (worst_mae is not None and worst_mae <= -0.40),
            "success_checklist I inverted (worst MAE > -40%)",
        ),
    ]
    for cid, fired, src in major:
        evaluated = fired is not False and (
            (cid.startswith("F") and top1_share is not None)
            or (cid.startswith("G") and top_sector_share is not None)
            or (cid.startswith("I") and worst_mae is not None)
        )
        checks.append(
            {
                "id": cid,
                "passed": not bool(fired) if evaluated else None,
                "evaluated": evaluated,
                "detail": {"fired": bool(fired) if evaluated else None},
                "source": src,
                "severity": "major",
            }
        )
        if evaluated and fired:
            review_reasons.append(cid)

    # Weak triggers (EXISTING checklist A/B/C/D/E/H inverted)
    weak_defs = [
        (
            "A_net_return_non_positive",
            metrics.get("total_return") is not None and float(metrics["total_return"]) <= 0,
            metrics.get("total_return") is not None,
        ),
        (
            "B_sharpe_non_positive",
            metrics.get("sharpe") is not None and float(metrics["sharpe"]) <= 0,
            metrics.get("sharpe") is not None,
        ),
        (
            "C_benchmark_excess_non_positive",
            metrics.get("benchmark_excess") is not None
            and float(metrics["benchmark_excess"]) <= 0,
            metrics.get("benchmark_excess") is not None,
        ),
        (
            "D_momentum_excess_non_positive",
            metrics.get("momentum_excess") is not None
            and float(metrics["momentum_excess"]) <= 0,
            metrics.get("momentum_excess") is not None,
        ),
        (
            "E_profit_factor_le_1",
            metrics.get("profit_factor") is not None
            and metrics.get("profit_factor") != float("inf")
            and float(metrics["profit_factor"]) <= 1.0,
            metrics.get("profit_factor") is not None,
        ),
        (
            "H_positive_month_ratio_low",
            pos_month is not None and pos_month < 0.40,
            pos_month is not None,
        ),
    ]
    weak_fails = 0
    weak_evaluated = 0
    for cid, fired, evaluated in weak_defs:
        if evaluated:
            weak_evaluated += 1
            if fired:
                weak_fails += 1
        checks.append(
            {
                "id": cid,
                "passed": (not fired) if evaluated else None,
                "evaluated": evaluated,
                "detail": {"fired": fired if evaluated else None},
                "source": "success_checklist inverted (holdout_diagnostics)",
                "severity": "weak",
            }
        )

    checks.append(
        {
            "id": "weak_fail_aggregation",
            "passed": weak_fails < weak_need,
            "detail": {
                "weak_fails": weak_fails,
                "weak_evaluated": weak_evaluated,
                "threshold": weak_need,
            },
            "source": (
                "NEW policy: combine weak inverted-checklist fails; "
                "holdout reports items individually without aggregation rule"
            ),
            "severity": "policy",
        }
    )
    if weak_fails >= weak_need:
        review_reasons.append(f"weak_fail_count_{weak_fails}_ge_{weak_need}")

    ranking_note = {
        "ic_ndcg_precision_at_k": "not_computable_from_paper_state_alone",
        "reason": (
            "Paper stores rankings snapshots optionally under reports/, but not "
            "aligned future returns for IC/NDCG; Phase 5B does not invent look-ahead labels."
        ),
    }

    if review_reasons:
        status = STATUS_REVIEW
    else:
        status = STATUS_CONTINUE

    return _finalize(
        status=status,
        model_id=model_id,
        asof=asof_s,
        sample=sample,
        metrics=metrics,
        benchmark=benchmark_block,
        checks=checks,
        review_reasons=review_reasons,
        checklist=checklist,
        extra={"ranking_monitoring": ranking_note, "success_checklist": checklist},
    )


def _finalize(
    *,
    status: str,
    model_id: str,
    asof: str | None,
    sample: dict[str, Any],
    metrics: dict[str, Any],
    benchmark: dict[str, Any],
    checks: list[dict[str, Any]],
    review_reasons: list[str],
    checklist: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_id": model_id,
        "locked_paper_model_id": LOCKED_PAPER_MODEL_ID,
        "asof": asof,
        "status": status,
        "sample": sample,
        "metrics": metrics,
        "benchmark": benchmark,
        "checks": checks,
        "review_reasons": review_reasons,
        "success_checklist": checklist,
        "auto_retrain": False,
        "auto_halt_trading": False,
        "generated_at_utc": _utc_now(),
    }
    if extra:
        out.update(extra)
    return out


def persist_validity_gate(
    result: dict[str, Any],
    *,
    paper_state_dir: Path,
    reports_paper_dir: Path | None = None,
) -> None:
    """Write latest JSON + asof-deduped history under paper state dir (Volume-safe).

    Also mirrors latest to reports/paper/validity_gate.json when provided (ephemeral OK).
    """
    paper_state_dir.mkdir(parents=True, exist_ok=True)
    latest = paper_state_dir / "validity_gate_latest.json"
    history = paper_state_dir / "validity_gate_history.csv"
    text = json.dumps(result, indent=2, default=str) + "\n"
    latest.write_text(text, encoding="utf-8")

    row = {
        "asof": result.get("asof"),
        "status": result.get("status"),
        "model_id": result.get("model_id"),
        "trading_days": (result.get("sample") or {}).get("trading_days"),
        "completed_trades": (result.get("sample") or {}).get("completed_trades"),
        "rebalance_count": (result.get("sample") or {}).get("rebalance_count"),
        "review_reasons": "|".join(result.get("review_reasons") or []),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    new_df = pd.DataFrame([row])
    if history.exists():
        prev = pd.read_csv(history)
        if "asof" in prev.columns and row["asof"] is not None:
            prev = prev.loc[prev["asof"].astype(str) != str(row["asof"])]
        out = pd.concat([prev, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(history, index=False)

    if reports_paper_dir is not None:
        reports_paper_dir.mkdir(parents=True, exist_ok=True)
        (reports_paper_dir / "validity_gate.json").write_text(text, encoding="utf-8")
