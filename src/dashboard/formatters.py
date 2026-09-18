"""Display formatters for Dashboard HTML (no I/O)."""

from __future__ import annotations

import html
import math
from typing import Any


def esc(value: Any) -> str:
    """HTML-escape any value for safe embedding in markup."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def dash(value: Any) -> str:
    """Render None / empty as an em dash (already safe literal)."""
    if value is None:
        return "—"
    text = str(value).strip()
    return "—" if not text else esc(text)


def _currency_symbol(currency: str | None) -> str:
    c = (currency or "").upper()
    if c == "JPY":
        return "¥"
    if c == "USD":
        return "$"
    if c == "EUR":
        return "€"
    return f"{c} " if c else ""


def format_money(
    value: float | int | None,
    *,
    currency: str | None,
    signed: bool = False,
) -> str:
    if value is None:
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    c = (currency or "").upper()
    abs_num = abs(num)
    if c == "JPY":
        body = f"{abs_num:,.0f}"
    else:
        body = f"{abs_num:,.2f}"
    sym = _currency_symbol(currency)
    if signed:
        if num > 0:
            return esc(f"+{sym}{body}")
        if num < 0:
            return esc(f"-{sym}{body}")
        return esc(f"{sym}{body}")
    if num < 0:
        return esc(f"-{sym}{body}")
    return esc(f"{sym}{body}")


def format_percent(value: float | int | None, *, signed: bool = True) -> str:
    if value is None:
        return "—"
    try:
        num = float(value) * 100.0
    except (TypeError, ValueError):
        return "—"
    if signed:
        return esc(f"{num:+.2f}%")
    return esc(f"{num:.2f}%")


def format_qty(value: float | int | None) -> str:
    if value is None:
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(num - round(num)) < 1e-9:
        return esc(f"{int(round(num)):,}")
    return esc(f"{num:,.4f}")


def format_price(value: float | int | None, *, currency: str | None = None) -> str:
    """Local / unit price — not necessarily base currency notionals."""
    if value is None:
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    c = (currency or "").upper()
    if c == "JPY":
        return esc(f"{num:,.2f}")
    return esc(f"{num:,.4f}" if abs(num) < 1 else f"{num:,.2f}")


def pnl_class(value: float | int | None) -> str:
    """CSS hint class; sign text remains the primary signal."""
    if value is None:
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if num > 0:
        return "pnl-pos"
    if num < 0:
        return "pnl-neg"
    return ""


_STATUS_LABELS_JA: dict[str, str] = {
    "SUCCESS": "正常",
    "SKIPPED": "スキップ",
    "ERROR": "エラー",
    "RUNNING": "実行中",
    "OK": "正常",
    "HOLD_INSUFFICIENT_DATA": "データ不足・検証継続中",
    "CONTINUE": "検証継続",
    "MODEL_REVIEW_REQUIRED": "モデル要確認",
}


def format_number(value: float | int | None, *, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(num):
        return "—"
    return esc(f"{num:.{digits}f}")


def format_ratio(value: float | int | None, *, digits: int = 2) -> str:
    """Sharpe / profit factor — no percent sign."""
    if value is None:
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(num):
        return "—"
    return esc(f"{num:.{digits}f}")


def status_label(status: str | None) -> str | None:
    """Map internal status codes to Japanese UI labels (values unchanged at source)."""
    if status is None:
        return None
    text = str(status).strip()
    if not text:
        return None
    return _STATUS_LABELS_JA.get(text.upper(), text)


def status_badge_class(status: str | None) -> str:
    if not status:
        return "badge-muted"
    s = status.upper()
    if s in ("SUCCESS", "CONTINUE", "OK", "RUNNING"):
        return "badge-ok"
    if s in ("SKIPPED", "HOLD_INSUFFICIENT_DATA"):
        return "badge-warn"
    if s in ("ERROR", "MODEL_REVIEW_REQUIRED"):
        return "badge-bad"
    return "badge-muted"
