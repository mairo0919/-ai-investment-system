"""Server-side SVG charts for Dashboard (no JS / no CDN)."""

from __future__ import annotations

import math
from typing import Any, Sequence

from src.dashboard.formatters import esc


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _scale_y(values: Sequence[float], *, height: float, pad: float = 8.0) -> tuple[float, float, list[float]]:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        # Flat series: invent a tiny band so the line is visible mid-chart.
        mid = lo
        span = max(abs(mid) * 0.01, 1.0)
        lo = mid - span
        hi = mid + span
    usable = height - 2 * pad

    def y(v: float) -> float:
        return pad + (hi - v) / (hi - lo) * usable

    return lo, hi, [y(v) for v in values]


def render_line_chart(
    points: Sequence[dict[str, Any]],
    *,
    y_key: str,
    initial_capital: float | None = None,
    width: int = 720,
    height: int = 280,
    stroke: str = "#5b8def",
    show_zero: bool = False,
    empty_message: str = "データがありません",
) -> str:
    """Render a responsive SVG line chart. Returns safe HTML/SVG markup."""
    series: list[tuple[str, float]] = []
    for p in points:
        y = _finite(p.get(y_key))
        if y is None:
            continue
        date = p.get("date")
        label = str(date) if date is not None else ""
        series.append((label, y))

    if not series:
        return (
            f'<div class="chart-empty">{esc(empty_message)}</div>'
        )

    pad_l, pad_r, pad_t, pad_b = 48.0, 16.0, 16.0, 36.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    ys = [v for _, v in series]

    scale_vals = list(ys)
    if initial_capital is not None and _finite(initial_capital) is not None:
        scale_vals.append(float(initial_capital))
    if show_zero:
        scale_vals.append(0.0)
    lo, hi, _ = _scale_y(scale_vals, height=plot_h, pad=0)

    def y_at(v: float) -> float:
        return pad_t + (hi - v) / (hi - lo) * plot_h

    n = len(series)
    if n == 1:
        x_coords = [pad_l + plot_w / 2.0]
    else:
        x_coords = [pad_l + i * plot_w / (n - 1) for i in range(n)]
    y_coords = [y_at(v) for _, v in series]

    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{esc(y_key)}" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#24314d" stroke-width="1"/>'
    )

    if initial_capital is not None and _finite(initial_capital) is not None:
        y_ref = y_at(float(initial_capital))
        parts.append(
            f'<line x1="{pad_l}" y1="{y_ref:.2f}" x2="{pad_l + plot_w}" y2="{y_ref:.2f}" '
            f'stroke="#93a0b8" stroke-width="1" stroke-dasharray="4 4"/>'
        )
        parts.append(
            f'<text x="{pad_l + 4}" y="{max(pad_t + 10, y_ref - 4):.2f}" fill="#93a0b8" '
            f'font-size="10">初期資金</text>'
        )

    if show_zero and lo < 0 < hi:
        y0 = y_at(0.0)
        parts.append(
            f'<line x1="{pad_l}" y1="{y0:.2f}" x2="{pad_l + plot_w}" y2="{y0:.2f}" '
            f'stroke="#3a5f9e" stroke-width="1" stroke-dasharray="2 3"/>'
        )

    if n == 1:
        parts.append(
            f'<circle cx="{x_coords[0]:.2f}" cy="{y_coords[0]:.2f}" r="4" fill="{esc(stroke)}"/>'
        )
    else:
        poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(x_coords, y_coords))
        parts.append(
            f'<polyline fill="none" stroke="{esc(stroke)}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{poly}"/>'
        )

    first_d, _first_v = series[0]
    last_d, _last_v = series[-1]
    parts.append(
        f'<text x="{pad_l}" y="{height - 12}" fill="#93a0b8" font-size="10">{esc(first_d)}</text>'
    )
    parts.append(
        f'<text x="{pad_l + plot_w}" y="{height - 12}" fill="#93a0b8" font-size="10" '
        f'text-anchor="end">{esc(last_d)}</text>'
    )
    parts.append(
        f'<text x="{pad_l - 6}" y="{y_at(hi) + 4:.2f}" fill="#93a0b8" font-size="10" '
        f'text-anchor="end">{esc(f"{hi:,.0f}")}</text>'
    )
    parts.append(
        f'<text x="{pad_l - 6}" y="{y_at(lo) + 4:.2f}" fill="#93a0b8" font-size="10" '
        f'text-anchor="end">{esc(f"{lo:,.0f}")}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def render_equity_chart(
    points: Sequence[dict[str, Any]],
    *,
    initial_capital: float | None,
    mini: bool = False,
) -> str:
    return render_line_chart(
        points,
        y_key="total_equity",
        initial_capital=initial_capital,
        width=360 if mini else 720,
        height=140 if mini else 280,
        stroke="#5b8def",
        empty_message="総資産の履歴がありません",
    )


def render_pnl_chart(points: Sequence[dict[str, Any]]) -> str:
    return render_line_chart(
        points,
        y_key="cumulative_pnl",
        initial_capital=None,
        show_zero=True,
        stroke="#6bcf8e",
        empty_message="損益推移データがありません",
    )


def render_drawdown_chart(points: Sequence[dict[str, Any]]) -> str:
    return render_line_chart(
        points,
        y_key="drawdown",
        initial_capital=None,
        show_zero=True,
        stroke="#e35d6a",
        empty_message="ドローダウン履歴がありません",
    )
