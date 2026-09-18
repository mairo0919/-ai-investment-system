"""Server-rendered HTML shell for Paper Dashboard (CSS only, no CDN)."""

from __future__ import annotations

from typing import Any

from src.dashboard.charts import (
    render_drawdown_chart,
    render_equity_chart,
    render_pnl_chart,
)
from src.dashboard.formatters import (
    dash,
    esc,
    format_money,
    format_percent,
    format_price,
    format_qty,
    format_ratio,
    pnl_class,
    status_badge_class,
    status_label,
)

NAV_ITEMS: tuple[tuple[str, str, bool], ...] = (
    ("概要", "/", True),
    ("運用成績", "/performance", True),
    ("保有銘柄", "#", False),
    ("売買履歴", "#", False),
    ("AIランキング", "#", False),
    ("システム", "#", False),
)

NAV_ACTIVE_OVERVIEW = "概要"
NAV_ACTIVE_PERFORMANCE = "運用成績"

EQUITY_TAIL = 20

CSS = """
:root {
  --bg: #0b1220;
  --bg-elev: #121a2b;
  --bg-card: #162036;
  --border: #24314d;
  --text: #e8eefc;
  --muted: #93a0b8;
  --accent: #5b8def;
  --accent-dim: #3a5f9e;
  --ok: #3dba7a;
  --warn: #d4a017;
  --bad: #e35d6a;
  --pos: #6bcf8e;
  --neg: #f07178;
  --sidebar-w: 240px;
  --radius: 12px;
  --shadow: 0 8px 24px rgba(0,0,0,.35);
  --font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; min-height: 100%;
  background: radial-gradient(1200px 600px at 10% -10%, #1a2744 0%, var(--bg) 55%);
  color: var(--text); font-family: var(--font); font-size: 15px; line-height: 1.45;
}
a { color: inherit; text-decoration: none; }
.layout { display: flex; min-height: 100vh; }
.sidebar {
  width: var(--sidebar-w); flex-shrink: 0;
  background: linear-gradient(180deg, #10182a 0%, #0c1422 100%);
  border-right: 1px solid var(--border);
  padding: 1.25rem 1rem; display: flex; flex-direction: column; gap: 1.5rem;
}
.brand-mark {
  font-size: .7rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
}
.brand-title { font-size: 1.05rem; font-weight: 700; margin-top: .25rem; }
.brand-sub { font-size: .8rem; color: var(--muted); margin-top: .15rem; }
.nav { display: flex; flex-direction: column; gap: .35rem; }
.nav a, .nav span {
  display: block; padding: .65rem .8rem; border-radius: 8px; font-size: .92rem;
}
.nav a.active { background: rgba(91,141,239,.18); color: #cfe0ff; border: 1px solid rgba(91,141,239,.35); }
.nav .soon {
  color: var(--muted); opacity: .7; border: 1px dashed transparent;
  display: flex; justify-content: space-between; align-items: center;
}
.nav .soon em {
  font-style: normal; font-size: .65rem; letter-spacing: .04em;
  text-transform: uppercase; color: var(--accent-dim);
}
.main { flex: 1; min-width: 0; padding: 1.5rem 1.75rem 2.5rem; }
.topbar {
  display: flex; flex-wrap: wrap; gap: 1rem; justify-content: space-between;
  align-items: flex-end; margin-bottom: 1.5rem;
}
.page-title { margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: -.01em; }
.page-sub { margin: .35rem 0 0; color: var(--muted); font-size: .9rem; }
.experiment-chip {
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 999px;
  padding: .45rem .9rem; font-family: var(--mono); font-size: .78rem; color: #c9d7f5;
}
.section { margin-top: 1.75rem; }
.section h2 {
  margin: 0 0 .85rem; font-size: .78rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted); font-weight: 600;
}
.cards {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .9rem;
}
.card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1rem 1.1rem; box-shadow: var(--shadow);
}
.card .label { font-size: .75rem; color: var(--muted); letter-spacing: .04em; text-transform: uppercase; }
.card .value { margin-top: .45rem; font-size: 1.35rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.card .hint { margin-top: .35rem; font-size: .75rem; color: var(--muted); }
.pnl-pos { color: var(--pos); }
.pnl-neg { color: var(--neg); }
.status-grid {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .75rem;
}
.status-item {
  background: var(--bg-elev); border: 1px solid var(--border);
  border-radius: 10px; padding: .85rem 1rem;
}
.status-item .k { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.status-item .v { margin-top: .35rem; font-family: var(--mono); font-size: .88rem; word-break: break-word; }
.badge {
  display: inline-block; padding: .2rem .55rem; border-radius: 999px;
  font-size: .72rem; font-weight: 650; letter-spacing: .04em; text-transform: uppercase;
  border: 1px solid transparent;
}
.badge-ok { background: rgba(61,186,122,.15); color: var(--ok); border-color: rgba(61,186,122,.35); }
.badge-warn { background: rgba(212,160,23,.15); color: var(--warn); border-color: rgba(212,160,23,.35); }
.badge-bad { background: rgba(227,93,106,.15); color: var(--bad); border-color: rgba(227,93,106,.35); }
.badge-muted { background: rgba(147,160,184,.12); color: var(--muted); border-color: rgba(147,160,184,.25); }
.panel {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow);
}
.table-wrap { overflow-x: auto; }
table.data {
  width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums;
}
table.data th, table.data td {
  padding: .7rem .85rem; text-align: left; border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
table.data th {
  font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); background: #10182a; font-weight: 600;
}
table.data tr:last-child td { border-bottom: none; }
table.data td.num { text-align: right; font-family: var(--mono); font-size: .86rem; }
.empty {
  padding: 1.25rem 1rem; color: var(--muted); font-size: .9rem;
}
.unavailable {
  max-width: 520px; margin: 4rem auto; text-align: center;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 2rem 1.5rem; box-shadow: var(--shadow);
}
.unavailable h1 { margin: 0 0 .75rem; font-size: 1.35rem; }
.unavailable p { margin: 0; color: var(--muted); }
.footer-note {
  margin-top: 2rem; color: var(--muted); font-size: .75rem;
}
.equity-bar {
  height: 6px; border-radius: 999px; background: #1c2942; overflow: hidden; min-width: 72px;
}
.equity-bar > span {
  display: block; height: 100%; background: linear-gradient(90deg, var(--accent-dim), var(--accent));
}
.chart-wrap {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1rem; box-shadow: var(--shadow);
}
.chart-svg { width: 100%; height: auto; display: block; }
.chart-empty {
  padding: 1.5rem 1rem; color: var(--muted); text-align: center; font-size: .9rem;
}
.chart-meta {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem;
  margin-top: .85rem;
}
.chart-meta .item {
  background: var(--bg-elev); border: 1px solid var(--border);
  border-radius: 8px; padding: .65rem .8rem;
}
.chart-meta .item .k { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.chart-meta .item .v { margin-top: .25rem; font-family: var(--mono); font-size: .88rem; }
.section-head {
  display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline;
  gap: .75rem; margin-bottom: .85rem;
}
.section-head h2 { margin: 0; }
.link-btn {
  display: inline-block; padding: .4rem .85rem; border-radius: 999px;
  border: 1px solid rgba(91,141,239,.45); background: rgba(91,141,239,.12);
  color: #cfe0ff; font-size: .82rem;
}
.note-muted { color: var(--muted); font-size: .85rem; margin: .5rem 0 0; }
@media (max-width: 1100px) {
  .cards, .status-grid, .chart-meta { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 800px) {
  .layout { flex-direction: column; }
  .sidebar { width: 100%; border-right: none; border-bottom: 1px solid var(--border); }
  .nav { flex-direction: row; flex-wrap: wrap; }
  .cards, .status-grid, .chart-meta { grid-template-columns: 1fr; }
  .main { padding: 1.1rem; }
}
"""


def _shell(
    *,
    title: str,
    experiment_label: str,
    active: str,
    body: str,
) -> str:
    nav_html: list[str] = []
    for label, href, enabled in NAV_ITEMS:
        is_active = label == active
        if enabled:
            cls = "active" if is_active else ""
            nav_html.append(f'<a class="{cls}" href="{esc(href)}">{esc(label)}</a>')
        else:
            nav_html.append(
                f'<span class="soon">{esc(label)}<em>準備中</em></span>'
            )
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{esc(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div>
        <div class="brand-mark">AI投資システム</div>
        <div class="brand-title">ペーパートレード</div>
        <div class="brand-sub">ダッシュボード</div>
      </div>
      <nav class="nav">
        {"".join(nav_html)}
      </nav>
    </aside>
    <main class="main">
      <div class="topbar">
        <div>
          <h1 class="page-title">{esc(title)}</h1>
          <p class="page-sub">ペーパートレード状態の読み取り専用ビュー</p>
        </div>
        <div class="experiment-chip" title="実験ID">{esc(experiment_label)}</div>
      </div>
      {body}
      <p class="footer-note">ペーパートレード実行プロセスとは独立しています。このダッシュボードはポートフォリオ状態を書き込みません。</p>
    </main>
  </div>
</body>
</html>
"""


def render_unavailable(*, experiment_id: str | None = None) -> str:
    label = experiment_id or "—"
    body = """
    <div class="unavailable">
      <h1>ダッシュボードを表示できません</h1>
      <p>ペーパーデータを安全に読み込めませんでした。しばらくして再試行するか、ペーパー実行サービスを確認してください。</p>
    </div>
    """
    return _shell(
        title="ダッシュボードを表示できません",
        experiment_label=str(label),
        active=NAV_ACTIVE_OVERVIEW,
        body=body,
    )


def render_not_found(*, experiment_id: str | None = None) -> str:
    label = experiment_id or "—"
    body = """
    <div class="unavailable">
      <h1>ページが見つかりません</h1>
      <p>指定されたページは存在しません。</p>
    </div>
    """
    return _shell(
        title="ページが見つかりません",
        experiment_label=str(label),
        active=NAV_ACTIVE_OVERVIEW,
        body=body,
    )


def _badge(status: str | None) -> str:
    if status is None or not str(status).strip():
        return f'<span class="badge {status_badge_class(None)}">—</span>'
    label = status_label(status) or str(status)
    return (
        f'<span class="badge {status_badge_class(status)}" title="{esc(status)}">'
        f"{esc(label)}</span>"
    )


def _kpi_card(label: str, value_html: str, hint: str | None = None) -> str:
    hint_html = f'<div class="hint">{esc(hint)}</div>' if hint else ""
    return f"""
    <div class="card">
      <div class="label">{esc(label)}</div>
      <div class="value">{value_html}</div>
      {hint_html}
    </div>
    """


def _status_item(label: str, value_html: str) -> str:
    return f"""
    <div class="status-item">
      <div class="k">{esc(label)}</div>
      <div class="v">{value_html}</div>
    </div>
    """


def render_overview(
    *,
    overview: dict[str, Any],
    system: dict[str, Any],
    positions: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
) -> str:
    currency = overview.get("base_currency") or system.get("base_currency")
    currency_label = currency or "—"
    experiment = overview.get("experiment_id") or system.get("experiment_id") or "—"

    eq = overview.get("current_equity")
    pnl = overview.get("total_pnl")
    ret = overview.get("total_return")
    cash = overview.get("cash")
    pos_val = overview.get("position_value")
    upnl = overview.get("unrealized_pnl")

    cards = "".join(
        [
            _kpi_card(
                "現在の資産評価額",
                format_money(eq, currency=currency),
                hint=f"基準通貨 {currency_label}",
            ),
            _kpi_card(
                "総損益",
                f'<span class="{pnl_class(pnl)}">{format_money(pnl, currency=currency, signed=True)}</span>',
                hint=f"初期資金比 · {currency_label}",
            ),
            _kpi_card(
                "総リターン",
                f'<span class="{pnl_class(ret)}">{format_percent(ret)}</span>',
            ),
            _kpi_card("現金残高", format_money(cash, currency=currency), hint=currency_label),
            _kpi_card(
                "保有株評価額",
                format_money(pos_val, currency=currency),
                hint=currency_label,
            ),
            _kpi_card(
                "含み損益",
                f'<span class="{pnl_class(upnl)}">{format_money(upnl, currency=currency, signed=True)}</span>',
                hint=currency_label,
            ),
        ]
    )

    run_status = system.get("run_status") or overview.get("run_status")
    validity = system.get("validity_status") or overview.get("validity_status")
    status_block = "".join(
        [
            _status_item(
                "実行状態",
                _badge(run_status if isinstance(run_status, str) else None),
            ),
            _status_item(
                "最終正常実行",
                dash(system.get("last_success_utc") or overview.get("last_success_utc")),
            ),
            _status_item(
                "データ基準日",
                dash(system.get("last_asof") or overview.get("latest_asof")),
            ),
            _status_item(
                "検証ステータス",
                _badge(validity if isinstance(validity, str) else None),
            ),
            _status_item("モデルID", dash(system.get("model_id") or overview.get("model_id"))),
            _status_item("実験ID", dash(experiment)),
        ]
    )

    if positions:
        pos_rows = []
        for p in positions:
            pos_rows.append(
                "<tr>"
                f"<td>{dash(p.get('symbol'))}</td>"
                f"<td>{dash(p.get('country'))}</td>"
                f"<td class='num'>{format_qty(p.get('quantity'))}</td>"
                f"<td class='num'>{format_price(p.get('entry_price'), currency=p.get('currency'))}</td>"
                f"<td class='num'>{format_price(p.get('current_price'), currency=p.get('currency'))}</td>"
                f"<td class='num'>{format_money(p.get('market_value'), currency=currency)}</td>"
                f"<td class='num {pnl_class(p.get('unrealized_pnl'))}'>"
                f"{format_money(p.get('unrealized_pnl'), currency=currency, signed=True)}</td>"
                f"<td class='num'>{dash(p.get('holding_days'))}</td>"
                f"<td class='num'>{dash(p.get('current_rank'))}</td>"
                "</tr>"
            )
        positions_html = (
            "<div class='panel table-wrap'><table class='data'>"
            "<thead><tr>"
            "<th>銘柄</th><th>国・地域</th><th>数量</th><th>取得価格</th>"
            "<th>現在価格</th><th>評価額</th><th>含み損益</th>"
            "<th>保有日数</th><th>現在順位</th>"
            "</tr></thead>"
            f"<tbody>{''.join(pos_rows)}</tbody></table></div>"
        )
    else:
        positions_html = "<div class='panel empty'>保有銘柄はありません。</div>"

    tail = equity_curve[-EQUITY_TAIL:] if equity_curve else []
    if tail:
        vals = [float(r["total_equity"]) for r in tail if r.get("total_equity") is not None]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        span = (hi - lo) or 1.0
        eq_rows = []
        for r in reversed(tail):
            te = r.get("total_equity")
            pct = 0.0
            if te is not None:
                pct = max(0.0, min(100.0, (float(te) - lo) / span * 100.0))
            eq_rows.append(
                "<tr>"
                f"<td>{dash(r.get('date'))}</td>"
                f"<td class='num'>{format_money(r.get('total_equity'), currency=currency)}</td>"
                f"<td class='num'>{format_money(r.get('cash'), currency=currency)}</td>"
                f"<td class='num'>{format_money(r.get('position_value'), currency=currency)}</td>"
                f"<td class='num {pnl_class(r.get('drawdown'))}'>{format_percent(r.get('drawdown'))}</td>"
                f"<td><div class='equity-bar' title='相対水準'><span style='width:{pct:.1f}%'></span></div></td>"
                "</tr>"
            )
        equity_html = (
            "<div class='panel table-wrap'><table class='data'>"
            "<thead><tr>"
            "<th>日付</th><th>総資産</th><th>現金残高</th>"
            "<th>保有株評価額</th><th>ドローダウン</th><th>相対水準</th>"
            "</tr></thead>"
            f"<tbody>{''.join(eq_rows)}</tbody></table></div>"
        )
    else:
        equity_html = "<div class='panel empty'>資産推移データはまだありません。</div>"

    mini_svg = render_equity_chart(
        equity_curve,
        initial_capital=_f_num(overview.get("initial_capital")),
        mini=True,
    )
    init_cap = overview.get("initial_capital")
    equity_summary = f"""
    <div class="chart-wrap">
      {mini_svg}
      <div class="chart-meta">
        <div class="item"><div class="k">初期資金</div>
          <div class="v">{format_money(init_cap, currency=currency)}</div></div>
        <div class="item"><div class="k">現在資産</div>
          <div class="v">{format_money(eq, currency=currency)}</div></div>
        <div class="item"><div class="k">累積損益</div>
          <div class="v {pnl_class(pnl)}">{format_money(pnl, currency=currency, signed=True)}</div></div>
        <div class="item"><div class="k">累積リターン</div>
          <div class="v {pnl_class(ret)}">{format_percent(ret)}</div></div>
      </div>
    </div>
    """

    body = f"""
    <section class="section">
      <h2>主要指標 · {esc(currency_label)}</h2>
      <div class="cards">{cards}</div>
    </section>
    <section class="section">
      <h2>ペーパートレード状況</h2>
      <div class="status-grid">{status_block}</div>
    </section>
    <section class="section">
      <h2>現在の保有銘柄</h2>
      {positions_html}
    </section>
    <section class="section">
      <div class="section-head">
        <h2>総資産の推移</h2>
        <a class="link-btn" href="/performance">詳細を見る</a>
      </div>
      {equity_summary}
    </section>
    <section class="section">
      <h2>最近の資産推移</h2>
      {equity_html}
    </section>
    """
    return _shell(
        title="概要",
        experiment_label=str(experiment),
        active=NAV_ACTIVE_OVERVIEW,
        body=body,
    )


def _f_num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_performance(*, performance: dict[str, Any]) -> str:
    currency = performance.get("base_currency")
    currency_label = currency or "—"
    experiment = performance.get("experiment_id") or "—"
    curve = performance.get("equity_curve") or []
    stats = performance.get("equity_stats") or {}
    trade = performance.get("trade_summary") or {}
    daily = performance.get("daily_rows") or []
    monthly = performance.get("monthly_rows") or []

    eq = performance.get("current_equity")
    pnl = performance.get("total_pnl")
    ret = performance.get("total_return")
    realized = performance.get("realized_pnl")
    unrealized = performance.get("unrealized_pnl")
    mdd = performance.get("max_drawdown")
    sharpe = performance.get("sharpe")
    win_rate = performance.get("win_rate")
    pf = performance.get("profit_factor")
    n_trades = performance.get("completed_trades")
    avg_win = performance.get("average_win")
    avg_loss = performance.get("average_loss")
    initial = performance.get("initial_capital")

    cards = "".join(
        [
            _kpi_card("現在の総資産", format_money(eq, currency=currency), hint=currency_label),
            _kpi_card(
                "累積損益",
                f'<span class="{pnl_class(pnl)}">{format_money(pnl, currency=currency, signed=True)}</span>',
            ),
            _kpi_card(
                "累積リターン",
                f'<span class="{pnl_class(ret)}">{format_percent(ret)}</span>',
            ),
            _kpi_card(
                "実現損益",
                f'<span class="{pnl_class(realized)}">{format_money(realized, currency=currency, signed=True)}</span>',
            ),
            _kpi_card(
                "含み損益",
                f'<span class="{pnl_class(unrealized)}">{format_money(unrealized, currency=currency, signed=True)}</span>',
            ),
            _kpi_card(
                "最大ドローダウン",
                f'<span class="{pnl_class(mdd)}">{format_percent(mdd)}</span>',
            ),
            _kpi_card("Sharpe Ratio", format_ratio(sharpe)),
            _kpi_card("勝率", format_percent(win_rate, signed=False)),
            _kpi_card("Profit Factor", format_ratio(pf)),
            _kpi_card("完了トレード数", dash(n_trades)),
            _kpi_card(
                "平均利益",
                f'<span class="pnl-pos">{format_money(avg_win, currency=currency, signed=True)}</span>',
            ),
            _kpi_card(
                "平均損失",
                f'<span class="pnl-neg">{format_money(avg_loss, currency=currency, signed=True)}</span>',
            ),
        ]
    )

    equity_svg = render_equity_chart(curve, initial_capital=_f_num(initial), mini=False)
    equity_meta = f"""
    <div class="chart-meta">
      <div class="item"><div class="k">期間</div>
        <div class="v">{dash(stats.get('start_date'))} → {dash(stats.get('end_date'))}</div></div>
      <div class="item"><div class="k">開始資産</div>
        <div class="v">{format_money(stats.get('start_equity'), currency=currency)}</div></div>
      <div class="item"><div class="k">現在資産</div>
        <div class="v">{format_money(stats.get('current_equity'), currency=currency)}</div></div>
      <div class="item"><div class="k">最高 / 最低</div>
        <div class="v">{format_money(stats.get('peak_equity'), currency=currency)}
         / {format_money(stats.get('trough_equity'), currency=currency)}</div></div>
    </div>
    """

    # PnL history: cumulative always from equity; realized/unrealized only if columns present
    has_r = bool(stats.get("has_realized_history"))
    has_u = bool(stats.get("has_unrealized_history"))
    pnl_svg = render_pnl_chart(curve)
    pnl_notes = []
    if has_r:
        pnl_notes.append("実現損益は equity_history の Realized PnL 列を使用")
    else:
        pnl_notes.append("実現損益の履歴データなし")
    if has_u:
        pnl_notes.append("含み損益は equity_history の Unrealized PnL 列を使用")
    else:
        pnl_notes.append("含み損益の履歴データなし")
    pnl_note_html = "<p class='note-muted'>" + " · ".join(esc(n) for n in pnl_notes) + "</p>"

    # Small table for realized/unrealized latest points if available
    if curve and (has_r or has_u):
        tail = list(reversed(curve[-12:]))
        pr_rows = []
        for r in tail:
            pr_rows.append(
                "<tr>"
                f"<td>{dash(r.get('date'))}</td>"
                f"<td class='num {pnl_class(r.get('cumulative_pnl'))}'>"
                f"{format_money(r.get('cumulative_pnl'), currency=currency, signed=True)}</td>"
                f"<td class='num'>{format_money(r.get('realized_pnl'), currency=currency, signed=True) if has_r else '—'}</td>"
                f"<td class='num'>{format_money(r.get('unrealized_pnl'), currency=currency, signed=True) if has_u else '—'}</td>"
                "</tr>"
            )
        pnl_table = (
            "<div class='panel table-wrap'><table class='data'>"
            "<thead><tr><th>日付</th><th>累積損益</th><th>実現損益</th><th>含み損益</th></tr></thead>"
            f"<tbody>{''.join(pr_rows)}</tbody></table></div>"
        )
    else:
        pnl_table = (
            "<div class='panel empty'>累積損益チャートのみ表示可能です。"
            "実現/含み損益の時系列が不足しています。</div>"
        )

    dd_svg = render_drawdown_chart(curve)

    if daily:
        d_rows = []
        for r in daily:
            d_rows.append(
                "<tr>"
                f"<td>{dash(r.get('date'))}</td>"
                f"<td class='num'>{format_money(r.get('total_equity'), currency=currency)}</td>"
                f"<td class='num {pnl_class(r.get('daily_pnl'))}'>"
                f"{format_money(r.get('daily_pnl'), currency=currency, signed=True)}</td>"
                f"<td class='num {pnl_class(r.get('daily_return'))}'>"
                f"{format_percent(r.get('daily_return'))}</td>"
                f"<td class='num {pnl_class(r.get('drawdown'))}'>"
                f"{format_percent(r.get('drawdown'))}</td>"
                f"<td class='num'>{dash(r.get('n_positions'))}</td>"
                "</tr>"
            )
        daily_html = (
            "<div class='panel table-wrap'><table class='data'>"
            "<thead><tr><th>日付</th><th>総資産</th><th>日次損益</th>"
            "<th>日次リターン</th><th>ドローダウン</th><th>保有銘柄数</th></tr></thead>"
            f"<tbody>{''.join(d_rows)}</tbody></table></div>"
        )
    else:
        daily_html = "<div class='panel empty'>日次成績データはありません。</div>"

    if monthly:
        m_rows = []
        for r in monthly:
            m_rows.append(
                "<tr>"
                f"<td>{dash(r.get('month'))}</td>"
                f"<td class='num {pnl_class(r.get('strategy_return'))}'>"
                f"{format_percent(r.get('strategy_return'))}</td>"
                f"<td class='num'>{format_money(r.get('month_end_equity'), currency=currency)}</td>"
                f"<td class='num {pnl_class(r.get('month_pnl'))}'>"
                f"{format_money(r.get('month_pnl'), currency=currency, signed=True)}</td>"
                "</tr>"
            )
        monthly_html = (
            "<div class='panel table-wrap'><table class='data'>"
            "<thead><tr><th>年月</th><th>月間リターン</th><th>月末総資産</th><th>月間損益</th></tr></thead>"
            f"<tbody>{''.join(m_rows)}</tbody></table></div>"
        )
    else:
        monthly_html = "<div class='panel empty'>月次成績データはありません。</div>"

    trade_cards = "".join(
        [
            _kpi_card("完了トレード数", dash(trade.get("completed_trades"))),
            _kpi_card("勝ち", dash(trade.get("wins"))),
            _kpi_card("負け", dash(trade.get("losses"))),
            _kpi_card("勝率", format_percent(trade.get("win_rate"), signed=False)),
            _kpi_card(
                "平均利益",
                format_money(trade.get("average_win"), currency=currency, signed=True),
            ),
            _kpi_card(
                "平均損失",
                format_money(trade.get("average_loss"), currency=currency, signed=True),
            ),
            _kpi_card("Profit Factor", format_ratio(trade.get("profit_factor"))),
            _kpi_card(
                "Best Trade",
                format_money(trade.get("best_trade"), currency=currency, signed=True),
            ),
            _kpi_card(
                "Worst Trade",
                format_money(trade.get("worst_trade"), currency=currency, signed=True),
            ),
        ]
    )

    body = f"""
    <section class="section">
      <h2>運用成績 KPI · {esc(currency_label)}</h2>
      <div class="cards">{cards}</div>
    </section>
    <section class="section">
      <h2>総資産の推移</h2>
      <div class="chart-wrap">{equity_svg}{equity_meta}</div>
    </section>
    <section class="section">
      <h2>損益推移</h2>
      <div class="chart-wrap">{pnl_svg}{pnl_note_html}</div>
      {pnl_table}
    </section>
    <section class="section">
      <h2>ドローダウン推移</h2>
      <div class="chart-wrap">{dd_svg}
        <p class="note-muted">最大ドローダウン: <span class="{pnl_class(mdd)}">{format_percent(mdd)}</span></p>
      </div>
    </section>
    <section class="section">
      <h2>日次成績（直近）</h2>
      {daily_html}
    </section>
    <section class="section">
      <h2>月次成績</h2>
      {monthly_html}
    </section>
    <section class="section">
      <h2>トレード要約</h2>
      <div class="cards">{trade_cards}</div>
    </section>
    """
    return _shell(
        title="運用成績",
        experiment_label=str(experiment),
        active=NAV_ACTIVE_PERFORMANCE,
        body=body,
    )
