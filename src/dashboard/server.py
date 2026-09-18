"""Stdlib HTTP server for Paper Trading Dashboard (Phase D2).

Read-only. Separate entrypoint from ``run-paper-trading``.
No directory listing, no arbitrary file routes, no secrets in responses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.config.settings import get_settings
from src.dashboard.data import DashboardDataSource
from src.dashboard.metrics import (
    build_current_portfolio,
    build_overview,
    build_performance,
    build_system_status,
    get_equity_curve,
)
from src.dashboard.templates import (
    render_not_found,
    render_overview,
    render_performance,
    render_unavailable,
)
logger = logging.getLogger(__name__)

SERVICE_NAME = "paper-dashboard"


def resolve_state_dir(explicit: str | Path | None = None) -> Path:
    """Resolve PAPER_STATE_DIR from arg / env / settings (never hardcoded)."""
    if explicit is not None and str(explicit).strip():
        return Path(str(explicit)).expanduser()
    env = os.getenv("PAPER_STATE_DIR")
    if env is not None and str(env).strip():
        return Path(str(env).strip()).expanduser()
    return get_settings().paper_state_dir


def resolve_experiment_id(explicit: str | None = None) -> str | None:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = os.getenv("DASHBOARD_EXPERIMENT_ID")
    if env is not None and str(env).strip():
        return str(env).strip()
    return None


def resolve_port(explicit: int | None = None) -> int:
    if explicit is not None:
        return int(explicit)
    raw = os.getenv("PORT", "8080")
    return int(raw)


class DashboardApp:
    """Request dispatcher — testable without binding a socket."""

    def __init__(
        self,
        *,
        state_dir: Path | str,
        experiment_id: str | None = None,
        source: DashboardDataSource | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.experiment_id = experiment_id
        self._source = source

    @property
    def source(self) -> DashboardDataSource:
        if self._source is not None:
            return self._source
        return DashboardDataSource(self.state_dir, experiment_id=self.experiment_id)

    def handle(self, method: str, path: str) -> tuple[int, dict[str, str], bytes]:
        method_u = method.upper()
        parsed = urlparse(path)
        route = parsed.path or "/"

        if method_u not in ("GET", "HEAD"):
            return self._text(405, "メソッドが許可されていません", content_type="text/plain; charset=utf-8")

        if route == "/health":
            return self._health()

        if route == "/":
            return self._overview()

        if route == "/performance":
            return self._performance()

        body = render_not_found(experiment_id=self.experiment_id).encode("utf-8")
        headers = {"Content-Type": "text/html; charset=utf-8", "Content-Length": str(len(body))}
        if method_u == "HEAD":
            return 404, headers, b""
        return 404, headers, body

    def _health(self) -> tuple[int, dict[str, str], bytes]:
        payload = {"status": "ok", "service": SERVICE_NAME}
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return (
            200,
            {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(raw)),
                "Cache-Control": "no-store",
            },
            raw,
        )

    def _overview(self) -> tuple[int, dict[str, str], bytes]:
        try:
            src = self.source
            overview = build_overview(src)
            system = build_system_status(src)
            positions = build_current_portfolio(src)
            equity = get_equity_curve(src)
            html = render_overview(
                overview=overview,
                system=system,
                positions=positions,
                equity_curve=equity,
            )
            raw = html.encode("utf-8")
            return (
                200,
                {
                    "Content-Type": "text/html; charset=utf-8",
                    "Content-Length": str(len(raw)),
                    "Cache-Control": "no-store",
                },
                raw,
            )
        except Exception:  # noqa: BLE001 — never leak internals to clients
            logger.exception("Dashboard overview failed")
            raw = render_unavailable(experiment_id=self.experiment_id).encode("utf-8")
            return (
                500,
                {
                    "Content-Type": "text/html; charset=utf-8",
                    "Content-Length": str(len(raw)),
                    "Cache-Control": "no-store",
                },
                raw,
            )

    def _performance(self) -> tuple[int, dict[str, str], bytes]:
        try:
            src = self.source
            performance = build_performance(src)
            html = render_performance(performance=performance)
            raw = html.encode("utf-8")
            return (
                200,
                {
                    "Content-Type": "text/html; charset=utf-8",
                    "Content-Length": str(len(raw)),
                    "Cache-Control": "no-store",
                },
                raw,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Dashboard performance failed")
            raw = render_unavailable(experiment_id=self.experiment_id).encode("utf-8")
            return (
                500,
                {
                    "Content-Type": "text/html; charset=utf-8",
                    "Content-Length": str(len(raw)),
                    "Cache-Control": "no-store",
                },
                raw,
            )

    def _text(
        self, status: int, message: str, *, content_type: str
    ) -> tuple[int, dict[str, str], bytes]:
        raw = message.encode("utf-8")
        return (
            status,
            {
                "Content-Type": content_type,
                "Content-Length": str(len(raw)),
                "Cache-Control": "no-store",
                "Allow": "GET, HEAD",
            },
            raw,
        )


def make_handler(app: DashboardApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _dispatch(self) -> None:
            status, headers, body = app.handle(self.command, self.path)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD" and body:
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch()

        def do_HEAD(self) -> None:  # noqa: N802
            self._dispatch()

        def do_POST(self) -> None:  # noqa: N802
            status, headers, body = app.handle("POST", self.path)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

    return Handler


def serve(
    *,
    host: str = "0.0.0.0",
    port: int | None = None,
    state_dir: Path | str | None = None,
    experiment_id: str | None = None,
) -> None:
    resolved_dir = resolve_state_dir(state_dir)
    resolved_exp = resolve_experiment_id(experiment_id)
    resolved_port = resolve_port(port)
    app = DashboardApp(state_dir=resolved_dir, experiment_id=resolved_exp)
    handler = make_handler(app)
    httpd = ThreadingHTTPServer((host, resolved_port), handler)
    logger.info(
        "Paper dashboard listening on %s:%s experiment_id=%s",
        host,
        resolved_port,
        resolved_exp,
    )
    # Do not log full state_dir path to stdout in production-facing messages beyond debug.
    logger.debug("Dashboard state_dir configured")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dashboard shut down")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Serve Paper Trading Dashboard (read-only)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="Override PORT env")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Override PAPER_STATE_DIR",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Override DASHBOARD_EXPERIMENT_ID",
    )
    args = parser.parse_args(argv)
    serve(
        host=args.host,
        port=args.port,
        state_dir=args.state_dir,
        experiment_id=args.experiment_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
