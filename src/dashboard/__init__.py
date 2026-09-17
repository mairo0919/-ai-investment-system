"""Dashboard read-only data layer (Phase D1). No HTTP / no Paper writers."""

from src.dashboard.data import DashboardDataSource
from src.dashboard.metrics import (
    build_current_portfolio,
    build_overview,
    build_performance,
    build_system_status,
    build_trade_history,
    get_equity_curve,
    get_rankings,
    list_ranking_dates,
)

__all__ = [
    "DashboardDataSource",
    "build_overview",
    "build_performance",
    "build_current_portfolio",
    "build_trade_history",
    "build_system_status",
    "get_equity_curve",
    "get_rankings",
    "list_ranking_dates",
]
