"""Phase 5 forward paper trading (no brokerage).

Do not eagerly import ``runner`` here. ``python -m src.paper.runner`` must load the
runner module only once; importing it from this package ``__init__`` triggers
runpy's double-load RuntimeWarning and can inflate startup memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["PaperTradingRunner"]

if TYPE_CHECKING:
    from src.paper.runner import PaperTradingRunner as PaperTradingRunner


def __getattr__(name: str) -> Any:
    if name == "PaperTradingRunner":
        from src.paper.runner import PaperTradingRunner

        return PaperTradingRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
