"""Point-in-time FX conversion to base currency (JPY). No silent 1:1."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.exceptions import TrainingError

logger = logging.getLogger(__name__)

# Local currency -> how to get units of base (JPY) per 1 local unit.
SUPPORTED_LOCAL = ("JPY", "USD", "EUR", "GBP", "CHF")


@dataclass(frozen=True)
class FxConfig:
    tickers: dict[str, str]
    base_currency: str = "JPY"

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base_currency: str = "JPY") -> FxConfig:
        return cls(tickers=dict(raw.get("tickers", {})), base_currency=base_currency)


class FxConverter:
    """Convert local quote currencies to JPY using asof(backward) closes.

    Required series (names in config tickers):
    - USDJPY via JPY=X
    - EURJPY / GBPJPY / CHFJPY preferred; EURUSD/GBPUSD fallbacks allowed.
    """

    def __init__(self, series: dict[str, pd.Series], *, base_currency: str = "JPY") -> None:
        if base_currency != "JPY":
            raise TrainingError("Only base_currency=JPY is supported in Phase 4")
        self.base_currency = base_currency
        self._series = {
            name: s.sort_index().astype(float) for name, s in series.items() if s is not None and not s.empty
        }

    @classmethod
    def from_frames(
        cls,
        frames: dict[str, pd.DataFrame],
        *,
        base_currency: str = "JPY",
    ) -> FxConverter:
        series: dict[str, pd.Series] = {}
        for name, frame in frames.items():
            if frame is None or frame.empty:
                continue
            tmp = frame.copy()
            tmp["Date"] = pd.to_datetime(tmp["Date"])
            s = tmp.set_index("Date")["Close"].astype(float).sort_index()
            series[name] = s
        return cls(series, base_currency=base_currency)

    def rate_to_base(self, currency: str, asof: pd.Timestamp) -> float:
        """Return JPY per 1 unit of ``currency`` as of ``asof`` (inclusive, backward)."""
        ccy = (currency or "").upper()
        asof = pd.Timestamp(asof).normalize()
        if ccy == "JPY":
            return 1.0
        if ccy == "USD":
            return self._lookup("USDJPY", asof)
        if ccy == "EUR":
            if "EURJPY" in self._series:
                return self._lookup("EURJPY", asof)
            return self._lookup("EURUSD", asof) * self._lookup("USDJPY", asof)
        if ccy == "GBP":
            if "GBPJPY" in self._series:
                return self._lookup("GBPJPY", asof)
            return self._lookup("GBPUSD", asof) * self._lookup("USDJPY", asof)
        if ccy == "CHF":
            return self._lookup("CHFJPY", asof)
        raise TrainingError(f"Unsupported currency for FX conversion: {currency}")

    def can_convert(self, currency: str) -> bool:
        ccy = (currency or "").upper()
        if ccy == "JPY":
            return True
        try:
            # Probe with a late date if series exist
            if not self._series:
                return False
            any_idx = next(iter(self._series.values())).index
            if len(any_idx) == 0:
                return False
            self.rate_to_base(ccy, any_idx[-1])
            return True
        except Exception:  # noqa: BLE001
            return False

    def _lookup(self, name: str, asof: pd.Timestamp) -> float:
        if name not in self._series:
            raise TrainingError(f"Missing FX series: {name}")
        s = self._series[name]
        # asof inclusive
        pos = s.index.searchsorted(asof, side="right") - 1
        if pos < 0:
            raise TrainingError(f"FX {name} not available on/before {asof.date()}")
        val = float(s.iloc[pos])
        if val <= 0 or pd.isna(val):
            raise TrainingError(f"Invalid FX {name} at {asof.date()}: {val}")
        return val


def fetch_fx_frames(
    provider: Any,
    cache: Any,
    fx_cfg: FxConfig,
    *,
    start: Any,
    end: Any,
    interval: str = "1d",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch configured FX tickers into named frames (USDJPY, EURJPY, ...)."""
    out: dict[str, pd.DataFrame] = {}
    for name, ticker in fx_cfg.tickers.items():
        try:
            frame = cache.get_or_fetch(
                provider,
                ticker,
                start=start,
                end=end,
                interval=interval,
                force_refresh=force_refresh,
            )
            out[name] = frame
            logger.info("FX ready %s (%s) rows=%d", name, ticker, len(frame))
        except Exception as exc:  # noqa: BLE001
            logger.warning("FX fetch failed %s (%s): %s", name, ticker, exc)
    return out
