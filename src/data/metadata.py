"""Instrument metadata for global universes (null-tolerant)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstrumentMeta:
    """Static / semi-static instrument attributes.

    Fields that cannot be obtained from the current provider remain ``None``.
    """

    symbol: str
    exchange: str | None = None
    country: str | None = None
    market: str | None = None
    currency: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    timezone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata for JSON/CSV persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> InstrumentMeta:
        """Build metadata from a universe config row."""
        if "symbol" not in raw or not str(raw["symbol"]).strip():
            raise ValueError("InstrumentMeta requires a non-empty 'symbol'")
        market_cap = raw.get("market_cap")
        return cls(
            symbol=str(raw["symbol"]).strip(),
            exchange=_optional_str(raw.get("exchange")),
            country=_optional_str(raw.get("country")),
            market=_optional_str(raw.get("market")),
            currency=_optional_str(raw.get("currency")),
            sector=_optional_str(raw.get("sector")),
            industry=_optional_str(raw.get("industry")),
            market_cap=float(market_cap) if market_cap is not None else None,
            timezone=_optional_str(raw.get("timezone")),
        )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def enrich_metadata_from_yfinance(meta: InstrumentMeta) -> InstrumentMeta:
    """Best-effort enrichment via yfinance ``Ticker.info``.

    Missing fields stay ``None``. Failures never raise to callers.
    This helper is provider-adjacent but optional; core pipelines must not require it.
    """
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance unavailable for metadata enrichment: %s", exc)
        return meta

    try:
        info = yf.Ticker(meta.symbol).info or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to enrich metadata for %s: %s", meta.symbol, exc)
        return meta

    def pick(*keys: str) -> Any:
        for key in keys:
            value = info.get(key)
            if value is not None and value != "":
                return value
        return None

    market_cap = pick("marketCap")
    try:
        market_cap_f = float(market_cap) if market_cap is not None else meta.market_cap
    except (TypeError, ValueError):
        market_cap_f = meta.market_cap

    return InstrumentMeta(
        symbol=meta.symbol,
        exchange=meta.exchange or _optional_str(pick("exchange", "fullExchangeName")),
        country=meta.country or _optional_str(pick("country")),
        market=meta.market or _optional_str(pick("exchange", "fullExchangeName")),
        currency=meta.currency or _optional_str(pick("currency")),
        sector=meta.sector or _optional_str(pick("sector")),
        industry=meta.industry or _optional_str(pick("industry")),
        market_cap=market_cap_f,
        timezone=meta.timezone
        or _optional_str(pick("timeZoneFullName", "exchangeTimezoneName")),
    )
