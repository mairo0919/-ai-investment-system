"""Global instrument universe loading (JP / US / EU mixable)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.metadata import InstrumentMeta

logger = logging.getLogger(__name__)

EUROPE_COUNTRIES = frozenset({"DE", "FR", "GB", "CH", "NL", "IT", "ES", "SE", "BE", "IE"})


def region_of(country: str | None) -> str | None:
    """Map ISO-like country codes to evaluation regions."""
    if country is None:
        return None
    code = country.strip().upper()
    if code in {"JP", "JAPAN"}:
        return "Japan"
    if code in {"US", "USA", "UNITED STATES"}:
        return "United States"
    if code in EUROPE_COUNTRIES or code == "EU":
        return "Europe"
    return code


@dataclass(frozen=True)
class Universe:
    """A bounded set of instruments for research (typically 50–100 names)."""

    name: str
    instruments: tuple[InstrumentMeta, ...]
    description: str | None = None
    known_limitations: tuple[str, ...] = ()

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(item.symbol for item in self.instruments)

    def by_country(self, country: str) -> tuple[InstrumentMeta, ...]:
        code = country.strip().upper()
        return tuple(
            item
            for item in self.instruments
            if item.country is not None and item.country.upper() == code
        )

    def by_region(self, region: str) -> tuple[InstrumentMeta, ...]:
        target = region.strip().lower()
        return tuple(
            item
            for item in self.instruments
            if region_of(item.country) is not None
            and region_of(item.country).lower() == target
        )

    def composition_report(self) -> dict[str, Any]:
        """Summarize country / region / sector counts for reporting."""
        countries = Counter(item.country or "null" for item in self.instruments)
        regions = Counter(region_of(item.country) or "null" for item in self.instruments)
        sectors = Counter(item.sector or "null" for item in self.instruments)
        markets = Counter(item.market or "null" for item in self.instruments)
        return {
            "universe_name": self.name,
            "size": len(self.instruments),
            "description": self.description,
            "known_limitations": list(self.known_limitations),
            "country_counts": dict(sorted(countries.items())),
            "region_counts": dict(sorted(regions.items())),
            "sector_counts": dict(sorted(sectors.items())),
            "market_counts": dict(sorted(markets.items())),
        }


def load_universe(path: Path) -> Universe:
    """Load a universe JSON file.

    Expected schema::

        {
          "name": "global100_research_fixed",
          "instruments": [
            {"symbol": "7203.T", "country": "JP", "market": "TSE", "currency": "JPY"},
            {"symbol": "AAPL", "country": "US", "market": "NASDAQ", "currency": "USD"}
          ]
        }
    """
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid universe JSON: {path}") from exc

    name = str(payload.get("name") or path.stem)
    raw_items = payload.get("instruments")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"Universe '{name}' must contain a non-empty instruments list")

    instruments = tuple(InstrumentMeta.from_dict(item) for item in raw_items)
    symbols = [item.symbol for item in instruments]
    if len(symbols) != len(set(symbols)):
        raise ValueError(f"Universe '{name}' contains duplicate symbols")

    required_missing = [
        item.symbol
        for item in instruments
        if not item.country or not item.market or not item.currency
    ]
    if required_missing:
        raise ValueError(
            "Universe instruments require country/market/currency; missing for: "
            f"{required_missing[:10]}"
        )

    limitations = tuple(str(x) for x in payload.get("known_limitations", []) or [])
    description = _optional_description(payload.get("description"))

    logger.info(
        "Loaded universe '%s' with %d instruments from %s",
        name,
        len(instruments),
        path,
    )
    return Universe(
        name=name,
        instruments=instruments,
        description=description,
        known_limitations=limitations,
    )


def _optional_description(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_universe_config(path: Path) -> dict[str, Any]:
    """Validate universe file and return composition report."""
    universe = load_universe(path)
    report = universe.composition_report()
    if report["size"] < 50 or report["size"] > 120:
        logger.warning(
            "Universe size=%d is outside the intended 50–100 (+buffer) research range",
            report["size"],
        )
    return report
