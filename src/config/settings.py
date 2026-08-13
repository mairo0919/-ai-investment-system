"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for data, features, and ML training."""

    data_provider: str = "yfinance"
    tickers: tuple[str, ...] = ("7203.T", "5333.T")
    lookback_years: int = 10
    interval: str = "1d"
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed"
    )
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    log_level: str = "INFO"
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    target_threshold: float = 0.02
    universe_path: Path | None = None
    project_root: Path = PROJECT_ROOT

    @property
    def raw_market_dir(self) -> Path:
        return self.raw_data_dir / "market"

    @property
    def processed_market_dir(self) -> Path:
        return self.processed_data_dir / "market"

    @property
    def experiments_dir(self) -> Path:
        return self.reports_dir / "experiments"


def _parse_tickers(raw: str) -> tuple[str, ...]:
    tickers = tuple(ticker.strip() for ticker in raw.split(",") if ticker.strip())
    if not tickers:
        raise ValueError("TICKERS must contain at least one symbol")
    return tickers


def _as_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return PROJECT_ROOT / path
    return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from `.env` and process environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")

    train_ratio = float(os.getenv("TRAIN_RATIO", "0.70"))
    valid_ratio = float(os.getenv("VALID_RATIO", "0.15"))
    test_ratio = float(os.getenv("TEST_RATIO", "0.15"))
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1.0, got {total}"
        )

    return Settings(
        data_provider=os.getenv("DATA_PROVIDER", "yfinance").strip().lower(),
        tickers=_parse_tickers(os.getenv("TICKERS", "7203.T,5333.T")),
        lookback_years=int(os.getenv("LOOKBACK_YEARS", "10")),
        interval=os.getenv("INTERVAL", "1d").strip(),
        raw_data_dir=_as_project_path(os.getenv("RAW_DATA_DIR", "data/raw")),
        processed_data_dir=_as_project_path(
            os.getenv("PROCESSED_DATA_DIR", "data/processed")
        ),
        models_dir=_as_project_path(os.getenv("MODELS_DIR", "models")),
        reports_dir=_as_project_path(os.getenv("REPORTS_DIR", "reports")),
        log_dir=_as_project_path(os.getenv("LOG_DIR", "logs")),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        target_threshold=float(os.getenv("TARGET_THRESHOLD", "0.02")),
        universe_path=_optional_project_path(os.getenv("UNIVERSE_PATH")),
        project_root=PROJECT_ROOT,
    )


def _optional_project_path(value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return _as_project_path(str(value).strip())
