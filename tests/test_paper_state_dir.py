"""PAPER_STATE_DIR resolution and PaperTradingRunner wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import PROJECT_ROOT, Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_paper_state_dir_default_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAPER_STATE_DIR", raising=False)
    settings = get_settings()
    assert settings.paper_state_dir == PROJECT_ROOT / "data" / "paper"


def test_paper_state_dir_relative_resolves_to_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAPER_STATE_DIR", "models/data/paper")
    settings = get_settings()
    assert settings.paper_state_dir == PROJECT_ROOT / "models" / "data" / "paper"
    assert settings.paper_state_dir.is_absolute()


def test_paper_state_dir_absolute_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    absolute = tmp_path / "persist" / "paper"
    monkeypatch.setenv("PAPER_STATE_DIR", str(absolute))
    settings = get_settings()
    assert settings.paper_state_dir == absolute


def test_paper_trading_runner_uses_settings_paper_state_dir(tmp_path: Path) -> None:
    from src.paper.runner import PaperTradingRunner

    paper_dir = tmp_path / "custom_paper"
    settings = Settings(paper_state_dir=paper_dir)
    runner = PaperTradingRunner(
        settings,
        Path("config/universe.global100.json"),
        Path("config/paper_trading.json"),
    )
    assert runner.store.root == paper_dir
    assert paper_dir.is_dir()
