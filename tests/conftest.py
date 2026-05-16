"""Shared pytest fixtures and configuration."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never touch real APIs and use a temp DB."""
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "False")
    monkeypatch.setenv("RANDOM_SEED", "42")


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    """Return a temp path for a test SQLite database."""
    return str(tmp_path / "test_trader.db")
