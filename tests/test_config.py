"""Tests for agent/config — settings loading and validators."""

from __future__ import annotations

import pytest


def _make_settings(monkeypatch: pytest.MonkeyPatch, **overrides: str):
    """Build a fresh Settings instance (bypassing lru_cache) with test env vars."""
    monkeypatch.setenv("ALPACA_API_KEY", overrides.pop("ALPACA_API_KEY", "test_key"))
    monkeypatch.setenv("ALPACA_SECRET_KEY", overrides.pop("ALPACA_SECRET_KEY", "test_secret"))
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)

    # Import fresh (lru_cache holds the real settings — create Settings directly in tests)
    from agent.config import Settings
    return Settings()


def test_settings_load_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _make_settings(monkeypatch)
    assert s.alpaca_api_key == "test_key"
    assert s.live_trading_enabled is False
    assert s.max_risk_per_trade_pct == 0.02


def test_url_normalisation_strips_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _make_settings(monkeypatch, ALPACA_BASE_URL="https://paper-api.alpaca.markets/v2")
    assert s.alpaca_base_url == "https://paper-api.alpaca.markets"


def test_url_normalisation_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _make_settings(monkeypatch, ALPACA_BASE_URL="https://paper-api.alpaca.markets/")
    assert s.alpaca_base_url == "https://paper-api.alpaca.markets"


def test_universe_property_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _make_settings(monkeypatch, TRADING_UNIVERSE="BTC/USD, ETH/USD, SOL/USD")
    assert s.universe == ["BTC/USD", "ETH/USD", "SOL/USD"]


def test_universe_default_has_eight_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _make_settings(monkeypatch)
    assert len(s.universe) == 8


def test_live_trading_enabled_true_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="LIVE_TRADING_ENABLED"):
        _make_settings(monkeypatch, LIVE_TRADING_ENABLED="True")


def test_llm_provider_default_is_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import LLMProvider
    s = _make_settings(monkeypatch)
    assert s.llm_provider == LLMProvider.CLAUDE


def test_llm_provider_local(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import LLMProvider
    s = _make_settings(monkeypatch, LLM_PROVIDER="local")
    assert s.llm_provider == LLMProvider.LOCAL
