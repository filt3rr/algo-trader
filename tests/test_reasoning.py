"""Tests for agent/reasoning — LLM adapter base types and factory."""

from __future__ import annotations

import pytest

from agent.reasoning.llm_base import Message, MessageRole, LLMResponse


# ── Message and LLMResponse ──────────────────────────────────────────────────

def test_message_roles() -> None:
    m = Message(role=MessageRole.USER, content="hello")
    assert m.role == MessageRole.USER
    assert m.content == "hello"


def test_llm_response_total_tokens() -> None:
    r = LLMResponse(content="ok", model="test-model", input_tokens=100, output_tokens=50)
    assert r.total_tokens == 150


def test_llm_response_defaults() -> None:
    r = LLMResponse(content="hi", model="m")
    assert r.input_tokens == 0
    assert r.output_tokens == 0
    assert r.total_tokens == 0
    assert r.raw == {}


def test_message_role_values() -> None:
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
    assert MessageRole.SYSTEM == "system"


# ── get_llm_adapter factory ───────────────────────────────────────────────────

def test_factory_returns_claude_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import patch

    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    from agent.config import Settings

    cfg = Settings()
    with patch("agent.config.get_settings", return_value=cfg), \
         patch("agent.reasoning.claude_adapter.anthropic.AsyncAnthropic"):
        from agent.reasoning.llm_base import get_llm_adapter
        adapter = get_llm_adapter()
        assert adapter.provider == "claude"


def test_factory_returns_local_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("LLM_PROVIDER", "local")

    from unittest.mock import patch
    from agent.config import Settings

    cfg = Settings()
    with patch("agent.config.get_settings", return_value=cfg):
        from agent.reasoning.llm_base import get_llm_adapter
        adapter = get_llm_adapter()
        assert adapter.provider == "local"
        assert adapter.model_id == cfg.local_llm_model


def test_factory_raises_on_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock, patch
    from agent.config import LLMProvider, Settings

    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")

    fake_cfg = MagicMock(spec=Settings)
    fake_cfg.llm_provider = "unknown_provider"

    with patch("agent.config.get_settings", return_value=fake_cfg):
        from agent.reasoning.llm_base import get_llm_adapter
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_llm_adapter()


def test_local_adapter_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("LLM_PROVIDER", "local")

    from agent.config import Settings
    from agent.reasoning.local_adapter import LocalLLMAdapter

    cfg = Settings()
    adapter = LocalLLMAdapter(cfg)
    assert "LocalLLMAdapter" in repr(adapter)
    assert cfg.local_llm_model in repr(adapter)
