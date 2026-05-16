"""Abstract LLM adapter interface.

Every LLM backend (Claude, Ollama, OpenAI-compatible, etc.) implements
LLMAdapter. The reasoning agent depends only on this interface — swapping
models requires zero changes to any other module.

Usage:
    from agent.reasoning.llm_base import LLMAdapter, Message, LLMResponse
    from agent.reasoning.claude_adapter import ClaudeAdapter

    adapter: LLMAdapter = ClaudeAdapter()
    response = await adapter.complete(messages, system="You are...")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: MessageRole
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMAdapter(ABC):
    """Abstract interface for all LLM backends."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a completion request and return the response.

        Args:
            messages: Conversation history (user/assistant turns).
            system:   System prompt string (handled differently per backend).
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            LLMResponse with .content as the raw text output.
        """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the active model identifier string."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name: 'claude', 'local', etc."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id!r})"


def get_llm_adapter() -> LLMAdapter:
    """Factory: return the configured LLM adapter based on LLM_PROVIDER env var."""
    from agent.config import get_settings, LLMProvider

    cfg = get_settings()

    if cfg.llm_provider == LLMProvider.CLAUDE:
        from agent.reasoning.claude_adapter import ClaudeAdapter
        return ClaudeAdapter(cfg)

    if cfg.llm_provider == LLMProvider.LOCAL:
        from agent.reasoning.local_adapter import LocalLLMAdapter
        return LocalLLMAdapter(cfg)

    raise ValueError(f"Unknown LLM_PROVIDER: {cfg.llm_provider!r}")
