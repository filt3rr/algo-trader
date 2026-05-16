"""Claude API adapter (Anthropic SDK).

Implements LLMAdapter using the Anthropic Python SDK with:
- Prompt caching on the system prompt (reduces cost ~90% on repeated calls).
- Exponential-backoff retry on rate limits and transient errors.
- Token usage tracking in LLMResponse.

Full prompt engineering and JSON-schema validation live in reasoning/agent.py
(Phase 3). This adapter is intentionally thin — just transport.
"""

from __future__ import annotations

import logging

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent.config import Settings, get_settings
from agent.reasoning.llm_base import LLMAdapter, LLMResponse, Message, MessageRole

log = logging.getLogger(__name__)


class ClaudeAdapter(LLMAdapter):
    """Anthropic Claude backend."""

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        if not cfg.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to .env before using Claude."
            )
        self._client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.claude_model
        log.info("ClaudeAdapter initialised, model=%s", self._model)

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        anthropic_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
            if msg.role != MessageRole.SYSTEM
        ]

        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }

        if system:
            # Use prompt caching for the system prompt — saves ~90% on repeated calls.
            # Requires Claude claude-sonnet-4-5+ models.
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        response = await self._client.messages.create(**kwargs)

        content = response.content[0].text if response.content else ""
        usage = response.usage

        log.debug(
            "Claude response | model=%s | in=%d out=%d | len=%d chars",
            self._model,
            usage.input_tokens,
            usage.output_tokens,
            len(content),
        )

        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            raw={"stop_reason": response.stop_reason},
        )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "claude"
