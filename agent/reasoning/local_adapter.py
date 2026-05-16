"""Local LLM adapter (Ollama / OpenAI-compatible REST API).

Connects to any OpenAI-compatible API served locally, defaulting to Ollama
at http://localhost:11434. Set LLM_PROVIDER=local and LOCAL_LLM_MODEL=<model>
in .env to activate.

Tested with:
- Ollama (llama3.1, mistral, deepseek-r1)
- LM Studio
- vLLM with openai-compatible mode

To run Ollama locally:
    ollama pull llama3.1:8b
    ollama serve
Then set LOCAL_LLM_MODEL=llama3.1:8b in .env.
"""

from __future__ import annotations

import logging

import httpx

from agent.config import Settings, get_settings
from agent.reasoning.llm_base import LLMAdapter, LLMResponse, Message, MessageRole

log = logging.getLogger(__name__)


class LocalLLMAdapter(LLMAdapter):
    """OpenAI-compatible local LLM adapter (Ollama, LM Studio, vLLM)."""

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self._base_url = cfg.local_llm_base_url.rstrip("/")
        self._model = cfg.local_llm_model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=120.0,
        )
        log.info(
            "LocalLLMAdapter initialised, model=%s, url=%s",
            self._model, self._base_url,
        )

    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        # Build OpenAI-format messages list
        oai_messages: list[dict] = []

        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue  # already handled above
            oai_messages.append({"role": msg.role.value, "content": msg.content})

        payload = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        # Ollama uses /api/chat; OpenAI-compatible servers use /v1/chat/completions
        # Try both, preferring OpenAI-compatible endpoint
        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                model=self._model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                raw=data,
            )
        except (httpx.HTTPStatusError, KeyError):
            # Fall back to Ollama native /api/chat
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return LLMResponse(
                content=content,
                model=self._model,
                input_tokens=0,
                output_tokens=0,
                raw=data,
            )

    async def health_check(self) -> bool:
        """Ping the local server and return True if reachable."""
        try:
            resp = await self._client.get("/", timeout=5.0)
            return resp.status_code < 500
        except Exception:
            return False

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "local"
