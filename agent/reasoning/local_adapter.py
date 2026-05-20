"""Local LLM adapter (Ollama / OpenAI-compatible REST API).

Connects to any OpenAI-compatible API served locally, defaulting to Ollama
at http://localhost:11434. Set LLM_PROVIDER=local and LOCAL_LLM_MODEL=<model>
in .env to activate.

Tested with:
- Ollama (llama3.1, mistral, deepseek-r1, qwen2.5)
- LM Studio
- vLLM with openai-compatible mode

To run Ollama locally:
    ollama pull qwen2.5:7b
    ollama serve
Then set LOCAL_LLM_MODEL=qwen2.5:7b in .env.
"""

from __future__ import annotations

import logging

import httpx

from agent.config import Settings, get_settings
from agent.reasoning.llm_base import LLMAdapter, LLMResponse, Message, MessageRole

log = logging.getLogger(__name__)

# Ollama's keep_alive keeps the model loaded in VRAM between calls.
# "10m" means the model stays loaded for 10 minutes after the last request,
# covering consecutive 5-minute trading cycles without a cold reload penalty.
# Set to "-1" to keep loaded indefinitely (more VRAM locked, but zero cold starts).
_OLLAMA_KEEP_ALIVE = "-1"  # never unload — agent runs 24/7 and cold starts take 2-5 min


class LocalLLMAdapter(LLMAdapter):
    """OpenAI-compatible local LLM adapter (Ollama, LM Studio, vLLM).

    Uses separate connect and read timeouts so slow GPU inference doesn't
    trigger a fast-fail connection timeout. TimeoutErrors are re-raised as
    Python TimeoutError so the ReasoningAgent retry loop catches them cleanly.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self._base_url = cfg.local_llm_base_url.rstrip("/")
        self._model = cfg.local_llm_model
        read_timeout = cfg.local_llm_timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            # Connect fast-fail (10s); read waits the full inference window.
            timeout=httpx.Timeout(connect=10.0, read=read_timeout, write=10.0, pool=10.0),
        )
        log.info(
            "LocalLLMAdapter initialised, model=%s, url=%s, read_timeout=%ds",
            self._model, self._base_url, read_timeout,
        )

    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        oai_messages: list[dict] = []

        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue
            oai_messages.append({"role": msg.role.value, "content": msg.content})

        payload = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            # Keeps the model hot in VRAM between consecutive calls.
            # Ollama honours this on both /v1/chat/completions and /api/chat.
            "keep_alive": _OLLAMA_KEEP_ALIVE,
        }

        try:
            return await self._call_openai_compat(payload)
        except (httpx.HTTPStatusError, KeyError):
            return await self._call_ollama_native(payload)
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Ollama inference timed out for model {self._model!r}. "
                "Check GPU load or increase LOCAL_LLM_TIMEOUT in .env."
            ) from exc

    async def _call_openai_compat(self, payload: dict) -> LLMResponse:
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

    async def _call_ollama_native(self, payload: dict) -> LLMResponse:
        try:
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
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Ollama inference timed out for model {self._model!r}. "
                "Check GPU load or increase LOCAL_LLM_TIMEOUT in .env."
            ) from exc

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
