"""ReasoningAgent — the Claude decision layer.

Assembles a Jinja2-rendered prompt from live market context, calls the
configured LLM adapter, parses the JSON response against TradingDecision,
and retries up to MAX_RETRIES times on malformed output.

Usage:
    agent = ReasoningAgent()
    decision = await agent.decide(decision_input)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from agent.reasoning.llm_base import LLMAdapter, Message, MessageRole, get_llm_adapter
from agent.reasoning.schemas import DecisionInput, DecisionRecord, TradingDecision

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_MAX_RETRIES = 3
_MAX_TOKENS = 600  # JSON output with detailed rationale is ~300 tokens; 600 gives headroom
_TEMPERATURE = 0.2

_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
)


def _render_prompt(ctx: DecisionInput) -> str:
    """Render decision.md with the given context."""
    template = _jinja_env.get_template("decision.md")
    return template.render(
        timestamp=ctx.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        symbol=ctx.symbol,
        dry_run=ctx.dry_run,
        signal=ctx.signal,
        account=ctx.account,
        open_positions=ctx.open_positions,
        recent_lessons=ctx.recent_lessons,
        recent_decisions=ctx.recent_decisions,
        btc_composite=ctx.btc_composite,
        mtf_signals=ctx.mtf_signals,
        json_schema=TradingDecision.json_schema_for_prompt(),
    )


def _extract_json(text: str) -> str:
    """Extract the first JSON object from raw LLM output.

    The model is instructed to return only JSON, but may occasionally wrap it
    in markdown code fences or add a brief preamble. We strip those here.
    """
    # Try markdown fence first: ```json ... ``` or ``` ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)

    # Fall back to the first { ... } block
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return brace.group(0)

    return text  # return raw and let json.loads raise


class ReasoningAgent:
    """LLM-backed trading decision maker.

    Args:
        adapter: An LLMAdapter instance. If None, the adapter is resolved from
                 the LLM_PROVIDER env var via get_llm_adapter().
    """

    def __init__(self, adapter: LLMAdapter | None = None) -> None:
        self._adapter = adapter or get_llm_adapter()

    @property
    def model_id(self) -> str:
        return self._adapter.model_id

    async def decide(self, ctx: DecisionInput) -> TradingDecision:
        """Run one reasoning cycle and return a validated TradingDecision.

        Retries up to MAX_RETRIES on malformed JSON or validation failure.
        On exhausted retries, returns a conservative hold decision.

        Args:
            ctx: Full decision context (signal + account + lessons).

        Returns:
            Validated TradingDecision.
        """
        prompt = _render_prompt(ctx)
        messages = [Message(role=MessageRole.USER, content=prompt)]

        last_error: Exception | None = None
        response = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._adapter.complete(
                    messages=messages,
                    system=self._system_prompt(),
                    max_tokens=_MAX_TOKENS,
                    temperature=_TEMPERATURE,
                )
                raw_json = _extract_json(response.content)
                data = json.loads(raw_json)
                decision = TradingDecision.model_validate(data)

                if decision.symbol != ctx.symbol:
                    raise ValueError(
                        f"LLM returned symbol {decision.symbol!r} "
                        f"but context was for {ctx.symbol!r}"
                    )
                if decision.dry_run != ctx.dry_run:
                    decision = decision.model_copy(update={"dry_run": ctx.dry_run})

                log.info(
                    "Decision made",
                    extra={
                        "symbol": ctx.symbol,
                        "action": decision.action,
                        "confidence": decision.confidence,
                        "attempt": attempt,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                )
                return decision

            except (json.JSONDecodeError, ValidationError, ValueError, KeyError, TimeoutError) as exc:
                last_error = exc
                log.warning(
                    "Reasoning parse failed (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=response.content if response is not None else "",
                        )
                    )
                    messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=(
                                f"Your previous response could not be parsed: {exc}. "
                                "Return ONLY the JSON object with no additional text."
                            ),
                        )
                    )

        log.error(
            "ReasoningAgent exhausted retries for %s, defaulting to hold. Last error: %s",
            ctx.symbol,
            last_error,
        )
        return TradingDecision(
            action="hold",
            symbol=ctx.symbol,
            rationale=(
                f"Safety hold: LLM returned unparseable output after {_MAX_RETRIES} attempts. "
                f"No valid JSON decision was produced. Holding to preserve capital. "
                f"Last error: {str(last_error)[:80]}"
            ),
            confidence=0.0,
            edge_estimate=0.0,
            risk_notes="Automatic fallback — LLM parse failure. Review model output and retry.",
            dry_run=ctx.dry_run,
        )

    def build_record(
        self,
        ctx: DecisionInput,
        decision: TradingDecision,
        input_tokens: int = 0,
        output_tokens: int = 0,
        order_id: str | None = None,
    ) -> DecisionRecord:
        """Assemble a DecisionRecord for persistence in SQLite."""
        return DecisionRecord(
            symbol=ctx.symbol,
            action=decision.action,
            rationale=decision.rationale,
            confidence=decision.confidence,
            edge_estimate=decision.edge_estimate,
            risk_notes=decision.risk_notes,
            composite_signal=ctx.signal.composite,
            signal_regime=ctx.signal.regime,
            portfolio_value_at_decision=ctx.account.portfolio_value,
            decided_at=ctx.timestamp,
            dry_run=ctx.dry_run,
            llm_model=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            order_id=order_id,
            close_price_at_decision=ctx.signal.close_price,
            signal_weights_json=json.dumps(ctx.signal.signal_weights),
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a disciplined quantitative crypto trading agent. "
            "You follow strict risk rules and output decisions as pure JSON only — "
            "no prose, no markdown fences, no commentary outside the JSON object. "
            "When uncertain, hold. Never fabricate data."
        )
