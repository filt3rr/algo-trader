"""Nightly reflection engine — the agent's self-improvement loop.

Each night (REFLECTION_HOUR UTC), the ReflectionEngine:
    1. Fetches all decisions from the past 24h that have been attributed
    2. Builds a ReflectionInput context object
    3. Calls the LLM with the reflection prompt
    4. Parses ReflectionOutput: lessons + weight adjustments
    5. Saves lessons to the `lessons` table (with deduplication — skips lessons
       that are semantically too similar to existing active lessons)
    6. Saves updated weights to the `signal_weights` table
    7. Applies the new weights to the live SignalEngine instance

Retries up to MAX_RETRIES on malformed output, falls back to keeping current
weights unchanged and logging a generic lesson.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.memory.queries import (
    get_current_weights,
    get_decisions_since,
    get_latest_performance,
    get_recent_lessons,
    save_lesson,
    save_weights,
)
from agent.reasoning.llm_base import LLMAdapter, Message, MessageRole
from agent.reasoning.schemas import (
    AttributedDecision,
    RecentLesson,
    ReflectionInput,
    ReflectionOutput,
)
from agent.strategy.signals import DEFAULT_WEIGHTS, SignalEngine

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "reasoning" / "prompts"
_MAX_RETRIES = 3
_MAX_TOKENS = 2048
_TEMPERATURE = 0.3
_LESSON_DEDUP_THRESHOLD = 0.55  # Jaccard similarity threshold for deduplication

_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
)


def _render_reflection_prompt(ctx: ReflectionInput) -> str:
    template = _jinja_env.get_template("reflection.md")
    return template.render(
        reflection_date=ctx.reflection_date.strftime("%Y-%m-%d %H:%M UTC"),
        decisions=ctx.decisions,
        current_weights=ctx.current_weights,
        portfolio_value=ctx.portfolio_value,
        cumulative_return_pct=ctx.cumulative_return_pct,
        recent_lessons=ctx.recent_lessons,
        json_schema=ReflectionOutput.json_schema_for_prompt(),
    )


def _extract_json(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return brace.group(0)
    return text


def _weights_from_row(row: Any) -> dict[str, float]:
    """Extract the actual signal weights used at decision time.

    Prefers signal_weights_json stored in the row (Phase 5+).
    Falls back to DEFAULT_WEIGHTS for older rows that predate this column.
    """
    weights_json = getattr(row, "signal_weights_json", None)
    if weights_json:
        try:
            parsed = json.loads(weights_json)
            if isinstance(parsed, dict) and {"ema", "rsi", "roc"}.issubset(parsed):
                return {k: float(parsed[k]) for k in ("ema", "rsi", "roc")}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return dict(DEFAULT_WEIGHTS)


def _build_attributed_decisions(rows) -> list[AttributedDecision]:
    """Convert DecisionRow ORM objects to AttributedDecision schema objects.

    Signal contributions use the weights actually applied at decision time
    (read from signal_weights_json). This gives the reflection LLM accurate
    per-signal credit/blame rather than assuming equal 1/3 weights.
    """
    result = []
    for r in rows:
        if r.outcome_pnl_pct is None or r.action == "hold":
            continue
        composite = r.composite_signal
        weights = _weights_from_row(r)
        result.append(
            AttributedDecision(
                symbol=r.symbol,
                action=r.action,
                composite_signal=composite,
                signal_regime=r.signal_regime,
                confidence=r.confidence,
                edge_estimate=r.edge_estimate,
                ema_contribution=composite * weights["ema"],
                rsi_contribution=composite * weights["rsi"],
                roc_contribution=composite * weights["roc"],
                outcome_pnl_pct=r.outcome_pnl_pct,
                decided_at=_parse_dt(r.decided_at),
            )
        )
    return result


def _parse_dt(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _is_duplicate_lesson(new_lesson: str, existing_lessons: list[str]) -> bool:
    """Return True if new_lesson is too similar to any existing lesson.

    Uses Jaccard similarity on word sets. Threshold 0.55 catches paraphrases
    of the same insight while allowing genuinely different lessons through.
    """
    new_words = set(new_lesson.lower().split())
    if len(new_words) < 4:
        return False  # too short to compare reliably
    for existing in existing_lessons:
        existing_words = set(existing.lower().split())
        if not existing_words:
            continue
        union = new_words | existing_words
        intersection = new_words & existing_words
        if union and len(intersection) / len(union) >= _LESSON_DEDUP_THRESHOLD:
            return True
    return False


class ReflectionEngine:
    """Runs the nightly self-improvement cycle.

    Args:
        adapter:       LLM adapter for the reflection call.
        signal_engine: Live SignalEngine instance — weights updated in-place.
    """

    def __init__(self, adapter: LLMAdapter, signal_engine: SignalEngine) -> None:
        self._adapter = adapter
        self._signal_engine = signal_engine

    async def run(self, session: AsyncSession) -> ReflectionOutput | None:
        """Execute one full reflection cycle.

        Returns the ReflectionOutput on success, None if no attributed decisions
        were available (skips LLM call — still saves a placeholder lesson).
        """
        now = datetime.now(tz=timezone.utc)
        since = now - timedelta(hours=48)  # cover attribution window + one day buffer

        rows = await get_decisions_since(session, since=since, with_outcome_only=True)
        attributed = _build_attributed_decisions(rows)

        current_weights = await get_current_weights(session) or dict(DEFAULT_WEIGHTS)
        recent_lessons = await get_recent_lessons(session, limit=5)

        perf = await get_latest_performance(session)
        portfolio_value = perf.portfolio_value if perf else "unknown"
        cumulative_return = perf.cumulative_return_pct if perf else None

        if not attributed:
            log.info("Reflection: no attributed decisions — saving patience lesson")
            await save_lesson(
                session,
                lesson=(
                    "No attributed decisions available for this reflection cycle. "
                    "Continue accumulating data before adjusting weights."
                ),
                source="reflection",
            )
            return None

        ctx = ReflectionInput(
            reflection_date=now,
            decisions=attributed,
            current_weights=current_weights,
            portfolio_value=portfolio_value,
            cumulative_return_pct=cumulative_return,
            recent_lessons=recent_lessons,
        )

        output = await self._call_llm(ctx)
        if output is None:
            return None

        # Save lessons with deduplication — skip those too similar to existing ones
        existing_texts = [l.lesson for l in recent_lessons]
        saved_count = 0
        for lesson_text in output.lessons:
            if _is_duplicate_lesson(lesson_text, existing_texts):
                log.debug("Skipping duplicate lesson: %.60s...", lesson_text)
                continue
            await save_lesson(session, lesson=lesson_text, source="reflection")
            existing_texts.append(lesson_text)  # prevent intra-batch duplication
            saved_count += 1

        # Persist and apply weights
        w = output.weight_adjustments
        await save_weights(
            session,
            ema=w["ema"],
            rsi=w["rsi"],
            roc=w["roc"],
            source="reflection",
            rationale=output.weight_rationale,
        )
        self._signal_engine.update_weights(w)

        log.info(
            "Reflection complete: %d/%d lessons saved, weights updated ema=%.3f rsi=%.3f roc=%.3f",
            saved_count, len(output.lessons),
            w["ema"], w["rsi"], w["roc"],
        )
        return output

    async def _call_llm(self, ctx: ReflectionInput) -> ReflectionOutput | None:
        prompt = _render_reflection_prompt(ctx)
        messages = [Message(role=MessageRole.USER, content=prompt)]
        system = (
            "You are the learning module of a quantitative crypto trading agent. "
            "Your job is to analyze today's trading outcomes and extract actionable lessons. "
            "Return ONLY the JSON object — no prose, no markdown fences."
        )

        last_error: Exception | None = None
        response = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._adapter.complete(
                    messages=messages,
                    system=system,
                    max_tokens=_MAX_TOKENS,
                    temperature=_TEMPERATURE,
                )
                raw_json = _extract_json(response.content)
                data = json.loads(raw_json)
                output = ReflectionOutput.model_validate(data)
                log.info(
                    "Reflection LLM call succeeded (attempt %d/%d)", attempt, _MAX_RETRIES
                )
                return output

            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                log.warning(
                    "Reflection parse failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc
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
                            content=f"Parse error: {exc}. Return ONLY the JSON object.",
                        )
                    )

        log.error(
            "Reflection exhausted retries. Last error: %s. Keeping current weights.", last_error
        )
        return None
