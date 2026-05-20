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


# ── Phase 3: Schema tests ─────────────────────────────────────────────────────

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from agent.reasoning.schemas import (
    AccountSnapshot,
    DecisionInput,
    DecisionRecord,
    PositionSnapshot,
    RecentDecision,
    RecentLesson,
    SignalSnapshot,
    TradingDecision,
)


def _signal_snap(
    symbol: str = "BTC/USD",
    direction: str = "buy",
    composite: float = 0.6,
) -> SignalSnapshot:
    return SignalSnapshot(
        symbol=symbol,
        direction=direction,
        composite=composite,
        confidence=abs(composite),
        ema_score=0.5,
        rsi_score=-0.2,
        roc_score=0.8,
        regime="trending_up",
        close_price=80_000.0,
        signal_weights={"ema": 1/3, "rsi": 1/3, "roc": 1/3},
    )


def _account_snap(circuit_breaker: bool = False) -> AccountSnapshot:
    return AccountSnapshot(
        portfolio_value="100000",
        cash="95000",
        cash_pct=0.95,
        daily_pnl_pct=0.01,
        open_position_count=1,
        circuit_breaker_active=circuit_breaker,
    )


def _decision_input(
    symbol: str = "BTC/USD",
    direction: str = "buy",
    dry_run: bool = True,
) -> DecisionInput:
    return DecisionInput(
        timestamp=datetime.now(tz=timezone.utc),
        symbol=symbol,
        signal=_signal_snap(symbol=symbol, direction=direction),
        account=_account_snap(),
        dry_run=dry_run,
    )


def _valid_decision_json(
    action: str = "buy",
    symbol: str = "BTC/USD",
    dry_run: bool = True,
) -> str:
    return json.dumps({
        "action": action,
        "symbol": symbol,
        "rationale": "Composite is +0.68 in trending_up regime with 90% volume confirmation. ROC and EMA both bullish with 100% sub-signal agreement. All three MTF timeframes aligned.",
        "confidence": 0.75,
        "edge_estimate": 0.6,
        "risk_notes": "Position near concentration limit.",
        "dry_run": dry_run,
    })


def _mock_adapter(response_content: str) -> MagicMock:
    from agent.reasoning.llm_base import LLMResponse
    adapter = MagicMock()
    adapter.model_id = "claude-test"
    adapter.complete = AsyncMock(
        return_value=LLMResponse(
            content=response_content,
            model="claude-test",
            input_tokens=500,
            output_tokens=100,
        )
    )
    return adapter


def test_trading_decision_valid_buy() -> None:
    d = TradingDecision(
        action="buy",
        symbol="BTC/USD",
        rationale="Composite is +0.72 in trending_up regime with 85% volume confirmation and full MTF alignment. EMA and ROC both bullish; RSI momentum confirms. Strong setup for entry.",
        confidence=0.8,
        edge_estimate=0.65,
        risk_notes="Momentum could fade if BTC reverses; watch for regime change.",
        dry_run=True,
    )
    assert d.action == "buy"
    assert d.confidence == 0.8


def test_trading_decision_valid_hold() -> None:
    d = TradingDecision(
        action="hold",
        symbol="ETH/USD",
        rationale="Composite is near zero at +0.03 in ranging regime with only 45% volume confirmation. Sub-signal agreement is split; no directional edge to act on.",
        confidence=0.65,
        edge_estimate=0.0,
        risk_notes="Regime may shift to trending_up if volume picks up; re-evaluate next cycle.",
        dry_run=True,
    )
    assert d.action == "hold"


def test_trading_decision_hold_clamps_edge() -> None:
    d = TradingDecision(
        action="hold",
        symbol="BTC/USD",
        rationale="Clamping validation: composite is -0.05 in ranging regime with 50% volume confirmation. Edge estimate provided is invalid for hold and should be clamped to zero.",
        confidence=0.6,
        edge_estimate=0.9,
        risk_notes="This is a schema validation test; edge_estimate must be clamped to 0.0 for holds.",
        dry_run=True,
    )
    assert d.edge_estimate == 0.0


def test_trading_decision_confidence_clamped_above_1() -> None:
    d = TradingDecision(
        action="buy",
        symbol="BTC/USD",
        rationale="Clamp test: composite is +0.80 in trending_up regime with full volume and MTF alignment. Confidence value of 5.0 exceeds the valid range and must be clamped to 1.0.",
        confidence=5.0,
        edge_estimate=0.5,
        dry_run=True,
    )
    assert d.confidence == 1.0


def test_trading_decision_confidence_clamped_below_0() -> None:
    d = TradingDecision(
        action="sell",
        symbol="SOL/USD",
        rationale="Clamp test: composite is -0.60 in trending_down regime with RSI overbought. Confidence of -0.5 is below valid range and must be clamped to 0.0 by the validator.",
        confidence=-0.5,
        edge_estimate=0.3,
        dry_run=True,
    )
    assert d.confidence == 0.0


def test_trading_decision_invalid_action() -> None:
    with pytest.raises(Exception):
        TradingDecision(
            action="maybe",  # type: ignore[arg-type]
            symbol="BTC/USD",
            rationale="Invalid action test: composite is +0.40 in trending_up regime. Action 'maybe' is not a valid literal and should fail schema validation immediately.",
            confidence=0.5,
            edge_estimate=0.5,
            dry_run=True,
        )


def test_trading_decision_rationale_too_short() -> None:
    with pytest.raises(Exception):
        TradingDecision(
            action="buy",
            symbol="BTC/USD",
            rationale="short",  # < 20 chars
            confidence=0.5,
            edge_estimate=0.5,
            dry_run=True,
        )


def test_trading_decision_json_schema_for_prompt() -> None:
    schema = TradingDecision.json_schema_for_prompt()
    assert "action" in schema
    assert "confidence" in schema
    assert "edge_estimate" in schema


def test_decision_input_symbol_mismatch_raises() -> None:
    with pytest.raises(Exception, match="signal.symbol"):
        DecisionInput(
            timestamp=datetime.now(tz=timezone.utc),
            symbol="BTC/USD",
            signal=_signal_snap(symbol="ETH/USD"),
            account=_account_snap(),
        )


def test_decision_input_with_positions() -> None:
    pos = PositionSnapshot(
        symbol="BTC/USD",
        qty="0.1",
        market_value="8000",
        unrealized_pnl="500",
        unrealized_pnl_pct=0.065,
        concentration_pct=0.08,
    )
    ctx = DecisionInput(
        timestamp=datetime.now(tz=timezone.utc),
        symbol="BTC/USD",
        signal=_signal_snap(),
        account=_account_snap(),
        open_positions=[pos],
    )
    assert len(ctx.open_positions) == 1


def test_decision_input_with_lessons() -> None:
    lesson = RecentLesson(
        lesson="ROC signal reliably precedes 4h momentum in BTC.",
        source="reflection",
        created_at=datetime.now(tz=timezone.utc),
    )
    ctx = DecisionInput(
        timestamp=datetime.now(tz=timezone.utc),
        symbol="BTC/USD",
        signal=_signal_snap(),
        account=_account_snap(),
        recent_lessons=[lesson],
    )
    assert len(ctx.recent_lessons) == 1


# ── ReasoningAgent tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reasoning_agent_happy_path() -> None:
    from agent.reasoning.agent import ReasoningAgent
    agent = ReasoningAgent(adapter=_mock_adapter(_valid_decision_json()))
    decision = await agent.decide(_decision_input())
    assert decision.action == "buy"
    assert decision.symbol == "BTC/USD"
    assert decision.confidence == pytest.approx(0.75)
    assert decision.dry_run is True


@pytest.mark.asyncio
async def test_reasoning_agent_sell_decision() -> None:
    from agent.reasoning.agent import ReasoningAgent
    agent = ReasoningAgent(adapter=_mock_adapter(_valid_decision_json(action="sell")))
    decision = await agent.decide(_decision_input(direction="sell"))
    assert decision.action == "sell"


@pytest.mark.asyncio
async def test_reasoning_agent_hold_decision() -> None:
    from agent.reasoning.agent import ReasoningAgent
    hold_json = json.dumps({
        "action": "hold",
        "symbol": "BTC/USD",
        "rationale": "Composite is +0.02 in ranging regime with 40% volume confirmation and mixed MTF signals. No directional edge exists; all three signals show weak and conflicting readings.",
        "confidence": 0.65,
        "edge_estimate": 0.0,
        "risk_notes": "Volume below threshold reduces signal reliability; hold until 4h and 1h align.",
        "dry_run": True,
    })
    agent = ReasoningAgent(adapter=_mock_adapter(hold_json))
    decision = await agent.decide(_decision_input())
    assert decision.action == "hold"
    assert decision.edge_estimate == 0.0


@pytest.mark.asyncio
async def test_reasoning_agent_strips_markdown_fence() -> None:
    from agent.reasoning.agent import ReasoningAgent
    fenced = f"```json\n{_valid_decision_json()}\n```"
    agent = ReasoningAgent(adapter=_mock_adapter(fenced))
    decision = await agent.decide(_decision_input())
    assert decision.action == "buy"


@pytest.mark.asyncio
async def test_reasoning_agent_retries_on_bad_json() -> None:
    from agent.reasoning.agent import ReasoningAgent
    from agent.reasoning.llm_base import LLMResponse

    bad = LLMResponse(content="I think you should buy.", model="t", input_tokens=10, output_tokens=10)
    good = LLMResponse(content=_valid_decision_json(), model="t", input_tokens=10, output_tokens=10)

    adapter = MagicMock()
    adapter.model_id = "claude-test"
    adapter.complete = AsyncMock(side_effect=[bad, good])

    agent = ReasoningAgent(adapter=adapter)
    decision = await agent.decide(_decision_input())
    assert decision.action == "buy"
    assert adapter.complete.call_count == 2


@pytest.mark.asyncio
async def test_reasoning_agent_fallback_after_all_retries_fail() -> None:
    from agent.reasoning.agent import ReasoningAgent
    from agent.reasoning.llm_base import LLMResponse

    always_bad = LLMResponse(content="not json at all", model="t", input_tokens=0, output_tokens=0)
    adapter = MagicMock()
    adapter.model_id = "claude-test"
    adapter.complete = AsyncMock(return_value=always_bad)

    agent = ReasoningAgent(adapter=adapter)
    decision = await agent.decide(_decision_input())
    assert decision.action == "hold"
    assert "parse failure" in decision.rationale.lower() or "safety hold" in decision.rationale.lower()
    assert adapter.complete.call_count == 3  # MAX_RETRIES


@pytest.mark.asyncio
async def test_reasoning_agent_symbol_mismatch_retries() -> None:
    from agent.reasoning.agent import ReasoningAgent
    from agent.reasoning.llm_base import LLMResponse

    wrong = LLMResponse(content=_valid_decision_json(symbol="ETH/USD"), model="t", input_tokens=0, output_tokens=0)
    correct = LLMResponse(content=_valid_decision_json(symbol="BTC/USD"), model="t", input_tokens=0, output_tokens=0)

    adapter = MagicMock()
    adapter.model_id = "claude-test"
    adapter.complete = AsyncMock(side_effect=[wrong, correct])

    agent = ReasoningAgent(adapter=adapter)
    decision = await agent.decide(_decision_input(symbol="BTC/USD"))
    assert decision.symbol == "BTC/USD"
    assert adapter.complete.call_count == 2


@pytest.mark.asyncio
async def test_reasoning_agent_dry_run_override() -> None:
    from agent.reasoning.agent import ReasoningAgent
    from agent.reasoning.llm_base import LLMResponse

    wrong_dry_run = LLMResponse(
        content=_valid_decision_json(dry_run=False),
        model="t", input_tokens=0, output_tokens=0,
    )
    adapter = MagicMock()
    adapter.model_id = "claude-test"
    adapter.complete = AsyncMock(return_value=wrong_dry_run)

    agent = ReasoningAgent(adapter=adapter)
    decision = await agent.decide(_decision_input(dry_run=True))
    assert decision.dry_run is True


@pytest.mark.asyncio
async def test_build_record_fields() -> None:
    from agent.reasoning.agent import ReasoningAgent
    agent = ReasoningAgent(adapter=_mock_adapter(_valid_decision_json()))
    ctx = _decision_input()
    decision = await agent.decide(ctx)
    record = agent.build_record(ctx=ctx, decision=decision, input_tokens=500, output_tokens=100, order_id="ord-abc")
    assert isinstance(record, DecisionRecord)
    assert record.symbol == "BTC/USD"
    assert record.action == "buy"
    assert record.llm_model == "claude-test"
    assert record.input_tokens == 500
    assert record.order_id == "ord-abc"
    assert record.dry_run is True


# ── DB round-trip ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_retrieve_decision(tmp_db_path) -> None:
    from agent.memory.db import init_db
    from agent.memory.queries import get_recent_decisions, save_decision

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")
    record = DecisionRecord(
        symbol="BTC/USD",
        action="buy",
        rationale="Test rationale: strong signal in trending regime today.",
        confidence=0.75,
        edge_estimate=0.6,
        risk_notes="",
        composite_signal=0.6,
        signal_regime="trending_up",
        portfolio_value_at_decision="100000",
        decided_at=datetime.now(tz=timezone.utc),
        dry_run=True,
        llm_model="claude-test",
        input_tokens=500,
        output_tokens=100,
    )

    async with factory() as session:
        did = await save_decision(session, record)
    assert did > 0

    async with factory() as session:
        decisions = await get_recent_decisions(session, symbol="BTC/USD", limit=5)
    assert len(decisions) == 1
    assert decisions[0].action == "buy"


@pytest.mark.asyncio
async def test_save_and_retrieve_lesson(tmp_db_path) -> None:
    from agent.memory.db import init_db
    from agent.memory.queries import get_recent_lessons, save_lesson

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    async with factory() as session:
        lid = await save_lesson(session, lesson="ROC leads reversals by 1-2 bars in BTC.", source="reflection", symbol="BTC/USD")
    assert lid > 0

    async with factory() as session:
        lessons = await get_recent_lessons(session, symbol="BTC/USD", limit=5)
    assert len(lessons) == 1
    assert "ROC" in lessons[0].lesson


@pytest.mark.asyncio
async def test_get_recent_decisions_empty(tmp_db_path) -> None:
    from agent.memory.db import init_db
    from agent.memory.queries import get_recent_decisions
    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")
    async with factory() as session:
        assert await get_recent_decisions(session) == []


@pytest.mark.asyncio
async def test_get_recent_lessons_empty(tmp_db_path) -> None:
    from agent.memory.db import init_db
    from agent.memory.queries import get_recent_lessons
    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")
    async with factory() as session:
        assert await get_recent_lessons(session) == []


@pytest.mark.asyncio
async def test_update_decision_outcome(tmp_db_path) -> None:
    from agent.memory.db import init_db
    from agent.memory.queries import get_recent_decisions, save_decision, update_decision_outcome

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")
    record = DecisionRecord(
        symbol="ETH/USD",
        action="sell",
        rationale="Downtrend detected; RSI overbought; exit signal confirmed clearly.",
        confidence=0.65,
        edge_estimate=0.55,
        risk_notes="",
        composite_signal=-0.5,
        signal_regime="trending_down",
        portfolio_value_at_decision="100000",
        decided_at=datetime.now(tz=timezone.utc),
        dry_run=True,
        llm_model="claude-test",
    )
    async with factory() as session:
        did = await save_decision(session, record)
        await update_decision_outcome(session, did, outcome_pnl_pct=0.034)

    async with factory() as session:
        decisions = await get_recent_decisions(session, symbol="ETH/USD")
    assert decisions[0].outcome_pnl_pct == pytest.approx(0.034)


# ── Prompt rendering ──────────────────────────────────────────────────────────

def test_prompt_renders_without_error() -> None:
    from agent.reasoning.agent import _render_prompt
    rendered = _render_prompt(_decision_input())
    assert "BTC/USD" in rendered
    assert "buy" in rendered.lower()


def test_prompt_contains_signal_values() -> None:
    from agent.reasoning.agent import _render_prompt
    rendered = _render_prompt(_decision_input())
    assert "trending_up" in rendered
    assert "80" in rendered  # price somewhere in output


def test_prompt_with_lessons_and_decisions() -> None:
    from agent.reasoning.agent import _render_prompt
    ctx = DecisionInput(
        timestamp=datetime.now(tz=timezone.utc),
        symbol="BTC/USD",
        signal=_signal_snap(),
        account=_account_snap(),
        recent_lessons=[
            RecentLesson(
                lesson="ROC leads price in trending regimes.",
                source="reflection",
                created_at=datetime.now(tz=timezone.utc),
            )
        ],
        recent_decisions=[
            RecentDecision(
                symbol="BTC/USD",
                action="buy",
                rationale_summary="Strong momentum in trending_up regime above threshold.",
                confidence=0.7,
                outcome_pnl_pct=0.02,
                decided_at=datetime.now(tz=timezone.utc),
            )
        ],
    )
    rendered = _render_prompt(ctx)
    assert "ROC leads price" in rendered
