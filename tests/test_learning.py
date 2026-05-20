"""Tests for Phase 4 — attribution, reflection, and performance metrics.

Coverage:
    - ReflectionOutput schema: lessons, weight normalisation, validation
    - ReflectionEngine: happy path, retry on bad JSON, no-data path
    - PriceBasedAttributor: attributes correctly, skips missing prices
    - PerformanceTracker: daily return, Sharpe, drawdown, win rate
    - DB: signal_weights and performance_snapshots tables, queries
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.reasoning.schemas import (
    AttributedDecision,
    ReflectionInput,
    ReflectionOutput,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _attributed_decision(
    symbol: str = "BTC/USD",
    action: str = "buy",
    outcome: float = 0.02,
    composite: float = 0.6,
) -> AttributedDecision:
    return AttributedDecision(
        symbol=symbol,
        action=action,
        composite_signal=composite,
        signal_regime="trending_up",
        confidence=0.7,
        edge_estimate=0.55,
        ema_contribution=composite * (1/3),
        rsi_contribution=composite * (1/3),
        roc_contribution=composite * (1/3),
        outcome_pnl_pct=outcome,
        decided_at=datetime.now(tz=timezone.utc) - timedelta(hours=25),
    )


def _mock_adapter(response_content: str) -> MagicMock:
    from agent.reasoning.llm_base import LLMResponse
    adapter = MagicMock()
    adapter.model_id = "claude-test"
    adapter.complete = AsyncMock(
        return_value=LLMResponse(
            content=response_content,
            model="claude-test",
            input_tokens=200,
            output_tokens=80,
        )
    )
    return adapter


def _valid_reflection_json(
    ema: float = 0.40,
    rsi: float = 0.20,
    roc: float = 0.40,
) -> str:
    return json.dumps({
        "lessons": [
            "ROC momentum correctly predicted BTC upward move in trending_up regime.",
            "RSI reversion signal produced mixed results; reduce weight incrementally.",
        ],
        "weight_adjustments": {"ema": ema, "rsi": rsi, "roc": roc},
        "weight_rationale": "ROC and EMA both positive in trending session; RSI misfired.",
    })


# ── ReflectionOutput schema ───────────────────────────────────────────────────

def test_reflection_output_valid() -> None:
    out = ReflectionOutput.model_validate(json.loads(_valid_reflection_json()))
    assert len(out.lessons) == 2
    assert abs(sum(out.weight_adjustments.values()) - 1.0) < 1e-9


def test_reflection_output_weights_normalised() -> None:
    # Weights don't sum to 1.0 — they must be normalised
    data = json.loads(_valid_reflection_json(ema=2.0, rsi=1.0, roc=1.0))
    out = ReflectionOutput.model_validate(data)
    assert abs(sum(out.weight_adjustments.values()) - 1.0) < 1e-9
    assert abs(out.weight_adjustments["ema"] - 0.5) < 1e-9


def test_reflection_output_zero_weights_get_defaults() -> None:
    data = {"lessons": ["Test lesson for zero-weight fallback scenario."],
            "weight_adjustments": {"ema": 0.0, "rsi": 0.0, "roc": 0.0},
            "weight_rationale": "All weights zeroed out in test scenario."}
    out = ReflectionOutput.model_validate(data)
    # Should fall back to equal weights
    assert abs(out.weight_adjustments["ema"] - 1/3) < 1e-9


def test_reflection_output_missing_key_raises() -> None:
    data = {"lessons": ["lesson"], "weight_adjustments": {"ema": 0.5, "rsi": 0.5},
            "weight_rationale": "Missing roc key test."}
    with pytest.raises(Exception):
        ReflectionOutput.model_validate(data)


def test_reflection_output_short_rationale_raises() -> None:
    data = {"lessons": ["lesson"],
            "weight_adjustments": {"ema": 0.4, "rsi": 0.3, "roc": 0.3},
            "weight_rationale": "short"}  # < 20 chars
    with pytest.raises(Exception):
        ReflectionOutput.model_validate(data)


def test_reflection_output_json_schema_for_prompt() -> None:
    schema = ReflectionOutput.json_schema_for_prompt()
    assert "lessons" in schema
    assert "weight_adjustments" in schema


def test_reflection_output_negative_weights_clamped() -> None:
    data = {"lessons": ["Test clamping negative weights."],
            "weight_adjustments": {"ema": -0.1, "rsi": 0.6, "roc": 0.5},
            "weight_rationale": "Testing negative weight clamping to zero minimum."}
    out = ReflectionOutput.model_validate(data)
    # ema should be clamped to 0, then normalised
    assert out.weight_adjustments["ema"] == pytest.approx(0.0)
    assert abs(sum(out.weight_adjustments.values()) - 1.0) < 1e-9


# ── AttributedDecision ────────────────────────────────────────────────────────

def test_attributed_decision_fields() -> None:
    d = _attributed_decision(outcome=0.03)
    assert d.outcome_pnl_pct == pytest.approx(0.03)
    assert d.action == "buy"


# ── ReflectionEngine ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflection_engine_happy_path(tmp_db_path) -> None:
    from agent.learning.reflection import ReflectionEngine
    from agent.memory.db import init_db
    from agent.memory.queries import (
        get_current_weights,
        get_recent_lessons,
        save_decision,
    )
    from agent.reasoning.schemas import DecisionRecord
    from agent.strategy.signals import SignalEngine

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    # Insert an attributed decision (outcome set)
    record = DecisionRecord(
        symbol="BTC/USD", action="buy",
        rationale="Strong ROC signal in trending_up regime confirmed clearly.",
        confidence=0.75, edge_estimate=0.6, risk_notes="",
        composite_signal=0.6, signal_regime="trending_up",
        portfolio_value_at_decision="100000",
        decided_at=datetime.now(tz=timezone.utc) - timedelta(hours=26),
        dry_run=True, llm_model="claude-test",
        outcome_pnl_pct=0.025,
    )
    async with factory() as session:
        await save_decision(session, record)

    adapter = _mock_adapter(_valid_reflection_json())
    engine_sig = SignalEngine()
    engine = ReflectionEngine(adapter=adapter, signal_engine=engine_sig)

    async with factory() as session:
        result = await engine.run(session)

    assert result is not None
    assert len(result.lessons) == 2
    assert abs(sum(result.weight_adjustments.values()) - 1.0) < 1e-9

    # Weights should have been persisted
    async with factory() as session:
        w = await get_current_weights(session)
    assert w is not None
    assert abs(w["ema"] - result.weight_adjustments["ema"]) < 1e-6

    # Lessons should be in DB
    async with factory() as session:
        lessons = await get_recent_lessons(session, limit=10)
    assert len(lessons) >= 2


@pytest.mark.asyncio
async def test_reflection_engine_no_data_saves_patience_lesson(tmp_db_path) -> None:
    from agent.learning.reflection import ReflectionEngine
    from agent.memory.db import init_db
    from agent.memory.queries import get_recent_lessons
    from agent.strategy.signals import SignalEngine

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    adapter = _mock_adapter(_valid_reflection_json())
    engine = ReflectionEngine(adapter=_mock_adapter(""), signal_engine=SignalEngine())

    async with factory() as session:
        result = await engine.run(session)

    assert result is None  # no attributed decisions

    async with factory() as session:
        lessons = await get_recent_lessons(session)
    assert len(lessons) == 1
    assert "patience" in lessons[0].lesson.lower() or "no attributed" in lessons[0].lesson.lower()


@pytest.mark.asyncio
async def test_reflection_engine_retries_on_bad_json(tmp_db_path) -> None:
    from agent.learning.reflection import ReflectionEngine
    from agent.memory.db import init_db
    from agent.memory.queries import save_decision
    from agent.reasoning.llm_base import LLMResponse
    from agent.reasoning.schemas import DecisionRecord
    from agent.strategy.signals import SignalEngine

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    record = DecisionRecord(
        symbol="ETH/USD", action="sell",
        rationale="Strong downtrend signal confirmed by RSI and ROC convergence.",
        confidence=0.65, edge_estimate=0.55, risk_notes="",
        composite_signal=-0.5, signal_regime="trending_down",
        portfolio_value_at_decision="100000",
        decided_at=datetime.now(tz=timezone.utc) - timedelta(hours=26),
        dry_run=True, llm_model="claude-test",
        outcome_pnl_pct=0.018,
    )
    async with factory() as session:
        await save_decision(session, record)

    bad = LLMResponse(content="This is not JSON.", model="t", input_tokens=0, output_tokens=0)
    good = LLMResponse(content=_valid_reflection_json(), model="t", input_tokens=0, output_tokens=0)

    adapter = MagicMock()
    adapter.model_id = "claude-test"
    adapter.complete = AsyncMock(side_effect=[bad, good])

    engine = ReflectionEngine(adapter=adapter, signal_engine=SignalEngine())
    async with factory() as session:
        result = await engine.run(session)

    assert result is not None
    assert adapter.complete.call_count == 2


@pytest.mark.asyncio
async def test_reflection_engine_updates_signal_weights(tmp_db_path) -> None:
    from agent.learning.reflection import ReflectionEngine
    from agent.memory.db import init_db
    from agent.memory.queries import save_decision
    from agent.reasoning.schemas import DecisionRecord
    from agent.strategy.signals import SignalEngine

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    record = DecisionRecord(
        symbol="BTC/USD", action="buy",
        rationale="EMA and ROC both bullish in strong trending regime today.",
        confidence=0.8, edge_estimate=0.65, risk_notes="",
        composite_signal=0.7, signal_regime="trending_up",
        portfolio_value_at_decision="100000",
        decided_at=datetime.now(tz=timezone.utc) - timedelta(hours=26),
        dry_run=True, llm_model="claude-test",
        outcome_pnl_pct=0.03,
    )
    async with factory() as session:
        await save_decision(session, record)

    sig_engine = SignalEngine()
    original_weights = dict(sig_engine.weights)

    adapter = _mock_adapter(_valid_reflection_json(ema=0.45, rsi=0.15, roc=0.40))
    engine = ReflectionEngine(adapter=adapter, signal_engine=sig_engine)

    async with factory() as session:
        await engine.run(session)

    # Signal engine weights should be updated
    assert abs(sig_engine.weights["ema"] - 0.45) < 1e-6
    assert abs(sig_engine.weights["rsi"] - 0.15) < 1e-6


# ── PriceBasedAttributor ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_attributor_buy_positive_outcome(tmp_db_path) -> None:
    from agent.learning.attribution import PriceBasedAttributor
    from agent.memory.db import init_db
    from agent.memory.queries import get_decisions_since, save_decision
    from agent.reasoning.schemas import DecisionRecord

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    record = DecisionRecord(
        symbol="BTC/USD", action="buy",
        rationale="ROC momentum confirmed upward trend with strong confluence.",
        confidence=0.75, edge_estimate=0.6, risk_notes="",
        composite_signal=0.6, signal_regime="trending_up",
        portfolio_value_at_decision="100000",
        decided_at=datetime.now(tz=timezone.utc) - timedelta(hours=26),
        dry_run=True, llm_model="claude-test",
    )
    async with factory() as session:
        await save_decision(session, record)

    attributor = PriceBasedAttributor(attribution_window_hours=24)
    entry_prices = {"BTC/USD": 80_000.0}
    current_prices = {"BTC/USD": 81_600.0}  # +2%

    async with factory() as session:
        n = await attributor.attribute_with_prices(session, entry_prices, current_prices)

    assert n == 1

    since = datetime.now(tz=timezone.utc) - timedelta(days=2)
    async with factory() as session:
        rows = await get_decisions_since(session, since=since, with_outcome_only=True)

    assert len(rows) == 1
    assert abs(rows[0].outcome_pnl_pct - 0.02) < 1e-6


@pytest.mark.asyncio
async def test_attributor_sell_positive_outcome(tmp_db_path) -> None:
    from agent.learning.attribution import PriceBasedAttributor
    from agent.memory.db import init_db
    from agent.memory.queries import get_decisions_since, save_decision
    from agent.reasoning.schemas import DecisionRecord

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    record = DecisionRecord(
        symbol="ETH/USD", action="sell",
        rationale="Downtrend confirmed by all three signals in ranging regime.",
        confidence=0.65, edge_estimate=0.55, risk_notes="",
        composite_signal=-0.5, signal_regime="trending_down",
        portfolio_value_at_decision="100000",
        decided_at=datetime.now(tz=timezone.utc) - timedelta(hours=26),
        dry_run=True, llm_model="claude-test",
    )
    async with factory() as session:
        await save_decision(session, record)

    attributor = PriceBasedAttributor(attribution_window_hours=24)
    entry_prices = {"ETH/USD": 3_000.0}
    current_prices = {"ETH/USD": 2_940.0}  # -2% → sell was correct → +2% P&L

    async with factory() as session:
        n = await attributor.attribute_with_prices(session, entry_prices, current_prices)

    assert n == 1

    since = datetime.now(tz=timezone.utc) - timedelta(days=2)
    async with factory() as session:
        rows = await get_decisions_since(session, since=since, with_outcome_only=True)

    assert abs(rows[0].outcome_pnl_pct - 0.02) < 1e-6


@pytest.mark.asyncio
async def test_attributor_skips_missing_price(tmp_db_path) -> None:
    from agent.learning.attribution import PriceBasedAttributor
    from agent.memory.db import init_db
    from agent.memory.queries import save_decision
    from agent.reasoning.schemas import DecisionRecord

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    record = DecisionRecord(
        symbol="SOL/USD", action="buy",
        rationale="Signal shows bullish divergence with upward momentum pattern.",
        confidence=0.6, edge_estimate=0.5, risk_notes="",
        composite_signal=0.4, signal_regime="ranging",
        portfolio_value_at_decision="100000",
        decided_at=datetime.now(tz=timezone.utc) - timedelta(hours=26),
        dry_run=True, llm_model="claude-test",
    )
    async with factory() as session:
        await save_decision(session, record)

    attributor = PriceBasedAttributor(attribution_window_hours=24)
    async with factory() as session:
        n = await attributor.attribute_with_prices(session, {}, {})  # no prices

    assert n == 0  # skipped


@pytest.mark.asyncio
async def test_attributor_skips_recent_decisions(tmp_db_path) -> None:
    from agent.learning.attribution import PriceBasedAttributor
    from agent.memory.db import init_db
    from agent.memory.queries import save_decision
    from agent.reasoning.schemas import DecisionRecord

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    # Decision only 1h old — should not be attributed yet (window=24h)
    record = DecisionRecord(
        symbol="BTC/USD", action="buy",
        rationale="Recent decision that should not yet be attributed by window.",
        confidence=0.7, edge_estimate=0.6, risk_notes="",
        composite_signal=0.5, signal_regime="trending_up",
        portfolio_value_at_decision="100000",
        decided_at=datetime.now(tz=timezone.utc) - timedelta(hours=1),
        dry_run=True, llm_model="claude-test",
    )
    async with factory() as session:
        await save_decision(session, record)

    attributor = PriceBasedAttributor(attribution_window_hours=24)
    async with factory() as session:
        n = await attributor.attribute_with_prices(
            session, {"BTC/USD": 80_000.0}, {"BTC/USD": 81_000.0}
        )

    assert n == 0  # too recent


# ── PerformanceTracker ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_performance_snapshot_first_cycle(tmp_db_path) -> None:
    from agent.learning.metrics import PerformanceTracker
    from agent.memory.db import init_db
    from agent.memory.queries import get_latest_performance

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")
    tracker = PerformanceTracker(initial_portfolio_value=Decimal("100000"))

    async with factory() as session:
        metrics = await tracker.snapshot(
            session, portfolio_value=Decimal("100500"),
            total_decisions=5, total_trades=2,
        )

    assert metrics.portfolio_value == "100500"
    assert metrics.total_decisions == 5
    assert metrics.total_trades == 2
    # Only one snapshot — Sharpe should be None
    assert metrics.sharpe_30d is None

    async with factory() as session:
        latest = await get_latest_performance(session)
    assert latest is not None
    assert latest.total_decisions == 5


@pytest.mark.asyncio
async def test_performance_cumulative_return(tmp_db_path) -> None:
    from agent.learning.metrics import PerformanceTracker
    from agent.memory.db import init_db

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")
    tracker = PerformanceTracker(initial_portfolio_value=Decimal("100000"))

    async with factory() as session:
        metrics = await tracker.snapshot(
            session, portfolio_value=Decimal("103000"),
            total_decisions=10, total_trades=4,
        )

    assert metrics.cumulative_return_pct == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_performance_win_rate(tmp_db_path) -> None:
    from agent.learning.metrics import PerformanceTracker
    from agent.memory.db import init_db

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")
    tracker = PerformanceTracker(initial_portfolio_value=Decimal("100000"))

    outcomes = [0.02, -0.01, 0.03, 0.015, -0.005]  # 3 wins out of 5

    async with factory() as session:
        metrics = await tracker.snapshot(
            session, portfolio_value=Decimal("101000"),
            total_decisions=5, total_trades=5,
            attributed_outcomes=outcomes,
        )

    assert metrics.win_rate == pytest.approx(3 / 5)


@pytest.mark.asyncio
async def test_performance_sharpe_computed_after_min_periods(tmp_db_path) -> None:
    from agent.learning.metrics import PerformanceTracker
    from agent.memory.db import init_db

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")
    tracker = PerformanceTracker(initial_portfolio_value=Decimal("100000"))

    # Need at least MIN_SHARPE_PERIODS (24) snapshots in history
    base = 100_000
    values = [base + i * 200 for i in range(24)]  # 24 monotone steps
    for v in values:
        async with factory() as session:
            await tracker.snapshot(
                session, portfolio_value=Decimal(str(v)),
                total_decisions=1, total_trades=0,
            )

    # 25th snapshot should have a computed Sharpe (24 periods of history available)
    async with factory() as session:
        metrics = await tracker.snapshot(
            session, portfolio_value=Decimal("104900"),
            total_decisions=1, total_trades=0,
        )

    assert metrics.sharpe_30d is not None
    assert isinstance(metrics.sharpe_30d, float)


def test_performance_win_rate_none_for_empty() -> None:
    from agent.learning.metrics import _compute_win_rate
    assert _compute_win_rate(None) is None
    assert _compute_win_rate([]) is None


def test_performance_max_drawdown() -> None:
    from agent.learning.metrics import _compute_max_drawdown

    class FakeRow:
        def __init__(self, v):
            self.portfolio_value = str(v)

    history = [FakeRow(100_000), FakeRow(105_000), FakeRow(98_000)]
    dd = _compute_max_drawdown(history, 99_000)
    # Peak was 105_000, trough is 98_000: drawdown = (105k-98k)/105k ≈ 6.67%
    assert dd == pytest.approx((105_000 - 98_000) / 105_000, rel=1e-3)


# ── Signal weights DB queries ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_get_weights(tmp_db_path) -> None:
    from agent.memory.db import init_db
    from agent.memory.queries import get_current_weights, save_weights

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    async with factory() as session:
        wid = await save_weights(
            session, ema=0.40, rsi=0.20, roc=0.40,
            source="reflection", rationale="ROC performed best this session.",
        )
    assert wid > 0

    async with factory() as session:
        w = await get_current_weights(session)
    assert w is not None
    assert w["ema"] == pytest.approx(0.40)
    assert w["roc"] == pytest.approx(0.40)


@pytest.mark.asyncio
async def test_get_current_weights_returns_none_when_empty(tmp_db_path) -> None:
    from agent.memory.db import init_db
    from agent.memory.queries import get_current_weights

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    async with factory() as session:
        w = await get_current_weights(session)
    assert w is None


@pytest.mark.asyncio
async def test_get_current_weights_returns_latest(tmp_db_path) -> None:
    from agent.memory.db import init_db
    from agent.memory.queries import get_current_weights, save_weights

    _, factory = await init_db(f"sqlite+aiosqlite:///{tmp_db_path}")

    async with factory() as session:
        await save_weights(session, ema=0.33, rsi=0.33, roc=0.34, rationale="First set.")
        await save_weights(session, ema=0.45, rsi=0.15, roc=0.40, rationale="Updated set.")

    async with factory() as session:
        w = await get_current_weights(session)
    assert w["ema"] == pytest.approx(0.45)


# ── Reflection prompt rendering ───────────────────────────────────────────────

def test_reflection_prompt_renders() -> None:
    from agent.learning.reflection import _render_reflection_prompt

    ctx = ReflectionInput(
        reflection_date=datetime.now(tz=timezone.utc),
        decisions=[_attributed_decision()],
        current_weights={"ema": 1/3, "rsi": 1/3, "roc": 1/3},
        portfolio_value="100000",
        cumulative_return_pct=0.015,
    )
    rendered = _render_reflection_prompt(ctx)
    assert "BTC/USD" in rendered
    assert "buy" in rendered.lower()
    assert "ema" in rendered.lower()


def test_reflection_prompt_no_decisions() -> None:
    from agent.learning.reflection import _render_reflection_prompt

    ctx = ReflectionInput(
        reflection_date=datetime.now(tz=timezone.utc),
        decisions=[],
        current_weights={"ema": 1/3, "rsi": 1/3, "roc": 1/3},
        portfolio_value="100000",
        cumulative_return_pct=None,
    )
    rendered = _render_reflection_prompt(ctx)
    assert "No attributed decisions" in rendered
