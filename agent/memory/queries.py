"""Named async queries for reading and writing agent memory.

All functions accept a SQLAlchemy AsyncSession. The caller (main loop /
reflection job) manages session lifecycle via context managers.

Naming conventions:
    save_*     — INSERT / UPDATE
    get_*      — SELECT (returns domain objects, not ORM rows)
    mark_*     — UPDATE a flag column
    count_*    — COUNT aggregate
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent.memory.db import (
    AgentStateRow,
    DecisionRow,
    LessonRow,
    PerformanceSnapshotRow,
    SignalWeightsRow,
    TradeRow,
)
from agent.reasoning.schemas import DecisionRecord, RecentDecision, RecentLesson

log = logging.getLogger(__name__)

# Reflection lessons expire after this many days; manual lessons never expire.
_LESSON_TTL_DAYS = 30


# ── Decision queries ──────────────────────────────────────────────────────────

async def save_decision(session: AsyncSession, record: DecisionRecord) -> int:
    """Insert a decision row and return its auto-assigned id."""
    row = DecisionRow(
        symbol=record.symbol,
        action=record.action,
        rationale=record.rationale,
        confidence=record.confidence,
        edge_estimate=record.edge_estimate,
        risk_notes=record.risk_notes,
        composite_signal=record.composite_signal,
        signal_regime=record.signal_regime,
        portfolio_value_at_decision=record.portfolio_value_at_decision,
        decided_at=record.decided_at.isoformat(),
        dry_run=record.dry_run,
        llm_model=record.llm_model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        order_id=record.order_id,
        outcome_pnl_pct=record.outcome_pnl_pct,
        close_price_at_decision=record.close_price_at_decision,
        signal_weights_json=record.signal_weights_json,
    )
    session.add(row)
    await session.flush()
    await session.commit()
    log.debug("Saved decision id=%d symbol=%s action=%s", row.id, row.symbol, row.action)
    return row.id  # type: ignore[return-value]


async def get_recent_decisions(
    session: AsyncSession,
    symbol: str | None = None,
    limit: int = 10,
) -> list[RecentDecision]:
    """Return recent decisions, optionally filtered by symbol (most recent first)."""
    stmt = select(DecisionRow).order_by(DecisionRow.decided_at.desc()).limit(limit)
    if symbol:
        stmt = stmt.where(DecisionRow.symbol == symbol)

    rows = (await session.execute(stmt)).scalars().all()
    return [
        RecentDecision(
            symbol=r.symbol,
            action=r.action,
            rationale_summary=r.rationale[:200],
            confidence=r.confidence,
            outcome_pnl_pct=r.outcome_pnl_pct,
            decided_at=_parse_dt(r.decided_at),
        )
        for r in rows
    ]


async def update_decision_outcome(
    session: AsyncSession,
    decision_id: int,
    outcome_pnl_pct: float,
) -> None:
    """Update the outcome P&L for a decision once attribution runs."""
    await session.execute(
        update(DecisionRow)
        .where(DecisionRow.id == decision_id)
        .values(outcome_pnl_pct=outcome_pnl_pct)
    )
    await session.commit()


async def link_order_to_decision(
    session: AsyncSession,
    decision_id: int,
    order_id: str,
) -> None:
    """Attach the broker order ID to the decision row."""
    await session.execute(
        update(DecisionRow)
        .where(DecisionRow.id == decision_id)
        .values(order_id=order_id)
    )
    await session.commit()


async def count_all_decisions(session: AsyncSession) -> int:
    """Return total number of decisions ever recorded."""
    result = await session.execute(select(func.count()).select_from(DecisionRow))
    return result.scalar() or 0


# ── Trade queries ─────────────────────────────────────────────────────────────

async def save_trade(
    session: AsyncSession,
    *,
    decision_id: int | None,
    order_id: str,
    symbol: str,
    side: str,
    qty: str,
    status: str,
    submitted_at: datetime,
    dry_run: bool,
    fill_price: str | None = None,
    order_value: float | None = None,
    filled_at: datetime | None = None,
) -> int:
    """Insert a trade row. Returns the new row id."""
    row = TradeRow(
        decision_id=decision_id,
        order_id=order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        fill_price=fill_price,
        order_value=order_value,
        status=status,
        submitted_at=submitted_at.isoformat(),
        filled_at=filled_at.isoformat() if filled_at else None,
        dry_run=dry_run,
    )
    session.add(row)
    await session.flush()
    await session.commit()
    return row.id  # type: ignore[return-value]


async def get_pending_trades(session: AsyncSession) -> list[TradeRow]:
    """Return all trades in pending or partially-filled status."""
    stmt = (
        select(TradeRow)
        .where(TradeRow.status.in_(["pending", "partially_filled"]))
        .order_by(TradeRow.submitted_at.asc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def update_trade_fill(
    session: AsyncSession,
    order_id: str,
    status: str,
    fill_price: str | None,
    filled_at: datetime | None,
    order_value: float | None,
) -> None:
    """Record a fill (or cancellation) for a pending trade."""
    values: dict = {"status": status}
    if fill_price is not None:
        values["fill_price"] = fill_price
    if filled_at is not None:
        values["filled_at"] = filled_at.isoformat()
    if order_value is not None:
        values["order_value"] = order_value
    if status in ("cancelled", "stale"):
        values["cancelled_at"] = datetime.now(tz=timezone.utc).isoformat()

    await session.execute(
        update(TradeRow).where(TradeRow.order_id == order_id).values(**values)
    )
    await session.commit()


async def count_all_trades(session: AsyncSession) -> int:
    """Return total number of trades ever recorded."""
    result = await session.execute(select(func.count()).select_from(TradeRow))
    return result.scalar() or 0


# ── Lesson queries ────────────────────────────────────────────────────────────

async def save_lesson(
    session: AsyncSession,
    lesson: str,
    source: str = "reflection",
    symbol: str | None = None,
) -> int:
    """Insert a lesson row and return its id.

    Reflection lessons expire after _LESSON_TTL_DAYS days.
    Manual lessons never expire (expires_at = None).
    """
    now = datetime.now(tz=timezone.utc)
    expires_at = None
    if source == "reflection":
        expires_at = (now + timedelta(days=_LESSON_TTL_DAYS)).isoformat()

    row = LessonRow(
        lesson=lesson,
        source=source,
        symbol=symbol,
        created_at=now.isoformat(),
        expires_at=expires_at,
        is_active=True,
    )
    session.add(row)
    await session.flush()
    await session.commit()
    log.debug("Saved lesson id=%d source=%s expires=%s", row.id, source, expires_at)
    return row.id  # type: ignore[return-value]


async def get_recent_lessons(
    session: AsyncSession,
    symbol: str | None = None,
    limit: int = 10,
) -> list[RecentLesson]:
    """Return recent active, non-expired lessons (most recent first).

    When symbol is provided, returns global lessons PLUS symbol-specific ones.
    Lessons whose expires_at is in the past are silently excluded.
    """
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    stmt = (
        select(LessonRow)
        .where(LessonRow.is_active == True)  # noqa: E712
        # Exclude expired lessons (NULL expires_at means never expires)
        .where(
            (LessonRow.expires_at.is_(None)) | (LessonRow.expires_at > now_iso)
        )
        .order_by(LessonRow.created_at.desc())
        .limit(limit)
    )
    if symbol:
        stmt = stmt.where(
            (LessonRow.symbol == symbol) | (LessonRow.symbol.is_(None))
        )

    rows = (await session.execute(stmt)).scalars().all()
    return [
        RecentLesson(
            lesson=r.lesson,
            source=r.source,
            created_at=_parse_dt(r.created_at),
        )
        for r in rows
    ]


async def deactivate_lesson(session: AsyncSession, lesson_id: int) -> None:
    """Soft-delete a lesson (marks is_active=False)."""
    await session.execute(
        update(LessonRow)
        .where(LessonRow.id == lesson_id)
        .values(is_active=False)
    )
    await session.commit()


# ── Attribution queries ───────────────────────────────────────────────────────

async def get_unattributed_decisions(
    session: AsyncSession,
    older_than_hours: int = 24,
) -> list[DecisionRow]:
    """Return buy/sell decisions older than N hours with no outcome yet."""
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=older_than_hours)).isoformat()
    stmt = (
        select(DecisionRow)
        .where(DecisionRow.outcome_pnl_pct.is_(None))
        .where(DecisionRow.action != "hold")
        .where(DecisionRow.decided_at <= cutoff)
        .order_by(DecisionRow.decided_at.asc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def get_next_opposing_prices(
    session: AsyncSession,
    decisions: list[DecisionRow],
) -> dict[int, float]:
    """Return the best available exit price for each decision.

    For a BUY decision, looks for the earliest subsequent SELL for the same symbol
    that has a stored close_price_at_decision. Uses that price as the exit proxy,
    which is more accurate than the current market price 24h later.

    Returns a mapping of {decision_id: exit_close_price} for decisions that have
    a matching subsequent opposing decision. Decisions with no match are absent
    from the dict — callers fall back to current market price.
    """
    result: dict[int, float] = {}
    for row in decisions:
        opposite = "sell" if row.action == "buy" else "buy"
        stmt = (
            select(DecisionRow.id, DecisionRow.close_price_at_decision)
            .where(DecisionRow.symbol == row.symbol)
            .where(DecisionRow.action == opposite)
            .where(DecisionRow.decided_at > row.decided_at)
            .where(DecisionRow.close_price_at_decision.is_not(None))
            .order_by(DecisionRow.decided_at.asc())
            .limit(1)
        )
        match = (await session.execute(stmt)).first()
        if match and match.close_price_at_decision:
            result[row.id] = float(match.close_price_at_decision)
    return result


async def get_decisions_since(
    session: AsyncSession,
    since: datetime,
    with_outcome_only: bool = False,
) -> list[DecisionRow]:
    """Return all decisions after `since`, optionally filtered to attributed ones."""
    stmt = (
        select(DecisionRow)
        .where(DecisionRow.decided_at >= since.isoformat())
        .order_by(DecisionRow.decided_at.asc())
    )
    if with_outcome_only:
        stmt = stmt.where(DecisionRow.outcome_pnl_pct.is_not(None))
    return list((await session.execute(stmt)).scalars().all())


# ── Signal weight queries ─────────────────────────────────────────────────────

async def save_weights(
    session: AsyncSession,
    *,
    ema: float,
    rsi: float,
    roc: float,
    source: str = "reflection",
    rationale: str = "",
) -> int:
    """Insert a new signal weights row and return its id."""
    row = SignalWeightsRow(
        ema=ema,
        rsi=rsi,
        roc=roc,
        set_at=datetime.now(tz=timezone.utc).isoformat(),
        source=source,
        rationale=rationale,
    )
    session.add(row)
    await session.flush()
    await session.commit()
    log.info("Signal weights saved: ema=%.3f rsi=%.3f roc=%.3f source=%s", ema, rsi, roc, source)
    return row.id  # type: ignore[return-value]


async def get_current_weights(
    session: AsyncSession,
) -> dict[str, float] | None:
    """Return the most recently saved signal weights, or None if no row exists."""
    stmt = select(SignalWeightsRow).order_by(SignalWeightsRow.set_at.desc()).limit(1)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return None
    return {"ema": row.ema, "rsi": row.rsi, "roc": row.roc}


async def get_weight_history(
    session: AsyncSession,
    limit: int = 30,
) -> list[SignalWeightsRow]:
    """Return recent weight rows for dashboard display."""
    stmt = select(SignalWeightsRow).order_by(SignalWeightsRow.set_at.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars().all())


# ── Performance snapshot queries ──────────────────────────────────────────────

async def save_performance_snapshot(
    session: AsyncSession,
    *,
    portfolio_value: str,
    daily_return_pct: float | None,
    cumulative_return_pct: float | None,
    sharpe_30d: float | None,
    max_drawdown_30d: float | None,
    win_rate: float | None,
    total_decisions: int,
    total_trades: int,
) -> int:
    """Insert a performance snapshot row and return its id."""
    row = PerformanceSnapshotRow(
        portfolio_value=portfolio_value,
        daily_return_pct=daily_return_pct,
        cumulative_return_pct=cumulative_return_pct,
        sharpe_30d=sharpe_30d,
        max_drawdown_30d=max_drawdown_30d,
        win_rate=win_rate,
        total_decisions=total_decisions,
        total_trades=total_trades,
        recorded_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    session.add(row)
    await session.flush()
    await session.commit()
    return row.id  # type: ignore[return-value]


async def get_performance_history(
    session: AsyncSession,
    days: int = 30,
) -> list[PerformanceSnapshotRow]:
    """Return performance snapshots from the last N days, oldest first."""
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
    stmt = (
        select(PerformanceSnapshotRow)
        .where(PerformanceSnapshotRow.recorded_at >= cutoff)
        .order_by(PerformanceSnapshotRow.recorded_at.asc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def get_latest_performance(
    session: AsyncSession,
) -> PerformanceSnapshotRow | None:
    """Return the most recent performance snapshot."""
    stmt = (
        select(PerformanceSnapshotRow)
        .order_by(PerformanceSnapshotRow.recorded_at.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


# ── Agent state queries ───────────────────────────────────────────────────────

async def save_agent_state(
    session: AsyncSession,
    key: str,
    value: str,
) -> None:
    """Upsert a durable agent state value (circuit breaker, counters, etc.)."""
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    existing = await session.get(AgentStateRow, key)
    if existing is not None:
        existing.value = value
        existing.updated_at = now_iso
    else:
        session.add(AgentStateRow(key=key, value=value, updated_at=now_iso))
    await session.commit()


async def get_agent_state(session: AsyncSession, key: str) -> str | None:
    """Return a durable agent state value, or None if not set."""
    row = await session.get(AgentStateRow, key)
    return row.value if row is not None else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(iso: str) -> datetime:
    """Parse an ISO-8601 string to a UTC-aware datetime."""
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
