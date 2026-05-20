"""SQLAlchemy async models for persistent agent memory.

Tables:
    decisions             — one row per LLM reasoning cycle
    trades                — one row per order (linked to a decision)
    lessons               — one row per insight from nightly reflection
    signal_weights        — history of signal weight snapshots (learning module)
    performance_snapshots — rolling portfolio metrics (dashboard / learning)
    agent_state           — key-value store for durable agent state (circuit breaker, etc.)

All tables use integer primary keys (SQLite autoincrement). Timestamps stored
as ISO-8601 strings (SQLite has no native datetime type).
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    event,
    text,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# ── Table definitions ─────────────────────────────────────────────────────────

class DecisionRow(Base):
    """One LLM decision cycle — may or may not result in an order."""

    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)           # buy | sell | hold
    rationale = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    edge_estimate = Column(Float, nullable=False)
    risk_notes = Column(Text, nullable=False, default="")
    composite_signal = Column(Float, nullable=False)
    signal_regime = Column(String(30), nullable=False)
    portfolio_value_at_decision = Column(String(30), nullable=False)
    decided_at = Column(String(30), nullable=False, index=True)  # ISO-8601 UTC
    dry_run = Column(Boolean, nullable=False, default=True)
    llm_model = Column(String(60), nullable=False)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    order_id = Column(String(60), nullable=True)
    outcome_pnl_pct = Column(Float, nullable=True)        # filled by learning module

    # Phase 5+ columns — added via migration for existing DBs
    close_price_at_decision = Column(Float, nullable=True)   # signal close price at decision time
    signal_weights_json = Column(Text, nullable=True)         # JSON of {ema,rsi,roc} weights used

    trades = relationship("TradeRow", back_populates="decision", lazy="select")


class TradeRow(Base):
    """An order linked to the decision that triggered it."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(Integer, ForeignKey("decisions.id"), nullable=True, index=True)
    order_id = Column(String(60), nullable=False, unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)              # buy | sell
    qty = Column(String(30), nullable=False)               # Decimal as string
    fill_price = Column(String(30), nullable=True)         # None until filled
    order_value = Column(Float, nullable=True)
    status = Column(String(20), nullable=False)            # pending | filled | cancelled | stale
    submitted_at = Column(String(30), nullable=False)      # ISO-8601 UTC
    filled_at = Column(String(30), nullable=True)
    cancelled_at = Column(String(30), nullable=True)       # set when order cancelled/staled
    dry_run = Column(Boolean, nullable=False, default=True)

    decision = relationship("DecisionRow", back_populates="trades")


class LessonRow(Base):
    """A lesson extracted by the nightly reflection job."""

    __tablename__ = "lessons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lesson = Column(Text, nullable=False)
    source = Column(String(30), nullable=False, default="reflection")  # reflection | manual
    symbol = Column(String(20), nullable=True)     # None = global lesson
    created_at = Column(String(30), nullable=False, index=True)   # ISO-8601 UTC
    expires_at = Column(String(30), nullable=True)                # None = never expires
    is_active = Column(Boolean, nullable=False, default=True)


class SignalWeightsRow(Base):
    """A snapshot of signal weights written by the reflection engine."""

    __tablename__ = "signal_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ema = Column(Float, nullable=False)
    rsi = Column(Float, nullable=False)
    roc = Column(Float, nullable=False)
    set_at = Column(String(30), nullable=False, index=True)    # ISO-8601 UTC
    source = Column(String(30), nullable=False, default="reflection")
    rationale = Column(Text, nullable=False, default="")


class PerformanceSnapshotRow(Base):
    """One portfolio performance reading, written each trading cycle."""

    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_value = Column(String(30), nullable=False)    # Decimal as string
    daily_return_pct = Column(Float, nullable=True)
    cumulative_return_pct = Column(Float, nullable=True)
    sharpe_30d = Column(Float, nullable=True)
    max_drawdown_30d = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    total_decisions = Column(Integer, nullable=False, default=0)
    total_trades = Column(Integer, nullable=False, default=0)
    recorded_at = Column(String(30), nullable=False, index=True)   # ISO-8601 UTC


class AgentStateRow(Base):
    """Durable key-value store for agent state that must survive restarts.

    Used for: circuit breaker state, day-open portfolio value, etc.
    Values are JSON strings. Primary key is the state key name.
    """

    __tablename__ = "agent_state"

    key = Column(String(60), primary_key=True)
    value = Column(Text, nullable=False, default="")
    updated_at = Column(String(30), nullable=False)


# ── Engine / session factory ──────────────────────────────────────────────────

def _make_engine(db_url: str) -> AsyncEngine:
    engine = create_async_engine(
        db_url,
        echo=False,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragma(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return engine


async def _run_migrations(engine: AsyncEngine) -> None:
    """Add new columns to existing tables without dropping data.

    SQLite supports ALTER TABLE … ADD COLUMN for new nullable columns.
    We check existing columns via PRAGMA before adding to stay idempotent.
    """
    # (table, new_column, DDL)
    migrations = [
        ("decisions", "close_price_at_decision",
         "ALTER TABLE decisions ADD COLUMN close_price_at_decision REAL"),
        ("decisions", "signal_weights_json",
         "ALTER TABLE decisions ADD COLUMN signal_weights_json TEXT"),
        ("lessons", "expires_at",
         "ALTER TABLE lessons ADD COLUMN expires_at TEXT"),
        ("trades", "cancelled_at",
         "ALTER TABLE trades ADD COLUMN cancelled_at TEXT"),
    ]

    async with engine.connect() as conn:
        for table, column, ddl in migrations:
            result = await conn.execute(text(f"PRAGMA table_info({table})"))
            existing = [row[1] for row in result.fetchall()]
            if column not in existing:
                await conn.execute(text(ddl))
                log.info("Migration applied: %s.%s", table, column)

        # Composite index for the unattributed-decisions query (IS NULL scan)
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_decisions_unattributed "
            "ON decisions(decided_at, action, outcome_pnl_pct)"
        ))
        # Index for per-symbol opposing-decision lookup used in attribution
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_decisions_symbol_action_time "
            "ON decisions(symbol, action, decided_at)"
        ))
        await conn.commit()


async def init_db(db_url: str) -> tuple[AsyncEngine, sessionmaker]:
    """Create tables (if not present), run migrations, return (engine, factory).

    Call this once at startup before any queries.
    """
    engine = _make_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await _run_migrations(engine)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    log.info("Database initialised: %s", db_url)
    return engine, factory
