"""Outcome attribution — simulates P&L for past buy/sell decisions.

For each buy or sell decision that is old enough (>= attribution_window_hours)
and has no outcome recorded, the attributor computes a simulated P&L using:

    1. close_price_at_decision (stored in DecisionRow) as the entry price
    2. exit price resolved in priority order:
         a. close_price_at_decision of the next OPPOSING decision for that symbol
            (e.g. for a buy: the next sell's entry price — closest to the actual
            holding period P&L rather than an arbitrary 24h window)
         b. current_prices[symbol] — current market price proxy (fallback)

P&L formula:
    buy  decision: pnl = (exit_price - entry_price) / entry_price
                   Positive = market rose after the buy signal  → correct call
    sell decision: pnl = (entry_price - exit_price) / entry_price
                   Positive = market fell after the sell signal → correct call

Entry price priority:
    1. row.close_price_at_decision  — stored at decision time (accurate)
    2. entry_prices[symbol]         — caller-supplied fallback (legacy / tests)
    If neither is available, the decision is skipped this cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from agent.memory.queries import (
    get_next_opposing_prices,
    get_unattributed_decisions,
    update_decision_outcome,
)

log = logging.getLogger(__name__)


def _compute_pnl(action: str, entry_price: float, exit_price: float) -> float:
    """Compute the simulated P&L percentage for a decision."""
    if entry_price <= 0:
        return 0.0
    if action == "buy":
        return (exit_price - entry_price) / entry_price
    if action == "sell":
        return (entry_price - exit_price) / entry_price
    return 0.0


class PriceBasedAttributor:
    """Attributes simulated P&L to past buy/sell decisions.

    The trading loop feeds this class the current market prices each cycle.
    The entry price is read from `close_price_at_decision` stored in the
    decision row (written by the reasoning agent at decision time).

    Exit price resolution order:
        1. close_price_at_decision of the next OPPOSING decision for the same symbol
           (e.g. for a BUY, the next SELL's entry price). This gives the actual
           holding-period P&L rather than an arbitrary 24h window measurement.
        2. current_prices[symbol] — current market price (fallback when no opposing
           decision exists yet, or for very recent position changes).

    Args:
        attribution_window_hours: Minimum age of a decision before attribution.
            Defaults to 24h so each decision gets a full day to play out.
    """

    def __init__(self, attribution_window_hours: int = 24) -> None:
        self._window_hours = attribution_window_hours

    async def attribute_with_prices(
        self,
        session: AsyncSession,
        entry_prices: dict[str, float],
        current_prices: dict[str, float],
    ) -> int:
        """Attribute all eligible unattributed decisions.

        Entry price resolution order:
            1. row.close_price_at_decision (stored at decision time)
            2. entry_prices[symbol]        (explicit caller parameter — fallback)

        Exit price resolution order:
            1. close_price_at_decision of the next opposing decision (best accuracy)
            2. current_prices[symbol]      (24h proxy — fallback)

        Args:
            session:       Active async SQLAlchemy session.
            entry_prices:  symbol → price at decision time (fallback for old rows).
            current_prices: symbol → current market price (exit price fallback).

        Returns:
            Number of decisions attributed this run.
        """
        decisions = await get_unattributed_decisions(
            session, older_than_hours=self._window_hours
        )

        if not decisions:
            return 0

        # Batch-fetch best exit prices from subsequent opposing decisions
        opposing_exit_prices = await get_next_opposing_prices(session, decisions)

        attributed = 0
        for row in decisions:
            # ── Resolve exit price ──────────────────────────────────────────
            # Priority 1: next opposing decision's entry price (most accurate)
            # Priority 2: current market price (24h proxy)
            exit_price: float | None = (
                opposing_exit_prices.get(row.id)
                or current_prices.get(row.symbol)
            )
            if exit_price is None:
                log.debug("No exit price for %s id=%s, skipping", row.symbol, row.id)
                continue

            # ── Resolve entry price ─────────────────────────────────────────
            entry: float | None = None
            stored = getattr(row, "close_price_at_decision", None)
            if stored is not None and float(stored) > 0:
                entry = float(stored)
            else:
                entry = entry_prices.get(row.symbol)

            if entry is None or entry <= 0:
                log.debug(
                    "Cannot resolve entry price for decision id=%s symbol=%s — skipping",
                    row.id, row.symbol,
                )
                continue

            pnl_pct = _compute_pnl(row.action, entry, exit_price)
            await update_decision_outcome(session, row.id, pnl_pct)

            exit_source = "opposing_decision" if row.id in opposing_exit_prices else "current_price"
            log.info(
                "Attributed id=%d %s %s entry=%.4f exit=%.4f pnl=%+.3f%% exit_source=%s",
                row.id, row.symbol, row.action,
                entry, exit_price, pnl_pct * 100, exit_source,
            )
            attributed += 1

        if attributed:
            log.info("Attribution complete: %d decisions attributed this cycle", attributed)

        return attributed
