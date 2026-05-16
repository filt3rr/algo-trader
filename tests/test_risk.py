"""Tests for agent/risk — 100% coverage on checks.py and sizing.py.

Every hard cap and circuit breaker path must be covered.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from agent.broker.base import AccountInfo, OrderRequest, OrderSide, OrderType, Position
from agent.risk.checks import CircuitBreakerState, RiskCheckResult, RiskChecker
from agent.risk.sizing import (
    compute_order_size,
    kelly_size,
    max_allowed_size,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _account(portfolio: float = 100_000, cash: float = 95_000) -> AccountInfo:
    p = Decimal(str(portfolio))
    return AccountInfo(
        id="acc-1",
        cash=Decimal(str(cash)),
        portfolio_value=p,
        equity=p,
        buying_power=Decimal(str(cash * 2)),
        is_paper=True,
    )


def _position(symbol: str, market_value: float, qty: float = 0.1) -> Position:
    return Position(
        symbol=symbol,
        qty=Decimal(str(qty)),
        avg_entry_price=Decimal("40000"),
        current_price=Decimal(str(market_value / qty)),
        market_value=Decimal(str(market_value)),
        unrealized_pnl=Decimal("0"),
        unrealized_pnl_pct=0.0,
    )


def _buy_order(symbol: str = "BTC/USD", qty: float = 0.001, limit_price: float = 50_000) -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY,
        qty=Decimal(str(qty)),
        type=OrderType.LIMIT,
        limit_price=Decimal(str(limit_price)),
    )


def _sell_order(symbol: str = "BTC/USD", qty: float = 0.001, limit_price: float = 50_000) -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        qty=Decimal(str(qty)),
        type=OrderType.LIMIT,
        limit_price=Decimal(str(limit_price)),
    )


def _checker() -> RiskChecker:
    return RiskChecker()


# ── RiskCheckResult ───────────────────────────────────────────────────────────

def test_risk_result_approved_str() -> None:
    r = RiskCheckResult(approved=True, rejection_reason=None)
    assert str(r) == "APPROVED"


def test_risk_result_rejected_str() -> None:
    r = RiskCheckResult(approved=False, rejection_reason="Too big")
    assert "REJECTED" in str(r)
    assert "Too big" in str(r)


# ── Circuit breaker ───────────────────────────────────────────────────────────

def test_circuit_breaker_not_triggered_initially() -> None:
    checker = _checker()
    assert checker.circuit_breaker_active is False


def test_circuit_breaker_allows_buys_when_not_triggered() -> None:
    checker = _checker()
    result = checker.check(_buy_order(), _account(), [])
    assert result.approved is True


def test_circuit_breaker_always_allows_sells() -> None:
    checker = _checker()
    today = datetime.now(tz=timezone.utc).date()
    checker._cb.triggered = True
    checker._cb.trigger_reason = "daily loss"
    checker._cb.reset_date = today
    result = checker.check(_sell_order(), _account(), [])
    assert result.approved is True
    assert result.checks.get("circuit_breaker") is True


def test_circuit_breaker_blocks_buys_when_triggered() -> None:
    checker = _checker()
    today = datetime.now(tz=timezone.utc).date()
    checker._cb.triggered = True
    checker._cb.trigger_reason = "daily loss exceeded -5%"
    checker._cb.reset_date = today  # must match today so reset doesn't clear it
    result = checker.check(_buy_order(), _account(), [])
    assert result.approved is False
    assert "Circuit breaker" in (result.rejection_reason or "")


def test_circuit_breaker_trips_on_daily_loss() -> None:
    checker = _checker()
    open_val = Decimal("100000")
    checker._cb.day_open_portfolio_value = open_val
    checker._cb.reset_date = datetime.now(tz=timezone.utc).date()

    # Simulate -6% daily loss (beyond -5% limit)
    down_account = _account(portfolio=94_000)
    checker.update_daily_pnl(down_account)

    assert checker.circuit_breaker_active is True
    assert "-6" in checker.circuit_breaker_reason or "loss" in checker.circuit_breaker_reason.lower()


def test_circuit_breaker_no_trip_on_small_loss() -> None:
    checker = _checker()
    checker._cb.day_open_portfolio_value = Decimal("100000")
    checker._cb.reset_date = datetime.now(tz=timezone.utc).date()

    # -3% loss, below -5% limit
    checker.update_daily_pnl(_account(portfolio=97_000))
    assert checker.circuit_breaker_active is False


def test_circuit_breaker_records_open_value_on_first_call() -> None:
    checker = _checker()
    checker.update_daily_pnl(_account(portfolio=100_000))
    assert checker._cb.day_open_portfolio_value == Decimal("100000")


def test_circuit_breaker_manual_reset() -> None:
    checker = _checker()
    checker._cb.triggered = True
    checker.reset_circuit_breaker("operator override")
    assert checker.circuit_breaker_active is False


def test_circuit_breaker_resets_on_new_day() -> None:
    checker = _checker()
    checker._cb.triggered = True
    checker._cb.trigger_reason = "old loss"
    checker._cb.reset_date = None  # force new-day detection
    checker._cb.reset_if_new_day()
    assert checker.circuit_breaker_active is False


def test_circuit_breaker_no_double_trip() -> None:
    checker = _checker()
    checker._cb.day_open_portfolio_value = Decimal("100000")
    checker._cb.reset_date = datetime.now(tz=timezone.utc).date()

    # Trip it once
    checker.update_daily_pnl(_account(portfolio=94_000))
    assert checker.circuit_breaker_active is True
    first_reason = checker.circuit_breaker_reason

    # Call again — should not override reason
    checker.update_daily_pnl(_account(portfolio=90_000))
    assert checker.circuit_breaker_reason == first_reason


# ── Position count cap ────────────────────────────────────────────────────────

def test_position_count_allows_new_when_under_limit() -> None:
    checker = _checker()
    positions = [_position(f"SYM{i}/USD", 5_000) for i in range(4)]  # 4 < 5
    result = checker.check(_buy_order("NEW/USD"), _account(), positions)
    # May fail for other reasons but not position count
    assert result.checks.get("position_count") is True


def test_position_count_rejects_new_when_at_limit() -> None:
    checker = _checker()
    positions = [_position(f"SYM{i}/USD", 5_000) for i in range(5)]  # 5 = limit
    result = checker.check(_buy_order("NEW/USD"), _account(), positions)
    assert result.approved is False
    assert "Max open positions" in (result.rejection_reason or "")


def test_position_count_allows_add_to_existing() -> None:
    checker = _checker()
    # 5 positions, but we're adding to BTC which already exists
    positions = [_position("BTC/USD", 5_000)] + [_position(f"SYM{i}/USD", 5_000) for i in range(4)]
    # Adding to BTC — not a new position, so count check is skipped
    result = checker.check(_buy_order("BTC/USD", qty=0.0001), _account(), positions)
    assert result.checks.get("position_count") is True


# ── Position concentration cap (10%) ─────────────────────────────────────────

def test_concentration_blocks_oversized_order() -> None:
    checker = _checker()
    # Trying to buy 0.02 BTC at $80k = $1,600 which is 1.6% of $100k — fine
    # But let's try to buy 0.02 BTC at $600k (hypothetical) = $12k > 10% of $100k
    result = checker.check(
        _buy_order("BTC/USD", qty=0.02, limit_price=600_000),  # $12k = 12% of portfolio
        _account(portfolio=100_000),
        [],
    )
    assert result.approved is False
    assert "concentration" in (result.rejection_reason or "").lower()


def test_concentration_accounts_for_existing_position() -> None:
    checker = _checker()
    # Already have $9k in BTC (9%), trying to add $2k more would = 11% > 10%
    existing = _position("BTC/USD", market_value=9_000, qty=0.1)
    result = checker.check(
        _buy_order("BTC/USD", qty=0.04, limit_price=50_000),  # $2k more
        _account(portfolio=100_000),
        [existing],
    )
    assert result.approved is False
    assert "concentration" in (result.rejection_reason or "").lower()


def test_concentration_allows_within_limit() -> None:
    checker = _checker()
    # $800 order at $80k price = 0.8% of $100k portfolio — fine
    result = checker.check(
        _buy_order("BTC/USD", qty=0.01, limit_price=80_000),
        _account(portfolio=100_000),
        [],
    )
    # Should not fail on concentration (may still fail trade risk if > 2%)
    assert result.checks.get("position_concentration") is True


# ── Per-trade risk cap (2%) ───────────────────────────────────────────────────

def test_trade_risk_blocks_oversized_trade() -> None:
    checker = _checker()
    # $3k order on $100k portfolio = 3% > 2% limit
    result = checker.check(
        _buy_order("BTC/USD", qty=0.06, limit_price=50_000),  # $3k
        _account(portfolio=100_000),
        [],
    )
    assert result.approved is False
    assert "risk" in (result.rejection_reason or "").lower()


def test_trade_risk_allows_within_limit() -> None:
    checker = _checker()
    # $1.5k order on $100k portfolio = 1.5% < 2% — fine
    result = checker.check(
        _buy_order("BTC/USD", qty=0.03, limit_price=50_000),  # $1.5k
        _account(portfolio=100_000),
        [],
    )
    assert result.checks.get("trade_risk") is True


def test_trade_risk_uses_entry_price_when_provided() -> None:
    checker = _checker()
    # MARKET order (no limit_price), but entry_price is supplied to the risk check
    market_order = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        qty=Decimal("0.001"),
        type=OrderType.MARKET,
        limit_price=None,
    )
    result = checker.check(
        market_order,
        _account(portfolio=100_000),
        [],
        entry_price=Decimal("50000"),  # $50 order = 0.05% — fine
    )
    assert result.checks.get("price_valid") is True


def test_price_invalid_when_none() -> None:
    checker = _checker()
    order = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        qty=Decimal("0.001"),
        type=OrderType.MARKET,
        limit_price=None,
    )
    result = checker.check(order, _account(), [])
    assert result.approved is False
    assert "price" in (result.rejection_reason or "").lower()


def test_zero_portfolio_value_rejected() -> None:
    checker = _checker()
    zero_account = AccountInfo(
        id="acc-0", cash=Decimal("0"), portfolio_value=Decimal("0"),
        equity=Decimal("0"), buying_power=Decimal("0"),
    )
    result = checker.check(_buy_order(), zero_account, [])
    assert result.approved is False


# ── Kelly sizing ──────────────────────────────────────────────────────────────

def test_kelly_size_returns_nonzero_for_valid_inputs() -> None:
    qty = kelly_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("80000"),
        confidence=0.8,
        edge_estimate=0.6,
    )
    assert qty > 0


def test_kelly_size_zero_for_zero_edge() -> None:
    qty = kelly_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("80000"),
        confidence=0.8,
        edge_estimate=0.0,
    )
    assert qty == Decimal("0")


def test_kelly_size_clamped_to_max_risk() -> None:
    # Even with confidence=1.0 and edge=0.99, risk must be <= 2%
    qty = kelly_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("50000"),
        confidence=1.0,
        edge_estimate=0.99,
    )
    order_value = qty * Decimal("50000")
    assert order_value <= Decimal("2000") * Decimal("1.001")  # ~2% with rounding


def test_kelly_size_below_minimum_returns_zero() -> None:
    qty = kelly_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("80000"),
        confidence=0.01,  # very low confidence → tiny size
        edge_estimate=0.01,
    )
    assert qty == Decimal("0")


def test_kelly_size_clamps_confidence_and_edge() -> None:
    # Should not raise even with out-of-range inputs
    qty = kelly_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("1000"),
        confidence=5.0,    # > 1, will be clamped
        edge_estimate=2.0, # > 1, will be clamped
    )
    assert qty >= 0


def test_max_allowed_size_respects_concentration() -> None:
    cap = max_allowed_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("50000"),
        current_position_value=Decimal("0"),
    )
    max_value = cap * Decimal("50000")
    assert max_value <= Decimal("10001")  # must be <= 10% of $100k


def test_max_allowed_size_zero_when_at_cap() -> None:
    cap = max_allowed_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("50000"),
        current_position_value=Decimal("10000"),  # already at 10%
    )
    assert cap == Decimal("0")


def test_max_allowed_size_zero_for_zero_price() -> None:
    cap = max_allowed_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("0"),
    )
    assert cap == Decimal("0")


def test_compute_order_size_integrates_both() -> None:
    qty = compute_order_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("80000"),
        confidence=0.8,
        edge_estimate=0.6,
    )
    assert qty > 0
    # Final order value must be <= 2% of portfolio
    order_value = qty * Decimal("80000")
    assert order_value <= Decimal("2001")


def test_compute_order_size_zero_for_low_confidence() -> None:
    qty = compute_order_size(
        portfolio_value=Decimal("100000"),
        entry_price=Decimal("80000"),
        confidence=0.0,
        edge_estimate=0.0,
    )
    assert qty == Decimal("0")
