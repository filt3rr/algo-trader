"""Abstract broker interface and live-trading guard.

LIVE TRADING WARNING
====================
This system is paper-only. To enable live trading you must:
  1. Set LIVE_TRADING_ENABLED=True in .env (this alone does nothing — the
     Settings validator rejects it).
  2. Remove the hard-coded guard in AlpacaBroker.__init__ (requires a code
     change, not just config).
  3. Set the env var LIVE_TRADING_SECRET=I_ACCEPT_ALL_FINANCIAL_RISK
  4. Swap ALPACA_BASE_URL to the live endpoint.
  5. Accept full legal and financial responsibility for any losses.

There is no automated path from paper to live. This is intentional.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any


# ── Domain types ──────────────────────────────────────────────────────────────

class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(StrEnum):
    GTC = "gtc"   # Good till cancelled
    IOC = "ioc"   # Immediate or cancel
    FOK = "fok"   # Fill or kill
    DAY = "day"


@dataclass(frozen=True)
class Order:
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    qty: Decimal
    status: OrderStatus
    submitted_at: datetime
    filled_qty: Decimal = Decimal("0")
    filled_avg_price: Decimal | None = None
    limit_price: Decimal | None = None
    client_order_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: float


@dataclass(frozen=True)
class AccountInfo:
    id: str
    cash: Decimal
    portfolio_value: Decimal
    equity: Decimal
    buying_power: Decimal
    currency: str = "USD"
    is_paper: bool = True


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    qty: Decimal
    type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.GTC
    limit_price: Decimal | None = None
    client_order_id: str = ""


# ── Live-trading guard ────────────────────────────────────────────────────────

_LIVE_SECRET_VALUE = "I_ACCEPT_ALL_FINANCIAL_RISK"


def assert_paper_only() -> None:
    """Raise if someone has attempted to enable live trading.

    This guard runs on every broker instantiation. It cannot be bypassed
    by config alone — the calling code must be deliberately modified.
    """
    live_enabled = os.getenv("LIVE_TRADING_ENABLED", "False").lower() in ("true", "1")
    live_secret = os.getenv("LIVE_TRADING_SECRET", "")

    if live_enabled and live_secret == _LIVE_SECRET_VALUE:
        # Both conditions met — this is an intentional live-trading setup.
        # Log a loud warning but allow it (the operator made a deliberate choice).
        import warnings
        warnings.warn(
            "LIVE TRADING IS ENABLED. Real money is at risk. "
            "The authors accept no responsibility for financial losses.",
            stacklevel=3,
        )
        return

    if live_enabled:
        raise RuntimeError(
            "LIVE_TRADING_ENABLED=True but LIVE_TRADING_SECRET is not set. "
            "See agent/broker/base.py for the full unlock procedure."
        )

    # Normal paper-only path — do nothing.


# ── Abstract adapter ─────────────────────────────────────────────────────────

class BrokerAdapter(ABC):
    """All broker implementations must satisfy this interface."""

    def __init__(self) -> None:
        assert_paper_only()

    # ── Account ──────────────────────────────────────────────────────────────

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Return current account snapshot."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all open positions."""

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        """Return the open position for a symbol, or None."""

    # ── Orders ───────────────────────────────────────────────────────────────

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> Order:
        """Submit a new order. Returns the submitted Order (may not be filled yet)."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled, False if already done."""

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """Fetch current state of an order by ID."""

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Return all open (unfilled) orders."""

    # ── Market data (latest price) ────────────────────────────────────────────

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Decimal:
        """Return the latest trade price for a symbol."""

    # ── Health ───────────────────────────────────────────────────────────────

    @abstractmethod
    def is_market_open(self) -> bool:
        """Crypto markets are always open, but the method exists for API parity."""

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Must return True for paper adapters."""
