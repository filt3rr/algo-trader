"""Pre-trade risk gate — every order must pass before broker submission.

ALL caps are enforced here in code. Config values are default starting points,
but the code never exceeds the hard limits defined as module-level constants.

Hard limits (cannot be overridden by config):
    MAX_RISK_PER_TRADE  = 2%   of portfolio value
    MAX_POSITION_PCT    = 10%  of portfolio value in any single asset
    MAX_OPEN_POSITIONS  = 5    concurrent open positions
    DAILY_LOSS_LIMIT    = -5%  of starting-day portfolio value

Circuit breaker behaviour:
    When daily loss >= DAILY_LOSS_LIMIT, new ENTRY orders are rejected.
    Exits (sells of existing positions) are always allowed.
    The circuit breaker resets at the start of each UTC trading day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

from agent.broker.base import AccountInfo, Order, OrderRequest, OrderSide, Position
from agent.config import Settings, get_settings

log = logging.getLogger(__name__)

# ── Hard limits — these are the law, config is just the dial ─────────────────
_HARD_MAX_RISK_PCT: float = 0.02        # 2% portfolio risk per trade
_HARD_MAX_POSITION_PCT: float = 0.10   # 10% in a single asset
_HARD_MAX_POSITIONS: int = 5            # concurrent open positions
_HARD_DAILY_LOSS_PCT: float = 0.05     # -5% daily triggers circuit breaker


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskCheckResult:
    approved: bool
    rejection_reason: Optional[str]
    checks: dict[str, bool] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.approved:
            return "APPROVED"
        return f"REJECTED: {self.rejection_reason}"


# ── Circuit breaker state (in-memory; resets on process restart / new day) ───

@dataclass
class CircuitBreakerState:
    triggered: bool = False
    trigger_reason: str = ""
    triggered_at: Optional[datetime] = None
    day_open_portfolio_value: Optional[Decimal] = None
    reset_date: Optional[date] = None

    def reset_if_new_day(self) -> None:
        today = datetime.now(tz=timezone.utc).date()
        if self.reset_date != today:
            self.triggered = False
            self.trigger_reason = ""
            self.triggered_at = None
            self.reset_date = today
            log.info("Circuit breaker reset for new trading day %s", today)

    def trip(self, reason: str) -> None:
        if not self.triggered:
            self.triggered = True
            self.trigger_reason = reason
            self.triggered_at = datetime.now(tz=timezone.utc)
            log.warning("CIRCUIT BREAKER TRIPPED: %s", reason)


# ── Main risk checker ─────────────────────────────────────────────────────────

class RiskChecker:
    """Stateful pre-trade risk gate.

    Holds circuit breaker state and validates each order against all hard caps.
    One instance should be shared across the trading loop.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        # Effective caps: min of config and hard limits
        self.max_risk_pct = min(cfg.max_risk_per_trade_pct, _HARD_MAX_RISK_PCT)
        self.max_position_pct = min(cfg.max_position_pct, _HARD_MAX_POSITION_PCT)
        self.max_open_positions = min(cfg.max_open_positions, _HARD_MAX_POSITIONS)
        self.daily_loss_pct = min(cfg.daily_loss_limit_pct, _HARD_DAILY_LOSS_PCT)
        self._cb = CircuitBreakerState()

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        order: OrderRequest,
        account: AccountInfo,
        positions: list[Position],
        entry_price: Optional[Decimal] = None,
    ) -> RiskCheckResult:
        """Run all risk checks for a proposed order.

        Args:
            order:       Proposed order (symbol, side, qty, type).
            account:     Current account snapshot.
            positions:   All currently open positions.
            entry_price: Expected fill price (uses order.limit_price if None).

        Returns:
            RiskCheckResult with approved=True or False + reason.
        """
        self._cb.reset_if_new_day()
        checks: dict[str, bool] = {}

        # 1. Circuit breaker — blocks new entries only
        cb_ok = self._check_circuit_breaker(order, account)
        checks["circuit_breaker"] = cb_ok
        if not cb_ok:
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Circuit breaker active: {self._cb.trigger_reason}",
                checks=checks,
            )

        # 2. Position count — only blocks new positions, not adds
        is_new_position = not any(p.symbol == order.symbol for p in positions)
        if is_new_position and order.side == OrderSide.BUY:
            pos_count_ok = len(positions) < self.max_open_positions
            checks["position_count"] = pos_count_ok
            if not pos_count_ok:
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=(
                        f"Max open positions reached ({len(positions)}/{self.max_open_positions})"
                    ),
                    checks=checks,
                )
        checks["position_count"] = True

        # 3. Resolve fill price
        price = entry_price or order.limit_price
        if price is None or price <= 0:
            checks["price_valid"] = False
            return RiskCheckResult(
                approved=False,
                rejection_reason="Cannot determine fill price for risk sizing",
                checks=checks,
            )
        checks["price_valid"] = True

        portfolio_value = account.portfolio_value
        if portfolio_value <= 0:
            return RiskCheckResult(
                approved=False,
                rejection_reason="Portfolio value is zero or negative",
                checks=checks,
            )

        # 4. Position concentration check
        order_value = order.qty * price
        existing_value = Decimal("0")
        for p in positions:
            if p.symbol == order.symbol:
                existing_value = p.market_value

        new_position_value = existing_value + order_value
        position_pct = float(new_position_value / portfolio_value)
        pos_pct_ok = position_pct <= self.max_position_pct
        checks["position_concentration"] = pos_pct_ok
        if not pos_pct_ok:
            return RiskCheckResult(
                approved=False,
                rejection_reason=(
                    f"Position concentration {position_pct:.1%} exceeds "
                    f"limit {self.max_position_pct:.0%} for {order.symbol}"
                ),
                checks=checks,
            )

        # 5. Per-trade risk check (order value as % of portfolio)
        trade_risk_pct = float(order_value / portfolio_value)
        risk_ok = trade_risk_pct <= self.max_risk_pct
        checks["trade_risk"] = risk_ok
        if not risk_ok:
            return RiskCheckResult(
                approved=False,
                rejection_reason=(
                    f"Trade risk {trade_risk_pct:.2%} exceeds limit {self.max_risk_pct:.0%}"
                ),
                checks=checks,
            )

        log.debug(
            "Risk check PASSED %s %s qty=%s price=%s risk=%.2f%% pos=%.2f%%",
            order.side, order.symbol, order.qty, price, trade_risk_pct * 100, position_pct * 100,
        )
        return RiskCheckResult(approved=True, rejection_reason=None, checks=checks)

    def update_daily_pnl(self, account: AccountInfo) -> None:
        """Call this at the start of each loop to update circuit breaker state.

        On the first call of each UTC day, records the opening portfolio value.
        On subsequent calls, checks if daily loss limit has been breached.
        """
        self._cb.reset_if_new_day()

        if self._cb.day_open_portfolio_value is None:
            self._cb.day_open_portfolio_value = account.portfolio_value
            log.info("Day open portfolio value: $%.2f", float(account.portfolio_value))
            return

        open_val = self._cb.day_open_portfolio_value
        if open_val <= 0:
            return

        daily_pnl_pct = float((account.portfolio_value - open_val) / open_val)

        if daily_pnl_pct <= -self.daily_loss_pct and not self._cb.triggered:
            self._cb.trip(
                f"Daily loss {daily_pnl_pct:.2%} reached limit -{self.daily_loss_pct:.0%}"
            )

    def reset_circuit_breaker(self, reason: str = "manual_reset") -> None:
        """Manually reset the circuit breaker (e.g. operator override)."""
        self._cb.triggered = False
        self._cb.trigger_reason = ""
        self._cb.triggered_at = None
        log.warning("Circuit breaker manually reset: %s", reason)

    @property
    def circuit_breaker_active(self) -> bool:
        return self._cb.triggered

    @property
    def circuit_breaker_reason(self) -> str:
        return self._cb.trigger_reason

    # ── Private helpers ───────────────────────────────────────────────────────

    def get_daily_pnl_pct(self, current_portfolio_value: Decimal) -> float:
        """Return today's P&L as a fraction (e.g. -0.02 for -2%).

        Returns 0.0 if the day-open value has not been recorded yet.
        """
        open_val = self._cb.day_open_portfolio_value
        if open_val is None or open_val <= 0:
            return 0.0
        return float((current_portfolio_value - open_val) / open_val)

    def export_state(self) -> dict:
        """Serialize circuit breaker state for DB persistence (JSON-serialisable)."""
        return {
            "triggered": self._cb.triggered,
            "trigger_reason": self._cb.trigger_reason,
            "triggered_at": self._cb.triggered_at.isoformat() if self._cb.triggered_at else None,
            "day_open_portfolio_value": str(self._cb.day_open_portfolio_value)
            if self._cb.day_open_portfolio_value is not None else None,
            "reset_date": self._cb.reset_date.isoformat() if self._cb.reset_date else None,
        }

    def import_state(self, state: dict) -> None:
        """Restore circuit breaker state from a previously exported dict.

        Called at startup to re-hydrate in-memory state from DB, so a process
        restart does not reset the daily loss limit mid-day.
        """
        self._cb.triggered = bool(state.get("triggered", False))
        self._cb.trigger_reason = state.get("trigger_reason", "")
        triggered_at = state.get("triggered_at")
        self._cb.triggered_at = datetime.fromisoformat(triggered_at) if triggered_at else None
        day_open = state.get("day_open_portfolio_value")
        self._cb.day_open_portfolio_value = Decimal(day_open) if day_open else None
        reset_date = state.get("reset_date")
        self._cb.reset_date = date.fromisoformat(reset_date) if reset_date else None
        # Let reset_if_new_day() clear stale state if we've crossed midnight
        self._cb.reset_if_new_day()
        if self._cb.triggered:
            log.info(
                "Circuit breaker state restored: triggered=True reason=%s",
                self._cb.trigger_reason,
            )

    def _check_circuit_breaker(self, order: OrderRequest, account: AccountInfo) -> bool:
        """Circuit breaker blocks new BUY entries only. Sells always pass."""
        if order.side == OrderSide.SELL:
            return True  # exits are always permitted
        return not self._cb.triggered
