"""Kelly-capped position sizing.

The Kelly criterion gives the theoretically optimal fraction of capital to
bet when you have an edge and known odds. We use a capped, fractional version:

    f* = (confidence * edge_estimate) / (1 - edge_estimate)  (classic Kelly)
    f  = min(f*, MAX_RISK_PCT) * portfolio_value / entry_price

Where:
    confidence      — Claude's stated confidence [0, 1] (from reasoning output)
    edge_estimate   — normalised composite signal strength |composite| [0, 1]
    MAX_RISK_PCT    — hard cap on fraction of portfolio risked (default 2%)

We apply a half-Kelly adjustment (multiply by 0.5) to reduce variance —
full Kelly is theoretically optimal but practically aggressive and sensitive
to edge mis-estimation.

Reference: Kelly (1956), Thorp (1969).
"""

from __future__ import annotations

import logging
from decimal import ROUND_DOWN, Decimal
from typing import Optional

from agent.config import Settings, get_settings

log = logging.getLogger(__name__)

# Minimum order size in quote currency (USD) — anything below is ignored
_MIN_ORDER_USD: Decimal = Decimal("15")

# Kelly multiplier — half-Kelly for variance reduction
_KELLY_FRACTION: float = 0.5

# Minimum LLM confidence to place any order — below this, skip regardless of Kelly
_MIN_CONFIDENCE: float = 0.30


def kelly_size(
    portfolio_value: Decimal,
    entry_price: Decimal,
    confidence: float,
    edge_estimate: float,
    settings: Optional[Settings] = None,
    min_qty_precision: int = 8,
) -> Decimal:
    """Compute Kelly-capped order quantity.

    Args:
        portfolio_value:   Total portfolio value in USD.
        entry_price:       Expected fill price for the asset.
        confidence:        Claude's decision confidence [0, 1].
        edge_estimate:     Signal strength from SignalResult.confidence [0, 1].
        settings:          Settings object (reads max_risk_per_trade_pct).
        min_qty_precision: Decimal places to round quantity to.

    Returns:
        Quantity to buy/sell, floored to min_qty_precision decimal places.
        Returns Decimal("0") if the computed size is below minimum order size.
    """
    cfg = settings or get_settings()
    max_risk_pct = Decimal(str(cfg.max_risk_per_trade_pct))

    # Clamp inputs
    confidence = max(0.0, min(1.0, confidence))
    edge_estimate = max(0.0, min(0.99, edge_estimate))  # cap at 0.99 to avoid div-by-zero

    # Classic Kelly fraction: f* = (p * b - q) / b  (simplified for symmetric bet)
    # Here: p = confidence, b = edge_estimate / (1 - edge_estimate) (implied odds)
    if edge_estimate > 0:
        kelly_f = (confidence * edge_estimate) / max(1 - edge_estimate, 0.01)
    else:
        kelly_f = 0.0

    # Apply half-Kelly and cap at max risk per trade
    adjusted_f = min(kelly_f * _KELLY_FRACTION, float(max_risk_pct))

    # Dollar amount to risk
    risk_usd = portfolio_value * Decimal(str(adjusted_f))

    if risk_usd < _MIN_ORDER_USD:
        log.debug(
            "Kelly size below minimum: risk_usd=%.2f < min=%.2f (confidence=%.2f, edge=%.2f)",
            float(risk_usd), float(_MIN_ORDER_USD), confidence, edge_estimate,
        )
        return Decimal("0")

    # Convert to asset quantity
    qty = risk_usd / entry_price
    qty = qty.quantize(Decimal(f"1e-{min_qty_precision}"), rounding=ROUND_DOWN)

    log.debug(
        "Kelly size: portfolio=$%.0f confidence=%.2f edge=%.2f f=%.4f "
        "risk_usd=$%.2f qty=%s @ $%.2f",
        float(portfolio_value), confidence, edge_estimate, adjusted_f,
        float(risk_usd), qty, float(entry_price),
    )
    return qty


def max_allowed_size(
    portfolio_value: Decimal,
    entry_price: Decimal,
    current_position_value: Decimal = Decimal("0"),
    settings: Optional[Settings] = None,
) -> Decimal:
    """Return the maximum quantity allowed by the position concentration cap.

    This is an upper bound independent of Kelly — the risk checker enforces
    it too, but sizing can use it to clamp without a round-trip to the checker.
    """
    cfg = settings or get_settings()
    max_pos_pct = Decimal(str(cfg.max_position_pct))
    max_pos_value = portfolio_value * max_pos_pct
    headroom = max_pos_value - current_position_value
    if headroom <= 0 or entry_price <= 0:
        return Decimal("0")
    return (headroom / entry_price).quantize(Decimal("1e-8"), rounding=ROUND_DOWN)


def compute_order_size(
    portfolio_value: Decimal,
    entry_price: Decimal,
    confidence: float,
    edge_estimate: float,
    current_position_value: Decimal = Decimal("0"),
    settings: Optional[Settings] = None,
) -> Decimal:
    """Unified sizing: Kelly size clamped by concentration cap.

    This is the function the trading loop should call. It returns 0 if either
    the confidence is below the minimum threshold, the Kelly size is below
    minimum, or the concentration cap is already hit.
    """
    if confidence < _MIN_CONFIDENCE:
        log.debug("Confidence %.2f below minimum %.2f, skipping order", confidence, _MIN_CONFIDENCE)
        return Decimal("0")

    kelly = kelly_size(portfolio_value, entry_price, confidence, edge_estimate, settings)
    if kelly == 0:
        return Decimal("0")

    cap = max_allowed_size(portfolio_value, entry_price, current_position_value, settings)
    final = min(kelly, cap)

    # Final minimum check after clamping
    if final * entry_price < _MIN_ORDER_USD:
        return Decimal("0")

    return final
