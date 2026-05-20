"""Portfolio performance tracker.

Computes and persists a performance snapshot after each trading loop cycle.
Metrics are used by:
    - The reflection prompt (daily context)
    - The dashboard (charts and KPI cards)
    - Gate validation (30-day Sharpe, cumulative return vs. benchmark)

All calculations are rolling over the available snapshot history. On the first
cycle there is only one data point — metrics that require a series (Sharpe,
drawdown) return None until sufficient history exists.

Sharpe annualization:
    The annualization factor is derived from the ACTUAL average interval between
    consecutive performance snapshots rather than being hardcoded. With an hourly
    trading loop the factor is ~8 760 (hours per year). With daily snapshots it
    would be 365. This prevents the 4.9× inflation that occurred when hourly
    returns were annualized with sqrt(365).

Daily return:
    Computed against the snapshot closest to 24 h ago rather than the oldest
    snapshot in the 30-day window (which would produce a 30-day return, not a
    daily one).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from agent.memory.queries import (
    get_performance_history,
    save_performance_snapshot,
)

log = logging.getLogger(__name__)

_MIN_SHARPE_PERIODS = 24   # require at least 24 snapshots (~1 day) before computing Sharpe
_HOURS_PER_YEAR = 8_760.0  # 365 × 24


@dataclass
class PerformanceMetrics:
    """Computed metrics for one snapshot instant."""
    portfolio_value: str
    daily_return_pct: float | None
    cumulative_return_pct: float | None
    sharpe_30d: float | None
    max_drawdown_30d: float | None
    win_rate: float | None
    total_decisions: int
    total_trades: int


class PerformanceTracker:
    """Computes rolling portfolio metrics and persists a snapshot.

    Usage (trading loop):
        tracker = PerformanceTracker(initial_portfolio_value)
        metrics = await tracker.snapshot(session, current_portfolio_value, ...)
    """

    def __init__(self, initial_portfolio_value: Decimal) -> None:
        self._initial_value = float(initial_portfolio_value)

    async def snapshot(
        self,
        session: AsyncSession,
        portfolio_value: Decimal,
        total_decisions: int,
        total_trades: int,
        attributed_outcomes: list[float] | None = None,
    ) -> PerformanceMetrics:
        """Compute metrics and persist to DB.

        Args:
            session:             Active async session.
            portfolio_value:     Current total portfolio value.
            total_decisions:     All-time decision count (restored from DB at startup).
            total_trades:        All-time trade count (restored from DB at startup).
            attributed_outcomes: List of outcome_pnl_pct values from attributed
                                 decisions. Used for win_rate calculation.
        Returns:
            PerformanceMetrics dataclass.
        """
        current = float(portfolio_value)
        history = await get_performance_history(session, days=31)

        daily_return = _compute_daily_return(history, current)
        cumulative = (
            (current - self._initial_value) / self._initial_value
            if self._initial_value > 0 else None
        )
        sharpe = _compute_rolling_sharpe(history)
        drawdown = _compute_max_drawdown(history, current)
        win_rate = _compute_win_rate(attributed_outcomes)

        metrics = PerformanceMetrics(
            portfolio_value=str(portfolio_value),
            daily_return_pct=daily_return,
            cumulative_return_pct=cumulative,
            sharpe_30d=sharpe,
            max_drawdown_30d=drawdown,
            win_rate=win_rate,
            total_decisions=total_decisions,
            total_trades=total_trades,
        )

        await save_performance_snapshot(
            session,
            portfolio_value=str(portfolio_value),
            daily_return_pct=daily_return,
            cumulative_return_pct=cumulative,
            sharpe_30d=sharpe,
            max_drawdown_30d=drawdown,
            win_rate=win_rate,
            total_decisions=total_decisions,
            total_trades=total_trades,
        )

        log.info(
            "Performance snapshot: value=%s daily=%.3f%% cumul=%.3f%% sharpe=%s dd=%s wr=%s",
            portfolio_value,
            (daily_return or 0) * 100,
            (cumulative or 0) * 100,
            f"{sharpe:.3f}" if sharpe is not None else "N/A",
            f"{(drawdown or 0)*100:.2f}%" if drawdown is not None else "N/A",
            f"{(win_rate or 0)*100:.1f}%" if win_rate is not None else "N/A",
        )
        return metrics


# ── Rolling metric calculations ───────────────────────────────────────────────

def _detect_snapshot_interval_hours(history) -> float:
    """Estimate the average interval between consecutive snapshots in hours.

    Used to determine the correct annualization factor for Sharpe. Returns
    1.0 (hourly) if there are fewer than 2 data points.
    """
    if len(history) < 2:
        return 1.0

    timestamps: list[datetime] = []
    for row in history:
        try:
            dt = datetime.fromisoformat(row.recorded_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            timestamps.append(dt)
        except (ValueError, AttributeError):
            continue

    if len(timestamps) < 2:
        return 1.0

    timestamps.sort()
    intervals = [
        (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
        for i in range(len(timestamps) - 1)
    ]
    avg = sum(intervals) / len(intervals)
    return max(avg, 1 / 60)  # floor at 1 minute to avoid div-by-zero


def _compute_daily_return(history, current_value: float) -> float | None:
    """Compare current portfolio value to the snapshot closest to 24 h ago.

    Uses the nearest snapshot to 24 h in the past rather than the oldest
    in the history window, giving a true 1-day return figure.
    """
    if not history:
        return None

    target = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    target_iso = target.isoformat()

    best_row = None
    best_delta = float("inf")
    for row in history:
        if row.recorded_at <= target_iso:
            try:
                dt = datetime.fromisoformat(row.recorded_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                delta = abs((target - dt).total_seconds())
                if delta < best_delta:
                    best_delta = delta
                    best_row = row
            except (ValueError, AttributeError):
                continue

    if best_row is None:
        return None

    day_open_value = _parse_float(best_row.portfolio_value)
    if day_open_value is None or day_open_value <= 0:
        return None
    return (current_value - day_open_value) / day_open_value


def _compute_rolling_sharpe(history) -> float | None:
    """Annualised Sharpe from consecutive returns in the snapshot history.

    The annualization factor is derived from the actual average interval
    between snapshots so the result is correct regardless of loop frequency.
    Returns None until at least _MIN_SHARPE_PERIODS snapshots are available.
    """
    if len(history) < _MIN_SHARPE_PERIODS:
        return None

    values: list[float] = []
    for row in history:
        v = _parse_float(row.portfolio_value)
        if v is not None and v > 0:
            values.append(v)

    if len(values) < 2:
        return None

    returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]
    n = len(returns)
    mean_r = sum(returns) / n
    if n < 2:
        return None
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0
    if std_r == 0:
        return None

    interval_hours = _detect_snapshot_interval_hours(history)
    periods_per_year = _HOURS_PER_YEAR / interval_hours
    return (mean_r / std_r) * math.sqrt(periods_per_year)


def _compute_max_drawdown(history, current_value: float) -> float | None:
    """Maximum percentage drawdown from peak over the history window."""
    if not history:
        return None

    values: list[float] = []
    for row in history:
        v = _parse_float(row.portfolio_value)
        if v is not None and v > 0:
            values.append(v)
    values.append(current_value)

    if len(values) < 2:
        return None

    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_win_rate(outcomes: list[float] | None) -> float | None:
    """Fraction of attributed decisions with positive P&L."""
    if not outcomes:
        return None
    wins = sum(1 for o in outcomes if o > 0)
    return wins / len(outcomes)


def _parse_float(s: str | None) -> float | None:
    try:
        return float(s) if s else None
    except (ValueError, TypeError):
        return None
