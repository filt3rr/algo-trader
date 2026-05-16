"""Signal generation engine — EMA crossover, RSI reversion, ROC momentum.

Each signal returns a normalised score in [-1, +1]:
    +1  = strong buy
     0  = neutral
    -1  = strong sell

Composite signal is a weighted average of the three raw scores. Weights are
loaded from the database (stored by the learning module) and default to equal
weighting.

The three signals were chosen for orthogonality:
    EMA crossover  — trend-following (captures sustained directional moves)
    RSI reversion  — mean-reversion  (captures overextended moves)
    ROC momentum   — rate-of-change  (captures acceleration without EMA lag)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import ta

log = logging.getLogger(__name__)


# ── Signal result types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalComponents:
    """Raw individual signal scores before weighting."""
    ema: float        # EMA crossover score  [-1, +1]
    rsi: float        # RSI reversion score  [-1, +1]
    roc: float        # ROC momentum score   [-1, +1]


@dataclass(frozen=True)
class SignalResult:
    """Full signal output for one symbol at one timestamp."""
    symbol: str
    timestamp: datetime
    components: SignalComponents
    composite: float          # weighted average of components [-1, +1]
    regime: str               # trending_up | trending_down | ranging
    weights: dict[str, float] # weights used (for audit / learning)
    close_price: float
    confidence: float         # |composite|, how certain is the signal

    @property
    def direction(self) -> str:
        if self.composite > 0.1:
            return "buy"
        if self.composite < -0.1:
            return "sell"
        return "hold"


# ── Default strategy weights (updated nightly by learning module) ─────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "ema": 1 / 3,
    "rsi": 1 / 3,
    "roc": 1 / 3,
}


# ── Individual signal computers ───────────────────────────────────────────────

def _ema_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """EMA crossover signal.

    Score = tanh((fast_ema - slow_ema) / slow_ema * 100)
    Positive when fast > slow (bullish), negative when fast < slow (bearish).
    tanh squashes to (-1, +1) and keeps proportionality.
    """
    fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
    raw = (fast_ema - slow_ema) / slow_ema * 100
    return np.tanh(raw * 2)  # *2 to steepen the curve near the zero cross


def _rsi_signal(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI mean-reversion signal.

    RSI < 30 → bullish reversion signal (+1)
    RSI > 70 → bearish reversion signal (-1)
    Score interpolates linearly between extremes and is zero at RSI=50.
    """
    rsi = ta.momentum.RSIIndicator(close=df["close"], window=period).rsi()
    # Map 0→+1 (oversold buy), 50→0 (neutral), 100→-1 (overbought sell)
    score = -(rsi - 50) / 50  # range (-1, +1)
    return score.clip(-1, 1)


def _roc_signal(df: pd.DataFrame, period: int = 10, lookback: int = 252) -> pd.Series:
    """ROC percentile-rank momentum signal.

    Computes the rolling percentile rank of ROC over the lookback window.
    High rank → momentum signal (buy); low rank → reversal signal (sell).
    Centred at 0.5 and scaled to (-1, +1).
    """
    roc = df["close"].pct_change(period)
    rank = roc.rolling(lookback, min_periods=max(20, period * 2)).rank(pct=True)
    return (rank - 0.5) * 2  # centre at 0, scale to (-1, +1)


def _market_regime(df: pd.DataFrame) -> pd.Series:
    """Simple regime classifier: trending_up | trending_down | ranging."""
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    slope = ema50.diff(5) / ema50.shift(5)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_pct = tr.rolling(14).mean() / df["close"]

    conditions = [
        (slope > 0.005) & (atr_pct < 0.04),
        (slope < -0.005) & (atr_pct < 0.04),
    ]
    choices = ["trending_up", "trending_down"]
    return pd.Series(
        np.select(conditions, choices, default="ranging"),
        index=df.index,
    )


# ── Main signal engine ────────────────────────────────────────────────────────

class SignalEngine:
    """Compute signals for a single-symbol OHLCV DataFrame."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        ema_fast: int = 12,
        ema_slow: int = 26,
        rsi_period: int = 14,
        roc_period: int = 10,
    ) -> None:
        self.weights = _normalise_weights(weights or DEFAULT_WEIGHTS.copy())
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.roc_period = roc_period

    def compute_all(self, df: pd.DataFrame, symbol: str) -> list[SignalResult]:
        """Compute signals for every row in df. Returns list ordered by timestamp."""
        if df.empty or len(df) < max(self.ema_slow, self.rsi_period, self.roc_period) + 10:
            log.warning("Insufficient data for %s (%d rows)", symbol, len(df))
            return []

        ema = _ema_signal(df, self.ema_fast, self.ema_slow)
        rsi = _rsi_signal(df, self.rsi_period)
        roc = _roc_signal(df, self.roc_period)
        regime = _market_regime(df)

        w = self.weights
        composite = (
            w["ema"] * ema
            + w["rsi"] * rsi
            + w["roc"] * roc
        )

        results: list[SignalResult] = []
        for ts in df.index:
            e = float(ema.get(ts, 0) or 0)
            r = float(rsi.get(ts, 0) or 0)
            o = float(roc.get(ts, 0) or 0)
            c = float(composite.get(ts, 0) or 0)
            reg = str(regime.get(ts, "ranging"))
            price = float(df.loc[ts, "close"])

            if any(np.isnan(v) for v in [e, r, o, c]):
                continue

            results.append(SignalResult(
                symbol=symbol,
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                components=SignalComponents(ema=e, rsi=r, roc=o),
                composite=c,
                regime=reg,
                weights=dict(w),
                close_price=price,
                confidence=abs(c),
            ))

        return results

    def compute_latest(self, df: pd.DataFrame, symbol: str) -> SignalResult | None:
        """Compute signal for the most recent bar only."""
        results = self.compute_all(df, symbol)
        return results[-1] if results else None

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Called by the learning module after each nightly reflection."""
        self.weights = _normalise_weights(new_weights)
        log.info("Signal weights updated: %s", self.weights)


def _normalise_weights(w: dict[str, float]) -> dict[str, float]:
    """Ensure weights sum to 1.0 and all are non-negative."""
    total = sum(abs(v) for v in w.values())
    if total == 0:
        return DEFAULT_WEIGHTS.copy()
    return {k: abs(v) / total for k, v in w.items()}
