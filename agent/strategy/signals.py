"""Signal generation engine — EMA crossover, RSI (regime-conditional), ROC momentum.

Each signal returns a normalised score in [-1, +1]:
    +1  = strong buy
     0  = neutral
    -1  = strong sell

Improvements over v1:
    - EMA parameters extended to 21/55 (less whipsaw on 1h crypto bars)
    - RSI is now regime-conditional: mean-reversion in ranging markets,
      momentum in trending markets — resolves the structural conflict where
      trend-following (EMA) and mean-reversion (RSI) cancel each other.
    - Market regime uses ADX + directional indicators instead of EMA slope,
      giving a much cleaner trending/ranging classification.
    - Volume confirmation multiplier: low-volume signals are dampened (possible
      noise / fake breakout); high-volume signals are amplified (confirmed move).
    - ROC lookback extended to 500 bars (~21 days) to capture crypto market cycles.
    - signal_agreement field tracks how many sub-signals align with the composite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import ta

log = logging.getLogger(__name__)


# ── Signal parameters ─────────────────────────────────────────────────────────

_EMA_FAST       = 21           # ~21h short-trend EMA
_EMA_SLOW       = 55           # ~55h long-trend EMA (less whipsaw than 12/26)
_RSI_PERIOD     = 14
_ROC_PERIOD     = 10
_ROC_LOOKBACK   = 500          # ~21 days of 1h bars for a stable percentile rank
_VOL_WINDOW     = 20           # bars for rolling volume average
_ADX_PERIOD     = 14
_ADX_TREND_THR  = 25.0         # ADX above this → trending regime


# ── Signal result types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalComponents:
    """Raw sub-signal scores before weighting and volume adjustment."""
    ema: float          # EMA crossover score        [-1, +1]
    rsi: float          # RSI regime-conditional      [-1, +1]
    roc: float          # ROC percentile-rank         [-1, +1]
    volume_mult: float = 1.0  # Volume confirmation multiplier [0.4, 1.0]


@dataclass(frozen=True)
class SignalResult:
    """Full signal output for one symbol at one timestamp."""
    symbol: str
    timestamp: datetime
    components: SignalComponents
    composite: float           # volume-adjusted weighted composite [-1, +1]
    regime: str                # trending_up | trending_down | ranging
    weights: dict[str, float]  # weights used for ema/rsi/roc
    close_price: float
    confidence: float          # |composite| — signal conviction
    signal_agreement: float = 0.0  # fraction of sub-signals aligned with composite [0, 1]

    @property
    def direction(self) -> str:
        if self.composite > 0.1:
            return "buy"
        if self.composite < -0.1:
            return "sell"
        return "hold"


# ── Default weights (updated nightly by learning module) ──────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "ema": 1 / 3,
    "rsi": 1 / 3,
    "roc": 1 / 3,
}

# ── Per-timeframe signal engine presets ───────────────────────────────────────
# Each dict is passed as **kwargs to SignalEngine.__init__.
# 15m: fast EMAs for intraday momentum; shorter ROC lookback (~2 days).
# 4h:  same EMA spans as 1h but operating on 4h bars → captures 3–9 day trends.
TIMEFRAME_PRESETS: dict[str, dict] = {
    "15m": dict(ema_fast=8,  ema_slow=21, rsi_period=14, roc_period=5,  roc_lookback=200, volume_window=20),
    "1h":  dict(ema_fast=21, ema_slow=55, rsi_period=14, roc_period=10, roc_lookback=500, volume_window=20),
    "4h":  dict(ema_fast=21, ema_slow=55, rsi_period=14, roc_period=10, roc_lookback=500, volume_window=20),
}

# Days of historical bars to fetch per timeframe (ensures enough bars for lookbacks)
TIMEFRAME_FETCH_DAYS: dict[str, int] = {
    "15m": 14,   # 14d × 24h × 4 = 1 344 bars  (> 200 ROC lookback)
    "1h":  60,   # 60d × 24    =   1 440 bars  (> 500 ROC lookback)
    "4h":  150,  # 150d × 6    =     900 bars  (> 500 ROC lookback)
}


# ── Individual signal computers ───────────────────────────────────────────────

def _market_regime(df: pd.DataFrame) -> pd.Series:
    """Regime classifier using ADX + directional indicators.

    trending_up:   ADX >= 25 AND +DI > -DI  (confirmed bullish trend)
    trending_down: ADX >= 25 AND +DI <= -DI (confirmed bearish trend)
    ranging:       ADX < 25                 (no clear directional trend)

    Falls back to all-ranging if high/low columns are absent.
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series("ranging", index=df.index)

    adx_ind  = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=_ADX_PERIOD
    )
    adx      = adx_ind.adx()
    plus_di  = adx_ind.adx_pos()
    minus_di = adx_ind.adx_neg()

    trending = adx >= _ADX_TREND_THR
    conditions = [
        trending & (plus_di > minus_di),
        trending & (plus_di <= minus_di),
    ]
    choices = ["trending_up", "trending_down"]
    return pd.Series(
        np.select(conditions, choices, default="ranging"),
        index=df.index,
    )


def _ema_signal(df: pd.DataFrame, fast: int = _EMA_FAST, slow: int = _EMA_SLOW) -> pd.Series:
    """EMA crossover signal.

    Score = tanh((fast_ema - slow_ema) / slow_ema * 200)
    Longer spans (21/55 vs old 12/26) reduce whipsaw on 1h crypto bars.
    Positive when fast > slow (bullish), negative when fast < slow (bearish).
    """
    fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
    raw = (fast_ema - slow_ema) / slow_ema * 100  # percentage gap
    # Scale factor 0.3: 1% gap→0.29, 3%→0.72, 5%→0.91 — graded, not binary
    return np.tanh(raw * 0.3).clip(-1, 1)


def _rsi_signal(df: pd.DataFrame, period: int = _RSI_PERIOD) -> pd.Series:
    """RSI signal, regime-conditional.

    Ranging markets  → mean-reversion: RSI < 30 → buy (+1), RSI > 70 → sell (-1)
    Trending markets → momentum:       RSI > 50 confirms trend direction

    This resolves the structural conflict where EMA (trend-following) and
    RSI mean-reversion cancel each other in trending markets.
    """
    rsi    = ta.momentum.RSIIndicator(close=df["close"], window=period).rsi()
    regime = _market_regime(df)

    reversion = -(rsi - 50) / 50   # overbought → sell, oversold → buy
    momentum  =  (rsi - 50) / 50   # RSI > 50 aligns with uptrend, < 50 with downtrend

    is_ranging = pd.Series(regime == "ranging", index=df.index)
    return reversion.where(is_ranging, momentum).clip(-1, 1)


def _roc_signal(df: pd.DataFrame, period: int = _ROC_PERIOD, lookback: int = _ROC_LOOKBACK) -> pd.Series:
    """ROC percentile-rank momentum signal.

    Longer lookback (500 bars ≈ 21 days) captures crypto's multi-month cycles.
    Rolling percentile rank normalises across different volatility regimes.
    """
    roc  = df["close"].pct_change(period)
    rank = roc.rolling(lookback, min_periods=max(30, period * 3)).rank(pct=True)
    return (rank - 0.5) * 2


def _volume_multiplier(df: pd.DataFrame, window: int = _VOL_WINDOW) -> pd.Series:
    """Volume confirmation multiplier in [0.4, 1.0].

    Maps volume relative to its rolling average to a signal scaling factor:
        vol < 0.5× avg  → 0.40 (likely noise; dampen signal significantly)
        vol = 1.0× avg  → 0.70 (neutral confirmation)
        vol ≥ 2.0× avg  → 1.00 (strong confirmation; signal at full strength)

    Returns a neutral multiplier (1.0) if volume column is absent.
    """
    if "volume" not in df.columns:
        return pd.Series(1.0, index=df.index)

    vol_ma    = df["volume"].rolling(window, min_periods=5).mean()
    vol_ratio = (df["volume"] / vol_ma.replace(0, np.nan)).fillna(1.0).clip(0.1, 5.0)
    return (0.4 + 0.6 * ((vol_ratio - 0.5) / 1.5).clip(0.0, 1.0))


# ── Main signal engine ────────────────────────────────────────────────────────

class SignalEngine:
    """Compute composite signals for a single-symbol OHLCV DataFrame."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        ema_fast: int = _EMA_FAST,
        ema_slow: int = _EMA_SLOW,
        rsi_period: int = _RSI_PERIOD,
        roc_period: int = _ROC_PERIOD,
        roc_lookback: int = _ROC_LOOKBACK,
        volume_window: int = _VOL_WINDOW,
    ) -> None:
        self.weights       = _normalise_weights(weights or DEFAULT_WEIGHTS.copy())
        self.ema_fast      = ema_fast
        self.ema_slow      = ema_slow
        self.rsi_period    = rsi_period
        self.roc_period    = roc_period
        self.roc_lookback  = roc_lookback
        self.volume_window = volume_window

    def compute_all(self, df: pd.DataFrame, symbol: str) -> list[SignalResult]:
        """Compute signals for every valid row in df."""
        min_bars = max(self.ema_slow, self.rsi_period, self.roc_period) + _ADX_PERIOD + 10
        if df.empty or len(df) < min_bars:
            log.warning("Insufficient data for %s (%d rows, need %d)", symbol, len(df), min_bars)
            return []

        ema      = _ema_signal(df, self.ema_fast, self.ema_slow)
        rsi      = _rsi_signal(df, self.rsi_period)
        roc      = _roc_signal(df, self.roc_period, self.roc_lookback)
        vol_mult = _volume_multiplier(df, self.volume_window)
        regime   = _market_regime(df)

        w = self.weights
        raw_composite = w["ema"] * ema + w["rsi"] * rsi + w["roc"] * roc
        composite = (raw_composite * vol_mult).clip(-1, 1)

        results: list[SignalResult] = []
        for ts in df.index:
            e   = float(ema.get(ts, 0) or 0)
            r   = float(rsi.get(ts, 0) or 0)
            o   = float(roc.get(ts, 0) or 0)
            vm  = float(vol_mult.get(ts, 0.7) or 0.7)
            c   = float(composite.get(ts, 0) or 0)
            reg = str(regime.get(ts, "ranging"))
            price = float(df.loc[ts, "close"])

            if any(np.isnan(v) for v in [e, r, o, c]):
                continue

            # Signal agreement: fraction of EMA/RSI/ROC that agree with composite direction
            if abs(c) > 0.01:
                comp_sign = np.sign(c)
                agreement = sum(1 for s in [e, r, o] if np.sign(s) == comp_sign) / 3
            else:
                agreement = 0.0

            results.append(SignalResult(
                symbol=symbol,
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                components=SignalComponents(ema=e, rsi=r, roc=o, volume_mult=vm),
                composite=c,
                regime=reg,
                weights=dict(w),
                close_price=price,
                confidence=abs(c),
                signal_agreement=agreement,
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
