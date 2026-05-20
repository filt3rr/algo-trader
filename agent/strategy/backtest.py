"""Vectorized backtesting of individual signals using backtesting.py.

Usage:
    from agent.strategy.backtest import run_backtest, BacktestResult
    result = run_backtest(df, "EMA", symbol="BTC/USD")
    print(result.summary())

The backtest runs each signal independently as a binary long/short/flat system.
This isolates signal quality — it does not simulate the full agent which
combines signals + Claude reasoning + risk management.

Parameters mirror production signal engine (1h bars, TIMEFRAME_PRESETS["1h"]):
    EMA:       fast=21, slow=55  (matches SignalEngine default)
    RSI:       period=14, regime-conditional thresholds (mean-rev in ranging, momentum in trending)
    ROC:       period=10, percentile-rank lookback=500
    COMPOSITE: equal-weight EMA+RSI+ROC composite, threshold ±0.2

Metrics reported:
    sharpe_ratio   — annualised Sharpe (risk-free rate = 0)
    max_drawdown   — peak-to-trough maximum drawdown (%)
    win_rate       — fraction of closed trades with positive return
    total_return   — total strategy return over the period (%)
    n_trades       — number of round-trip trades
    calmar_ratio   — total_return / abs(max_drawdown)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

log = logging.getLogger(__name__)

SignalName = Literal["EMA", "RSI", "ROC", "COMPOSITE"]

# ADX threshold: above this is trending (momentum RSI), below is ranging (mean-reversion RSI)
_ADX_TREND_THRESHOLD = 25


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    symbol: str
    signal: SignalName
    start_date: str
    end_date: str
    n_bars: int
    n_trades: int
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    calmar_ratio: float
    buy_hold_return_pct: float

    def summary(self) -> str:
        lines = [
            f"Signal: {self.signal} | Symbol: {self.symbol}",
            f"Period: {self.start_date} to {self.end_date} ({self.n_bars} bars)",
            f"  Return:       {self.total_return_pct:+.2f}%  (buy-hold: {self.buy_hold_return_pct:+.2f}%)",
            f"  Sharpe:       {self.sharpe_ratio:.3f}",
            f"  Max drawdown: {self.max_drawdown_pct:.2f}%",
            f"  Win rate:     {self.win_rate:.1%}",
            f"  Calmar:       {self.calmar_ratio:.3f}",
            f"  Trades:       {self.n_trades}",
        ]
        return "\n".join(lines)


# ── Strategy classes ──────────────────────────────────────────────────────────

class _EMACrossStrategy(Strategy):
    """EMA 21/55 crossover — matches production SignalEngine parameters."""
    fast = 21
    slow = 55

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        self.fast_ema = self.I(lambda c: c.ewm(span=self.fast, adjust=False).mean(), close)
        self.slow_ema = self.I(lambda c: c.ewm(span=self.slow, adjust=False).mean(), close)

    def next(self) -> None:
        if crossover(self.fast_ema, self.slow_ema):
            if not self.position.is_long:
                self.position.close()
                self.buy(size=0.95)
        elif crossover(self.slow_ema, self.fast_ema):
            if not self.position.is_short:
                self.position.close()
                self.sell(size=0.95)


class _RSIReversionStrategy(Strategy):
    """Regime-conditional RSI: mean-reversion in ranging markets, momentum in trending.

    Mirrors the production SignalEngine logic: ADX < 25 → contrarian thresholds;
    ADX >= 25 → momentum (buy on >50, sell on <50).
    """
    period = 14
    adx_period = 14

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.rsi = self.I(
            lambda c: ta.momentum.RSIIndicator(close=c, window=self.period).rsi(), close
        )

        def _adx(h, l, c):
            return ta.trend.ADXIndicator(high=h, low=l, close=c, window=self.adx_period).adx()

        self.adx = self.I(_adx, high, low, close)

    def next(self) -> None:
        rsi = self.rsi[-1]
        adx = self.adx[-1]
        if np.isnan(rsi) or np.isnan(adx):
            return

        trending = adx >= _ADX_TREND_THRESHOLD

        if trending:
            # Momentum: buy when RSI shows strength, sell when weak
            if rsi > 55 and not self.position.is_long:
                self.position.close()
                self.buy(size=0.95)
            elif rsi < 45 and not self.position.is_short:
                self.position.close()
                self.sell(size=0.95)
            elif 45 <= rsi <= 55:
                self.position.close()
        else:
            # Mean-reversion: buy oversold, sell overbought
            if rsi < 30 and not self.position.is_long:
                self.position.close()
                self.buy(size=0.95)
            elif rsi > 70 and not self.position.is_short:
                self.position.close()
                self.sell(size=0.95)
            elif 35 < rsi < 65:
                self.position.close()


class _ROCMomentumStrategy(Strategy):
    """ROC percentile-rank momentum — lookback=500 matches production."""
    period = 10
    lookback = 500

    def init(self) -> None:
        close = pd.Series(self.data.Close)

        def roc_rank(c: pd.Series) -> pd.Series:
            roc = c.pct_change(self.period)
            return roc.rolling(self.lookback, min_periods=20).rank(pct=True)

        self.rank = self.I(roc_rank, close)

    def next(self) -> None:
        rank = self.rank[-1]
        if np.isnan(rank):
            return
        if rank > 0.75 and not self.position.is_long:
            self.position.close()
            self.buy(size=0.95)
        elif rank < 0.25 and not self.position.is_short:
            self.position.close()
            self.sell(size=0.95)


class _CompositeStrategy(Strategy):
    """Equal-weight composite of EMA + regime-conditional RSI + ROC rank.

    Uses production parameters (1h). Enters long when composite > +0.2,
    short when < -0.2. Mirrors production signal thresholds.
    """
    ema_fast = 21
    ema_slow = 55
    rsi_period = 14
    adx_period = 14
    roc_period = 10
    roc_lookback = 500
    entry_threshold = 0.2

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.fast_ema = self.I(lambda c: c.ewm(span=self.ema_fast, adjust=False).mean(), close)
        self.slow_ema = self.I(lambda c: c.ewm(span=self.ema_slow, adjust=False).mean(), close)

        self.rsi = self.I(
            lambda c: ta.momentum.RSIIndicator(close=c, window=self.rsi_period).rsi(), close
        )

        def _adx(h, l, c):
            return ta.trend.ADXIndicator(high=h, low=l, close=c, window=self.adx_period).adx()
        self.adx = self.I(_adx, high, low, close)

        def roc_rank(c: pd.Series) -> pd.Series:
            roc = c.pct_change(self.roc_period)
            return roc.rolling(self.roc_lookback, min_periods=20).rank(pct=True)
        self.roc_rank = self.I(roc_rank, close)

    def _ema_score(self) -> float:
        fe, se = self.fast_ema[-1], self.slow_ema[-1]
        if np.isnan(fe) or np.isnan(se) or se == 0:
            return 0.0
        diff = (fe - se) / se
        return float(np.clip(diff * 10, -1.0, 1.0))

    def _rsi_score(self) -> float:
        rsi = self.rsi[-1]
        adx = self.adx[-1]
        if np.isnan(rsi) or np.isnan(adx):
            return 0.0
        if adx >= _ADX_TREND_THRESHOLD:
            # Momentum: positive when RSI > 50
            return float(np.clip((rsi - 50) / 50, -1.0, 1.0))
        else:
            # Mean-reversion: positive when oversold
            return float(np.clip((50 - rsi) / 50, -1.0, 1.0))

    def _roc_score(self) -> float:
        rank = self.roc_rank[-1]
        if np.isnan(rank):
            return 0.0
        return float(np.clip((rank - 0.5) * 2, -1.0, 1.0))

    def next(self) -> None:
        ema = self._ema_score()
        rsi = self._rsi_score()
        roc = self._roc_score()
        composite = (ema + rsi + roc) / 3.0

        if composite > self.entry_threshold and not self.position.is_long:
            self.position.close()
            self.buy(size=0.95)
        elif composite < -self.entry_threshold and not self.position.is_short:
            self.position.close()
            self.sell(size=0.95)
        elif abs(composite) < self.entry_threshold * 0.5:
            self.position.close()


_STRATEGY_MAP: dict[str, type[Strategy]] = {
    "EMA": _EMACrossStrategy,
    "RSI": _RSIReversionStrategy,
    "ROC": _ROCMomentumStrategy,
    "COMPOSITE": _CompositeStrategy,
}


# ── Main backtest runner ──────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    signal: SignalName,
    symbol: str = "UNKNOWN",
    initial_cash: float = 100_000,
    commission: float = 0.002,  # 0.2% per side (realistic for crypto including spread)
) -> BacktestResult:
    """Run a vectorized backtest for one signal on one symbol's OHLCV data.

    Args:
        df:           DataFrame with columns open/high/low/close/volume, DatetimeIndex.
        signal:       Which signal strategy to run ('EMA', 'RSI', 'ROC', 'COMPOSITE').
        symbol:       Symbol name for labelling.
        initial_cash: Starting portfolio value in USD.
        commission:   Per-trade commission fraction (both sides). Defaults to 0.2%.

    Returns:
        BacktestResult with all performance metrics.
    """
    if signal not in _STRATEGY_MAP:
        raise ValueError(f"Unknown signal: {signal!r}. Choose from {list(_STRATEGY_MAP)}")

    # backtesting.py needs capitalised OHLCV columns
    bt_df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]].copy()

    bt_df = bt_df.dropna()
    bt_df.index = pd.to_datetime(bt_df.index)

    strategy_cls = _STRATEGY_MAP[signal]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt = Backtest(bt_df, strategy_cls, cash=initial_cash, commission=commission)
        stats = bt.run()

    buy_hold = (bt_df["Close"].iloc[-1] / bt_df["Close"].iloc[0] - 1) * 100

    trades = stats.get("_trades")
    if trades is not None and len(trades) > 0:
        profitable = (trades["PnL"] > 0).sum()
        win_rate = float(profitable / len(trades))
    else:
        win_rate = 0.0

    raw_dd = stats.get("Max. Drawdown [%]", 0.0)
    max_dd = abs(float(raw_dd)) if raw_dd is not None else 0.0

    total_ret = float(stats.get("Return [%]", 0.0) or 0.0)
    calmar = total_ret / max_dd if max_dd > 0 else 0.0

    sharpe = float(stats.get("Sharpe Ratio", 0.0) or 0.0)
    n_trades = int(stats.get("# Trades", 0) or 0)

    return BacktestResult(
        symbol=symbol,
        signal=signal,
        start_date=str(bt_df.index[0].date()),
        end_date=str(bt_df.index[-1].date()),
        n_bars=len(bt_df),
        n_trades=n_trades,
        total_return_pct=total_ret,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        calmar_ratio=calmar,
        buy_hold_return_pct=float(buy_hold),
    )


def run_all_signals(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    initial_cash: float = 100_000,
) -> dict[SignalName, BacktestResult]:
    """Run all four signal strategies and return a results dict."""
    results: dict[SignalName, BacktestResult] = {}
    for sig in ("EMA", "RSI", "ROC", "COMPOSITE"):
        try:
            results[sig] = run_backtest(df, sig, symbol, initial_cash)
            log.info("Backtest %s/%s done: Sharpe=%.3f", symbol, sig, results[sig].sharpe_ratio)
        except Exception as exc:
            log.error("Backtest %s/%s failed: %s", symbol, sig, exc)
    return results
