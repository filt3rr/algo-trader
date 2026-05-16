"""Vectorized backtesting of individual signals using backtesting.py.

Usage:
    from agent.strategy.backtest import run_backtest, BacktestResult
    result = run_backtest(df, "EMA", symbol="BTC/USD")
    print(result.summary())

The backtest runs each signal independently as a binary long/short/flat system.
This isolates signal quality — it does not simulate the full agent which
combines signals + Claude reasoning + risk management.

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
    fast = 12
    slow = 26

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
    period = 14
    oversold = 30
    overbought = 70

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        self.rsi = self.I(
            lambda c: ta.momentum.RSIIndicator(close=c, window=self.period).rsi(), close
        )

    def next(self) -> None:
        rsi = self.rsi[-1]
        if rsi < self.oversold and not self.position.is_long:
            self.position.close()
            self.buy(size=0.95)
        elif rsi > self.overbought and not self.position.is_short:
            self.position.close()
            self.sell(size=0.95)
        elif self.oversold + 5 < rsi < self.overbought - 5:
            self.position.close()


class _ROCMomentumStrategy(Strategy):
    period = 10
    lookback = 126  # ~6 months of daily bars for rank

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


_STRATEGY_MAP: dict[str, type[Strategy]] = {
    "EMA": _EMACrossStrategy,
    "RSI": _RSIReversionStrategy,
    "ROC": _ROCMomentumStrategy,
}


# ── Main backtest runner ──────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    signal: SignalName,
    symbol: str = "UNKNOWN",
    initial_cash: float = 100_000,
    commission: float = 0.001,  # 0.1% per side (realistic for crypto)
) -> BacktestResult:
    """Run a vectorized backtest for one signal on one symbol's OHLCV data.

    Args:
        df:           DataFrame with columns open/high/low/close/volume, DatetimeIndex.
        signal:       Which signal strategy to run ('EMA', 'RSI', 'ROC').
        symbol:       Symbol name for labelling.
        initial_cash: Starting portfolio value in USD.
        commission:   Per-trade commission fraction (both sides).

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
    """Run all three signal strategies and return a results dict."""
    results: dict[SignalName, BacktestResult] = {}
    for sig in ("EMA", "RSI", "ROC"):
        try:
            results[sig] = run_backtest(df, sig, symbol, initial_cash)
            log.info("Backtest %s/%s done: Sharpe=%.3f", symbol, sig, results[sig].sharpe_ratio)
        except Exception as exc:
            log.error("Backtest %s/%s failed: %s", symbol, sig, exc)
    return results
