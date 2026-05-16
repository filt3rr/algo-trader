"""Feature engineering on top of OHLCV DataFrames.

All functions are pure (no side effects) and operate on a single-symbol
DataFrame indexed by timestamp. They return the same DataFrame with new
columns appended.

Convention: feature columns are lowercase, snake_case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_return and pct_return columns."""
    df = df.copy()
    df["pct_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_volatility(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling volatility (std of log returns, annualised) for each window."""
    df = df.copy()
    if "log_return" not in df.columns:
        df = add_returns(df)
    windows = windows or [14, 30]
    for w in windows:
        df[f"volatility_{w}"] = df["log_return"].rolling(w).std() * np.sqrt(252 * 24)
    return df


def add_vwap(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Add rolling VWAP. If vwap column already exists (from Alpaca), rename it."""
    df = df.copy()
    if "vwap" in df.columns:
        # Alpaca provides per-bar VWAP; compute rolling cumulative as well
        df["vwap_raw"] = df["vwap"]
    # Typical price × volume rolling VWAP
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_rolling"] = (
        (typical * df["volume"]).rolling(window).sum()
        / df["volume"].rolling(window).sum()
    )
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column."""
    import ta  # lazy import — only when needed
    df = df.copy()
    df[f"rsi_{period}"] = ta.momentum.RSIIndicator(
        close=df["close"], window=period
    ).rsi()
    return df


def add_ema(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Add EMA columns for each period."""
    df = df.copy()
    periods = periods or [12, 26, 50, 200]
    for p in periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD, MACD signal, and MACD histogram."""
    import ta
    df = df.copy()
    macd = ta.trend.MACD(close=df["close"], window_slow=slow, window_fast=fast, window_sign=signal)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands: upper, middle, lower, and %B."""
    import ta
    df = df.copy()
    bb = ta.volatility.BollingerBands(close=df["close"], window=period, window_dev=std)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct_b"] = bb.bollinger_pband()
    return df


def add_roc(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Add Rate of Change (momentum) for each period."""
    df = df.copy()
    periods = periods or [5, 10, 20]
    for p in periods:
        df[f"roc_{p}"] = df["close"].pct_change(p) * 100
    return df


def add_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add volume SMA, volume ratio, and OBV."""
    import ta
    df = df.copy()
    df["volume_sma"] = df["volume"].rolling(window).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()
    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple market regime label: trending_up, trending_down, ranging.

    Uses 50-period EMA slope and 20-period ATR relative to price.
    """
    df = df.copy()
    if "ema_50" not in df.columns:
        df = add_ema(df, [50])

    ema_slope = df["ema_50"].diff(5) / df["ema_50"].shift(5)
    atr_window = 14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_pct = tr.rolling(atr_window).mean() / df["close"]

    conditions = [
        (ema_slope > 0.01) & (atr_pct < 0.03),
        (ema_slope < -0.01) & (atr_pct < 0.03),
    ]
    choices = ["trending_up", "trending_down"]
    df["market_regime"] = np.select(conditions, choices, default="ranging")
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full standard feature pipeline to a single-symbol OHLCV DataFrame."""
    df = add_returns(df)
    df = add_volatility(df)
    df = add_ema(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_bollinger(df)
    df = add_roc(df)
    df = add_volume_features(df)
    df = add_vwap(df)
    df = add_market_regime(df)
    return df
