"""Tests for agent/data — cache, features, symbol normalisation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.data.cache import OHLCVCache
from agent.data.features import (
    add_bollinger,
    add_ema,
    add_macd,
    add_market_regime,
    add_returns,
    add_roc,
    add_rsi,
    add_volatility,
    add_volume_features,
    add_vwap,
    build_feature_set,
)
from agent.data.ingestion import _normalise_symbol


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 45000 + np.cumsum(rng.normal(0, 200, n))
    high = close + rng.uniform(50, 300, n)
    low = close - rng.uniform(50, 300, n)
    open_ = close + rng.normal(0, 100, n)
    volume = rng.uniform(10, 500, n)
    vwap = (high + low + close) / 3

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "vwap": vwap, "trade_count": 100},
        index=idx,
    )


# ── Symbol normalisation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("BTCUSD", "BTC/USD"),
    ("ETHUSD", "ETH/USD"),
    ("SOLUSD", "SOL/USD"),
    ("BTC/USD", "BTC/USD"),
    ("ETH/USD", "ETH/USD"),
    ("AVAXUSD", "AVAX/USD"),
])
def test_normalise_symbol(raw: str, expected: str) -> None:
    assert _normalise_symbol(raw) == expected


# ── Feature engineering ───────────────────────────────────────────────────────

def test_add_returns_columns(tmp_path: Path) -> None:
    df = _make_ohlcv()
    out = add_returns(df)
    assert "pct_return" in out.columns
    assert "log_return" in out.columns
    assert out["pct_return"].notna().sum() > 90


def test_add_volatility_windows(tmp_path: Path) -> None:
    df = add_returns(_make_ohlcv())
    out = add_volatility(df, windows=[14, 30])
    assert "volatility_14" in out.columns
    assert "volatility_30" in out.columns
    assert out["volatility_14"].dropna().gt(0).all()


def test_add_ema_periods() -> None:
    df = _make_ohlcv()
    out = add_ema(df, periods=[12, 26])
    assert "ema_12" in out.columns
    assert "ema_26" in out.columns
    # EMA(26) should lag EMA(12) in a trending market
    assert out["ema_26"].notna().sum() > 0


def test_add_rsi_bounded() -> None:
    df = _make_ohlcv(200)
    out = add_rsi(df, period=14)
    valid = out["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_add_macd_columns() -> None:
    df = _make_ohlcv()
    out = add_macd(df)
    assert all(c in out.columns for c in ["macd", "macd_signal", "macd_hist"])


def test_add_bollinger_bands() -> None:
    df = _make_ohlcv()
    out = add_bollinger(df)
    assert "bb_upper" in out.columns and "bb_lower" in out.columns
    # Upper band should always be above lower
    valid = out[["bb_upper", "bb_lower"]].dropna()
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_add_roc_columns() -> None:
    df = _make_ohlcv()
    out = add_roc(df, periods=[5, 10])
    assert "roc_5" in out.columns and "roc_10" in out.columns


def test_add_volume_features() -> None:
    df = _make_ohlcv()
    out = add_volume_features(df)
    assert "volume_ratio" in out.columns and "obv" in out.columns


def test_add_market_regime_values() -> None:
    df = _make_ohlcv(200)
    out = add_market_regime(df)
    assert "market_regime" in out.columns
    valid_regimes = {"trending_up", "trending_down", "ranging"}
    assert set(out["market_regime"].unique()).issubset(valid_regimes)


def test_build_feature_set_no_original_columns_dropped() -> None:
    df = _make_ohlcv()
    out = build_feature_set(df)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in out.columns


# ── OHLCVCache ────────────────────────────────────────────────────────────────

def test_cache_save_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.data.cache.get_settings",
                        lambda: _mock_settings(str(tmp_path)))
    cache = OHLCVCache(base_dir=tmp_path)
    df = _make_ohlcv()
    cache.save(df, "BTC/USD", "1h")

    loaded = cache.load("BTC/USD", "1h")
    assert loaded is not None
    assert len(loaded) == len(df)


def test_cache_deduplication(tmp_path: Path) -> None:
    cache = OHLCVCache(base_dir=tmp_path)
    df = _make_ohlcv(50)
    # Save same data twice — should not duplicate rows
    cache.save(df, "ETH/USD", "1h")
    cache.save(df, "ETH/USD", "1h")
    loaded = cache.load("ETH/USD", "1h")
    assert loaded is not None
    assert len(loaded) == 50


def test_cache_returns_none_for_missing(tmp_path: Path) -> None:
    cache = OHLCVCache(base_dir=tmp_path)
    result = cache.load("NONEXISTENT/USD", "1h")
    assert result is None


def test_cache_is_fresh_false_for_missing(tmp_path: Path) -> None:
    cache = OHLCVCache(base_dir=tmp_path)
    assert cache.is_fresh("BTC/USD", "1h") is False


def test_cache_load_range(tmp_path: Path) -> None:
    cache = OHLCVCache(base_dir=tmp_path)
    df = _make_ohlcv(100)
    cache.save(df, "SOL/USD", "1h")

    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end = datetime(2024, 1, 4, tzinfo=timezone.utc)
    sliced = cache.load_range("SOL/USD", "1h", start, end)
    assert sliced is not None
    assert len(sliced) > 0
    assert sliced.index.min() >= pd.Timestamp(start)


def test_cache_load_range_empty_when_no_data(tmp_path: Path) -> None:
    cache = OHLCVCache(base_dir=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    result = cache.load_range("BTC/USD", "1h", start, end)
    assert result is None


def test_cache_missing_range_fully_covered(tmp_path: Path) -> None:
    cache = OHLCVCache(base_dir=tmp_path)
    df = _make_ohlcv(200)  # 2024-01-01 to 2024-01-09 (200 hours)
    cache.save(df, "BTC/USD", "1h")

    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end = datetime(2024, 1, 4, tzinfo=timezone.utc)
    gap = cache.missing_range("BTC/USD", "1h", start, end)
    # Cached data fully covers this range
    assert gap is None


def test_cache_missing_range_not_cached(tmp_path: Path) -> None:
    cache = OHLCVCache(base_dir=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 7, tzinfo=timezone.utc)
    gap = cache.missing_range("NEW/USD", "1h", start, end)
    assert gap == (start, end)


def test_cache_is_fresh_stale_data(tmp_path: Path) -> None:
    from datetime import timedelta
    cache = OHLCVCache(base_dir=tmp_path)
    # Create a dataframe with old timestamps (well outside 2h staleness window for 1h bars)
    old_idx = pd.date_range("2020-01-01", periods=50, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0, "vwap": 1.0, "trade_count": 1}, index=old_idx)
    cache.save(df, "STALE/USD", "1h")
    assert cache.is_fresh("STALE/USD", "1h") is False


# ── helpers ───────────────────────────────────────────────────────────────────

class _mock_settings:
    def __init__(self, parquet_dir: str):
        self.parquet_dir = parquet_dir
