"""Tests for agent/strategy — signal engine and backtest runner."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from agent.strategy.signals import (
    DEFAULT_WEIGHTS,
    SignalComponents,
    SignalEngine,
    SignalResult,
    _ema_signal,
    _normalise_weights,
    _roc_signal,
    _rsi_signal,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, seed: int = 42, trending: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1d", tz="UTC")
    if trending:
        # Upward trend
        close = 40_000 + np.cumsum(rng.normal(100, 500, n))
    else:
        # Sideways / mean-reverting
        close = 40_000 + rng.normal(0, 800, n)
    close = np.maximum(close, 1)  # floor at $1
    high = close + rng.uniform(100, 1000, n)
    low = close - rng.uniform(100, 1000, n)
    low = np.maximum(low, 1)
    volume = rng.uniform(100, 5000, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ── Weight normalisation ──────────────────────────────────────────────────────

def test_normalise_weights_sums_to_one() -> None:
    w = _normalise_weights({"ema": 2.0, "rsi": 1.0, "roc": 1.0})
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_normalise_weights_zero_returns_default() -> None:
    w = _normalise_weights({"ema": 0.0, "rsi": 0.0, "roc": 0.0})
    assert w == DEFAULT_WEIGHTS


def test_normalise_weights_handles_negatives() -> None:
    w = _normalise_weights({"ema": -1.0, "rsi": 2.0, "roc": 1.0})
    assert all(v >= 0 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-9


# ── Individual signal functions ───────────────────────────────────────────────

def test_ema_signal_range(tmp_path) -> None:
    df = _make_ohlcv()
    scores = _ema_signal(df)
    valid = scores.dropna()
    assert (valid >= -1).all() and (valid <= 1).all()


def test_ema_signal_positive_in_uptrend() -> None:
    df = _make_ohlcv(trending=True)
    scores = _ema_signal(df)
    # In an uptrend, the fast EMA should be above slow → positive scores dominate
    assert scores.dropna().mean() > 0


def test_rsi_signal_range() -> None:
    df = _make_ohlcv()
    scores = _rsi_signal(df)
    valid = scores.dropna()
    assert (valid >= -1).all() and (valid <= 1).all()


def test_rsi_signal_neutral_at_rsi_50() -> None:
    # For a sideways-oscillating series, RSI stays near 50 → score near 0 on average
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=200, freq="1D", tz="UTC")
    prices = 40000 + np.cumsum(rng.normal(0, 50, 200))  # zero-drift noise
    sideways = pd.DataFrame({"close": prices}, index=idx)
    scores = _rsi_signal(sideways)
    # Mean score across a zero-drift series should be near 0 (within 0.3 tolerance)
    assert abs(scores.dropna().mean()) < 0.3


def test_roc_signal_range() -> None:
    df = _make_ohlcv(n=400)
    scores = _roc_signal(df)
    valid = scores.dropna()
    assert (valid >= -1).all() and (valid <= 1).all()


# ── SignalEngine ──────────────────────────────────────────────────────────────

def test_signal_engine_compute_all_returns_results() -> None:
    engine = SignalEngine()
    df = _make_ohlcv()
    results = engine.compute_all(df, "BTC/USD")
    assert len(results) > 0
    assert all(isinstance(r, SignalResult) for r in results)


def test_signal_engine_composite_in_range() -> None:
    engine = SignalEngine()
    df = _make_ohlcv()
    results = engine.compute_all(df, "BTC/USD")
    for r in results:
        assert -1.0 <= r.composite <= 1.0


def test_signal_engine_confidence_is_abs_composite() -> None:
    engine = SignalEngine()
    df = _make_ohlcv()
    results = engine.compute_all(df, "BTC/USD")
    for r in results:
        assert abs(r.confidence - abs(r.composite)) < 1e-9


def test_signal_result_direction_buy() -> None:
    r = SignalResult(
        symbol="BTC/USD",
        timestamp=datetime.now(tz=timezone.utc),
        components=SignalComponents(ema=0.8, rsi=0.5, roc=0.6),
        composite=0.5,
        regime="trending_up",
        weights={"ema": 1/3, "rsi": 1/3, "roc": 1/3},
        close_price=80000.0,
        confidence=0.5,
    )
    assert r.direction == "buy"


def test_signal_result_direction_sell() -> None:
    r = SignalResult(
        symbol="BTC/USD",
        timestamp=datetime.now(tz=timezone.utc),
        components=SignalComponents(ema=-0.8, rsi=-0.5, roc=-0.6),
        composite=-0.5,
        regime="trending_down",
        weights={"ema": 1/3, "rsi": 1/3, "roc": 1/3},
        close_price=80000.0,
        confidence=0.5,
    )
    assert r.direction == "sell"


def test_signal_result_direction_hold() -> None:
    r = SignalResult(
        symbol="BTC/USD",
        timestamp=datetime.now(tz=timezone.utc),
        components=SignalComponents(ema=0.05, rsi=-0.05, roc=0.02),
        composite=0.0,
        regime="ranging",
        weights={"ema": 1/3, "rsi": 1/3, "roc": 1/3},
        close_price=80000.0,
        confidence=0.0,
    )
    assert r.direction == "hold"


def test_signal_engine_compute_latest_returns_last() -> None:
    engine = SignalEngine()
    df = _make_ohlcv()
    all_results = engine.compute_all(df, "BTC/USD")
    latest = engine.compute_latest(df, "BTC/USD")
    assert latest is not None
    assert latest.timestamp == all_results[-1].timestamp


def test_signal_engine_returns_empty_for_insufficient_data() -> None:
    engine = SignalEngine()
    df = _make_ohlcv(n=10)  # too few bars
    results = engine.compute_all(df, "BTC/USD")
    assert results == []


def test_signal_engine_compute_latest_returns_none_for_empty() -> None:
    engine = SignalEngine()
    df = _make_ohlcv(n=5)
    result = engine.compute_latest(df, "BTC/USD")
    assert result is None


def test_signal_engine_update_weights() -> None:
    engine = SignalEngine()
    engine.update_weights({"ema": 0.6, "rsi": 0.2, "roc": 0.2})
    assert abs(engine.weights["ema"] - 0.6) < 1e-9
    assert abs(sum(engine.weights.values()) - 1.0) < 1e-9


def test_signal_engine_regime_labels() -> None:
    engine = SignalEngine()
    df = _make_ohlcv(n=300)
    results = engine.compute_all(df, "BTC/USD")
    valid_regimes = {"trending_up", "trending_down", "ranging"}
    for r in results:
        assert r.regime in valid_regimes


# ── Backtest (smoke test with synthetic data — no live API) ───────────────────

def test_backtest_ema_runs_without_error() -> None:
    from agent.strategy.backtest import run_backtest
    df = _make_ohlcv(n=365)
    result = run_backtest(df, "EMA", symbol="TEST/USD")
    assert result.signal == "EMA"
    assert isinstance(result.sharpe_ratio, float)
    assert result.n_bars > 0


def test_backtest_rsi_runs_without_error() -> None:
    from agent.strategy.backtest import run_backtest
    df = _make_ohlcv(n=365, trending=False)
    result = run_backtest(df, "RSI", symbol="TEST/USD")
    assert result.signal == "RSI"


def test_backtest_roc_runs_without_error() -> None:
    from agent.strategy.backtest import run_backtest
    df = _make_ohlcv(n=365)
    result = run_backtest(df, "ROC", symbol="TEST/USD")
    assert result.signal == "ROC"


def test_backtest_summary_format() -> None:
    from agent.strategy.backtest import run_backtest
    df = _make_ohlcv(n=365)
    result = run_backtest(df, "EMA", symbol="TEST/USD")
    summary = result.summary()
    assert "EMA" in summary
    assert "TEST/USD" in summary
    assert "Sharpe" in summary


def test_backtest_unknown_signal_raises() -> None:
    from agent.strategy.backtest import run_backtest
    df = _make_ohlcv(n=365)
    with pytest.raises(ValueError, match="Unknown signal"):
        run_backtest(df, "INVALID", symbol="TEST/USD")  # type: ignore[arg-type]


def test_run_all_signals_returns_four() -> None:
    from agent.strategy.backtest import run_all_signals
    df = _make_ohlcv(n=365)
    results = run_all_signals(df, symbol="TEST/USD")
    assert set(results.keys()) == {"EMA", "RSI", "ROC", "COMPOSITE"}
