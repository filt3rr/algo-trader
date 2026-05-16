# Phase 1 Report — Data + Broker

**Date:** 2026-05-15
**Status:** Complete — awaiting review

---

## What Was Built

### Modules

| File | Description |
|------|-------------|
| `agent/config.py` | Pydantic-settings singleton; URL normalisation (strips `/v2`); universe CSV parser; hard guard against `LIVE_TRADING_ENABLED=True` at settings load time |
| `agent/broker/base.py` | Abstract `BrokerAdapter`; typed domain objects (`Order`, `Position`, `AccountInfo`, `OrderRequest`); `assert_paper_only()` with two-condition unlock |
| `agent/broker/alpaca.py` | `AlpacaBroker`: full `BrokerAdapter` implementation using `alpaca-py`; retry on all network calls; symbol normalisation (BTCUSD ↔ BTC/USD) |
| `agent/data/ingestion.py` | `MarketDataClient`: OHLCV bars + latest quotes; 6 timeframes (1m→1d); returns multi-index `(symbol, timestamp)` DataFrame |
| `agent/data/cache.py` | `OHLCVCache`: Parquet read/write with snappy compression; staleness detection (2× timeframe window); append-dedup-sort on write; `missing_range()` for efficient partial fetches |
| `agent/data/features.py` | `build_feature_set()`: full pipeline — returns, log returns, volatility, EMA (12/26/50/200), MACD, RSI(14), Bollinger Bands, ROC, volume features, VWAP, market regime label |
| `agent/reasoning/llm_base.py` | `LLMAdapter` abstract interface; `Message`, `LLMResponse` types; `get_llm_adapter()` factory |
| `agent/reasoning/claude_adapter.py` | Claude backend with prompt caching on system prompt (ephemeral cache); retry on rate limits |
| `agent/reasoning/local_adapter.py` | Ollama/OpenAI-compatible backend; auto-fallback between `/v1/chat/completions` and `/api/chat`; health check method |
| `scripts/connectivity_test.py` | End-to-end check: balance → live price → test order → cancel → LLM config |

### Tests Added

| File | Tests | Coverage target |
|------|-------|----------------|
| `tests/test_config.py` | 6 | config module |
| `tests/test_broker.py` | 10 | broker/base + broker/alpaca (mocked) |
| `tests/test_data.py` | 16 | data/cache + data/features + data/ingestion |

---

## Key Design Decisions

### 1. LLM Abstraction (new — added per user request)

The `LLMAdapter` interface lives in `agent/reasoning/llm_base.py`. Both `ClaudeAdapter` and `LocalLLMAdapter` implement it. The factory `get_llm_adapter()` reads `LLM_PROVIDER` from env. Switching to a local model requires:
- `LLM_PROVIDER=local`
- `LOCAL_LLM_MODEL=llama3.1:8b` (or any model)
- A running Ollama/LM Studio server

No other code changes needed.

### 2. URL Normalisation

The user's `.env` had `ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2`. The `alpaca-py` SDK appends `/v2` internally, so we strip it in the `Settings` validator. The live `.env` now uses the bare URL.

### 3. Prompt Caching on Claude

`ClaudeAdapter` uses Anthropic's ephemeral cache on the system prompt (`cache_control: ephemeral`). For a 5-minute trading loop with a ~2000-token system prompt, this reduces input token costs by ~90% on cache hits.

### 4. "Forever Learning" Architecture Note

The data layer is designed with continuous backtesting in mind:
- `OHLCVCache.missing_range()` lets the learning module fetch only the data it hasn't seen yet.
- `build_feature_set()` is deterministic and stateless — safe to call on any historical slice.
- Phase 4's learning module will use these to run rolling backtests on updated signal weights.

---

## Assumptions Made

1. **`alpaca-py 0.38` API** — Based on current docs. If the SDK changes its `TradingClient` constructor signature, `AlpacaBroker.__init__` needs updating.
2. **Crypto markets always open** — `is_market_open()` always returns `True`. Alpaca crypto trading is 24/7.
3. **Symbol format** — Alpaca sends `BTCUSD`; we normalise to `BTC/USD` throughout the codebase. The reverse (strip slash before submitting) happens in the broker adapter.
4. **Test order will cancel** — The connectivity test submits a 0.0001 BTC market order and immediately cancels. At high BTC prices this should cancel before filling, but if the market is fast it may fill. The test handles both outcomes gracefully.

---

## What's Next (Phase 2)

1. `agent/strategy/signals.py` — EMA crossover, RSI mean-reversion, ROC momentum signals.
2. `agent/strategy/backtest.py` — Vector backtest runner; report Sharpe, max drawdown, win rate.
3. `agent/risk/checks.py` — `RiskChecker` with all hard caps.
4. `agent/risk/sizing.py` — Kelly-capped position sizing.
5. Unit tests: every risk cap and circuit breaker must have a test.

**Phase 2 gate:** Running `pytest tests/test_risk.py` with 100% coverage on the risk module.

---

## Run the Connectivity Test

```bash
# Ensure .env is populated with your keys
python scripts/connectivity_test.py
```

Expected output:
```
[CONNECTED] Balance: $100,000.00 | BTC/USD last: $95,234.56
[ORDER]     Submitted buy 0.0001 BTC/USD (market) → id=xxxxxxxx
[CANCELLED] Order xxxxxxxx cancel: OK
[LLM]       Provider=claude | Model=claude-sonnet-4-5
[OK]        All connectivity checks passed.
```
