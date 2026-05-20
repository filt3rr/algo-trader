# Crypto Paper-Trading Agent

> A reflective, self-improving autonomous trading agent — technical signals meet LLM reasoning.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-dashboard-009688)](https://fastapi.tiangolo.com)
[![Claude](https://img.shields.io/badge/LLM-Claude%20%7C%20Ollama-purple)](https://anthropic.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/coverage-94%25-brightgreen)](tests/)

> **Disclaimer** — This software is for **educational and research purposes only**. It is not financial advice. Paper trading eliminates slippage, partial fills, funding costs, and the psychological pressure of real capital — any edge observed here may vanish entirely in live markets. The authors accept no responsibility for financial losses. **This system only executes paper trades.** Enabling live trading requires deliberate multi-step code changes described in `agent/broker/base.py`.

---

## What It Does

Every 5 minutes, the agent:

1. **Fetches live OHLCV data** from Alpaca's paper trading API for your configured universe of crypto pairs
2. **Computes three technical signals** per symbol — EMA crossover (trend), regime-conditional RSI (mean-reversion or momentum), and ROC percentile (cycle positioning) — weighted by a volume multiplier and gated by ADX regime classification
3. **Assembles multi-timeframe context** — 4h macro trend + 15m entry timing signals alongside the primary 1h read
4. **Calls Claude (or a local LLM)** with the full market state, account snapshot, open positions, recent decision outcomes, and accumulated lessons — receiving a structured `{action, symbol, confidence, edge_estimate, rationale}` response
5. **Enforces hard risk caps** — per-trade risk ≤ 2%, position concentration ≤ 10%, max 5 open positions, and a −5% daily-loss circuit breaker — all in code, non-negotiable
6. **Sizes the position** using Half-Kelly capped at the hard risk limit
7. **Submits a paper order** to Alpaca (or skips entirely in `DRY_RUN=True` mode)

Every night at 23:00 UTC:

8. **Attributes outcomes** — compares entry price to current price for all decisions ≥ 24h old
9. **Reflects with Claude** — receives a summary of wins/losses broken down by signal contribution, returns 3–5 lessons and updated signal weights
10. **Applies weight changes** to the live `SignalEngine` for the next trading day

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                  DASHBOARD (FastAPI + HTMX)                   │
│  Overview | Positions | Signals | Decisions | Trades | Lessons │
└──────────────────────────┬────────────────────────────────────┘
                           │ polls every 60s (HTMX partials)
┌──────────────────────────▼────────────────────────────────────┐
│                     SQLite (WAL mode)                         │
│  decisions · trades · lessons · signal_weights · performance  │
└───────┬──────────┬──────────────────────┬─────────────────────┘
        │          │                      │
┌───────▼──┐  ┌────▼──────────────┐  ┌───▼──────────────┐
│ TRADING  │  │ NIGHTLY           │  │ ORDER            │
│ LOOP     │  │ REFLECTION        │  │ RECONCILIATION   │
│ (5 min)  │  │ (23:00 UTC)       │  │ (5 min)          │
│          │  │                   │  │                  │
│ 1. Fetch │  │ 1. Attribution    │  │ Stale order      │
│ 2. Signal│  │ 2. LLM reflection │  │ cleanup          │
│ 3. MTF   │  │ 3. Lessons +      │  └──────────────────┘
│ 4. LLM   │  │    weight updates │
│ 5. Risk  │  └───────────────────┘
│ 6. Size  │
│ 7. Order │
└───────┬──┘
        │
┌───────▼──────────────────────────────────────────────────────┐
│                     PROVIDER LAYER                           │
│  Alpaca (bars + orders) │ CoinGecko (universe) │ Ollama / Claude │
└──────────────────────────────────────────────────────────────┘
```

---

## Features

- **Regime-conditional RSI** — mean-reversion in ranging markets (ADX < 25), momentum in trending (ADX ≥ 25), resolving the classic EMA vs. RSI conflict
- **Volume multiplier** — dampens low-volume noise, amplifies high-volume confirmed moves [0.4×–1.0×]
- **ROC percentile rank** — 10-bar ROC ranked against a 500-bar window to capture crypto cycle positioning
- **Multi-timeframe context** — 4h (macro trend) and 15m (entry timing) signals fed into the LLM prompt alongside the primary 1h read
- **Prompt caching on Claude** — 2,000-token system prompt cached ephemeral, ~90% cost reduction on repeated 5-min calls
- **Dual LLM path** — Claude for decisions (configurable model), local Ollama/LM Studio as a zero-cost alternative
- **Nightly self-improvement** — LLM reflection updates per-signal weights from real outcome data; weights persist across sessions
- **Lesson injection** — recent lessons included in every future decision prompt; deduplication prevents redundancy (Jaccard threshold 0.55)
- **Two-layer risk defense** — `RiskChecker` enforces all hard caps before order submission; broker adapter enforces a second time
- **Live-trading guard** — requires a code change + env var change + secret; a single flag cannot enable live trading
- **Full audit trail** — every decision, trade, lesson, and weight change written to SQLite with timestamps
- **HTMX dashboard** — no JavaScript framework; FastAPI + Jinja2 templates with partial refresh every 60s
- **Dynamic universe** — optional CoinGecko screening refreshes the trading universe daily by market cap and volume

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Alpaca paper trading account](https://alpaca.markets) (free)
- [Anthropic API key](https://console.anthropic.com) — or a local LLM via [Ollama](https://ollama.ai)

### 1. Clone and install

```bash
git clone https://github.com/filt3rr/crypto-trading-agent.git
cd crypto-trading-agent

python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — minimum required:

```env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ANTHROPIC_API_KEY=sk-ant-...   # or set LLM_PROVIDER=local for Ollama
```

### 3. Verify connectivity

```bash
python scripts/connectivity_test.py
# [CONNECTED]  Balance: $100,000.00 | BTC/USD last: $95,234.56
# [LLM]        Provider=claude | Model=claude-sonnet-4-6
# [OK]         All connectivity checks passed.
```

### 4. Run tests

```bash
pytest
# 180+ tests, 94% coverage
```

### 5. Start

```bash
# Trading agent (5-min loop + nightly reflection)
python -m agent.main

# Dashboard (separate terminal)
python -m agent.dashboard
# → http://localhost:8000
```

---

## Dashboard

Six live views, updated every 60 seconds via HTMX:

| View | Contents |
|------|---------|
| **Overview** | Portfolio KPIs, 30-day return chart vs. BTC buy-and-hold, recent decisions |
| **Positions** | Open positions — side, qty, avg price, notional, concentration % |
| **Signals** | Current signal weights, per-symbol latest composite signal, weight history chart |
| **Decisions** | Paginated decision log, filterable by symbol / action / outcome |
| **Trades** | Order history with fill prices and status |
| **Lessons** | Numbered lesson log + weight adjustment history |

---

## Signal Rules

Three signals combined into a weighted composite, scaled by a volume multiplier:

| Signal | Indicator | Logic |
|--------|-----------|-------|
| **EMA Crossover** | 21/55 EMA (1h) | `tanh((fast - slow) / slow × 0.3)` → −1 to +1 |
| **RSI** | 14-period (1h) | Ranging: mean-reversion `−(RSI − 50) / 50`; Trending: momentum `(RSI − 50) / 50` |
| **ROC Momentum** | 10-bar ROC (1h) | Percentile rank over 500 bars, remapped to −1 to +1 |
| **Volume multiplier** | 20-bar rolling avg | Ratio clamped to [0.4, 1.0], multiplies composite |

Regime classification (ADX-based): `trending_up` · `trending_down` · `ranging`

Signal weights start at equal thirds and are updated nightly by the reflection engine based on per-signal win rates.

---

## Risk Controls

All caps are enforced in `agent/risk/checks.py` — not configuration, not easily bypassed:

| Control | Limit |
|---------|-------|
| Per-trade risk | ≤ 2% of portfolio value |
| Position concentration | ≤ 10% of portfolio value |
| Max open positions | 5 |
| Daily loss circuit breaker | −5% → blocks new BUY entries; SELL always allowed |

Position sizing uses Half-Kelly: `f = (confidence × edge_estimate) / odds_ratio`, capped at the per-trade risk limit.

---

## LLM Backends

| Backend | Config | Use case |
|---------|--------|----------|
| **Anthropic Claude** | `LLM_PROVIDER=claude` | Default. Highest reasoning quality. Prompt caching reduces cost ~90%. |
| **Ollama / LM Studio** | `LLM_PROVIDER=local` | Zero marginal cost. Set `LLM_LOCAL_URL` and `LLM_LOCAL_MODEL`. |

The reflection engine (nightly) defaults to Claude for complex multi-step reasoning. Decisions can use either backend independently.

---

## Backtesting

```bash
python scripts/run_backtest.py
```

Runs a 2-year vectorized backtest on daily bars for each symbol in your universe:

```
Symbol    Signal  Return   B&H    Sharpe  MaxDD  WinRate  Trades
BTC/USD   ROC    +30.8%  +18.0%  0.407   25.4%   34.4%      32
ETH/USD   EMA    +22.1%  +14.6%  0.381   31.2%   38.1%      27
```

---

## Configuration

All settings via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | — | Alpaca paper account key |
| `ALPACA_SECRET_KEY` | — | Alpaca paper account secret |
| `ALPACA_BASE_URL` | paper endpoint | Never change to live without reading `broker/base.py` |
| `ANTHROPIC_API_KEY` | — | Claude API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Anthropic model ID |
| `LLM_PROVIDER` | `claude` | `claude` or `local` |
| `LLM_LOCAL_URL` | `http://localhost:11434` | Ollama/LM Studio base URL |
| `LLM_LOCAL_MODEL` | `qwen2.5:7b` | Local model name |
| `TRADING_UNIVERSE` | BTC/USD,ETH/USD,... | Comma-separated crypto pairs |
| `DRY_RUN` | `True` | Log decisions only — no orders submitted |
| `MAX_RISK_PER_TRADE_PCT` | `0.02` | Hard cap: 2% per trade |
| `MAX_POSITION_PCT` | `0.10` | Hard cap: 10% concentration |
| `MAX_OPEN_POSITIONS` | `5` | Hard cap: 5 concurrent positions |
| `DAILY_LOSS_LIMIT_PCT` | `0.05` | Circuit breaker threshold |
| `TRADING_LOOP_SECONDS` | `300` | How often the trading loop fires |
| `REFLECTION_HOUR` | `23` | UTC hour for nightly reflection |
| `DB_PATH` | `data/trader.db` | SQLite database path |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11, asyncio |
| Broker | Alpaca (paper endpoint) via alpaca-py |
| LLM (cloud) | Anthropic Claude — claude-sonnet-4-6 |
| LLM (local) | Ollama / LM Studio (OpenAI-compatible) |
| Data | Alpaca OHLCV bars via alpaca-py; Parquet cache via PyArrow |
| Indicators | ta library (EMA, RSI, ADX, Bollinger) + NumPy |
| Backtesting | backtesting.py (vectorized, 2-year daily) |
| Universe screening | CoinGecko API (optional) |
| Database | SQLite (WAL mode) + SQLAlchemy 2.0 async |
| Scheduling | APScheduler (5-min trading loop, nightly reflection) |
| Config | pydantic-settings + python-dotenv |
| Dashboard | FastAPI + Jinja2 + HTMX + Chart.js |
| Logging | structlog (JSON structured logs) |
| Retries | tenacity (exponential backoff on all API calls) |
| Testing | pytest + pytest-asyncio, 94% coverage |
| Linting | ruff + black + pre-commit |

---

## Project Structure

```
crypto-trading-agent/
├── agent/
│   ├── main.py                   # Orchestrator — APScheduler, all async jobs
│   ├── config.py                 # Pydantic-settings singleton, env validation
│   ├── broker/
│   │   ├── base.py               # Abstract BrokerAdapter + live-trading guard
│   │   └── alpaca.py             # Alpaca paper-trading implementation
│   ├── data/
│   │   ├── ingestion.py          # OHLCV fetcher + DataFrame assembly
│   │   ├── cache.py              # Parquet read/write with staleness detection
│   │   ├── features.py           # Feature engineering pipeline
│   │   └── universe.py           # Dynamic universe screening (CoinGecko)
│   ├── strategy/
│   │   ├── signals.py            # EMA, regime-conditional RSI, ROC, volume multiplier
│   │   └── backtest.py           # Vectorized 2-year backtesting framework
│   ├── reasoning/
│   │   ├── agent.py              # ReasoningAgent: prompt → LLM → JSON → decision
│   │   ├── schemas.py            # Pydantic models for all LLM I/O
│   │   ├── llm_base.py           # LLMAdapter abstract interface
│   │   ├── claude_adapter.py     # Claude API with prompt caching
│   │   ├── local_adapter.py      # Ollama/LM Studio implementation
│   │   └── prompts/
│   │       ├── decision.md       # Jinja2 decision prompt template
│   │       └── reflection.md     # Jinja2 reflection prompt template
│   ├── risk/
│   │   ├── checks.py             # Hard caps — circuit breaker, concentration, count
│   │   └── sizing.py             # Half-Kelly position sizing with clamps
│   ├── memory/
│   │   ├── db.py                 # SQLAlchemy async models (6 tables)
│   │   └── queries.py            # Named CRUD queries
│   ├── learning/
│   │   ├── reflection.py         # Nightly reflection engine
│   │   ├── attribution.py        # Price-based outcome attribution
│   │   └── metrics.py            # Rolling KPI tracker
│   └── dashboard/
│       ├── app.py                # FastAPI factory
│       ├── routes.py             # All route handlers
│       ├── templates/            # Jinja2 HTML templates
│       └── static/               # CSS design system
├── scripts/
│   ├── connectivity_test.py      # End-to-end sanity check
│   └── run_backtest.py           # 2-year backtest CLI
├── tests/                        # pytest suite — 180+ tests, 94% coverage
├── data/                         # SQLite DB + Parquet cache (gitignored)
├── logs/                         # Structured JSON logs (gitignored)
├── DESIGN.md                     # Architecture decisions and tradeoffs
├── pyproject.toml
└── .env.example
```

---

## Development

```bash
# Lint + format
ruff check . && black --check .
ruff check . --fix && black .

# Tests with coverage
pytest

# Single module
pytest tests/test_risk.py -v

# Install pre-commit hooks (runs black + ruff on every commit)
pre-commit install
```

---

## Safety

- `DRY_RUN=True` is the default — no orders are submitted without explicitly disabling it
- `LIVE_TRADING_ENABLED=False` is enforced in `.env` **and** in `agent/broker/base.py` — enabling live trading requires editing two separate files plus setting a secret environment variable; this friction is intentional
- All API keys are read from environment variables and never logged
- Every decision, its reasoning, and its outcome are written to SQLite before any order is placed

---

## License

MIT — see [LICENSE](LICENSE). Use at your own risk. Not financial advice.
