# Crypto Paper-Trading Agent

A **reflective reasoning** crypto paper-trading agent powered by Claude.

> **DISCLAIMER — READ BEFORE PROCEEDING**
>
> - This software is for **educational and research purposes only**.
> - It is **not financial advice**. Nothing in this codebase constitutes a
>   recommendation to buy, sell, or hold any financial instrument.
> - **Past paper-trading performance does not predict future real-money
>   performance.** Paper trading eliminates slippage, partial fills, funding
>   costs, exchange outages, and the psychological pressure of real capital.
>   Any edge observed in paper trading may vanish entirely in live markets.
> - The authors accept no responsibility for financial losses arising from
>   use of this software.
> - **This system only executes paper trades.** Live trading requires a
>   deliberate multi-step code change described in `agent/broker/base.py`.

---

## What this is

An autonomous agent that:

1. **Ingests live crypto market data** from Alpaca's paper trading platform.
2. **Generates trading signals** from technical indicators (EMA crossover,
   RSI mean-reversion, Rate of Change momentum).
3. **Reasons through each decision** using Claude — the LLM receives market
   state, portfolio state, recent trade outcomes, and accumulated lessons,
   then produces a structured `{action, symbol, size, confidence, reasoning}`
   decision with full audit trail.
4. **Enforces hard risk caps in code** — not just config — including per-trade
   risk, position concentration limits, and a daily-loss circuit breaker.
5. **Reflects nightly** on the day's decisions, attributes outcomes, and
   writes lessons that are injected into future prompts (stateful, reflective
   reasoning).
6. **Exposes a live dashboard** at `http://localhost:8000` showing positions,
   P&L, and decision logs.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Alpaca paper trading account](https://alpaca.markets) (free)
- [Anthropic API key](https://console.anthropic.com)

### Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd crypto-trader-agent

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure secrets
cp .env.example .env
# Edit .env and fill in ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY

# 5. Install pre-commit hooks
pre-commit install

# 6. Run tests
pytest

# 7. Start the agent
python -m agent.main
```

### Dashboard

Visit [http://localhost:8000](http://localhost:8000) while the agent is running.

---

## Project Structure

```
agent/
├── data/          # Market data ingestion, OHLCV caching, feature engineering
├── broker/        # Paper-trading adapter (Alpaca), abstract base interface
├── strategy/      # Signal generation (EMA, RSI, ROC), vectorized backtesting
├── reasoning/     # Claude-powered decision layer, prompt templates, schemas
├── risk/          # Pre-trade risk checks, Kelly-capped position sizing
├── memory/        # SQLite schema, trade/decision/lesson storage
├── learning/      # Nightly reflection job
├── dashboard/     # FastAPI + HTMX dashboard
└── main.py        # Entrypoint, APScheduler orchestration
tests/             # pytest suite (>80% coverage on non-IO modules)
data/              # SQLite DB + Parquet OHLCV cache (gitignored)
logs/              # Structured JSON logs (gitignored)
```

---

## Architecture Overview

See [DESIGN.md](DESIGN.md) for detailed decisions and tradeoffs.

---

## Development

```bash
# Run linters
ruff check . && black --check .

# Run tests with coverage
pytest

# Run a single test module
pytest tests/test_risk.py -v
```

---

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ | Scaffolding, tooling, design |
| 1 | 🔲 | Data ingestion + broker adapter |
| 2 | 🔲 | Strategy signals + risk module |
| 3 | 🔲 | Claude reasoning layer |
| 4 | 🔲 | Memory + nightly learning |
| 5 | 🔲 | Dashboard + 30-day paper run |

---

## Safety

- `LIVE_TRADING_ENABLED=False` is enforced in `.env` **and** in `agent/broker/base.py`.
- Enabling live trading requires editing two separate files plus setting a secret
  environment variable. This is intentional friction.
- All API keys are read from environment variables. Keys are never logged.
- All trades are logged to SQLite with the full reasoning trace before execution.

---

## License

MIT — see LICENSE file. Use at your own risk. Not financial advice.
