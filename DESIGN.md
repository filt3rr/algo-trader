# Design Document — Crypto Paper-Trading Agent

**Date:** 2026-05-15
**Author:** Tyler (assisted by Claude)
**Status:** Phase 0 — Scaffolding

---

## 1. Broker / Market Data: Why Alpaca?

**Chosen:** Alpaca Crypto API (paper endpoint)
**Alternative considered:** CCXT against Binance Testnet

### Decision rationale

| Criterion | Alpaca Paper | CCXT / Binance Testnet |
|-----------|-------------|----------------------|
| Paper trading support | First-class, dedicated endpoint | Testnet is unofficial, often stale |
| Python SDK quality | `alpaca-py` — typed, maintained, async | CCXT is unified but lowest-common-denominator |
| Market data included | Yes, same API key | Separate stream, more config |
| WebSocket support | Yes, for real-time quotes | Yes via CCXT pro (paid) |
| KYC friction | None for paper | Binance account + VPN risk |
| Reliability | High (US company, regulated) | Testnet uptime is unreliable |
| Crypto universe | 50+ pairs | Larger but unnecessary for our scope |

Alpaca also provides a unified interface for paper order management + market data,
which simplifies the architecture (fewer credentials, fewer failure points).

**Concrete Alpaca paper endpoints used:**
- REST: `https://paper-api.alpaca.markets`
- Market data: `https://data.alpaca.markets` (same key)
- WebSocket: `wss://stream.data.alpaca.markets/v1beta3/crypto/us`

---

## 2. LLM Reasoning Layer: Claude

**Model:** `claude-sonnet-4-5` (configurable via `CLAUDE_MODEL` env var)

We use Claude as the **reasoning kernel**, not for signal generation. The agent:

1. Computes signals deterministically (technical indicators, statistics).
2. Packages signals + portfolio state + recent lessons into a structured prompt.
3. Calls Claude and expects a strict JSON response (validated by Pydantic).
4. Retries up to 3× with exponential backoff on malformed output.

Claude is **not** asked to predict prices. It is asked to weigh evidence,
apply lessons, and justify a position-sizing decision — the same role a
discretionary risk manager would play on top of a quant signal.

---

## 3. Data Store

| Store | Purpose | Format |
|-------|---------|--------|
| SQLite (`data/trader.db`) | Trades, decisions, reflections, lessons | SQLAlchemy ORM |
| Parquet (`data/parquet/`) | OHLCV history per symbol | `{symbol}_{timeframe}.parquet` |

SQLite is sufficient for a single-agent system and keeps the stack simple.
Parquet is columnar, compresses well, and integrates naturally with pandas.
No external database is needed — the entire state fits on one machine.

---

## 4. Risk Architecture

Risk rules are enforced in **two places** intentionally:

1. **`agent/risk/checks.py`** — pre-trade gate that every order must pass.
   Returns a typed `RiskCheckResult` with pass/fail reason.
2. **`agent/broker/base.py`** — broker adapter refuses any order that was
   not pre-approved by the risk module (defense in depth).

Hard caps (non-overridable from config):

| Rule | Limit | Enforcement |
|------|-------|-------------|
| Risk per trade | 2% of portfolio | `checks.py` |
| Max position size | 10% of portfolio | `checks.py` |
| Max open positions | 5 | `checks.py` |
| Daily loss circuit breaker | −5% of portfolio | `checks.py` + scheduler |

Position sizing uses a Kelly-capped formula:
```
f = (confidence * edge_estimate) / odds_ratio
size = min(f, MAX_RISK_PER_TRADE_PCT) * portfolio_value / entry_price
```
Where `confidence` comes from Claude's output (0–1) and `edge_estimate` is
the signal strength normalized from the technical layer.

---

## 5. The Reflective Reasoning Loop

What "self-improving" means concretely:

```
Every N minutes (default: 5):
  1. Fetch market state (OHLCV, order book, portfolio)
  2. Compute signals (EMA, RSI, ROC, volatility)
  3. Query recent lessons from memory (last 20, regime-filtered)
  4. Build reasoning prompt
  5. Call Claude → parse JSON decision
  6. Run risk checks
  7. If approved: submit paper order, log to SQLite
  8. If rejected: log rejection reason

Every night at 23:00 UTC:
  9. Review all trades from the past 24h
  10. Attribute wins/losses to specific signals or reasoning patterns
  11. Write 3–5 structured lessons to the lessons table
  12. Lessons are retrieved in step 3 of the next day's loop
```

**Important honesty note:** The agent does not learn in the ML sense between
nightly runs. What it does is:
- Maintain a growing text corpus of specific, attributed lessons.
- Inject the most relevant lessons (by regime and recency) into the context window.
- Let Claude reason differently given this additional context.

This is **prompt-based memory**, not weight update. We document this clearly
in both code comments and the final report.

---

## 6. Signals (Phase 2 preview)

Three baseline signals selected for orthogonality:

| Signal | Type | Indicator | Rationale |
|--------|------|-----------|-----------|
| EMA Crossover | Trend-following | 12/26 EMA cross | Captures sustained directional moves |
| RSI Reversion | Mean-reversion | RSI(14) < 30 or > 70 | Captures overextended moves |
| ROC Momentum | Momentum | ROC(10) percentile rank | Captures acceleration without lag |

These will be vector-backtested on 2 years of Alpaca historical data in Phase 2,
with Sharpe, max drawdown, and win rate reported.

---

## 7. Dashboard

FastAPI + Jinja2 templates with HTMX for partial page updates. No JS framework.
HTMX polls `/api/state` every 5 seconds and updates:
- Open positions table
- P&L chart (server-rendered SVG or plain numbers)
- Recent decisions log with reasoning snippets
- Circuit breaker status

The dashboard is **read-only** — it cannot submit orders or change settings.

---

## 8. Dependency Choices

- **`alpaca-py`** over `alpaca-trade-api`: the newer, typed SDK (v0.38+)
- **`ta`** over `pandas-ta`: simpler, fewer dependencies, actively maintained
- **`backtesting.py`** for Phase 2 vectorized backtests: lightweight, no Zipline overhead
- **`APScheduler 3.x`** (not 4.x): 4.x is still in pre-release, 3.x is battle-tested
- **`structlog`**: JSON-structured logs that are queryable and parseable by the reflection job
- **`pydantic v2`**: required for FastAPI 0.100+ and used for all data validation

---

## 9. What This Does NOT Prove

(Required honest section, repeated verbatim in the final report)

1. **Survivorship bias:** Backtested signals were selected in hindsight.
2. **Paper vs. live slippage:** Paper fills assume best-bid/ask with no market impact.
   Crypto is volatile; large orders move the market.
3. **Overfitting to the paper period:** 30 days is statistically insufficient to
   distinguish skill from luck in any market.
4. **LLM reliability:** Claude's reasoning may be inconsistent across identical
   inputs (temperature > 0). The agent is not deterministic.
5. **Regime change:** Any strategy trained/tuned in one market regime (bull, bear,
   sideways, high-volatility) may fail immediately in another.
6. **Exchange risk:** Alpaca is a broker, not an exchange. In live mode, you'd
   face custody risk, withdrawal limits, and platform outages.

---

## 10. File-by-File Responsibility Summary

```
agent/
├── data/
│   ├── ingestion.py     — Alpaca REST + WebSocket → pandas DataFrame
│   ├── cache.py         — Parquet read/write, staleness checks
│   └── features.py      — Feature engineering on top of OHLCV
├── broker/
│   ├── base.py          — Abstract BrokerAdapter + live-trading guard
│   └── alpaca.py        — AlpacaBroker implementation
├── strategy/
│   ├── signals.py       — EMA, RSI, ROC signal computation
│   └── backtest.py      — Vector backtest runner, metric reporting
├── reasoning/
│   ├── agent.py         — ReasoningAgent: prompt build → Claude call → parse
│   ├── schemas.py       — Pydantic models for input/output
│   └── prompts/
│       └── decision.md  — Prompt template (Jinja2)
├── risk/
│   ├── checks.py        — RiskChecker: all hard caps, circuit breaker
│   └── sizing.py        — Kelly-capped position sizing
├── memory/
│   ├── db.py            — SQLAlchemy models + session factory
│   └── queries.py       — Named queries (recent lessons, trade history)
├── learning/
│   └── reflection.py    — Nightly job: review trades → write lessons
├── dashboard/
│   ├── app.py           — FastAPI application
│   └── templates/
│       └── index.html   — HTMX dashboard
└── main.py              — APScheduler setup, graceful shutdown, entrypoint
```
