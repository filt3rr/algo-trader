# Phase 3 Report — Claude Reasoning Layer

**Date:** 2026-05-15
**Status:** Complete — awaiting review

---

## What Was Built

| Module | Description |
|--------|-------------|
| `agent/reasoning/schemas.py` | `DecisionInput` (full context fed to LLM) + `TradingDecision` (validated JSON response) + `DecisionRecord` (DB row) — all Pydantic v2 with strict validation |
| `agent/reasoning/prompts/decision.md` | Jinja2 prompt template: signal state table, account snapshot, open positions, recent lessons, recent decisions, hard constraint summary, JSON schema |
| `agent/reasoning/agent.py` | `ReasoningAgent`: renders prompt, calls `LLMAdapter.complete()`, extracts JSON (strips markdown fences), validates against `TradingDecision`, retries up to 3× on parse failure, falls back to hold |
| `agent/memory/db.py` | SQLAlchemy async models: `DecisionRow`, `TradeRow`, `LessonRow`; WAL journal mode; `init_db()` returns `(engine, session_factory)` |
| `agent/memory/queries.py` | `save_decision`, `get_recent_decisions`, `update_decision_outcome`, `link_order_to_decision`, `save_trade`, `save_lesson`, `get_recent_lessons`, `deactivate_lesson` |
| `agent/main.py` | APScheduler async trading loop (5-min), nightly reflection placeholder (Phase 4), graceful SIGINT/SIGTERM shutdown, DRY_RUN guard |
| `tests/test_reasoning.py` | 28 new tests covering all schema paths, retry logic, fallback hold, DB round-trips, prompt rendering |

---

## Pipeline Architecture

```
MarketDataClient (Alpaca 1h bars)
        |
    SignalEngine
        |
    DecisionInput assembly
    (signal + account + positions + lessons + recent decisions)
        |
    ReasoningAgent.decide()
        |--- renders Jinja2 prompt
        |--- calls LLMAdapter.complete() (Claude or local)
        |--- extracts + validates TradingDecision JSON
        |--- retries up to 3x on malformed output
        |--- fallback: hold with rationale
        |
    DRY_RUN check
        |
    [DRY_RUN=True]          [DRY_RUN=False]
    Log decision only        RiskChecker.check()
                                 |
                             kelly_size() -> OrderRequest
                                 |
                             AlpacaBroker.submit_order()
        |
    save_decision() -> SQLite decisions table
```

---

## Key Design Decisions

### JSON extraction with fence stripping
The model is instructed to return bare JSON. In practice, models occasionally
wrap output in markdown code fences. `_extract_json()` in `agent.py` tries the
fence pattern first (```` ```json ... ``` ````), then falls back to the first
`{...}` block found via regex.

### Retry with conversation continuation
On a parse failure, the agent appends the bad response and a correction
request to the message list and retries. This preserves context so the model
can self-correct rather than starting cold each time.

### Dry-run override
If the LLM echoes `dry_run=False` when the context says `True`, the agent
silently corrects it. The context (not the model's output) is authoritative.

### Hold as safe default
After 3 failed parse attempts, the agent returns `action="hold"` with
`confidence=0.0`. This means a malformed response never triggers an order.

### DRY_RUN=True in .env
Added `DRY_RUN=True` to Settings (pydantic-settings field) and to both
`.env` and `.env.example`. The trading loop reads `cfg.dry_run` and skips
order submission when True.

---

## Test Results

```
153 passed, 18 warnings
Coverage: 94.57% (above 80% threshold)
```

New tests (28 added to `tests/test_reasoning.py`):
- Schema validation: all action types, clamp logic, mismatch guards
- DecisionInput: symbol mismatch, positions, lessons
- ReasoningAgent: happy path, sell, hold, fence stripping, retry on bad JSON,
  exhausted retries → fallback hold, symbol mismatch retry, dry_run override
- DecisionRecord assembly
- DB round-trip: save/retrieve decisions, save/retrieve lessons, update outcome
- Prompt rendering: renders without error, contains signal values, handles lessons

---

## DRY_RUN Gate

Phase 3 gate requirement: *"A dry-run of the full decision pipeline that logs
a structured decision to SQLite without placing any order."*

Status: **Met.** With `DRY_RUN=True` (the default):
- Market data is fetched
- Signals are computed
- The LLM makes a decision
- The decision is logged to `data/trader.db`
- **No order is submitted**

To run:
```bash
python -m agent.main
```

---

## What's Next (Phase 4)

1. **Nightly reflection job** — LLM reviews yesterday's decisions vs. outcomes
   and extracts lessons into the `lessons` table
2. **Outcome attribution** — `update_decision_outcome()` called when trades close
3. **Signal weight updates** — reflection loop calls `SignalEngine.update_weights()`
   based on per-signal attribution analysis
4. **Rolling backtest** — after each reflection cycle, re-run backtest on the
   latest 30-day window to track strategy drift
5. **Performance metrics** — cumulative return, Sharpe, max drawdown stored for
   Phase 5 dashboard display

**Phase 4 gate:** Nightly reflection produces a measurable change in signal
weights after at least one cycle of paper-trading decisions with outcomes.
