# Phase 0 Report — Scaffolding

**Date:** 2026-05-15
**Status:** Complete — awaiting review

---

## What Was Built

### Repository Structure

```
Trader/
├── agent/
│   ├── __init__.py
│   ├── data/__init__.py
│   ├── broker/__init__.py
│   ├── strategy/__init__.py
│   ├── reasoning/__init__.py
│   │   └── prompts/           (prompt templates go here in Phase 3)
│   ├── risk/__init__.py
│   ├── memory/__init__.py
│   ├── learning/__init__.py
│   └── dashboard/__init__.py
│       └── templates/
├── tests/
│   ├── __init__.py
│   ├── conftest.py            (shared fixtures, env isolation)
│   └── test_scaffolding.py   (structure smoke tests)
├── data/
│   └── parquet/
├── logs/
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── DESIGN.md
└── PHASE_0_REPORT.md
```

### Key Decisions Made

1. **Broker: Alpaca** — See DESIGN.md §1 for full comparison vs CCXT/Binance Testnet.
   Summary: Alpaca is purpose-built for paper trading, has a first-class typed Python SDK,
   and bundles market data with the same API key.

2. **Dependency pinning** — All versions pinned in `pyproject.toml` for reproducibility.
   Notable: `APScheduler==3.10.4` (not 4.x, which is pre-release), `pydantic==2.10.4`
   (v2 for performance and FastAPI compatibility).

3. **Pre-commit hooks** — `black` + `ruff` (with auto-fix) enforce style on every commit.
   `detect-private-key` hook prevents accidental secret commits.

4. **Test isolation** — `conftest.py` uses `monkeypatch` to inject fake env vars for all
   tests. No real API calls in any test (verified by fixture `autouse=True`).

5. **Coverage threshold** — Set to 80% in `pyproject.toml`, excluding IO-heavy modules
   (broker adapter, dashboard, main entrypoint) per the project spec.

6. **Live trading guard** — Two-layer design documented in DESIGN.md §4 and `.env.example`.
   The actual code-level guard will be implemented in Phase 1 (`agent/broker/base.py`).

---

## What Was Tested

- `test_module_importable` — all 9 agent sub-packages import without error
- `test_env_example_exists` — `.env.example` present
- `test_design_md_exists` — `DESIGN.md` present
- `test_required_directories_exist` — all 11 required directories present
- `test_live_trading_disabled_by_default` — live mode defaults to False

All 5 tests pass. Coverage for Phase 0 is structural only; substantive coverage
begins in Phase 1.

---

## Assumptions Made

1. **Python 3.11 available on the target machine.** The project uses `match` statements
   and `tomllib` from stdlib, both requiring 3.11+.

2. **Alpaca paper account will be created by the user.** We cannot create one
   programmatically. Instructions are in README.md.

3. **`claude-sonnet-4-5` is the target model.** If this model is retired before
   Phase 3, we switch via `CLAUDE_MODEL` env var — no code change required.

4. **Single-machine deployment.** No distributed concerns (Kafka, Redis, etc.).
   APScheduler runs in-process; SQLite has no concurrent write bottleneck at our
   5-minute loop frequency.

---

## What's Next (Phase 1)

1. Implement `agent/broker/base.py` — abstract `BrokerAdapter` with live-trading guard.
2. Implement `agent/broker/alpaca.py` — paper-endpoint Alpaca adapter.
3. Connectivity test: print account balance, place + cancel a test order.
4. Implement `agent/data/ingestion.py` — OHLCV fetcher for 8-asset default universe.
5. Implement `agent/data/cache.py` — Parquet read/write with staleness detection.
6. Implement `agent/data/features.py` — base feature engineering (returns, vol, VWAP).

**Phase 1 gate:** A script that runs end-to-end and prints
`[CONNECTED] Balance: $X | BTC/USD last: $Y` to stdout.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Alpaca API rate limits on free tier | Medium | Cache OHLCV aggressively in Parquet |
| `backtesting.py` incompatible with pandas 2.x | Low | Pinned versions; test in Phase 2 |
| Claude API latency > 5s causes missed candles | Medium | Async decision call with timeout |
| SQLite lock under concurrent writes | Low | Single writer (scheduler thread) |
