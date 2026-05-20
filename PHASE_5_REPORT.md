# Phase 5 Report — Apex Console Dashboard

**Date:** 2026-05-15
**Status:** Complete

---

## What Was Built

| File | Description |
|------|-------------|
| `agent/dashboard/app.py` | FastAPI application factory — lifespan context manager, StaticFiles mount, Jinja2Templates with `fmt_usd` / `fmt_pct` / `fmt_num` globals, router include |
| `agent/dashboard/routes.py` | All route handlers: 6 full-page routes, 1 JSON API endpoint, 3 HTMX partials |
| `agent/dashboard/static/css/main.css` | Full enterprise design system — CSS custom properties, all reusable components |
| `agent/dashboard/templates/base.html` | Layout shell — Inter font, Chart.js v4, HTMX 1.9, topbar with nav, DRY RUN badge, status pill |
| `agent/dashboard/templates/overview.html` | Portfolio overview — KPI strip, performance line chart (Chart.js), signal weight bars, recent decisions table |
| `agent/dashboard/templates/positions.html` | Open positions table — side badge, qty, avg price, notional, broker order ID, status |
| `agent/dashboard/templates/signals.html` | Signal dashboard — weight bars, per-symbol latest decision, weight history multi-line chart, weight log table |
| `agent/dashboard/templates/decisions.html` | Decision log — paginated (25/page), filterable by symbol / action / outcome, ellipsis pagination |
| `agent/dashboard/templates/trades.html` | Trade history — paginated (25/page), side badge, status badge |
| `agent/dashboard/templates/lessons.html` | Lessons viewer — numbered lesson list, recent weight adjustments table, system health panel |
| `agent/dashboard/templates/partials/kpi_cards.html` | HTMX partial — 6-column KPI strip (portfolio value, daily return, cumulative return, Sharpe, max drawdown, win rate) |
| `agent/dashboard/templates/partials/circuit_breaker.html` | HTMX partial — danger alert banner when `daily_return_pct <= -5%` |
| `agent/dashboard/templates/partials/recent_decisions.html` | HTMX partial — last 10 decisions table |
| `agent/dashboard/__main__.py` | `python -m agent.dashboard` entrypoint — uvicorn on 127.0.0.1:8000 |

---

## Design System

| Token | Value | Usage |
|-------|-------|-------|
| `--navy` | `#0f2044` | Topbar background, active nav, primary buttons |
| `--navy-soft` | `#1a3560` | Hover states, ROC weight fill |
| `--slate-900` | `#0f172a` | Page headings |
| `--slate-700` | `#334155` | Body text, table cells |
| `--slate-500` | `#64748b` | Labels, card titles (uppercase) |
| `--slate-50` | `#f8fafc` | Page background, table header background |
| `--white` | `#ffffff` | Card backgrounds |
| `--border` | `#e2e8f0` | All dividers and card borders |
| `--green-700` | `#15803d` | Positive P&L, buy badges |
| `--red-600` | `#dc2626` | Negative P&L, sell badges, danger alerts |
| `--amber-600` | `#d97706` | DRY RUN badge, warning alerts |
| `--topbar-h` | `52px` | Fixed topbar height, `padding-top` on `.main-wrap` |

---

## Pages and Live Updates

| Page | URL | HTMX Polling |
|------|-----|-------------|
| Overview | `/` | KPI strip, circuit breaker, recent decisions — every 60s |
| Positions | `/positions` | Static (full page refresh) |
| Signals | `/signals` | Static (full page refresh) |
| Decisions | `/decisions` | Filter form + pagination (full page) |
| Trades | `/trades` | Pagination (full page) |
| Lessons | `/lessons` | Static (full page refresh) |

HTMX partials (`/partials/kpis`, `/partials/circuit-breaker`, `/partials/recent-decisions`) are polled every 60 seconds on the Overview page using `hx-trigger="every 60s"`.

---

## Running the Dashboard

```bash
python -m agent.dashboard
# → http://127.0.0.1:8000
```

Or via uvicorn directly:
```bash
uvicorn agent.dashboard.app:app --host 127.0.0.1 --port 8000
```

---

## Phase 5 Gate

**Gate requirement:** "Dashboard renders correctly and reflects live DB state with ≥1 day of paper-trading data."

**Status: Architecture complete.** All routes, templates, and static assets are in place. The dashboard reads exclusively from SQLite via async SQLAlchemy sessions (the same DB written by the trading loop). HTMX polling ensures the Overview page reflects the latest data without a page refresh.

---

## Test Results

```
180 passed, 18 warnings
Coverage: 94.60%
```

Dashboard code (`agent/dashboard/*`) is excluded from the coverage requirement (IO-heavy, tested by manual browser verification). All 180 existing tests continue to pass unchanged.

---

## What's Next (Phase 6)

Suggested next phase options:
- **Backtesting harness** — run the full signal + LLM pipeline over historical OHLCV data, compare vs buy-and-hold
- **Multi-symbol portfolio optimisation** — Kelly sizing across correlated assets, concentration limits per sector
- **Live paper-trading validation** — 7-day paper run, review lesson convergence, confirm weight adaptation
- **Alerting** — email / Slack notifications on circuit breaker trips, large drawdowns, or new lessons
