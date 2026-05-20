"""Route handlers for the Apex Console dashboard.

All routes are read-only except DELETE /lessons/{id} (soft-delete).
Data flows: Request -> SQLite (via session_factory on app.state) -> Jinja2 template.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from math import ceil

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import asc, desc, func, select

from agent.memory.db import (
    AgentStateRow,
    DecisionRow,
    LessonRow,
    PerformanceSnapshotRow,
    SignalWeightsRow,
    TradeRow,
)
from agent.memory.queries import (
    get_current_weights,
    get_latest_performance,
    get_performance_history,
    get_recent_lessons,
    get_weight_history,
)
from agent.strategy.signals import DEFAULT_WEIGHTS


def _mask_key(key: str) -> str:
    """Return a masked version of an API key for safe display."""
    if not key:
        return "— not set —"
    if len(key) <= 10:
        return "••••••••"
    return key[:6] + "••••••" + key[-4:]

router = APIRouter()
PAGE_SIZE = 25


# ── Helpers ───────────────────────────────────────────────────────────────────

def _templates(request: Request):
    return request.app.state.templates


def _factory(request: Request):
    return request.app.state.session_factory


def _cfg(request: Request):
    return request.app.state.cfg


def _base_ctx(request: Request) -> dict:
    cfg = _cfg(request)
    return {
        "request": request,
        "dry_run": cfg.dry_run,
        "now": datetime.now(tz=timezone.utc),
    }


async def _count(factory, model) -> int:
    async with factory() as session:
        result = await session.execute(select(func.count()).select_from(model))
        return result.scalar_one()


def _pct_color(value: float | None) -> str:
    if value is None:
        return "text-muted"
    return "text-green" if value >= 0 else "text-red"


def _infer_cb_active(perf) -> bool:
    if perf is None:
        return False
    return (perf.daily_return_pct or 0) <= -0.05


# ── Overview ──────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    factory = _factory(request)
    async with factory() as session:
        latest_perf = await get_latest_performance(session)
        perf_history = await get_performance_history(session, days=30)
        current_weights = await get_current_weights(session) or dict(DEFAULT_WEIGHTS)

        result = await session.execute(
            select(DecisionRow).order_by(desc(DecisionRow.decided_at)).limit(10)
        )
        recent_decisions = result.scalars().all()

    total_decisions = await _count(factory, DecisionRow)
    total_trades = await _count(factory, TradeRow)

    # Chart data
    chart_labels = [row.recorded_at[:16].replace("T", " ") for row in perf_history]
    chart_values = [round((row.cumulative_return_pct or 0) * 100, 3) for row in perf_history]

    cb_active = _infer_cb_active(latest_perf)

    ctx = {
        **_base_ctx(request),
        "page": "overview",
        "latest_perf": latest_perf,
        "current_weights": current_weights,
        "recent_decisions": recent_decisions,
        "total_decisions": total_decisions,
        "total_trades": total_trades,
        "cb_active": cb_active,
        "chart_labels": json.dumps(chart_labels),
        "chart_values": json.dumps(chart_values),
    }
    return _templates(request).TemplateResponse(request, "overview.html", ctx)


# ── Positions ─────────────────────────────────────────────────────────────────

@router.get("/positions", response_class=HTMLResponse)
async def positions(request: Request):
    factory = _factory(request)
    async with factory() as session:
        result = await session.execute(
            select(TradeRow)
            .where(TradeRow.side == "buy")
            .where(TradeRow.status == "open")
            .order_by(desc(TradeRow.submitted_at))
        )
        open_positions = result.scalars().all()

        latest_perf = await get_latest_performance(session)

    ctx = {
        **_base_ctx(request),
        "page": "positions",
        "open_positions": open_positions,
        "cb_active": _infer_cb_active(latest_perf),
    }
    return _templates(request).TemplateResponse(request, "positions.html", ctx)


# ── Signals ───────────────────────────────────────────────────────────────────

@router.get("/signals", response_class=HTMLResponse)
async def signals(request: Request):
    factory = _factory(request)
    async with factory() as session:
        current_weights = await get_current_weights(session) or dict(DEFAULT_WEIGHTS)
        weight_history = await get_weight_history(session, limit=30)

        # Latest decision per symbol — shows most recent signal composite
        symbols_result = await session.execute(
            select(DecisionRow.symbol).distinct()
        )
        symbols = [row[0] for row in symbols_result.all()]

        latest_by_symbol: dict[str, DecisionRow] = {}
        for sym in symbols:
            r = await session.execute(
                select(DecisionRow)
                .where(DecisionRow.symbol == sym)
                .order_by(desc(DecisionRow.decided_at))
                .limit(1)
            )
            row = r.scalar_one_or_none()
            if row:
                latest_by_symbol[sym] = row

    # Weight history chart data (multi-line: ema, rsi, roc over time)
    wh_labels = [row.set_at[:10] for row in weight_history][::-1]
    wh_ema = [round(row.ema * 100, 1) for row in weight_history][::-1]
    wh_rsi = [round(row.rsi * 100, 1) for row in weight_history][::-1]
    wh_roc = [round(row.roc * 100, 1) for row in weight_history][::-1]

    ctx = {
        **_base_ctx(request),
        "page": "signals",
        "current_weights": current_weights,
        "weight_history": weight_history,
        "latest_by_symbol": latest_by_symbol,
        "wh_labels": json.dumps(wh_labels),
        "wh_ema": json.dumps(wh_ema),
        "wh_rsi": json.dumps(wh_rsi),
        "wh_roc": json.dumps(wh_roc),
    }
    return _templates(request).TemplateResponse(request, "signals.html", ctx)


# ── Decisions ─────────────────────────────────────────────────────────────────

@router.get("/decisions", response_class=HTMLResponse)
async def decisions(request: Request):
    factory = _factory(request)
    page = max(1, int(request.query_params.get("page", 1)))
    filter_symbol = request.query_params.get("symbol", "").strip()
    filter_action = request.query_params.get("action", "").strip()
    filter_outcome = request.query_params.get("outcome", "").strip()

    async with factory() as session:
        stmt = select(DecisionRow).order_by(desc(DecisionRow.decided_at))
        if filter_symbol:
            stmt = stmt.where(DecisionRow.symbol == filter_symbol)
        if filter_action:
            stmt = stmt.where(DecisionRow.action == filter_action)
        if filter_outcome == "attributed":
            stmt = stmt.where(DecisionRow.outcome_pnl_pct.is_not(None))
        elif filter_outcome == "pending":
            stmt = stmt.where(DecisionRow.outcome_pnl_pct.is_(None))

        count_result = await session.execute(
            select(func.count()).select_from(stmt.subquery())
        )
        total = count_result.scalar_one()
        total_pages = max(1, ceil(total / PAGE_SIZE))

        paged = stmt.offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
        result = await session.execute(paged)
        rows = result.scalars().all()

        symbols_result = await session.execute(select(DecisionRow.symbol).distinct())
        all_symbols = sorted(row[0] for row in symbols_result.all())

    ctx = {
        **_base_ctx(request),
        "page": "decisions",
        "rows": rows,
        "total": total,
        "current_page": page,
        "total_pages": total_pages,
        "filter_symbol": filter_symbol,
        "filter_action": filter_action,
        "filter_outcome": filter_outcome,
        "all_symbols": all_symbols,
    }
    return _templates(request).TemplateResponse(request, "decisions.html", ctx)


# ── Trades ────────────────────────────────────────────────────────────────────

@router.get("/trades", response_class=HTMLResponse)
async def trades(request: Request):
    factory = _factory(request)
    page = max(1, int(request.query_params.get("page", 1)))

    async with factory() as session:
        count_result = await session.execute(
            select(func.count()).select_from(TradeRow)
        )
        total = count_result.scalar_one()
        total_pages = max(1, ceil(total / PAGE_SIZE))

        result = await session.execute(
            select(TradeRow)
            .order_by(desc(TradeRow.submitted_at))
            .offset((page - 1) * PAGE_SIZE)
            .limit(PAGE_SIZE)
        )
        rows = result.scalars().all()

    ctx = {
        **_base_ctx(request),
        "page": "trades",
        "rows": rows,
        "total": total,
        "current_page": page,
        "total_pages": total_pages,
    }
    return _templates(request).TemplateResponse(request, "trades.html", ctx)


# ── Lessons ───────────────────────────────────────────────────────────────────

@router.get("/lessons", response_class=HTMLResponse)
async def lessons(request: Request):
    factory = _factory(request)
    async with factory() as session:
        all_lessons = await get_recent_lessons(session, limit=50)
        weight_history = await get_weight_history(session, limit=20)

    ctx = {
        **_base_ctx(request),
        "page": "lessons",
        "lessons": all_lessons,
        "weight_history": weight_history,
    }
    return _templates(request).TemplateResponse(request, "lessons.html", ctx)


# ── API: performance JSON (for Chart.js) ──────────────────────────────────────

@router.get("/api/performance")
async def api_performance(request: Request):
    factory = _factory(request)
    async with factory() as session:
        history = await get_performance_history(session, days=30)
    return JSONResponse({
        "labels": [r.recorded_at[:16].replace("T", " ") for r in history],
        "values": [round((r.cumulative_return_pct or 0) * 100, 3) for r in history],
        "portfolio": [float(r.portfolio_value) for r in history],
    })


# ── HTMX partials ─────────────────────────────────────────────────────────────

@router.get("/partials/kpis", response_class=HTMLResponse)
async def partial_kpis(request: Request):
    factory = _factory(request)
    async with factory() as session:
        latest_perf = await get_latest_performance(session)
        total_decisions = await _count(factory, DecisionRow)
        total_trades = await _count(factory, TradeRow)
    ctx = {
        **_base_ctx(request),
        "latest_perf": latest_perf,
        "total_decisions": total_decisions,
        "total_trades": total_trades,
    }
    return _templates(request).TemplateResponse(request, "partials/kpi_cards.html", ctx)


@router.get("/partials/circuit-breaker", response_class=HTMLResponse)
async def partial_circuit_breaker(request: Request):
    factory = _factory(request)
    async with factory() as session:
        latest_perf = await get_latest_performance(session)
    cb_active = _infer_cb_active(latest_perf)
    ctx = {**_base_ctx(request), "cb_active": cb_active, "latest_perf": latest_perf}
    return _templates(request).TemplateResponse(request, "partials/circuit_breaker.html", ctx)


@router.get("/partials/recent-decisions", response_class=HTMLResponse)
async def partial_recent_decisions(request: Request):
    factory = _factory(request)
    async with factory() as session:
        result = await session.execute(
            select(DecisionRow).order_by(desc(DecisionRow.decided_at)).limit(10)
        )
        recent_decisions = result.scalars().all()
    ctx = {**_base_ctx(request), "recent_decisions": recent_decisions}
    return _templates(request).TemplateResponse(request, "partials/recent_decisions.html", ctx)


# ── System Logs ───────────────────────────────────────────────────────────────

@router.get("/system", response_class=HTMLResponse)
async def system_logs(request: Request):
    factory = _factory(request)
    cfg = _cfg(request)

    async with factory() as session:
        # ── Decision breakdown ────────────────────────────────────────────
        action_rows = (await session.execute(
            select(DecisionRow.action, func.count(DecisionRow.id))
            .group_by(DecisionRow.action)
        )).all()
        action_counts = {r[0]: r[1] for r in action_rows}
        total_decisions = sum(action_counts.values())
        buy_count  = action_counts.get("buy", 0)
        sell_count = action_counts.get("sell", 0)
        hold_count = action_counts.get("hold", 0)

        # ── Token usage ───────────────────────────────────────────────────
        token_row = (await session.execute(
            select(
                func.sum(DecisionRow.input_tokens),
                func.sum(DecisionRow.output_tokens),
                func.avg(DecisionRow.input_tokens),
                func.avg(DecisionRow.output_tokens),
            )
        )).one()
        total_input_tokens  = int(token_row[0] or 0)
        total_output_tokens = int(token_row[1] or 0)
        avg_input_tokens    = round(float(token_row[2] or 0), 1)
        avg_output_tokens   = round(float(token_row[3] or 0), 1)

        # ── Attribution ───────────────────────────────────────────────────
        eligible = buy_count + sell_count
        attributed_count = (await session.execute(
            select(func.count()).select_from(DecisionRow)
            .where(DecisionRow.outcome_pnl_pct.is_not(None))
            .where(DecisionRow.action != "hold")
        )).scalar_one()

        win_count = (await session.execute(
            select(func.count()).select_from(DecisionRow)
            .where(DecisionRow.outcome_pnl_pct > 0)
        )).scalar_one()

        avg_confidence = (await session.execute(
            select(func.avg(DecisionRow.confidence))
            .where(DecisionRow.action != "hold")
        )).scalar_one()

        # ── Trades ────────────────────────────────────────────────────────
        total_trades   = await _count(factory, TradeRow)
        filled_trades  = (await session.execute(
            select(func.count()).select_from(TradeRow).where(TradeRow.status == "filled")
        )).scalar_one()
        pending_trades = (await session.execute(
            select(func.count()).select_from(TradeRow)
            .where(TradeRow.status.in_(["pending", "partially_filled"]))
        )).scalar_one()

        # ── Lessons / weights ─────────────────────────────────────────────
        active_lesson_count = (await session.execute(
            select(func.count()).select_from(LessonRow).where(LessonRow.is_active == True)  # noqa: E712
        )).scalar_one()
        total_lesson_count  = (await session.execute(
            select(func.count()).select_from(LessonRow)
        )).scalar_one()
        weight_revisions = (await session.execute(
            select(func.count()).select_from(SignalWeightsRow)
        )).scalar_one()
        reflection_count = (await session.execute(
            select(func.count()).select_from(LessonRow).where(LessonRow.source == "reflection")
        )).scalar_one()

        current_weights = await get_current_weights(session) or dict(DEFAULT_WEIGHTS)
        latest_perf = await get_latest_performance(session)

        # ── Agent state ───────────────────────────────────────────────────
        cb_state_row = await session.get(AgentStateRow, "circuit_breaker")
        cb_state = cb_state_row.value if cb_state_row else None

        # ── Recent activity log (last 30 decisions) ───────────────────────
        recent_activity = (await session.execute(
            select(DecisionRow)
            .order_by(desc(DecisionRow.decided_at))
            .limit(30)
        )).scalars().all()

        # ── First decision timestamp (uptime proxy) ───────────────────────
        first_decision = (await session.execute(
            select(DecisionRow).order_by(asc(DecisionRow.decided_at)).limit(1)
        )).scalar_one_or_none()

    # ── Computed stats ────────────────────────────────────────────────────
    attribution_rate = round(attributed_count / eligible * 100, 1) if eligible > 0 else 0.0
    win_rate = round(win_count / attributed_count * 100, 1) if attributed_count > 0 else 0.0
    avg_conf_pct = round(float(avg_confidence or 0) * 100, 1)
    trade_pct = buy_count + sell_count
    hold_pct = round(hold_count / total_decisions * 100, 1) if total_decisions > 0 else 0.0
    llm_call_pct = round(trade_pct / total_decisions * 100, 1) if total_decisions > 0 else 0.0

    # ── Universe from config ──────────────────────────────────────────────
    universe = cfg.universe

    ctx = {
        **_base_ctx(request),
        "page": "system",
        # Config
        "cfg": cfg,
        "decision_model": cfg.local_llm_model if cfg.llm_provider == "local" else cfg.claude_model,
        "reflection_model": cfg.claude_model if (cfg.reflection_llm_provider or cfg.llm_provider) == "claude" else cfg.local_llm_model,
        "llm_provider": str(cfg.llm_provider),
        # Masked keys
        "alpaca_key_masked": _mask_key(cfg.alpaca_api_key),
        "anthropic_key_masked": _mask_key(cfg.anthropic_api_key),
        "coingecko_key_masked": _mask_key(cfg.coingecko_api_key),
        # Decision stats
        "total_decisions": total_decisions,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "hold_pct": hold_pct,
        "llm_call_pct": llm_call_pct,
        "avg_conf_pct": avg_conf_pct,
        # Token usage
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        # Attribution
        "attributed_count": attributed_count,
        "attribution_rate": attribution_rate,
        "win_count": win_count,
        "win_rate": win_rate,
        # Trades
        "total_trades": total_trades,
        "filled_trades": filled_trades,
        "pending_trades": pending_trades,
        # Lessons
        "active_lesson_count": active_lesson_count,
        "total_lesson_count": total_lesson_count,
        "weight_revisions": weight_revisions,
        "reflection_count": reflection_count,
        # Misc
        "current_weights": current_weights,
        "latest_perf": latest_perf,
        "cb_state": cb_state,
        "universe": universe,
        "first_decision": first_decision,
        "recent_activity": recent_activity,
        "cb_active": _infer_cb_active(latest_perf),
    }
    return _templates(request).TemplateResponse(request, "system.html", ctx)
