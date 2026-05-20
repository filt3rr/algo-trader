"""FastAPI application for the Apex Console dashboard.

Run standalone:
    python -m agent.dashboard
    uvicorn agent.dashboard.app:create_app --factory --host 127.0.0.1 --port 8000

The dashboard is read-only — it shares the same SQLite DB as the trading loop
but never writes to it except for lesson deactivation (soft-delete via HTMX).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent.config import get_settings
from agent.memory.db import init_db

_HERE = Path(__file__).parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"


def _fmt_usd(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(value, decimals: int = 2) -> str:
    try:
        v = float(value)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(value, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def create_app() -> FastAPI:
    cfg = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine, factory = await init_db(cfg.db_url)
        app.state.session_factory = factory
        app.state.engine = engine
        app.state.cfg = cfg
        yield
        await engine.dispose()

    app = FastAPI(
        title="Apex Console",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.globals["fmt_usd"] = _fmt_usd
    templates.env.globals["fmt_pct"] = _fmt_pct
    templates.env.globals["fmt_num"] = _fmt_num
    templates.env.filters["fmt_usd"] = _fmt_usd
    templates.env.filters["fmt_pct"] = _fmt_pct
    templates.env.filters["fmt_num"] = _fmt_num

    from agent.dashboard.routes import router
    app.include_router(router)
    app.state.templates = templates

    return app


app = create_app()
