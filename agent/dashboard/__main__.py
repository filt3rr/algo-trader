"""python -m agent.dashboard — launch the Apex Console dashboard."""

import uvicorn

uvicorn.run(
    "agent.dashboard.app:app",
    host="127.0.0.1",
    port=8000,
    reload=False,
    log_level="info",
)
