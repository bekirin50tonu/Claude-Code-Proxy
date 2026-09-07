import asyncio
import contextlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from api.dashboard import router as dashboard_router
from api.mcp import mcp_router
from api.metrics import metrics_router
from bot import start_all_bots, stop_all_bots
from core.gateway import router as api_router
from core.interceptor import JSONRepairMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycles."""
    import signal

    from config import model_registry, settings, stats

    logger.info("Initializing Claude Code Proxy Server...")

    # Register SIGHUP signal handler for zero-downtime hot-reload of config and models
    def _handle_sighup(signum: int, frame: Any) -> None:
        logger.info("Received SIGHUP signal. Reloading settings & model registry in-memory...")
        try:
            settings.reload()
            model_registry.reload()
            logger.success("In-memory settings & model registry successfully reloaded on SIGHUP.")
        except Exception as err:
            logger.error("Failed to reload configuration on SIGHUP: {}", err)

    if hasattr(signal, "SIGHUP"):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGHUP, _handle_sighup)

    # Automatically sync settings
    from api.settings_manager import claude_settings_manager
    claude_settings_manager.sync_proxy_to_claude()

    # Start configured Telegram & Discord bots via BotFactory
    await start_all_bots()

    yield

    # Shutdown hooks
    logger.info("Gracefully shutting down proxy server...")
    await stop_all_bots()

    import time
    drain_timeout = 5.0
    start_drain = time.monotonic()
    while stats.active_concurrency > 0 and (time.monotonic() - start_drain) < drain_timeout:
        await asyncio.sleep(0.1)

    logger.info("Claude Code Proxy Server shutdown complete.")


app = FastAPI(
    title="Claude Code Proxy Server",
    description="FastAPI gateway routing Claude Code requests to NVIDIA NIM, OpenRouter, and local models.",
    version="0.1.0",
    lifespan=lifespan,
)

# Register JSON Repair Middleware
app.add_middleware(JSONRepairMiddleware)

# Register static files directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Register endpoints
app.include_router(api_router)
app.include_router(dashboard_router)
app.include_router(mcp_router)
app.include_router(metrics_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "status": "online",
        "name": "Claude Code Proxy Server (Gateway)",
        "docs": "https://github.com/bekirin50tonu/Claude-Code-Proxy",
    }


@app.head("/api/hello")
@app.get("/api/hello")
async def api_hello() -> dict[str, str]:
    """Health check endpoint requested by Claude Code CLI."""
    return {"status": "ok", "service": "claude-code-proxy"}


@app.get("/health")
async def health_liveness() -> dict[str, str]:
    """Liveness probe returning 200 OK if proxy server is running."""
    return {"status": "ok", "service": "claude-code-proxy"}


@app.get("/health/ready")
async def health_readiness() -> JSONResponse:
    """Readiness probe checking server state and model circuit breaker headroom."""
    from config import stats
    from core.router.selector import model_selector

    status_data = model_selector.get_status()
    all_open = all(
        v.get("circuit_breaker", {}).get("state") == "open"
        for v in status_data.values()
        if isinstance(v, dict)
    ) if status_data else False

    if all_open:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "All upstream circuit breakers are OPEN"},
        )

    return JSONResponse(
        content={
            "status": "ready",
            "active_concurrency": stats.active_concurrency,
            "total_requests": stats.total_requests,
        }
    )


if __name__ == "__main__":
    import os

    from config import settings

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", os.getenv("GATEWAY_PORT", 8090)))
    uvicorn.run("server:app", host=host, port=port, reload=settings.RELOAD)
