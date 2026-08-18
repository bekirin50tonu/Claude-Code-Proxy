from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

from api.dashboard import router as dashboard_router
from api.mcp import mcp_router
from bot import start_all_bots, stop_all_bots
from core.gateway import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycles."""
    logger.info("Initializing Claude Code Proxy Server...")

    # Start configured Telegram & Discord bots via BotFactory
    await start_all_bots()

    yield

    # Shutdown hooks
    logger.info("Shutting down bots...")
    await stop_all_bots()
    logger.info("Claude Code Proxy Server shutdown complete.")


app = FastAPI(
    title="Claude Code Proxy Server",
    description="FastAPI gateway routing Claude Code requests to NVIDIA NIM, OpenRouter, and local models.",
    version="0.1.0",
    lifespan=lifespan,
)

# Register static files directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Register endpoints
app.include_router(api_router)
app.include_router(dashboard_router)
app.include_router(mcp_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "status": "online",
        "name": "Claude Code Proxy Server (Gateway)",
        "docs": "https://github.com/Alishahryar1/free-claude-code",
    }


if __name__ == "__main__":
    import os

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", os.getenv("GATEWAY_PORT", 8090)))
    uvicorn.run("server:app", host=host, port=port, reload=True)
