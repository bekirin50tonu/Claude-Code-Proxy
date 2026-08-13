import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.dashboard import router as dashboard_router
from api.routes import router as api_router
from messaging.discord_bot import init_discord_bot
from messaging.manager import messaging_manager
from messaging.telegram_bot import init_telegram_bot

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("proxy_server")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycles."""
    logger.info("Initializing Claude Code Proxy Server...")

    # Initialize Telegram Bot
    tg_bot = init_telegram_bot()

    # Initialize Discord Bot
    ds_bot = await init_discord_bot()

    # Set references in messaging manager
    messaging_manager.set_bots(tg_bot, ds_bot)

    yield

    # Shutdown hooks
    logger.info("Shutting down bots...")
    if ds_bot:
        await ds_bot.close()
    logger.info("Claude Code Proxy Server shutdown complete.")


app = FastAPI(
    title="Claude Code Proxy Server",
    description="FastAPI gateway routing Claude Code requests to NVIDIA NIM, OpenRouter, and local models.",
    version="0.1.0",
    lifespan=lifespan,
)

# Register endpoints
app.include_router(api_router)
app.include_router(dashboard_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "status": "online",
        "name": "Claude Code Proxy Server (Gateway)",
        "docs": "https://github.com/Alishahryar1/free-claude-code",
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8090, reload=True)
