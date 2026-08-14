"""Telegram Bot v2 main module — async Application lifecycle & proactive Circuit Breaker alert dispatcher."""

import asyncio

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from bot.formatters import format_circuit_breaker_alert
from bot.handlers.callbacks import button_callback_handler
from bot.handlers.commands import (
    help_command,
    reset_circuit_command,
    run_command,
    set_model_command,
    start_command,
    status_command,
)
from config import settings
from core.router.circuit_breaker import circuit_breaker_registry

# Global singleton PTB Application instance
ptb_app: Application | None = None
_bot_task: asyncio.Task[None] | None = None


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global Telegram bot error handler.

    Ensures Telegram API/network errors are caught and logged without affecting FastAPI.
    """
    logger.error(f"Telegram Bot error encountered: {context.error}")


def init_telegram_bot_v2() -> Application | None:
    """Initialize python-telegram-bot v20+ Application and register handlers."""
    global ptb_app
    token = settings.TELEGRAM_BOT_TOKEN.strip()
    if not token:
        logger.info("Telegram Bot token not provided. Skipping Telegram Bot v2 initialization.")
        return None

    try:
        app = ApplicationBuilder().token(token).build()

        # Register slash commands
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("status", status_command))
        app.add_handler(CommandHandler("reset_circuit", reset_circuit_command))
        app.add_handler(CommandHandler("set_model", set_model_command))
        app.add_handler(CommandHandler("run", run_command))

        # Register inline button callback query handler
        app.add_handler(CallbackQueryHandler(button_callback_handler))

        # Register global error handler
        app.add_error_handler(error_handler)

        ptb_app = app
        logger.info("Telegram Bot v2 application initialized successfully.")

        # Register proactive alert callback on CircuitBreakerRegistry
        circuit_breaker_registry.register_trip_callback(on_circuit_breaker_trip)

        return app
    except Exception as e:
        logger.error(f"Failed to build Telegram Bot v2 application: {e}")
        return None


async def start_telegram_bot_v2(app: Application | None = None) -> None:
    """Start Telegram Bot v2 polling task within FastAPI's async event loop."""
    global ptb_app, _bot_task
    target_app = app or ptb_app
    if not target_app:
        return

    try:
        await target_app.initialize()
        await target_app.start()
        if target_app.updater:
            await target_app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram Bot v2 started and listening for updates.")
    except Exception as e:
        logger.error(f"Error starting Telegram Bot v2: {e}")


async def stop_telegram_bot_v2(app: Application | None = None) -> None:
    """Shutdown Telegram Bot v2 polling and release resources cleanly."""
    global ptb_app
    target_app = app or ptb_app
    if not target_app:
        return

    try:
        if target_app.updater and target_app.updater.running:
            await target_app.updater.stop()
        if target_app.running:
            await target_app.stop()
        await target_app.shutdown()
        logger.info("Telegram Bot v2 shutdown complete.")
    except Exception as e:
        logger.error(f"Error shutting down Telegram Bot v2: {e}")


def on_circuit_breaker_trip(model_id: str, reason: str) -> None:
    """Trip event listener invoked synchronously or asynchronously by CircuitBreakerRegistry."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(send_circuit_breaker_alert(model_id, reason))
    except RuntimeError:
        pass


async def send_circuit_breaker_alert(
    model_id: str,
    reason: str = "Upstream error / timeout",
    fallback_model: str | None = None,
) -> None:
    """Send proactive push notification alert with inline reset button to Telegram user."""
    global ptb_app
    if not ptb_app or not settings.ALLOWED_TELEGRAM_USER_ID:
        return

    chat_id = settings.ALLOWED_TELEGRAM_USER_ID.strip()
    if not chat_id:
        return

    try:
        alert_text = format_circuit_breaker_alert(model_id, reason, fallback_model)

        keyboard = [
            [
                InlineKeyboardButton(
                    "🔌 Devreyi Sıfırla (Reset)",
                    callback_data=f"reset_cb:{model_id}",
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await ptb_app.bot.send_message(
            chat_id=chat_id,
            text=alert_text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup,
        )
        logger.info(f"Proactive Circuit Breaker alert sent to Telegram chat {chat_id} for model {model_id}")
    except Exception as e:
        logger.warning(f"Failed to send proactive Circuit Breaker alert to Telegram: {e}")
