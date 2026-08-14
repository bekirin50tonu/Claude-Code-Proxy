"""Callback query handlers for interactive Inline Keyboard buttons in Telegram Bot v2."""

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.formatters import escape_markdown_v2, format_status_overview
from bot.handlers.commands import is_authorized
from core.router.circuit_breaker import circuit_breaker_registry


async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Central handler for all inline button callbacks."""
    query = update.callback_query
    if not query:
        return

    # Check access security
    if not is_authorized(update):
        await query.answer("❌ Unauthorized user", show_alert=True)
        return

    data = query.data or ""
    logger.info(f"Telegram Bot callback query received: {data}")

    # 1. Circuit Breaker Reset Callback (reset_cb:<model_id>)
    if data.startswith("reset_cb:"):
        model_id = data.split("reset_cb:", 1)[1]
        
        # Perform reset
        cb = circuit_breaker_registry.get(model_id)
        cb.reset()

        await query.answer("🔌 Devre sıfırlandı! Model yeniden aktif.", show_alert=True)

        esc_mid = escape_markdown_v2(model_id, is_code_block=True)
        updated_text = (
            "✅ *Circuit Breaker Sıfırlandı\\!*\n\n"
            f"🔌 *Sağlayıcı/Model:* `{esc_mid}`\n"
            "📊 *Durum:* `CLOSED` \\(Yeniden İsteklere Açık\\)\n\n"
            "✨ _Model trafiği normale döndü\\._"
        )

        try:
            await query.edit_message_text(
                text=updated_text,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=None,
            )
        except Exception as e:
            logger.warning(f"Failed to edit alert message on reset callback: {e}")

    # 2. Refresh Status Panel Callback
    elif data == "status_refresh":
        await query.answer("🔄 Status updated!")
        status_text = format_status_overview()
        keyboard = [[InlineKeyboardButton("🔄 Refresh Status", callback_data="status_refresh")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            await query.edit_message_text(
                text=status_text,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.debug(f"Status text unchanged or edit error: {e}")

    # 3. Show Help Callback
    elif data == "show_help":
        await query.answer()
        help_text = (
            "🛠️ *Claude Code Proxy Bot Commands*\n\n"
            "• `/status` \\- View proxy configuration and Circuit Breaker states\n"
            "• `/reset_circuit <id|name>` \\- Reset Circuit Breaker for a model/provider back to `CLOSED` state\n"
            "• `/set_model <KEY> <PROVIDER/MODEL>` \\- Dynamically update model mapping\n"
            "• `/run <command>` \\- Execute bash command in proxy workspace\n"
            "• `/help` \\- Show this help menu"
        )
        try:
            await query.edit_message_text(
                text=help_text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except Exception as e:
            logger.debug(f"Help text edit error: {e}")

    else:
        await query.answer()
