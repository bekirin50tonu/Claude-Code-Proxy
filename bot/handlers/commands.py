"""Command handlers for Telegram Bot v2 (/start, /status, /reset_circuit, /set_model, /run, /help)."""

import asyncio
import os
import subprocess

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.formatters import escape_markdown_v2, format_status_overview
from config import settings
from core.router.circuit_breaker import circuit_breaker_registry


def is_authorized(update: Update) -> bool:
    """Check if the requesting user or chat ID matches ALLOWED_TELEGRAM_USER_ID."""
    allowed = settings.ALLOWED_TELEGRAM_USER_ID.strip()
    if not allowed:
        return False
    user_id = str(update.effective_user.id) if update.effective_user else ""
    chat_id = str(update.effective_chat.id) if update.effective_chat else ""
    return user_id == allowed or chat_id == allowed


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /start command with interactive inline menu."""
    if not is_authorized(update):
        if update.message:
            await update.message.reply_text("❌ Access Denied: Unauthorized Telegram User.")
        return

    keyboard = [
        [
            InlineKeyboardButton("📊 Status Overview", callback_data="status_refresh"),
            InlineKeyboardButton("❓ Help", callback_data="show_help"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    welcome_text = (
        "🚀 *Claude Code Proxy Bot v2 Active\\!*\n\n"
        "Welcome to the enhanced gateway manager with proactive alerting, Markdown V2 streaming, "
        "and interactive resiliency controls\\.\n\n"
        "Use the buttons below or send `/help` for available commands\\."
    )
    if update.message:
        await update.message.reply_text(
            text=welcome_text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup,
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /help command."""
    if not is_authorized(update):
        return

    help_text = (
        "🛠️ *Claude Code Proxy Bot Commands*\n\n"
        "• `/status` \\- View proxy configuration and Circuit Breaker states\n"
        "• `/reset_circuit <id|name>` \\- Reset Circuit Breaker for a model/provider back to `CLOSED` state\n"
        "• `/set_model <KEY> <PROVIDER/MODEL>` \\- Dynamically update model mapping\n"
        "• `/run <command>` \\- Execute bash command in proxy workspace\n"
        "• `/help` \\- Show this help menu\n\n"
        "💡 *Examples:*\n"
        "`/reset_circuit nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct`\n"
        "`/reset_circuit nvidia_nim`\n"
        "`/set_model MODEL_SONNET open_router/anthropic/claude-3.5-sonnet`"
    )
    if update.message:
        await update.message.reply_text(
            text=help_text,
            parse_mode=ParseMode.MARKDOWN_V2,
        )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /status command with live refresh inline button."""
    if not is_authorized(update):
        return

    keyboard = [[InlineKeyboardButton("🔄 Refresh Status", callback_data="status_refresh")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    status_text = format_status_overview()
    if update.message:
        await update.message.reply_text(
            text=status_text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup,
        )


async def reset_circuit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /reset_circuit <id|name> command."""
    if not is_authorized(update):
        return

    if not context.args or len(context.args) < 1:
        if update.message:
            await update.message.reply_text(
                "⚠️ *Usage:* `/reset_circuit <model_id|provider_name>`\n"
                "Example: `/reset_circuit nvidia_nim/meta/llama-3.1-70b-instruct` or `/reset_circuit nvidia_nim`",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        return

    target = context.args[0].strip()
    all_breakers = circuit_breaker_registry._breakers
    reset_models: list[str] = []

    # Search by exact model_id or provider prefix
    for mid, cb in list(all_breakers.items()):
        if mid == target or mid.startswith(f"{target}/") or target in mid:
            cb.reset()
            reset_models.append(mid)

    # If not registered yet, attempt to get and reset explicitly
    if not reset_models:
        cb = circuit_breaker_registry.get(target)
        cb.reset()
        reset_models.append(target)

    models_str = ", ".join(f"`{escape_markdown_v2(m, is_code_block=True)}`" for m in reset_models)
    reply_text = (
        "✅ *Devre Kesici Sıfırlandı\\!*\n\n"
        f"🔌 *Sıfırlanan Model\\(ler\\):* {models_str}\n"
        "📊 *Yeni Durum:* `CLOSED` \\(İsteklere Açık\\)"
    )

    if update.message:
        await update.message.reply_text(
            text=reply_text,
            parse_mode=ParseMode.MARKDOWN_V2,
        )


async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /set_model <key> <val> command."""
    if not is_authorized(update):
        return

    if not context.args or len(context.args) < 2:
        if update.message:
            await update.message.reply_text(
                "⚠️ *Usage:* `/set_model <MODEL_KEY> <PROVIDER/MODEL>`\n"
                "Example: `/set_model MODEL_SONNET open_router/openai/gpt-4o`",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        return

    key = context.args[0].upper().strip()
    val = context.args[1].strip()

    if key == "MODEL_OPUS":
        settings.MODEL_OPUS = val
    elif key == "MODEL_SONNET":
        settings.MODEL_SONNET = val
    elif key == "MODEL_HAIKU":
        settings.MODEL_HAIKU = val
    elif key == "MODEL":
        settings.MODEL = val
    else:
        if update.message:
            esc_key = escape_markdown_v2(key, is_code_block=True)
            await update.message.reply_text(
                f"❌ Invalid config key `{esc_key}`\\. Must be MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU, or MODEL\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        return

    esc_k = escape_markdown_v2(key, is_code_block=True)
    esc_v = escape_markdown_v2(val, is_code_block=True)
    if update.message:
        await update.message.reply_text(
            f"✅ Updated `{esc_k}` to `{esc_v}` successfully\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )


async def run_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /run <bash_command> command."""
    if not is_authorized(update):
        return

    if not context.args:
        if update.message:
            await update.message.reply_text("⚠️ Usage: `/run <bash command>`", parse_mode=ParseMode.MARKDOWN_V2)
        return

    cmd = " ".join(context.args)
    workspace = os.path.abspath(settings.CLAUDE_WORKSPACE)
    os.makedirs(workspace, exist_ok=True)

    esc_cmd = escape_markdown_v2(cmd, is_code_block=True)
    esc_ws = escape_markdown_v2(workspace, is_code_block=True)

    if update.message:
        status_msg = await update.message.reply_text(
            f"⏳ Executing `{esc_cmd}` in `{esc_ws}`\\...",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

        try:
            # Run in asyncio executor thread
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    shell=True,
                    cwd=workspace,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=60,
                ),
            )
            output = res.stdout or "Command finished with no output."
            if len(output) > 3500:
                output = output[:3500] + "\n...(truncated)"

            esc_output = escape_markdown_v2(output, is_code_block=True)
            await status_msg.edit_text(
                f"```\n{esc_output}\n```",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except subprocess.TimeoutExpired:
            await status_msg.edit_text("❌ Command execution timed out \\(60s limit\\)\\.", parse_mode=ParseMode.MARKDOWN_V2)
        except Exception as e:
            esc_err = escape_markdown_v2(str(e), is_code_block=True)
            await status_msg.edit_text(f"❌ Execution error: `{esc_err}`", parse_mode=ParseMode.MARKDOWN_V2)
