import os
import subprocess
import threading
from typing import Any

import telebot
from loguru import logger

from config import settings


def check_access(message: Any) -> bool:
    """Ensure message is sent by the authorized user identifier."""
    allowed = settings.ALLOWED_TELEGRAM_USER_ID.strip()
    if not allowed:
        return False
    return str(message.from_user.id) == allowed or str(message.chat.id) == allowed


def init_telegram_bot() -> telebot.TeleBot | None:
    """Initialize Telegram Bot client and register slash command listeners."""
    token = settings.TELEGRAM_BOT_TOKEN.strip()
    if not token:
        logger.info(
            "Telegram Bot token not provided. Skipping Telegram bot integration."
        )
        return None

    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def send_welcome(message: Any) -> None:
        if not check_access(message):
            bot.reply_to(message, "❌ Unauthorized access.")
            return
        bot.reply_to(
            message,
            "🚀 *Claude Code Proxy Bot Active!*\n\n"
            "Commands:\n"
            "/status - View settings and mappings\n"
            "/set_model <key> <val> - Update model mapping\n"
            "/run <command> - Execute command in workspace",
            parse_mode="Markdown",
        )

    @bot.message_handler(commands=["status"])
    def show_status(message: Any) -> None:
        if not check_access(message):
            return
        status_text = (
            "⚙️ *Proxy Configuration Status*\n\n"
            f"• *OPUS Model:* `{settings.MODEL_OPUS}`\n"
            f"• *SONNET Model:* `{settings.MODEL_SONNET}`\n"
            f"• *HAIKU Model:* `{settings.MODEL_HAIKU}`\n"
            f"• *Default Model:* `{settings.MODEL}`\n"
            f"• *Rate Limits:* `{settings.PROVIDER_RATE_LIMIT} req / {settings.PROVIDER_RATE_WINDOW}s`\n"
            f"• *Max Concurrency:* `{settings.PROVIDER_MAX_CONCURRENCY}`\n"
            f"• *Fast Prefix Detection:* `{settings.FAST_PREFIX_DETECTION}`\n"
            f"• *Network Probe Mock:* `{settings.ENABLE_NETWORK_PROBE_MOCK}`\n"
            f"• *Filepath Mock:* `{settings.ENABLE_FILEPATH_EXTRACTION_MOCK}`\n"
            f"• *Workspace:* `{settings.CLAUDE_WORKSPACE}`"
        )
        bot.reply_to(message, status_text, parse_mode="Markdown")

    @bot.message_handler(commands=["set_model"])
    def set_model_mapping(message: Any) -> None:
        if not check_access(message):
            return
        parts = message.text.split(maxsplit=2)
        if len(parts) < 3:
            bot.reply_to(
                message,
                "⚠️ Usage: `/set_model <MODEL_KEY> <PROVIDER/MODEL>`\nExample: `/set_model MODEL_SONNET open_router/openai/gpt-4o`",
                parse_mode="Markdown",
            )
            return

        key, val = parts[1].upper(), parts[2]
        if key == "MODEL_OPUS":
            settings.MODEL_OPUS = val
        elif key == "MODEL_SONNET":
            settings.MODEL_SONNET = val
        elif key == "MODEL_HAIKU":
            settings.MODEL_HAIKU = val
        elif key == "MODEL":
            settings.MODEL = val
        else:
            bot.reply_to(
                message,
                f"❌ Invalid config key `{key}`. Must be MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU, or MODEL.",
                parse_mode="Markdown",
            )
            return

        bot.reply_to(
            message,
            f"✅ Updated `{key}` to `{val}` successfully.",
            parse_mode="Markdown",
        )

    @bot.message_handler(commands=["run"])
    def execute_command(message: Any) -> None:
        if not check_access(message):
            return
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            bot.reply_to(message, "⚠️ Usage: `/run <bash command>`")
            return

        cmd = parts[1]
        workspace = os.path.abspath(settings.CLAUDE_WORKSPACE)
        os.makedirs(workspace, exist_ok=True)

        bot.reply_to(message, f"⏳ Executing `{cmd}` in `{workspace}`...")

        try:
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
            )
            output = res.stdout or "Command finished with no output."
            # Split message if it exceeds Telegram limits
            if len(output) > 4000:
                output = output[:3900] + "\n...(truncated)"
            bot.reply_to(message, f"```\n{output}\n```", parse_mode="Markdown")
        except subprocess.TimeoutExpired:
            bot.reply_to(message, "❌ Command execution timed out (60s limit).")
        except Exception as e:
            bot.reply_to(message, f"❌ Execution error: {e}")

    # Launch bot in a separate background thread
    t = threading.Thread(target=bot.infinity_polling, daemon=True)
    t.start()
    logger.info("Telegram Bot thread started successfully.")

    return bot
