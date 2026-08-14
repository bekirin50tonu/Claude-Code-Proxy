"""Telegram Bot platform adapter using python-telegram-bot v20+."""

import asyncio
import os
import subprocess

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

from bot.auth import is_authorized_telegram
from bot.base import BaseBotAdapter
from bot.formatters import (
    escape_markdown_v2,
    format_circuit_breaker_alert_tg,
    format_status_overview_tg,
)
from bot.handlers.ask import handle_ask_command
from bot.handlers.inline import inline_query_handler
from bot.handlers.voice import handle_voice_message
from bot.live_bridge import live_bridge_manager
from config import settings, stats
from core.router.circuit_breaker import circuit_breaker_registry


class TelegramBotAdapter(BaseBotAdapter):
    """Adapter encapsulating Telegram Bot operations via python-telegram-bot v20+."""

    def __init__(self) -> None:
        self.token = settings.TELEGRAM_BOT_TOKEN.strip()
        self.app: Application | None = None
        self.session_threads: dict[str, int] = {}

    @property
    def platform_name(self) -> str:
        return "telegram"

    async def _send_typing(self, update: Update) -> None:
        """Send 'typing...' chat action indicator to Telegram."""
        if update.effective_chat:
            try:
                await update.effective_chat.send_action(ChatAction.TYPING)
            except Exception as e:
                logger.debug(f"Failed to send typing indicator: {e}")

    async def start(self) -> None:
        """Build and launch the Telegram bot application in background."""
        if not self.token:
            logger.info("Telegram Bot token not provided. Skipping Telegram adapter.")
            return

        try:
            self.app = ApplicationBuilder().token(self.token).build()

            # Command Handlers
            self.app.add_handler(CommandHandler("start", self._cmd_start))
            self.app.add_handler(CommandHandler("help", self._cmd_help))
            self.app.add_handler(CommandHandler("status", self._cmd_status))
            self.app.add_handler(CommandHandler("ask", handle_ask_command))
            self.app.add_handler(CommandHandler("workspace", self._cmd_workspace))
            self.app.add_handler(CommandHandler("reset_circuit", self._cmd_reset_circuit))
            self.app.add_handler(CommandHandler("set_model", self._cmd_set_model))
            self.app.add_handler(CommandHandler("run", self._cmd_run))
            self.app.add_handler(CommandHandler("live", self._cmd_live))
            self.app.add_handler(CommandHandler("stop", self._cmd_stop))
            self.app.add_handler(CommandHandler("clear", self._cmd_clear))
            self.app.add_handler(CommandHandler("stats", self._cmd_stats))

            # Direct Text Messages -> AI Prompt Handler
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._cmd_direct_text))

            # Inline Query Autocomplete Handler (Hermes Agent Previews)
            self.app.add_handler(InlineQueryHandler(inline_query_handler))

            # Voice Note Message Handler
            self.app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

            # Callback Query Handler
            self.app.add_handler(CallbackQueryHandler(self._handle_callback))

            # Error Handler
            self.app.add_error_handler(self._handle_error)

            await self.app.initialize()
            await self.app.start()
            if self.app.updater:
                await self.app.updater.start_polling(drop_pending_updates=True)

            logger.info("TelegramBotAdapter started successfully.")
        except Exception as e:
            logger.error(f"Failed to start TelegramBotAdapter: {e}")

    async def stop(self) -> None:
        """Shutdown Telegram bot polling and application cleanly."""
        if not self.app:
            return
        try:
            if self.app.updater and self.app.updater.running:
                await self.app.updater.stop()
            if self.app.running:
                await self.app.stop()
            await self.app.shutdown()
            logger.info("TelegramBotAdapter stopped cleanly.")
        except Exception as e:
            logger.error(f"Error stopping TelegramBotAdapter: {e}")

    async def send_circuit_breaker_alert(
        self,
        model_id: str,
        reason: str,
        fallback_model: str | None = None,
    ) -> None:
        """Send proactive Circuit Breaker trip alert notification to authorized user."""
        if not self.app or not settings.ALLOWED_TELEGRAM_USER_ID:
            return

        chat_id = settings.ALLOWED_TELEGRAM_USER_ID.strip()
        if not chat_id:
            return

        try:
            alert_text = format_circuit_breaker_alert_tg(model_id, reason, fallback_model)
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🔌 Reset Circuit",
                        callback_data=f"reset_cb:{model_id}",
                    )
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await self.app.bot.send_message(
                chat_id=chat_id,
                text=alert_text,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=reply_markup,
            )
            logger.info(f"Proactive alert sent to Telegram chat {chat_id} for model {model_id}")
        except Exception as e:
            logger.warning(f"Failed to send Telegram proactive alert: {e}")

    async def send_to_thread(self, session_id: str, title: str, text: str) -> None:
        """Send message under session parent thread on Telegram."""
        if not self.app or not settings.ALLOWED_TELEGRAM_USER_ID:
            return
        chat_id = settings.ALLOWED_TELEGRAM_USER_ID.strip()
        if not chat_id:
            return

        try:
            parent_id = self.session_threads.get(session_id)
            esc_text = escape_markdown_v2(text, is_code_block=False)

            if not parent_id:
                esc_title = escape_markdown_v2(title, is_code_block=True)
                esc_sid = escape_markdown_v2(session_id, is_code_block=True)
                parent_msg = await self.app.bot.send_message(
                    chat_id=chat_id,
                    text=f"📂 *Session:* `{esc_title}`\nID: `{esc_sid}`",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                parent_id = parent_msg.message_id
                self.session_threads[session_id] = parent_id

            await self.app.bot.send_message(
                chat_id=chat_id,
                text=esc_text,
                reply_to_message_id=parent_id,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except Exception as e:
            logger.warning(f"Failed to send Telegram thread message: {e}")

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error(f"Telegram API Error: {context.error}")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            if update.message:
                await update.message.reply_text("❌ Access Denied: Unauthorized Telegram User.")
            return

        await self._send_typing(update)

        keyboard = [
            [
                InlineKeyboardButton("📊 Status Overview", callback_data="status_refresh"),
                InlineKeyboardButton("❓ Help", callback_data="show_help"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        welcome_text = (
            "🚀 *Claude Code Proxy Bot v2 Active\\!*\n\n"
            "Welcome to the enhanced gateway manager with proactive alerting, live stream bridge, "
            "Hermes autocomplete previews, and Whisper voice note support\\.\n\n"
            "Use the buttons below or send `/help` for available commands\\."
        )
        if update.message:
            await update.message.reply_text(
                text=welcome_text,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=reply_markup,
            )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

        help_text = (
            "🛠️ *Claude Code Proxy Bot Commands*\n\n"
            "• Direct message \\- Simply type your prompt directly to talk to AI\n"
            "• `/ask <prompt>` \\- Ask AI (Claude Code) a prompt/coding question\n"
            "• `/workspace <path>` \\- View or set active workspace directory for `/run`\n"
            "• `/run <command>` \\- Execute bash command in workspace\n"
            "• `/status` \\- View proxy configuration and Circuit Breaker states\n"
            "• `/live <on|off>` \\- Toggle real\\-time reasoning stream and tool call tracking\n"
            "• `/reset_circuit <id|name>` \\- Reset Circuit Breaker for model/provider to `CLOSED`\n"
            "• `/set_model <KEY> <PROVIDER/MODEL>` \\- Update model mapping\n"
            "• `/stats` \\- View live RPM/TPM token budget & request statistics\n"
            "• `/stop` \\- Cancel active streaming task\n"
            "• `/clear` \\- Clear current session context\n"
            "• `/help` \\- Show this help menu\n\n"
            "💡 *Tip:* Type `@bot /` in any chat to view Hermes Agent autocomplete command previews\\!"
        )
        if update.message:
            await update.message.reply_text(text=help_text, parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_workspace(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

        if context.args:
            new_ws = " ".join(context.args).strip()
            abs_path = os.path.abspath(new_ws)
            os.makedirs(abs_path, exist_ok=True)
            settings.CLAUDE_WORKSPACE = abs_path
            esc_ws = escape_markdown_v2(abs_path, is_code_block=True)
            if update.message:
                await update.message.reply_text(
                    f"✅ *Workspace Directory Updated\\!*\n\n📂 `{esc_ws}`",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
        else:
            curr_ws = os.path.abspath(settings.CLAUDE_WORKSPACE)
            esc_ws = escape_markdown_v2(curr_ws, is_code_block=True)
            if update.message:
                await update.message.reply_text(
                    f"📂 *Current Workspace Directory:*\n`{esc_ws}`\n\n"
                    "💡 *Usage:* `/workspace <path_to_project>` to change active execution directory for `/run`\\.",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )

    async def _cmd_direct_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle direct text messages without slash prefix by forwarding to handle_ask_command."""
        if not is_authorized_telegram(update) or not update.message or not update.message.text:
            return

        text = update.message.text.strip()
        if text.startswith("/"):
            return

        context.args = text.split()
        await handle_ask_command(update, context)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

        keyboard = [[InlineKeyboardButton("🔄 Refresh Status", callback_data="status_refresh")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        status_text = format_status_overview_tg()

        if update.message:
            await update.message.reply_text(
                text=status_text,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=reply_markup,
            )

    async def _cmd_live(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

        chat_id = str(update.effective_chat.id) if update.effective_chat else ""
        sub = context.args[0].lower() if context.args else ""

        if sub == "on":
            active = await live_bridge_manager.toggle_live(chat_id, enable=True)
        elif sub == "off":
            active = await live_bridge_manager.toggle_live(chat_id, enable=False)
        else:
            active = live_bridge_manager.is_watcher(chat_id)

        status_str = "🟢 *ACTIVE*" if active else "🔴 *DISABLED*"
        text = (
            f"📡 *Live Stream Tracking Status:* {status_str}\n\n"
            "Use `/live on` to enable real\\-time reasoning stream and tool call tracking\\.\n"
            "Use `/live off` to disable stream tracking\\."
        )

        if update.message:
            await update.message.reply_text(text=text, parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

        if update.message:
            await update.message.reply_text("🛑 *Active task execution stopped\\.*", parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

        if update.message:
            await update.message.reply_text("🧹 *Session context cleared successfully\\.*", parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

        total_req = stats.total_requests
        mocked_req = stats.mocked_requests
        error_cnt = stats.error_count
        active_conc = stats.active_concurrency

        stats_text = (
            "📊 *Claude Code Proxy Gateway Statistics*\n\n"
            f"• *Total Requests Captured:* `{total_req}`\n"
            f"• *Mocked Requests:* `{mocked_req}`\n"
            f"• *Error Count:* `{error_cnt}`\n"
            f"• *Active Concurrency:* `{active_conc}` / `{settings.PROVIDER_MAX_CONCURRENCY}`\n"
            f"• *Configured Rate Limit:* `{settings.PROVIDER_RATE_LIMIT}` req / `{settings.PROVIDER_RATE_WINDOW}`s"
        )

        if update.message:
            await update.message.reply_text(text=stats_text, parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_reset_circuit(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

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

        for mid, cb in list(all_breakers.items()):
            if mid == target or mid.startswith(f"{target}/") or target in mid:
                cb.reset()
                reset_models.append(mid)

        if not reset_models:
            cb = circuit_breaker_registry.get(target)
            cb.reset()
            reset_models.append(target)

        models_str = ", ".join(f"`{escape_markdown_v2(m, is_code_block=True)}`" for m in reset_models)
        reply_text = (
            "✅ *Circuit Breaker Reset\\!*\n\n"
            f"🔌 *Reset Model\\(s\\):* {models_str}\n"
            "📊 *New Status:* `CLOSED` \\(Requests Allowed\\)"
        )

        if update.message:
            await update.message.reply_text(text=reply_text, parse_mode=ParseMode.MARKDOWN_V2)

    async def _cmd_set_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

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
                    f"❌ Invalid key `{esc_key}`\\. Must be MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU, or MODEL\\.",
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

    async def _cmd_run(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_authorized_telegram(update):
            return

        await self._send_typing(update)

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
                f"⏳ *Executing:* `{esc_cmd}`\n📂 *Workspace:* `{esc_ws}`",
                parse_mode=ParseMode.MARKDOWN_V2,
            )

            stop_typing = asyncio.Event()

            async def keep_typing_loop() -> None:
                while not stop_typing.is_set():
                    await self._send_typing(update)
                    try:
                        await asyncio.sleep(4.0)
                    except asyncio.CancelledError:
                        break

            typing_task = asyncio.create_task(keep_typing_loop())

            try:
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
            finally:
                stop_typing.set()
                typing_task.cancel()

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query:
            return

        if not is_authorized_telegram(update):
            await query.answer("❌ Unauthorized user", show_alert=True)
            return

        data = query.data or ""
        logger.info(f"Telegram callback query: {data}")

        if data.startswith("reset_cb:"):
            model_id = data.split("reset_cb:", 1)[1]
            cb = circuit_breaker_registry.get(model_id)
            cb.reset()

            await query.answer("🔌 Circuit reset! Model reactivated.", show_alert=True)
            esc_mid = escape_markdown_v2(model_id, is_code_block=True)
            updated_text = (
                "✅ *Circuit Breaker Reset\\!*\n\n"
                f"🔌 *Provider/Model:* `{esc_mid}`\n"
                "📊 *Status:* `CLOSED` \\(Requests Allowed\\)\n\n"
                "✨ _Model traffic restored to normal\\._"
            )
            try:
                await query.edit_message_text(
                    text=updated_text,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=None,
                )
            except Exception as e:
                logger.warning(f"Failed to edit alert message on reset callback: {e}")

        elif data == "status_refresh":
            await query.answer("🔄 Status updated!")
            status_text = format_status_overview_tg()
            keyboard = [[InlineKeyboardButton("🔄 Refresh Status", callback_data="status_refresh")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            try:
                await query.edit_message_text(
                    text=status_text,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=reply_markup,
                )
            except Exception as e:
                logger.debug(f"Status text unchanged: {e}")

        elif data == "show_help":
            await query.answer()
            help_text = (
                "🛠️ *Claude Code Proxy Bot Commands*\n\n"
                "• `/status` \\- View proxy configuration and Circuit Breaker states\n"
                "• `/live <on|off>` \\- Toggle real\\-time reasoning stream and tool call tracking\n"
                "• `/reset_circuit <id|name>` \\- Reset Circuit Breaker for model/provider to `CLOSED`\n"
                "• `/set_model <KEY> <PROVIDER/MODEL>` \\- Update model mapping\n"
                "• `/run <command>` \\- Execute bash command in workspace\n"
                "• `/stats` \\- View live RPM/TPM token budget & request statistics\n"
                "• `/stop` \\- Cancel active streaming task\n"
                "• `/clear` \\- Clear current session context\n"
                "• `/help` \\- Show this help menu"
            )
            try:
                await query.edit_message_text(text=help_text, parse_mode=ParseMode.MARKDOWN_V2)
            except Exception as e:
                logger.debug(f"Help text edit error: {e}")
