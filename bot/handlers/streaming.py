"""Live streaming message updater with anti-flood rate throttling for Telegram Bot v2."""

import asyncio
import time

from loguru import logger
from telegram import Bot
from telegram.constants import ParseMode

from bot.formatters import escape_markdown_v2


class TelegramStreamUpdater:
    """Manages live updates (thinking process, tool calling, bash output) to a Telegram message."""

    def __init__(
        self,
        bot: Bot,
        chat_id: str | int,
        parent_msg_id: int | None = None,
        min_edit_interval: float = 1.0,
    ) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.msg_id = parent_msg_id
        self.min_edit_interval = min_edit_interval
        self._last_edit_time = 0.0
        self._lock = asyncio.Lock()
        self._current_text = ""

    async def initialize(self, initial_header: str) -> None:
        """Send the initial message to Telegram."""
        async with self._lock:
            esc_header = escape_markdown_v2(initial_header, is_code_block=False)
            try:
                msg = await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=esc_header,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                self.msg_id = msg.message_id
                self._last_edit_time = time.monotonic()
                self._current_text = initial_header
            except Exception as e:
                logger.warning(f"Failed to initialize Telegram stream message: {e}")

    async def update(self, content: str, force: bool = False) -> None:
        """Edit message with updated content, respecting the rate limit throttle."""
        if not self.msg_id:
            await self.initialize(content)
            return

        async with self._lock:
            now = time.monotonic()
            if not force and (now - self._last_edit_time) < self.min_edit_interval:
                return

            if content == self._current_text:
                return

            self._current_text = content
            self._last_edit_time = now

            # Truncate content if too long for single Telegram message
            display_text = content
            if len(display_text) > 3800:
                display_text = display_text[:3700] + "\n...(stream truncated)"

            esc_text = escape_markdown_v2(display_text, is_code_block=False)

            try:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self.msg_id,
                    text=esc_text,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
            except Exception as e:
                # Ignore MessageNotModified errors
                if "Message is not modified" not in str(e):
                    logger.warning(f"Telegram stream edit message error: {e}")
