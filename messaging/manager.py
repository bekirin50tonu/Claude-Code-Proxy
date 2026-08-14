import asyncio
from typing import Any

from loguru import logger

from config import settings


class MessagingManager:
    def __init__(self) -> None:
        self.telegram_bot: Any = None
        self.discord_bot: Any = None
        self.loop: asyncio.AbstractEventLoop | None = None

        # Maps session_id/thread_key -> (telegram_msg_id, discord_thread_id)
        self.session_threads: dict[str, tuple[int | None, int | None]] = {}
        self.thread_lock = asyncio.Lock()

    def set_bots(self, telegram_bot: Any, discord_bot: Any) -> None:
        self.telegram_bot = telegram_bot
        self.discord_bot = discord_bot

    async def get_or_create_thread(
        self, session_id: str, title: str
    ) -> tuple[int | None, int | None]:
        """Resolve parent message IDs for tree-based thread tracking."""
        async with self.thread_lock:
            if session_id in self.session_threads:
                return self.session_threads[session_id]

            tg_parent = None
            ds_parent = None

            # 1. Setup Telegram Thread parent
            if (
                self.telegram_bot
                and settings.TELEGRAM_BOT_TOKEN
                and settings.ALLOWED_TELEGRAM_USER_ID
            ):
                try:
                    # Send a parent message for this session
                    msg = await asyncio.to_thread(
                        self.telegram_bot.send_message,
                        chat_id=settings.ALLOWED_TELEGRAM_USER_ID,
                        text=f"📂 *Session: {title}*\nID: `{session_id}`",
                        parse_mode="Markdown",
                    )
                    tg_parent = msg.message_id
                except Exception as e:
                    logger.warning(f"Failed to create Telegram session thread: {e}")

            # 2. Setup Discord Thread parent
            if (
                self.discord_bot
                and settings.DISCORD_BOT_TOKEN
                and settings.ALLOWED_DISCORD_CHANNELS
            ):
                try:
                    # Resolve channel and start a thread
                    channel_ids = [
                        c.strip()
                        for c in settings.ALLOWED_DISCORD_CHANNELS.split(",")
                        if c.strip()
                    ]
                    if channel_ids:
                        # Call bot custom method to create thread asynchronously
                        ds_parent = await self.discord_bot.create_session_thread(
                            channel_ids[0], session_id, title
                        )
                except Exception as e:
                    logger.warning(f"Failed to create Discord session thread: {e}")

            self.session_threads[session_id] = (tg_parent, ds_parent)
            return tg_parent, ds_parent

    async def send_to_thread(self, session_id: str, title: str, text: str) -> None:
        """Send message as a reply/thread update under the session parent."""
        tg_parent, ds_parent = await self.get_or_create_thread(session_id, title)

        # Send Telegram Thread reply
        if (
            self.telegram_bot
            and settings.TELEGRAM_BOT_TOKEN
            and settings.ALLOWED_TELEGRAM_USER_ID
        ):
            try:
                await asyncio.to_thread(
                    self.telegram_bot.send_message,
                    chat_id=settings.ALLOWED_TELEGRAM_USER_ID,
                    text=text,
                    reply_to_message_id=tg_parent,
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.warning(f"Failed to send Telegram thread message: {e}")

        # Send Discord Thread reply
        if self.discord_bot and settings.DISCORD_BOT_TOKEN and ds_parent:
            try:
                await self.discord_bot.send_to_thread(ds_parent, text)
            except Exception as e:
                logger.warning(f"Failed to send Discord thread message: {e}")

    async def broadcast_session_event(
        self, session_id: str, title: str, event_type: str, details: str
    ) -> None:
        """Helper to post agent updates (thinking/tool execution) inside threads."""
        emoji = "🧠" if event_type == "thinking" else "🛠️"
        header = (
            "*Thinking Process*" if event_type == "thinking" else "*Tool Invocation*"
        )
        formatted_msg = f"{emoji} {header}\n{details}"
        await self.send_to_thread(session_id, title, formatted_msg)


# Shared messaging manager
messaging_manager = MessagingManager()
