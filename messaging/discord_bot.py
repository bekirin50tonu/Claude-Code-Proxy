import asyncio
import logging
import os
import subprocess
from typing import Any

import discord

from config import settings

logger = logging.getLogger("proxy_discord")


class ProxyDiscordBot(discord.Client):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(*args, intents=intents, **kwargs)

    async def on_ready(self) -> None:
        logger.info(f"Logged into Discord as {self.user}")

    async def create_session_thread(
        self, channel_id: str, session_id: str, title: str
    ) -> int | None:
        """Create a new thread channel under the target text channel."""
        try:
            channel = self.get_channel(int(channel_id))
            if not channel:
                channel = await self.fetch_channel(int(channel_id))

            if isinstance(channel, discord.TextChannel):
                thread = await channel.create_thread(
                    name=f"Session {session_id[:8]}: {title[:40]}",
                    auto_archive_duration=60,
                )
                return thread.id
        except Exception as e:
            logger.warning(f"Failed to create Discord thread: {e}")
        return None

    async def send_to_thread(self, thread_id: int, text: str) -> None:
        """Send message update directly to thread."""
        try:
            thread = self.get_channel(thread_id)
            if not thread:
                thread = await self.fetch_channel(thread_id)
            if isinstance(thread, (discord.Thread, discord.TextChannel)):
                # Strip Markdown format if Discord block constraints occur, or wrap cleanly
                await thread.send(text)
        except Exception as e:
            logger.warning(f"Failed to send message to Discord thread {thread_id}: {e}")

    async def on_message(self, message: discord.Message) -> None:
        # Ignore messages from self
        if message.author == self.user:
            return

        # Check access against allowed channel list
        allowed = [
            c.strip() for c in settings.ALLOWED_DISCORD_CHANNELS.split(",") if c.strip()
        ]
        if not allowed or str(message.channel.id) not in allowed:
            return

        content = message.content.strip()

        # Handle !status
        if content.startswith("!status"):
            status_text = (
                "⚙️ **Proxy Configuration Status**\n\n"
                f"• **OPUS Model:** `{settings.MODEL_OPUS}`\n"
                f"• **SONNET Model:** `{settings.MODEL_SONNET}`\n"
                f"• **HAIKU Model:** `{settings.MODEL_HAIKU}`\n"
                f"• **Default Model:** `{settings.MODEL}`\n"
                f"• **Rate Limits:** `{settings.PROVIDER_RATE_LIMIT} req / {settings.PROVIDER_RATE_WINDOW}s`\n"
                f"• **Workspace:** `{settings.CLAUDE_WORKSPACE}`"
            )
            await message.reply(status_text)

        # Handle !set_model
        elif content.startswith("!set_model"):
            parts = content.split(maxsplit=2)
            if len(parts) < 3:
                await message.reply(
                    "⚠️ Usage: `!set_model <MODEL_KEY> <PROVIDER/MODEL>`\nExample: `!set_model MODEL_SONNET open_router/openai/gpt-4o`"
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
                await message.reply(
                    f"❌ Invalid key `{key}`. Choose MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU, or MODEL."
                )
                return

            await message.reply(f"✅ Updated `{key}` to `{val}` successfully.")

        # Handle !run
        elif content.startswith("!run"):
            parts = content.split(maxsplit=1)
            if len(parts) < 2:
                await message.reply("⚠️ Usage: `!run <bash command>`")
                return

            cmd = parts[1]
            workspace = os.path.abspath(settings.CLAUDE_WORKSPACE)
            os.makedirs(workspace, exist_ok=True)

            status_msg = await message.reply(
                f"⏳ Executing `{cmd}` in `{workspace}`..."
            )

            # Run in executor to avoid blocking the loop
            try:

                def run() -> subprocess.CompletedProcess[str]:
                    return subprocess.run(
                        cmd,
                        shell=True,
                        cwd=workspace,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=60,
                    )

                res = await asyncio.get_event_loop().run_in_executor(None, run)
                output = res.stdout or "Command finished with no output."
                if len(output) > 1900:
                    output = output[:1800] + "\n...(truncated)"

                await status_msg.edit(content=f"```\n{output}\n```")
            except subprocess.TimeoutExpired:
                await status_msg.edit(
                    content="❌ Command execution timed out (60s limit)."
                )
            except Exception as e:
                await status_msg.edit(content=f"❌ Execution error: {e}")


async def init_discord_bot() -> ProxyDiscordBot | None:
    """Instantiate discord bot client and launch it in the current loop."""
    token = settings.DISCORD_BOT_TOKEN.strip()
    if not token:
        logger.info("Discord Bot token not provided. Skipping Discord bot integration.")
        return None

    bot = ProxyDiscordBot()
    # Schedule the client to start running in the background
    asyncio.create_task(bot.start(token))
    logger.info("Discord Bot task scheduled successfully.")
    return bot
