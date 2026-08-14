"""Discord Bot platform adapter using discord.py v2.7+ with Embeds and UI View components."""

import asyncio
import os
import subprocess
from typing import Any

import discord
from loguru import logger

from bot.base import BaseBotAdapter
from bot.formatters import (
    format_circuit_breaker_alert_discord,
    format_status_overview_discord,
)
from config import settings
from core.router.circuit_breaker import circuit_breaker_registry


class DiscordResetButtonView(discord.ui.View):
    """Interactive Discord UI View containing a Reset Circuit button."""

    def __init__(self, model_id: str, timeout: float = 86400) -> None:
        super().__init__(timeout=timeout)
        self.model_id = model_id

    @discord.ui.button(label="🔌 Reset Circuit", style=discord.ButtonStyle.danger, custom_id="reset_cb_button")
    async def reset_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button[Any]) -> None:
        """Handler when user clicks 'Reset Circuit' button on Discord message."""
        cb = circuit_breaker_registry.get(self.model_id)
        cb.reset()

        button.disabled = True
        button.label = "✅ Circuit Reset"
        button.style = discord.ButtonStyle.success

        embed = discord.Embed(
            title="✅ Circuit Breaker Reset!",
            description=f"Circuit Breaker for `{self.model_id}` has been reset back to `CLOSED`.",
            color=0x2ECC71,  # Green
        )
        embed.add_field(name="🔌 Provider / Model", value=f"`{self.model_id}`", inline=False)
        embed.add_field(name="📊 Status", value="`CLOSED` (Requests Allowed)", inline=True)
        embed.set_footer(text="Claude Code Proxy Gateway • Resiliency System")

        await interaction.response.edit_message(embed=embed, view=self)


class DiscordBotClient(discord.Client):
    """Discord Client handling messages, slash/prefix commands, and threads."""

    def __init__(self, adapter: "DiscordBotAdapter", *args: Any, **kwargs: Any) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(*args, intents=intents, **kwargs)
        self.adapter = adapter

    async def on_ready(self) -> None:
        logger.info(f"DiscordBotAdapter logged in as {self.user}")

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        allowed = [c.strip() for c in settings.ALLOWED_DISCORD_CHANNELS.split(",") if c.strip()]
        if not allowed or str(message.channel.id) not in allowed:
            return

        content = message.content.strip()

        # Handle !start or !help
        if content in ("!start", "!help"):
            help_text = (
                "🛠️ **Claude Code Proxy Gateway Bot Commands**\n\n"
                "• `!status` - View configuration and Circuit Breaker states\n"
                "• `!reset_circuit <model_id|provider>` - Reset Circuit Breaker for model/provider to `CLOSED`\n"
                "• `!set_model <KEY> <PROVIDER/MODEL>` - Update model mapping dynamically\n"
                "• `!run <command>` - Execute bash command in workspace\n"
                "• `!help` - Display this help message"
            )
            await message.reply(help_text)

        # Handle !status
        elif content.startswith("!status"):
            status_text = format_status_overview_discord()
            await message.reply(status_text)

        # Handle !reset_circuit <model_id>
        elif content.startswith("!reset_circuit"):
            parts = content.split(maxsplit=1)
            if len(parts) < 2:
                await message.reply("⚠️ Usage: `!reset_circuit <model_id|provider_name>`")
                return

            target = parts[1].strip()
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

            models_str = ", ".join(f"`{m}`" for m in reset_models)
            await message.reply(
                "✅ **Circuit Breaker Reset!**\n\n"
                f"🔌 **Reset Model(s):** {models_str}\n"
                "📊 **New Status:** `CLOSED` (Requests Allowed)"
            )

        # Handle !set_model
        elif content.startswith("!set_model"):
            parts = content.split(maxsplit=2)
            if len(parts) < 3:
                await message.reply(
                    "⚠️ Usage: `!set_model <MODEL_KEY> <PROVIDER/MODEL>`\n"
                    "Example: `!set_model MODEL_SONNET open_router/openai/gpt-4o`"
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

            status_msg = await message.reply(f"⏳ Executing `{cmd}` in `{workspace}`...")

            try:
                loop = asyncio.get_event_loop()
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
                if len(output) > 1900:
                    output = output[:1800] + "\n...(truncated)"

                await status_msg.edit(content=f"```\n{output}\n```")
            except subprocess.TimeoutExpired:
                await status_msg.edit(content="❌ Command execution timed out (60s limit).")
            except Exception as e:
                await status_msg.edit(content=f"❌ Execution error: {e}")


class DiscordBotAdapter(BaseBotAdapter):
    """Adapter encapsulating Discord Bot operations via discord.py."""

    def __init__(self) -> None:
        self.token = settings.DISCORD_BOT_TOKEN.strip()
        self.client: DiscordBotClient | None = None
        self.session_threads: dict[str, int] = {}
        self._bg_task: asyncio.Task[None] | None = None

    @property
    def platform_name(self) -> str:
        return "discord"

    async def start(self) -> None:
        """Launch Discord client in background task."""
        if not self.token:
            logger.info("Discord Bot token not provided. Skipping Discord adapter.")
            return

        try:
            self.client = DiscordBotClient(adapter=self)
            self._bg_task = asyncio.create_task(self.client.start(self.token))
            logger.info("DiscordBotAdapter background task scheduled.")
        except Exception as e:
            logger.error(f"Failed to start DiscordBotAdapter: {e}")

    async def stop(self) -> None:
        """Close Discord client connection cleanly."""
        if self.client and not self.client.is_closed():
            try:
                await self.client.close()
                logger.info("DiscordBotAdapter stopped cleanly.")
            except Exception as e:
                logger.error(f"Error stopping DiscordBotAdapter: {e}")

    async def send_circuit_breaker_alert(
        self,
        model_id: str,
        reason: str,
        fallback_model: str | None = None,
    ) -> None:
        """Send proactive Circuit Breaker trip alert Embed with interactive Reset button."""
        if not self.client or self.client.is_closed():
            return

        channel_ids = [c.strip() for c in settings.ALLOWED_DISCORD_CHANNELS.split(",") if c.strip()]
        if not channel_ids:
            return

        for cid in channel_ids:
            try:
                channel = self.client.get_channel(int(cid))
                if not channel:
                    channel = await self.client.fetch_channel(int(cid))

                if isinstance(channel, (discord.TextChannel, discord.Thread)):
                    payload = format_circuit_breaker_alert_discord(model_id, reason, fallback_model)
                    embed = discord.Embed(
                        title=payload["title"],
                        description=payload["description"],
                        color=payload["color"],
                    )
                    for f in payload["fields"]:
                        embed.add_field(name=f["name"], value=f["value"], inline=f["inline"])
                    embed.set_footer(text=payload["footer"]["text"])

                    view = DiscordResetButtonView(model_id=model_id)
                    await channel.send(embed=embed, view=view)
                    logger.info(f"Proactive alert Embed sent to Discord channel {cid} for model {model_id}")
            except Exception as e:
                logger.warning(f"Failed to send Discord proactive alert to channel {cid}: {e}")

    async def send_to_thread(self, session_id: str, title: str, text: str) -> None:
        """Send message update directly to a Discord thread."""
        if not self.client or self.client.is_closed():
            return

        channel_ids = [c.strip() for c in settings.ALLOWED_DISCORD_CHANNELS.split(",") if c.strip()]
        if not channel_ids:
            return

        cid = int(channel_ids[0])
        try:
            thread_id = self.session_threads.get(session_id)
            if not thread_id:
                channel = self.client.get_channel(cid) or await self.client.fetch_channel(cid)
                if isinstance(channel, discord.TextChannel):
                    thread = await channel.create_thread(
                        name=f"Session {session_id[:8]}: {title[:40]}",
                        auto_archive_duration=60,
                    )
                    thread_id = thread.id
                    self.session_threads[session_id] = thread_id

            if thread_id:
                thread = self.client.get_channel(thread_id) or await self.client.fetch_channel(thread_id)
                if isinstance(thread, (discord.Thread, discord.TextChannel)):
                    await thread.send(text)
        except Exception as e:
            logger.warning(f"Failed to send Discord thread message: {e}")
