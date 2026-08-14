"""BotFactory pattern coordinator for Telegram and Discord platform adapters."""

import asyncio

from loguru import logger

from bot.base import BaseBotAdapter
from bot.platforms.discord import DiscordBotAdapter
from bot.platforms.telegram import TelegramBotAdapter
from core.router.circuit_breaker import circuit_breaker_registry


class BotFactory:
    """Factory maintaining active bot platform adapters and coordinating broadcasts."""

    def __init__(self) -> None:
        self.adapters: dict[str, BaseBotAdapter] = {}
        self._init_adapters()
        # Register factory broadcast listener with CircuitBreakerRegistry
        circuit_breaker_registry.register_trip_callback(self._on_circuit_breaker_trip_event)

    def _init_adapters(self) -> None:
        """Instantiate available bot adapters."""
        tg = TelegramBotAdapter()
        ds = DiscordBotAdapter()
        self.adapters[tg.platform_name] = tg
        self.adapters[ds.platform_name] = ds

    def get_adapter(self, name: str) -> BaseBotAdapter | None:
        """Retrieve a specific adapter by platform name."""
        return self.adapters.get(name)

    async def start_all(self) -> None:
        """Start all configured bot adapters asynchronously."""
        logger.info("BotFactory initializing and starting all configured bot adapters...")
        for name, adapter in self.adapters.items():
            try:
                await adapter.start()
            except Exception as e:
                logger.error(f"Failed to start bot adapter '{name}': {e}")

    async def stop_all(self) -> None:
        """Stop all running bot adapters cleanly."""
        logger.info("BotFactory shutting down all active bot adapters...")
        for name, adapter in self.adapters.items():
            try:
                await adapter.stop()
            except Exception as e:
                logger.error(f"Error stopping bot adapter '{name}': {e}")

    def _on_circuit_breaker_trip_event(self, model_id: str, reason: str) -> None:
        """Synchronous event callback triggered by CircuitBreakerRegistry on trip OPEN."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast_circuit_breaker_alert(model_id, reason))
        except RuntimeError:
            pass

    async def broadcast_circuit_breaker_alert(
        self,
        model_id: str,
        reason: str = "Upstream error / timeout",
        fallback_model: str | None = None,
    ) -> None:
        """Dispatch proactive Circuit Breaker trip alert to all active bot platforms."""
        for name, adapter in self.adapters.items():
            try:
                await adapter.send_circuit_breaker_alert(model_id, reason, fallback_model)
            except Exception as e:
                logger.warning(f"Error sending proactive alert via '{name}': {e}")

    async def send_to_thread(self, session_id: str, title: str, text: str) -> None:
        """Dispatch session thread message to all active bot platforms."""
        for name, adapter in self.adapters.items():
            try:
                await adapter.send_to_thread(session_id, title, text)
            except Exception as e:
                logger.warning(f"Error sending thread message via '{name}': {e}")


# Singleton BotFactory instance
bot_factory = BotFactory()
