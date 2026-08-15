"""Bot platform management package for Claude Code Proxy Gateway."""

from bot.factory import bot_factory


async def start_all_bots() -> None:
    """Start all active bot platform adapters."""
    await bot_factory.start_all()


async def stop_all_bots() -> None:
    """Stop all active bot platform adapters cleanly."""
    await bot_factory.stop_all()


async def send_circuit_breaker_alert(
    model_id: str,
    reason: str,
    fallback_model: str | None = None,
) -> None:
    """Send proactive Circuit Breaker trip alert notification across all platforms."""
    await bot_factory.broadcast_circuit_breaker_alert(model_id, reason, fallback_model)


__all__ = [
    "bot_factory",
    "send_circuit_breaker_alert",
    "start_all_bots",
    "stop_all_bots",
]
