"""BaseBotAdapter abstract interface for bot platforms in Claude Code Proxy Gateway."""

from abc import ABC, abstractmethod


class BaseBotAdapter(ABC):
    """Abstract base class defining interface contract for messaging bot adapters."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of the bot platform (e.g. 'telegram', 'discord')."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Initialize and start the bot client task asynchronously."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop and shutdown the bot client instance."""
        pass

    @abstractmethod
    async def send_circuit_breaker_alert(
        self,
        model_id: str,
        reason: str,
        fallback_model: str | None = None,
    ) -> None:
        """Send proactive alert when a circuit breaker trips OPEN."""
        pass

    @abstractmethod
    async def send_circuit_breaker_recovery(self, model_id: str) -> None:
        """Send proactive alert when a circuit breaker recovers (CLOSED)."""
        pass

    @abstractmethod
    async def send_to_thread(self, session_id: str, title: str, text: str) -> None:
        """Send an update to a session thread/parent message."""
        pass
