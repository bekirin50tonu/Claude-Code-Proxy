"""Base atomic parser abstract class (ABC)."""

from abc import ABC, abstractmethod
from typing import Any

from shared.schemas.anthropic import SSEBaseEvent


class BaseAtomicParser(ABC):
    """Abstract base class for stateful stream parsers in the Atomic layer."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal parser state."""

    @abstractmethod
    async def process_chunk(self, chunk: dict[str, Any] | str) -> list[SSEBaseEvent]:
        """Process an incoming chunk or string fragment and return zero or more SSE event objects."""

    @abstractmethod
    async def flush(self) -> list[SSEBaseEvent]:
        """Flush any remaining buffered state at stream end."""
