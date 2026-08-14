"""
Circuit Breaker Data Models & Enums.

Defines CircuitState enum and CircuitStatus dataclass for per-model fault isolation.
"""

from dataclasses import dataclass
from enum import StrEnum


class CircuitState(StrEnum):
    """Possible operational states for a Circuit Breaker instance."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStatus:
    """Snapshot representation of a Circuit Breaker's state and health."""

    model_id: str
    state: CircuitState
    failure_count: int
    failure_threshold: int
    recovery_timeout_s: float
    opened_at_wall: str | None = None
    reopens_at_wall: str | None = None
    recovery_remaining_s: float = 0.0
    last_failure_reason: str = ""

    @property
    def is_available(self) -> bool:
        return self.state != CircuitState.OPEN
