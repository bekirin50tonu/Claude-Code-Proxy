"""
Circuit Breaker — per-model fault isolation.

States
------
CLOSED   : Normal operation. Requests pass through.
OPEN     : Too many failures. Requests are rejected immediately.
HALF_OPEN: Recovery probe state. One request is let through;
           success → CLOSED, failure → OPEN again.

Transitions
-----------
CLOSED  + failure_threshold consecutive failures  → OPEN
OPEN    + recovery_timeout seconds elapsed        → HALF_OPEN
HALF_OPEN + success                               → CLOSED
HALF_OPEN + failure                               → OPEN (reset timer)
"""

import asyncio
import time
from enum import StrEnum


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        model_id: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.model_id = model_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def force_open(self) -> None:
        """Force circuit breaker into OPEN state."""
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()

    def is_open(self) -> bool:
        """Return True when requests should be blocked (OPEN state)."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed → transition to HALF_OPEN
            if (
                self._opened_at
                and (time.monotonic() - self._opened_at) >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                return False  # Allow one probe request through
            return True
        return False

    async def record_success(self) -> None:
        async with self._lock:
            self._failure_count = 0
            self._opened_at = None
            self._state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            if (
                self._state == CircuitState.HALF_OPEN
                or self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._failure_count = 0  # Reset for next cycle

    def status_dict(self) -> dict[str, object]:
        elapsed = (time.monotonic() - self._opened_at) if self._opened_at else None
        remaining: float | None = None
        if self._state == CircuitState.OPEN and elapsed is not None:
            remaining = max(0.0, self.recovery_timeout - elapsed)
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "recovery_remaining_s": round(remaining, 1)
            if remaining is not None
            else None,
        }


class CircuitBreakerRegistry:
    """Singleton registry — one CircuitBreaker per model_id."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, model_id: str) -> CircuitBreaker:
        if model_id not in self._breakers:
            self._breakers[model_id] = CircuitBreaker(model_id)
        return self._breakers[model_id]

    def all_statuses(self) -> dict[str, dict[str, object]]:
        return {mid: cb.status_dict() for mid, cb in self._breakers.items()}


# Module-level singleton
circuit_breaker_registry = CircuitBreakerRegistry()
