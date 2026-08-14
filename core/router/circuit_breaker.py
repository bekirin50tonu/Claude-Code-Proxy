"""Circuit Breaker — state machine fault isolation in Core layer."""

import asyncio
import time

from models import CircuitState

TIMEOUT_STEPS = [60.0, 300.0, 600.0, 900.0, 1800.0, 3600.0]  # 1m, 5m, 10m, 15m, 30m, 60m


class CircuitBreaker:
    """Model-based CLOSED -> OPEN -> HALF_OPEN state machine."""

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
        self._opened_at_wall: str | None = None
        self._reopens_at_wall: str | None = None
        self._last_failure_reason: str = ""
        self._manual_timeout_index: int = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def reset(self) -> None:
        """Reset circuit breaker back to CLOSED state and clear failure counter and timeout."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
        self._opened_at_wall = None
        self._reopens_at_wall = None
        self._last_failure_reason = ""
        self._manual_timeout_index = 0
        self.recovery_timeout = 60.0
        provider = self.model_id.split("/", 1)[0] if "/" in self.model_id else "nvidia_nim"
        from core.router.daily_tracker import daily_request_tracker
        daily_request_tracker.reset_provider(provider)

    def trip_or_extend(self, reason: str = "Manually forced OPEN via Dashboard") -> float:
        """Trip circuit breaker into OPEN state or extend timeout step-wise (1m -> 5m -> 10m -> 15m -> 30m -> 60m)."""
        if self._state == CircuitState.OPEN:
            self._manual_timeout_index = min(
                self._manual_timeout_index + 1, len(TIMEOUT_STEPS) - 1
            )
        else:
            self._manual_timeout_index = 0

        new_timeout = TIMEOUT_STEPS[self._manual_timeout_index]
        self._state = CircuitState.OPEN
        self.recovery_timeout = new_timeout
        self._opened_at = time.monotonic()
        self._opened_at_wall = time.strftime("%H:%M:%S")
        self._reopens_at_wall = time.strftime(
            "%H:%M:%S", time.localtime(time.time() + new_timeout)
        )
        self._last_failure_reason = f"{reason} ({int(new_timeout // 60)} min timeout)"
        return new_timeout

    def force_open(self, reason: str = "Upstream error / EOL / Not Found") -> None:
        """Force circuit breaker into OPEN state with given failure reason."""
        self.trip_or_extend(reason=reason)

    def is_open(self) -> bool:
        """Return True when requests should be blocked (OPEN state)."""
        provider = self.model_id.split("/", 1)[0] if "/" in self.model_id else "nvidia_nim"
        from core.router.daily_tracker import daily_request_tracker
        exceeded, cur, limit = daily_request_tracker.is_exceeded(provider)

        if exceeded:
            if self._state != CircuitState.OPEN or "RPD" not in self._last_failure_reason:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._opened_at_wall = time.strftime("%H:%M:%S")
                self._reopens_at_wall = "Midnight (RPD Reset)"
                self._last_failure_reason = f"Daily RPD limit reached ({cur}/{limit})"
            return True

        if self._state == CircuitState.OPEN:
            if self._opened_at and (time.monotonic() - self._opened_at) >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                return False
            return True
        return False

    async def record_success(self) -> None:
        async with self._lock:
            self.reset()


    async def record_failure(self, reason: str = "Upstream execution failure") -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_reason = reason
            if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._opened_at_wall = time.strftime("%H:%M:%S")
                self._reopens_at_wall = time.strftime(
                    "%H:%M:%S", time.localtime(time.time() + self.recovery_timeout)
                )
                self._failure_count = 0

    def status_dict(self) -> dict[str, object]:
        elapsed = (time.monotonic() - self._opened_at) if self._opened_at else None
        remaining: float | None = None
        if self._state == CircuitState.OPEN and elapsed is not None:
            remaining = max(0.0, self.recovery_timeout - elapsed)
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "opened_at_wall": self._opened_at_wall,
            "reopens_at_wall": self._reopens_at_wall,
            "recovery_remaining_s": round(remaining, 1) if remaining is not None else None,
            "last_failure_reason": self._last_failure_reason or "None",
        }


class CircuitBreakerRegistry:
    """Registry maintaining a CircuitBreaker per model_id."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, model_id: str) -> CircuitBreaker:
        if model_id not in self._breakers:
            self._breakers[model_id] = CircuitBreaker(model_id)
        return self._breakers[model_id]

    def all_statuses(self) -> dict[str, dict[str, object]]:
        return {mid: cb.status_dict() for mid, cb in self._breakers.items()}


circuit_breaker_registry = CircuitBreakerRegistry()
