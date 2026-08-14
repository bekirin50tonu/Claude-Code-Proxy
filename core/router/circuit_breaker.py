"""Circuit Breaker — state machine fault isolation in Core layer."""

import asyncio
import time

from models import CircuitState

TIMEOUT_STEPS = [
    60.0,      # 1st trip: 1m
    120.0,     # 2nd trip: 2m
    300.0,     # 3rd trip: 5m
    600.0,     # 4th trip: 10m
    1800.0,    # 5th trip: 30m
    3600.0,    # 6th trip: 60m (1 hour)
    7200.0,    # 7th trip: 120m (2 hours)
    14400.0,   # 8th trip: 240m (4 hours)
    28800.0,   # 9th trip: 480m (8 hours)
    86400.0,   # 10th trip+: 1440m (24 hours / 1 day)
]


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
        self.initial_recovery_timeout = recovery_timeout
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._opened_at_wall: str | None = None
        self._reopens_at_wall: str | None = None
        self._last_failure_reason: str = ""
        self._manual_timeout_index = 0
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
        self.recovery_timeout = self.initial_recovery_timeout
        provider = self.model_id.split("/", 1)[0] if "/" in self.model_id else "nvidia_nim"
        from core.router.daily_tracker import daily_request_tracker
        daily_request_tracker.reset_provider(provider)

    def trip_or_extend(self, reason: str = "Manually forced OPEN via Dashboard") -> float:
        """Trip circuit breaker into OPEN state or extend timeout step-wise (1m -> 2m -> 5m -> 10m -> 30m -> 60m -> 120m -> 240m -> 480m -> 1440m)."""
        if self._state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
            self._manual_timeout_index = min(
                self._manual_timeout_index + 1, len(TIMEOUT_STEPS) - 1
            )
            new_timeout = TIMEOUT_STEPS[self._manual_timeout_index]
        else:
            self._manual_timeout_index = 0
            new_timeout = self.recovery_timeout

        self._state = CircuitState.OPEN
        self.recovery_timeout = new_timeout
        self._opened_at = time.monotonic()
        self._opened_at_wall = time.strftime("%H:%M:%S")
        self._reopens_at_wall = time.strftime(
            "%H:%M:%S", time.localtime(time.time() + new_timeout)
        )
        mins = int(new_timeout // 60)
        self._last_failure_reason = f"{reason} ({mins} min timeout)"
        circuit_breaker_registry.notify_trip(self.model_id, self._last_failure_reason)
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
                circuit_breaker_registry.notify_trip(self.model_id, self._last_failure_reason)
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
                self.trip_or_extend(reason=reason)
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
    """Registry maintaining a CircuitBreaker per model_id with trip notification callbacks."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._trip_callbacks: list[object] = []

    def register_trip_callback(self, callback: object) -> None:
        """Register a callback function (sync or async) invoked when any circuit breaker trips OPEN."""
        if callback not in self._trip_callbacks:
            self._trip_callbacks.append(callback)

    def notify_trip(self, model_id: str, reason: str) -> None:
        """Dispatch trip event to all registered listeners."""
        for cb in self._trip_callbacks:
            try:
                if callable(cb):
                    res = cb(model_id, reason)
                    if asyncio.iscoroutine(res):
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(res)
                        except RuntimeError:
                            pass
            except Exception:
                pass

    def get(self, model_id: str) -> CircuitBreaker:
        if model_id not in self._breakers:
            self._breakers[model_id] = CircuitBreaker(model_id)
        return self._breakers[model_id]

    def all_statuses(self) -> dict[str, dict[str, object]]:
        return {mid: cb.status_dict() for mid, cb in self._breakers.items()}


circuit_breaker_registry = CircuitBreakerRegistry()
