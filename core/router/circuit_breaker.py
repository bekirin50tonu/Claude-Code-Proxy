"""Circuit Breaker — state machine fault isolation in Core layer with persistent storage."""

import asyncio
import os
import time
from typing import Any

import yaml
from loguru import logger

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

STORAGE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".data", "circuit_breakers.yaml")
)


class CircuitBreaker:
    """Model-based CLOSED -> OPEN -> HALF_OPEN state machine with wall-clock persistence."""

    def __init__(
        self,
        model_id: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        registry: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.failure_threshold = failure_threshold
        self.initial_recovery_timeout = recovery_timeout
        self.recovery_timeout = recovery_timeout
        self._registry = registry

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self.started_at: float | None = None  # Wall-clock epoch time when OPEN started
        self.expired_at: float | None = None  # Wall-clock epoch time when OPEN expires
        self._opened_at_wall: str | None = None
        self._reopens_at_wall: str | None = None
        self._last_failure_reason: str = ""
        self._manual_timeout_index = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def _opened_at(self) -> float | None:
        return self.started_at

    @_opened_at.setter
    def _opened_at(self, val: float | None) -> None:
        self.started_at = val

    def _get_registry(self) -> Any:
        return self._registry or circuit_breaker_registry

    def reset(self, save: bool = True) -> None:
        """Reset circuit breaker back to CLOSED state and clear failure counter and timeout."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self.started_at = None
        self.expired_at = None
        self._opened_at_wall = None
        self._reopens_at_wall = None
        self._last_failure_reason = ""
        self._manual_timeout_index = 0
        self.recovery_timeout = self.initial_recovery_timeout
        provider = self.model_id.split("/", 1)[0] if "/" in self.model_id else "nvidia_nim"
        from core.router.daily_tracker import daily_request_tracker
        daily_request_tracker.reset_provider(provider)
        if save:
            self._get_registry().save_to_file()

    def trip_or_extend(self, reason: str = "Manually forced OPEN via Dashboard", save: bool = True) -> float:
        """Trip circuit breaker into OPEN state or extend timeout step-wise."""
        now_wall = time.time()
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
        self.started_at = now_wall
        self.expired_at = now_wall + new_timeout
        self._opened_at_wall = time.strftime("%H:%M:%S", time.localtime(now_wall))
        self._reopens_at_wall = time.strftime("%H:%M:%S", time.localtime(self.expired_at))
        mins = int(new_timeout // 60)
        self._last_failure_reason = f"{reason} ({mins} min timeout)"

        if save:
            reg = self._get_registry()
            reg.save_to_file()
            reg.notify_trip(self.model_id, self._last_failure_reason)

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
                now_wall = time.time()
                self.started_at = now_wall
                self.expired_at = None
                self._opened_at_wall = time.strftime("%H:%M:%S", time.localtime(now_wall))
                self._reopens_at_wall = "Midnight (RPD Reset)"
                self._last_failure_reason = f"Daily RPD limit reached ({cur}/{limit})"
                reg = self._get_registry()
                reg.save_to_file()
                reg.notify_trip(self.model_id, self._last_failure_reason)
            return True

        if self._state == CircuitState.OPEN:
            now_wall = time.time()
            if self.expired_at and now_wall >= self.expired_at:
                self._state = CircuitState.HALF_OPEN
                self._get_registry().save_to_file()
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
        now_wall = time.time()
        remaining: float | None = None
        if self._state == CircuitState.OPEN and self.expired_at is not None:
            remaining = max(0.0, self.expired_at - now_wall)

        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "started_at": self.started_at,
            "expired_at": self.expired_at,
            "opened_at_wall": self._opened_at_wall,
            "reopens_at_wall": self._reopens_at_wall,
            "recovery_remaining_s": round(remaining, 1) if remaining is not None else None,
            "last_failure_reason": self._last_failure_reason or "None",
        }


class CircuitBreakerRegistry:
    """Registry maintaining a CircuitBreaker per model_id with persistent file storage and trip notifications."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._trip_callbacks: list[object] = []
        self._load_from_file()
        self.save_to_file()

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
            self._breakers[model_id] = CircuitBreaker(model_id, registry=self)
        return self._breakers[model_id]

    def all_statuses(self) -> dict[str, dict[str, object]]:
        return {mid: cb.status_dict() for mid, cb in self._breakers.items()}

    def save_to_file(self, force: bool = False) -> None:
        """Persist active OPEN circuit breakers to storage file."""
        if "PYTEST_CURRENT_TEST" in os.environ and not force:
            return
        try:
            os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)
            data: dict[str, Any] = {}
            for mid, cb in self._breakers.items():
                if cb.state == CircuitState.OPEN:
                    data[mid] = {
                        "state": cb.state.value,
                        "failure_count": cb._failure_count,
                        "started_at": cb.started_at,
                        "expired_at": cb.expired_at,
                        "recovery_timeout": cb.recovery_timeout,
                        "manual_timeout_index": cb._manual_timeout_index,
                        "last_failure_reason": cb._last_failure_reason,
                        "opened_at_wall": cb._opened_at_wall,
                        "reopens_at_wall": cb._reopens_at_wall,
                    }
            with open(STORAGE_FILE, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
        except Exception as e:
            logger.warning(f"Failed to persist circuit breaker states: {e}")

    def _load_from_file(self, force: bool = False) -> None:
        """Load and restore persisted OPEN circuit breaker states from storage file."""
        if ("PYTEST_CURRENT_TEST" in os.environ and not force) or not os.path.exists(STORAGE_FILE):
            return
        try:
            with open(STORAGE_FILE, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            now_wall = time.time()
            for mid, rec in data.items():
                expired_at = rec.get("expired_at")
                if expired_at and now_wall < expired_at:
                    cb = self.get(mid)
                    cb._state = CircuitState.OPEN
                    cb.started_at = rec.get("started_at", now_wall)
                    cb.expired_at = expired_at
                    cb.recovery_timeout = rec.get("recovery_timeout", 60.0)
                    cb._manual_timeout_index = rec.get("manual_timeout_index", 0)
                    cb._last_failure_reason = rec.get("last_failure_reason", "Restored from persistent storage")
                    cb._opened_at_wall = rec.get("opened_at_wall", time.strftime("%H:%M:%S", time.localtime(cb.started_at)))
                    cb._reopens_at_wall = rec.get("reopens_at_wall", time.strftime("%H:%M:%S", time.localtime(expired_at)))
                    logger.info(
                        f"CircuitBreakerRegistry: Restored OPEN breaker for '{mid}' (Expires in {int(expired_at - now_wall)}s)"
                    )
        except Exception as e:
            logger.warning(f"Failed to load persistent circuit breaker states: {e}")


circuit_breaker_registry = CircuitBreakerRegistry()
