"""Proactive Asynchronous Rate Limiter, Concurrency Guard, and Queue/Throttle Management for NVIDIA NIM."""

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from loguru import logger

from config import settings
from shared.exceptions import NimQueueTimeoutError


class NimThrottleGuard:
    """Proactive Rate Limiter, Single-Lane Concurrency Guard, and Queue Controller for NVIDIA NIM.

    Guarantees:
    1. Single-Lane Concurrency Guard: Max 1 active connection to NVIDIA NIM (serialized execution).
    2. Sliding Window Throttling: Max 38 RPM per 60-second window with proactive async sleep.
    3. Queue & Overflow Timeout Control: Maximum 30s queue wait budget before raising NimQueueTimeoutError
       to trigger seamless fallback to secondary providers (e.g. OpenRouter).
    """

    def __init__(
        self,
        rpm_limit: int | None = None,
        window_seconds: float | None = None,
        max_queue_wait: float | None = None,
    ) -> None:
        self._rpm_limit = rpm_limit
        self._window_seconds = window_seconds
        self._max_queue_wait = max_queue_wait

        self._concurrency_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._timestamps: list[float] = []

    @property
    def rpm_limit(self) -> int:
        if self._rpm_limit is not None:
            return self._rpm_limit
        with contextlib.suppress(Exception):
            nim_cfg = settings.get_provider_config("nvidia_nim")
            if nim_cfg and "rpm" in nim_cfg and nim_cfg["rpm"] is not None:
                return int(nim_cfg["rpm"])
        return settings.NVIDIA_NIM_SAFE_RPM

    @property
    def window_seconds(self) -> float:
        return self._window_seconds if self._window_seconds is not None else float(settings.NVIDIA_NIM_WINDOW_SECONDS)

    @property
    def max_queue_wait(self) -> float:
        return self._max_queue_wait if self._max_queue_wait is not None else float(settings.NVIDIA_NIM_MAX_QUEUE_WAIT)

    @asynccontextmanager
    async def acquire(
        self,
        model_name: str = "nvidia_nim",
        custom_max_queue_wait: float | None = None,
    ) -> AsyncGenerator[None, None]:
        """Acquire single-lane concurrency lock and sliding window rate limit slot.

        Raises NimQueueTimeoutError if queue wait or throttle sleep exceeds max queue timeout.
        """
        start_time = time.monotonic()
        timeout_budget = custom_max_queue_wait if custom_max_queue_wait is not None else self.max_queue_wait

        # Phase 1: Single-Lane Concurrency Guard
        remaining = timeout_budget - (time.monotonic() - start_time)
        if remaining <= 0:
            logger.warning(
                "NVIDIA NIM queue timeout before acquiring concurrency lock for model '%s'",
                model_name,
            )
            raise NimQueueTimeoutError(
                model_name=model_name,
                waited_seconds=round(time.monotonic() - start_time, 2),
                max_queue_wait=timeout_budget,
            )

        try:
            await asyncio.wait_for(self._concurrency_lock.acquire(), timeout=remaining)
        except TimeoutError as err:
            waited = time.monotonic() - start_time
            logger.warning(
                "NVIDIA NIM queue wait exceeded %.1fs timeout for model '%s'",
                timeout_budget,
                model_name,
            )
            raise NimQueueTimeoutError(
                model_name=model_name,
                waited_seconds=round(waited, 2),
                max_queue_wait=timeout_budget,
            ) from err

        try:
            # Phase 2: Sliding Window Throttling (38 RPM / 60s)
            while True:
                now = time.monotonic()
                sleep_needed = 0.0

                async with self._state_lock:
                    # Clean up timestamps older than window_seconds
                    self._timestamps = [
                        ts for ts in self._timestamps if now - ts < self.window_seconds
                    ]
                    if len(self._timestamps) < self.rpm_limit:
                        self._timestamps.append(now)
                        break

                    # Queue is full for current 60s sliding window
                    oldest_ts = self._timestamps[0]
                    sleep_needed = (oldest_ts + self.window_seconds) - now

                waited_so_far = time.monotonic() - start_time
                remaining_budget = timeout_budget - waited_so_far

                if remaining_budget <= 0 or sleep_needed > remaining_budget:
                    total_expected_wait = waited_so_far + max(0.0, sleep_needed)
                    logger.warning(
                        "NVIDIA NIM sliding window throttle delay (%.2fs) exceeds remaining queue budget (%.2fs) for '%s'",
                        sleep_needed,
                        remaining_budget,
                        model_name,
                    )
                    raise NimQueueTimeoutError(
                        model_name=model_name,
                        waited_seconds=round(total_expected_wait, 2),
                        max_queue_wait=timeout_budget,
                    )

                logger.info(
                    "NVIDIA NIM sliding window full (%d/%d RPM). Throttling request for %.2fs...",
                    len(self._timestamps),
                    self.rpm_limit,
                    sleep_needed,
                )
                await asyncio.sleep(max(0.01, sleep_needed))

            # Phase 3: Single-lane execution lock held across yield
            yield
        finally:
            if self._concurrency_lock.locked():
                self._concurrency_lock.release()

    def reset(self) -> None:
        """Reset state for testing."""
        self._timestamps.clear()
        if self._concurrency_lock.locked():
            with contextlib.suppress(RuntimeError):
                self._concurrency_lock.release()


nim_throttle_guard = NimThrottleGuard()
