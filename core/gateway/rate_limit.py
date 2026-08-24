"""Sliding window rate limiter and global concurrency control for Core Gateway."""

import asyncio
import time
from collections import deque

from config import settings


class SlidingWindowRateLimiter:
    """Sliding window rate limiter tracking request timestamps in memory."""

    def __init__(self, limit: int, window: float | int):
        self.limit = limit
        self.window = float(window)
        self.requests: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Attempt to acquire a slot within the sliding window."""
        async with self.lock:
            now = time.monotonic()
            while self.requests and (now - self.requests[0]) >= self.window:
                self.requests.popleft()
            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            return False


gateway_rate_limiter = SlidingWindowRateLimiter(
    settings.PROVIDER_RATE_LIMIT, settings.PROVIDER_RATE_WINDOW
)
rate_limiter = gateway_rate_limiter
concurrency_semaphore = asyncio.Semaphore(settings.PROVIDER_MAX_CONCURRENCY)
