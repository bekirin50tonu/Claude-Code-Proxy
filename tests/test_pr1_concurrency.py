"""Unit and concurrency tests for PR #1 critical fixes."""

import asyncio
import os

import pytest

from core.gateway import SlidingWindowRateLimiter
from core.router.circuit_breaker import STORAGE_FILE, circuit_breaker_registry
from providers.openai import _select_key


@pytest.mark.asyncio
async def test_sliding_window_rate_limiter_deque() -> None:
    limiter = SlidingWindowRateLimiter(limit=3, window=0.5)

    assert await limiter.acquire() is True
    assert await limiter.acquire() is True
    assert await limiter.acquire() is True
    assert await limiter.acquire() is False  # 4th request within 0.5s window should fail

    await asyncio.sleep(0.6)
    assert await limiter.acquire() is True  # Window expired, acquisition allowed


@pytest.mark.asyncio
async def test_key_rotation_concurrency() -> None:
    raw_keys = "key_a, key_b, key_c"

    tasks = [_select_key(raw_keys, "test_provider") for _ in range(30)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 30
    assert results.count("key_a") == 10
    assert results.count("key_b") == 10
    assert results.count("key_c") == 10


@pytest.mark.asyncio
async def test_circuit_breaker_atomic_persistence() -> None:
    cb = circuit_breaker_registry.get("test_provider/atomic_model")
    await cb.trip_or_extend("Testing atomic persistence")

    circuit_breaker_registry.save_to_file(force=True)

    assert os.path.exists(STORAGE_FILE)
    # Ensure no leftover temp files
    parent_dir = os.path.dirname(STORAGE_FILE)
    tmp_files = [f for f in os.listdir(parent_dir) if f.startswith(os.path.basename(STORAGE_FILE) + ".tmp")]
    assert len(tmp_files) == 0

    await cb.reset()
