"""Unit tests for NVIDIA NIM Proactive Async Rate Limiter & Concurrency Guard."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from atomic.guards.nim_guard import NimThrottleGuard
from shared.exceptions import NimQueueTimeoutError


@pytest.mark.asyncio
async def test_nim_single_lane_concurrency_guard():
    """Verify that requests to NVIDIA NIM execute serially (concurrency = 1)."""
    guard = NimThrottleGuard(rpm_limit=10, window_seconds=60.0, max_queue_wait=5.0)
    execution_order: list[str] = []
    active_count = 0
    max_observed_concurrency = 0

    async def worker(worker_id: str):
        nonlocal active_count, max_observed_concurrency
        async with guard.acquire("nvidia_nim/test"):
            active_count += 1
            if active_count > max_observed_concurrency:
                max_observed_concurrency = active_count
            execution_order.append(f"start_{worker_id}")
            await asyncio.sleep(0.05)
            execution_order.append(f"end_{worker_id}")
            active_count -= 1

    await asyncio.gather(
        worker("A"),
        worker("B"),
        worker("C"),
    )

    assert max_observed_concurrency == 1
    assert len(execution_order) == 6


@pytest.mark.asyncio
async def test_nim_sliding_window_rate_limiter_throttling():
    """Verify that sliding window throttling delays requests when 38 RPM limit is reached."""
    guard = NimThrottleGuard(rpm_limit=2, window_seconds=0.2, max_queue_wait=5.0)

    start_time = time.monotonic()

    # Request 1 & Request 2 take the 2 available slots in current window
    async with guard.acquire("nvidia_nim/test"):
        pass
    async with guard.acquire("nvidia_nim/test"):
        pass

    # Request 3 should throttle and sleep until window clears (~0.2s)
    async with guard.acquire("nvidia_nim/test"):
        pass

    elapsed = time.monotonic() - start_time
    assert elapsed >= 0.18, f"Expected throttling sleep (~0.2s), but completed in {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_nim_queue_timeout_raises_error():
    """Verify that NimQueueTimeoutError is raised when queue wait time exceeds max_queue_wait."""
    guard = NimThrottleGuard(rpm_limit=2, window_seconds=60.0, max_queue_wait=0.1)

    # Acquire lock and hold it for longer than max_queue_wait (0.2s > 0.1s)
    async def slow_holder():
        async with guard.acquire("nvidia_nim/test"):
            await asyncio.sleep(0.25)

    async def waiting_request():
        await asyncio.sleep(0.02)  # ensure slow_holder gets lock first
        with pytest.raises(NimQueueTimeoutError) as exc_info:
            async with guard.acquire("nvidia_nim/test"):
                pass
        assert "queue timeout" in str(exc_info.value).lower()

    await asyncio.gather(slow_holder(), waiting_request())


def test_gateway_nim_queue_timeout_triggers_openrouter_fallback():
    """Integration test: Gateway redirects to OpenRouter fallback model when NVIDIA NIM times out in queue."""
    from server import app

    client = TestClient(app)
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello fallback test"}],
        "stream": False,
    }

    # Mock provider.complete:
    # First call (for nvidia_nim model): raises NimQueueTimeoutError
    # Second call (for open_router fallback model): succeeds
    mock_provider = AsyncMock()

    async def mock_complete(model, **kwargs):
        if model.startswith("nvidia_nim/"):
            raise NimQueueTimeoutError(model_name=model, waited_seconds=30.0, max_queue_wait=30.0)
        return (
            {
                "id": "msg_fallback_123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from OpenRouter fallback!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {"x-ratelimit-remaining-requests": "100"},
        )

    mock_provider.complete.side_effect = mock_complete

    with (
        patch("core.gateway._get_provider", return_value=mock_provider),
        patch("core.gateway._check_auth", return_value=True),
    ):
        response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    body = response.json()
    assert body["content"][0]["text"] == "Hello from OpenRouter fallback!"
    # Verify complete was called twice: once for nvidia_nim (which timed out) and once for open_router fallback
    assert mock_provider.complete.call_count == 2
