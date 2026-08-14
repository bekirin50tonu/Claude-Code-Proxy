"""Unit tests for Requests Per Day (RPD) tracking and Circuit Breaker integration."""

import pytest
from fastapi.testclient import TestClient

from config import settings
from core.router.circuit_breaker import circuit_breaker_registry
from core.router.daily_tracker import daily_request_tracker
from core.router.selector import model_selector
from server import app

client = TestClient(app)


def test_daily_request_tracker_increment_and_reset() -> None:
    """Test recording requests, reaching RPD limits, and resetting daily tracker."""
    daily_request_tracker.reset_provider("groq")
    assert daily_request_tracker.get_count("groq") == 0

    count = daily_request_tracker.record_request("groq")
    assert count == 1
    assert daily_request_tracker.get_count("groq") == 1

    daily_request_tracker.reset_provider("groq")
    assert daily_request_tracker.get_count("groq") == 0


def test_circuit_breaker_trips_on_rpd_exceeded() -> None:
    """Test that CircuitBreaker automatically trips OPEN when provider RPD limit is exceeded."""
    model_id = "groq/llama-3.3-70b"
    cb = circuit_breaker_registry.get(model_id)
    cb.reset()
    assert not cb.is_open()

    # Artificially set Groq RPD limit to 2
    settings.PROVIDER_GROQ_RPD = 2
    daily_request_tracker.reset_provider("groq")
    daily_request_tracker.record_request("groq")
    daily_request_tracker.record_request("groq")

    # Now Groq has 2 requests = RPD limit 2 -> cb.is_open() should return True
    assert cb.is_open()
    status = cb.status_dict()
    assert status["state"] == "open"
    assert "Daily RPD limit reached" in str(status["last_failure_reason"])

    # Resetting CB should also clear daily RPD count
    cb.reset()
    assert not cb.is_open()


@pytest.mark.asyncio
async def test_selector_record_outcome_quota_429() -> None:
    """Test that upstream 429 Daily Quota error triggers an RPD circuit trip."""
    model_id = "deepseek/deepseek-chat"
    cb = circuit_breaker_registry.get(model_id)
    cb.reset()

    await model_selector.record_outcome(
        model_id,
        success=False,
        reason="HTTP 429 Daily Quota Exceeded (RPD limit)",
    )

    assert cb.is_open()
    assert "Daily RPD quota exceeded" in cb.status_dict()["last_failure_reason"]

    # Clean up
    cb.reset()


def test_router_status_endpoint_returns_daily_rpd() -> None:
    """Test that GET /api/router-status returns daily_rpd section."""
    resp = client.get("/api/router-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "daily_rpd" in data
    daily_rpd = data["daily_rpd"]
    assert "nvidia_nim" in daily_rpd
    assert "rpd_limit" in daily_rpd["nvidia_nim"]
    assert "count" in daily_rpd["nvidia_nim"]
