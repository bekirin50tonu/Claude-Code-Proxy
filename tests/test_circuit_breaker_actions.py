"""Unit tests for Circuit Breaker action endpoints and step-wise timeout extension."""

from fastapi.testclient import TestClient

from core.router.circuit_breaker import CircuitBreaker
from models import CircuitState
from server import app

client = TestClient(app)


def test_circuit_breaker_reset_and_trip_or_extend() -> None:
    cb = CircuitBreaker("test_model_actions")

    # Initially CLOSED
    assert cb.state == CircuitState.CLOSED

    # First trip: should be OPEN with 1 min timeout (60s)
    t1 = cb.trip_or_extend("Manual trip 1")
    assert cb.state == CircuitState.OPEN
    assert t1 == 60.0
    assert cb.recovery_timeout == 60.0

    # Second trip while OPEN: should extend timeout to 2 min (120s)
    t2 = cb.trip_or_extend("Manual trip 2")
    assert cb.state == CircuitState.OPEN
    assert t2 == 120.0
    assert cb.recovery_timeout == 120.0

    # Third trip: extends to 5 min (300s)
    t3 = cb.trip_or_extend("Manual trip 3")
    assert t3 == 300.0

    # Fourth trip: extends to 10 min (600s)
    t4 = cb.trip_or_extend("Manual trip 4")
    assert t4 == 600.0

    # 10th trip (max): extends to 1440 min (86400s / 24h / 1 day)
    for i in range(5, 12):
        t_max = cb.trip_or_extend(f"Manual trip {i}")
    assert t_max == 86400.0

    # Reset: reverts to CLOSED and forgets timeout
    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.recovery_timeout == 60.0
    assert cb._opened_at is None
    assert cb._last_failure_reason == ""


def test_circuit_breaker_api_action_endpoint() -> None:
    model_id = "nvidia_nim/test-action-api"

    # 1. Trip circuit breaker via API
    resp = client.post(
        "/api/circuit-breaker/action",
        json={"model_id": model_id, "action": "trip"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "forced OPEN" in data["message"]
    assert data["circuit_breaker"]["state"] == "open"

    # Second trip extends timeout to 2 min
    resp2 = client.post(
        "/api/circuit-breaker/action",
        json={"model_id": model_id, "action": "close"},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "2 min" in data2["message"]

    # 2. Reset circuit breaker via API
    resp3 = client.post(
        "/api/circuit-breaker/action",
        json={"model_id": model_id, "action": "open"},
    )
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["status"] == "success"
    assert "CLOSED" in data3["message"]
    assert data3["circuit_breaker"]["state"] == "closed"
