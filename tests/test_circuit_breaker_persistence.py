"""Unit tests for persistent Circuit Breaker state machine."""

import os
import time

from core.router.circuit_breaker import (
    STORAGE_FILE,
    CircuitBreakerRegistry,
    CircuitState,
)


def test_circuit_breaker_started_at_and_expired_at_persistence() -> None:
    """Test that started_at and expired_at are persisted to file and restored on restart."""
    # Ensure clean file state
    if os.path.exists(STORAGE_FILE):
        os.remove(STORAGE_FILE)

    registry1 = CircuitBreakerRegistry()
    cb1 = registry1.get("test_provider/persist_model")

    now = time.time()
    # Trip breaker for 300 seconds (5 min)
    cb1.trip_or_extend("Test persistent failure")

    assert cb1.state == CircuitState.OPEN
    assert cb1.started_at is not None
    assert cb1.expired_at is not None
    assert cb1.started_at >= now - 5
    assert cb1.expired_at == cb1.started_at + 60.0  # First trip recovery timeout

    # Verify JSON file was created
    assert os.path.exists(STORAGE_FILE)

    # Re-initialize a new registry simulating a server restart
    registry2 = CircuitBreakerRegistry()
    cb2 = registry2.get("test_provider/persist_model")

    # Verify state was restored with exact wall-clock expiration
    assert cb2.state == CircuitState.OPEN
    assert cb2.started_at == cb1.started_at
    assert cb2.expired_at == cb1.expired_at
    assert cb2.is_open() is True

    # Cleanup
    cb1.reset()
    if os.path.exists(STORAGE_FILE):
        os.remove(STORAGE_FILE)
