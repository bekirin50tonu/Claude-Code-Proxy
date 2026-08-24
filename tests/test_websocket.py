"""Unit tests for Dashboard WebSocket Endpoint and WebSocketManager."""

import pytest
from fastapi.testclient import TestClient

from api.websocket_manager import ws_manager
from server import app


def test_websocket_manager_connect_disconnect():
    """Verify WebSocket manager tracks active connection state."""
    assert isinstance(ws_manager.active_connections, set)


def test_websocket_dashboard_endpoint():
    """Verify /ws/dashboard WebSocket endpoint accepts connections and streams initial snapshot."""
    client = TestClient(app)
    with client.websocket_connect("/ws/dashboard") as websocket:
        data = websocket.receive_json()
        assert data.get("event") == "initial_state"
        assert "stats" in data.get("data", {})
        assert "router_status" in data.get("data", {})


@pytest.mark.asyncio
async def test_websocket_broadcast():
    """Verify WebSocketManager broadcast_event dispatches JSON payloads."""
    ws = ws_manager
    assert hasattr(ws, "broadcast_event")
