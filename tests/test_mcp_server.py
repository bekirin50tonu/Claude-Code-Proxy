"""Unit tests for MCP Server HTTP endpoints and JSON-RPC tools."""

import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_mcp_initialize(client: TestClient) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("jsonrpc") == "2.0"
    assert data.get("id") == 1
    result = data.get("result", {})
    assert result.get("protocolVersion") == "2024-11-05"
    assert "serverInfo" in result


def test_mcp_tools_list(client: TestClient) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    tools = data.get("result", {}).get("tools", [])
    tool_names = [t.get("name") for t in tools]
    assert "get_models" in tool_names
    assert "set_model_mapping" in tool_names
    assert "get_system_config" in tool_names
    assert "update_system_config" in tool_names
    assert "get_metrics" in tool_names
    assert "control_circuit_breaker" in tool_names


def test_mcp_tool_call_get_models(client: TestClient) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_models",
            "arguments": {},
        },
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    result = data.get("result", {})
    content = result.get("content", [])
    assert len(content) > 0
    assert content[0]["type"] == "text"
    text = content[0]["text"]
    assert "active_client_mappings" in text


def test_mcp_tool_call_get_metrics(client: TestClient) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "get_metrics",
            "arguments": {},
        },
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    result = data.get("result", {})
    content = result.get("content", [])
    assert len(content) > 0
    assert "provider_metrics" in content[0]["text"]


def test_mcp_tool_call_get_system_config(client: TestClient) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "get_system_config",
            "arguments": {},
        },
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    result = data.get("result", {})
    content = result.get("content", [])
    assert len(content) > 0
    assert "configs" in content[0]["text"]
