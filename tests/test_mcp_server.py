"""Comprehensive unit and integration tests for MCP Server HTTP/SSE endpoints, JSON-RPC tools, and Stdio process transport."""

import asyncio
import json
import subprocess
import sys
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
    assert result["serverInfo"]["name"] == "claude-code-proxy-mcp"


def test_mcp_ping_and_initialized(client: TestClient) -> None:
    req_ping = {"jsonrpc": "2.0", "id": 10, "method": "ping"}
    resp = client.post("/mcp", json=req_ping)
    assert resp.status_code == 200
    assert resp.json().get("result") == {}

    req_init_notif = {"jsonrpc": "2.0", "id": 11, "method": "notifications/initialized"}
    resp2 = client.post("/mcp", json=req_init_notif)
    assert resp2.status_code == 200
    assert resp2.json().get("result") == {}


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
    assert "available_models_by_provider" in text


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
    assert "global_metrics" in content[0]["text"]


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


def test_mcp_tool_call_set_model_mapping(client: TestClient) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "set_model_mapping",
            "arguments": {
                "alias_key": "MODEL_HAIKU",
                "target_model": "nvidia_nim/meta/llama-3.1-8b-instruct",
            },
        },
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    result = data.get("result", {})
    content = result.get("content", [])
    assert len(content) > 0
    assert "success" in content[0]["text"]


def test_mcp_tool_call_update_system_config(client: TestClient) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "update_system_config",
            "arguments": {
                "configs": {
                    "subagents_enabled": True,
                    "PROVIDER_GROQ_RPM": 30,
                },
            },
        },
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    result = data.get("result", {})
    content = result.get("content", [])
    assert len(content) > 0
    assert "success" in content[0]["text"]


def test_mcp_tool_call_control_circuit_breaker(client: TestClient) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "control_circuit_breaker",
            "arguments": {
                "model_id": "nvidia_nim/test-mcp-model",
                "action": "reset",
            },
        },
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    result = data.get("result", {})
    content = result.get("content", [])
    assert len(content) > 0
    assert "reset" in content[0]["text"]


def test_mcp_sse_endpoint(client: TestClient) -> None:
    resp = client.get("/mcp/sse")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert "event: endpoint" in resp.text
    assert "data: /mcp" in resp.text


def test_stdio_mcp_server_process() -> None:
    """Integration test executing mcp_server.py as a subprocess over stdio."""
    input_rpc = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}}) + "\n"

    proc = subprocess.Popen(
        [sys.executable, "-u", "mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(input_rpc)
        proc.stdin.flush()

        response_line = proc.stdout.readline()
        assert response_line
        res = json.loads(response_line)
        assert res.get("jsonrpc") == "2.0"
        assert res.get("id") == 99
        tools = res.get("result", {}).get("tools", [])
        tool_names = [t.get("name") for t in tools]
        assert "get_models" in tool_names
        assert "get_metrics" in tool_names
    finally:
        proc.kill()
        proc.wait()
