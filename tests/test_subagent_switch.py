"""Unit tests for Subagent Emergency Switch (ON/OFF Bypass) state management and payload interception."""

import pytest
from fastapi.testclient import TestClient

from atomic.guards.subagent import SubagentGuard
from server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_subagents_api_endpoints(client):
    # GET initial state
    resp = client.get("/api/settings/subagents")
    assert resp.status_code == 200
    data = resp.json()
    assert "subagents_enabled" in data

    # POST toggle to False (OFF Bypass)
    post_resp = client.post("/api/settings/subagents", json={"enabled": False})
    assert post_resp.status_code == 200
    post_data = post_resp.json()
    assert post_data["status"] == "success"
    assert post_data["subagents_enabled"] is False

    # GET updated state
    resp_updated = client.get("/api/settings/subagents")
    assert resp_updated.json()["subagents_enabled"] is False

    # POST toggle back to True (ON)
    client.post("/api/settings/subagents", json={"enabled": True})
    assert client.get("/api/settings/subagents").json()["subagents_enabled"] is True


def test_subagent_guard_sanitize_payload_when_on():
    guard = SubagentGuard()
    payload = {
        "model": "claude-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Task",
                        "input": {"run_in_background": True, "prompt": "build feature"}
                    }
                ]
            }
        ],
        "run_in_background": True
    }

    # When enabled is True (ON), payload remains untouched
    result = guard.sanitize_payload(payload.copy(), enabled=True)
    assert result["run_in_background"] is True
    assert result["messages"][0]["content"][0]["input"]["run_in_background"] is True


def test_subagent_guard_sanitize_payload_when_off():
    guard = SubagentGuard()
    payload = {
        "model": "claude-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Task",
                        "input": {"run_in_background": True, "prompt": "build feature"}
                    }
                ]
            }
        ],
        "run_in_background": True
    }

    # When enabled is False (OFF Bypass), run_in_background is forced to False
    result = guard.sanitize_payload(payload, enabled=False)
    assert result["run_in_background"] is False
    assert result["messages"][0]["content"][0]["input"]["run_in_background"] is False


@pytest.mark.asyncio
async def test_subagent_guard_enforce_tool_call_when_on_and_off():
    guard = SubagentGuard()
    tool_input = {"run_in_background": True, "command": "pytest"}

    # When enabled is True (ON), run_in_background remains True
    res_on = await guard.enforce_tool_call("Task", tool_input.copy(), enabled=True)
    assert res_on["run_in_background"] is True

    # When enabled is False (OFF Bypass), run_in_background is overridden to False
    res_off = await guard.enforce_tool_call("Task", tool_input.copy(), enabled=False)
    assert res_off["run_in_background"] is False

