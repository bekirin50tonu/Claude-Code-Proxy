"""Unit tests for enhancements: Subagent Policy Engine, Prompt Queue, Metrics, and Claude Settings."""

import pytest
from fastapi.testclient import TestClient

from api.mcp import execute_mcp_tool
from atomic.guards.subagent_policy import subagent_policy_engine
from core.interceptor.prompt_queue import prompt_queue_manager
from server import app

client = TestClient(app)


def test_subagent_policy_engine() -> None:
    policy = subagent_policy_engine.get_policy()
    assert "subagents" in policy

    allowed, reason = subagent_policy_engine.evaluate_action("run_command", {"cmd": "ls"})
    assert allowed is True

    blocked, reason_blocked = subagent_policy_engine.evaluate_action("run_command", {"cmd": "rm -rf /"})
    assert blocked is False
    assert "blocked pattern" in reason_blocked

    decisions = subagent_policy_engine.get_decisions()
    assert len(decisions) >= 2


@pytest.mark.asyncio
async def test_prompt_queue_advanced_operations() -> None:
    session_id = "test_queue_session"
    prompt_queue_manager.clear_queue(session_id)

    await prompt_queue_manager.push_prompt("Prompt 1", session_id)
    await prompt_queue_manager.push_prompt("Prompt 2", session_id)

    peeked = prompt_queue_manager.peek_prompts(session_id)
    assert len(peeked) == 2
    assert peeked[0] == "Prompt 1"

    await prompt_queue_manager.inject_prompt_at("Injected Prompt", index=0, session_id=session_id)
    peeked_after_inject = prompt_queue_manager.peek_prompts(session_id)
    assert peeked_after_inject[0] == "Injected Prompt"

    await prompt_queue_manager.replace_prompt(index=0, new_prompt="Replaced Prompt", session_id=session_id)
    peeked_after_replace = prompt_queue_manager.peek_prompts(session_id)
    assert peeked_after_replace[0] == "Replaced Prompt"

    cleared = prompt_queue_manager.clear_queue(session_id)
    assert cleared == 3
    assert not prompt_queue_manager.has_pending(session_id)


def test_healthz_readyz_and_metrics_endpoints() -> None:
    res_health = client.get("/healthz")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    res_ready = client.get("/readyz")
    assert res_ready.status_code == 200
    assert res_ready.json()["status"] == "ready"

    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    text = res_metrics.text
    assert "proxy_requests_total" in text
    assert "proxy_circuit_state" in text
    assert "proxy_rate_limit_headroom" in text


@pytest.mark.asyncio
async def test_new_mcp_tools() -> None:
    # Test get_subagent_policy
    res = await execute_mcp_tool("get_subagent_policy", {})
    assert "content" in res
    assert "subagents" in res["content"][0]["text"]

    # Test get_claude_settings
    res_settings = await execute_mcp_tool("get_claude_settings", {})
    assert "content" in res_settings

    # Test sync_proxy_to_claude
    res_sync = await execute_mcp_tool("sync_proxy_to_claude", {})
    assert "status" in res_sync["content"][0]["text"]


@pytest.mark.asyncio
async def test_action_continuation_emits_tool_call(monkeypatch) -> None:
    """Verify that when the model emits only thinking, action continuation emits a tool call instead of closing the turn."""
    from unittest.mock import AsyncMock

    from core.transformer.stream_engine import StreamEngine
    from providers.openai import OpenAICompatibleProvider

    tools = [{
        "name": "Bash",
        "description": "Run bash",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
    }]
    engine = StreamEngine(target_model="claude-opus-5", tools=tools, session_id="test_continuation")

    mock_complete = AsyncMock(return_value={
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call-123",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": '{"command": "mkdir -p src/app"}'},
                }]
            }
        }]
    })
    monkeypatch.setattr(OpenAICompatibleProvider, "complete", mock_complete)

    async def mock_upstream():
        yield {"choices": [{"delta": {"reasoning_content": "Let me create the directory structure."}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    events = []
    async for ev in engine.stream_response(mock_upstream()):
        events.append(ev)

    assert any("tool_use" in ev for ev in events)
    assert any("mkdir -p src/app" in ev for ev in events)
    assert engine.text_or_tool_emitted is True
    assert engine.final_stop_reason == "tool_use"

