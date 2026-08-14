"""Unit tests for GeminiPayloadSanitizer."""

import pytest

from atomic.sanitizers.gemini_sanitizer import (
    GEMINI_MAX_TOKENS_LIMIT,
    GeminiPayloadSanitizer,
)


@pytest.mark.asyncio
async def test_sanitize_parameters_max_tokens_clamping() -> None:
    payload = {
        "model": "gemini-3.7-flash",
        "max_tokens": 16384,
        "temperature": 2.5,
        "stop": "",
        "thinking": "some_thinking_context",
    }
    sanitized = await GeminiPayloadSanitizer.sanitize(payload)
    assert sanitized["max_tokens"] == GEMINI_MAX_TOKENS_LIMIT
    assert sanitized["temperature"] == 2.0
    assert "stop" not in sanitized
    assert "thinking" not in sanitized


@pytest.mark.asyncio
async def test_sanitize_parameters_stop_sequences() -> None:
    payload = {
        "model": "gemini-3.7-flash",
        "stop": ["STOP_1", "STOP_2", None, ""],
    }
    sanitized = await GeminiPayloadSanitizer.sanitize(payload)
    assert sanitized["stop"] == ["STOP_1", "STOP_2"]


@pytest.mark.asyncio
async def test_dummy_tool_call_injection_preceding_assistant() -> None:
    messages = [
        {"role": "user", "content": "Run bash command"},
        {"role": "assistant", "content": "I will run the command now."},
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "name": "execute_bash",
            "content": "output text",
        },
    ]
    payload = {"model": "gemini-3.7-flash", "messages": messages}
    sanitized = await GeminiPayloadSanitizer.sanitize(payload)

    res_msgs = sanitized["messages"]
    assert len(res_msgs) == 3
    assistant_msg = res_msgs[1]
    assert assistant_msg["role"] == "assistant"
    assert "tool_calls" in assistant_msg
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["id"] == "call_abc123"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "execute_bash"


@pytest.mark.asyncio
async def test_dummy_tool_call_injection_no_preceding_assistant() -> None:
    messages = [
        {"role": "user", "content": "Run tool without assistant step"},
        {
            "role": "tool",
            "tool_call_id": "call_orphan",
            "name": "str_replace_editor",
            "content": "file replaced",
        },
    ]
    payload = {"model": "gemini-3.7-flash", "messages": messages}
    sanitized = await GeminiPayloadSanitizer.sanitize(payload)

    res_msgs = sanitized["messages"]
    assert len(res_msgs) == 3
    assert res_msgs[0]["role"] == "user"
    assert res_msgs[1]["role"] == "assistant"
    assert res_msgs[1]["content"] is None
    assert res_msgs[1]["tool_calls"][0]["id"] == "call_orphan"
    assert res_msgs[2]["role"] == "tool"


@pytest.mark.asyncio
async def test_consecutive_user_roles_merging() -> None:
    messages = [
        {"role": "user", "content": "First instruction"},
        {"role": "user", "content": "Second instruction"},
        {"role": "user", "content": "Third instruction"},
    ]
    payload = {"model": "gemini-3.7-flash", "messages": messages}
    sanitized = await GeminiPayloadSanitizer.sanitize(payload)

    res_msgs = sanitized["messages"]
    assert len(res_msgs) == 1
    assert res_msgs[0]["role"] == "user"
    assert "First instruction" in res_msgs[0]["content"]
    assert "Second instruction" in res_msgs[0]["content"]
    assert "Third instruction" in res_msgs[0]["content"]


@pytest.mark.asyncio
async def test_consecutive_assistant_roles_merging() -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Part 1",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f1"}}],
        },
        {
            "role": "assistant",
            "content": "Part 2",
            "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "f2"}}],
        },
    ]
    payload = {"model": "gemini-3.7-flash", "messages": messages}
    sanitized = await GeminiPayloadSanitizer.sanitize(payload)

    res_msgs = sanitized["messages"]
    assert len(res_msgs) == 2
    assert res_msgs[0]["role"] == "user"
    assert res_msgs[1]["role"] == "assistant"
    assert res_msgs[1]["content"] == "Part 1\n\nPart 2"
    assert len(res_msgs[1]["tool_calls"]) == 2
    assert res_msgs[1]["tool_calls"][0]["id"] == "t1"
    assert res_msgs[1]["tool_calls"][1]["id"] == "t2"


@pytest.mark.asyncio
async def test_system_prompt_extraction_and_merging() -> None:
    messages = [
        {"role": "system", "content": "System prompt 1"},
        {"role": "user", "content": "User prompt"},
        {"role": "system", "content": "System prompt 2"},
    ]
    payload = {"model": "gemini-3.7-flash", "messages": messages}
    sanitized = await GeminiPayloadSanitizer.sanitize(payload)

    res_msgs = sanitized["messages"]
    assert len(res_msgs) == 2
    assert res_msgs[0]["role"] == "system"
    assert res_msgs[0]["content"] == "System prompt 1\n\nSystem prompt 2"
    assert res_msgs[1]["role"] == "user"
    assert res_msgs[1]["content"] == "User prompt"
