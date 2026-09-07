import pytest

from atomic.sanitizers.nim_sanitizer import (
    NVIDIA_NIM_MAX_OUTPUT_CAP,
    NimPayloadSanitizer,
)


@pytest.mark.asyncio
async def test_nim_sanitizer_clamps_max_tokens():
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64000,
        "thinking": {"type": "adaptive"},
        "context_management": {"edits": []},
        "output_config": {"effort": "medium"},
    }

    sanitized = await NimPayloadSanitizer.sanitize(payload)

    # Verify max_tokens is clamped to 8192
    assert sanitized["max_tokens"] == NVIDIA_NIM_MAX_OUTPUT_CAP
    # Verify non-standard root parameters are removed
    assert "thinking" not in sanitized
    assert "context_management" not in sanitized
    assert "output_config" not in sanitized


@pytest.mark.asyncio
async def test_nim_sanitizer_cleans_role_tool_messages():
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [
            {"role": "user", "content": "Run command"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "1", "name": "bash", "content": "output"},
        ],
        "tool_choice": {"type": "auto"},
    }

    sanitized = await NimPayloadSanitizer.sanitize(payload)

    # Verify assistant empty string content is converted to None
    assert sanitized["messages"][1]["content"] is None
    # Verify role=tool name parameter is stripped
    assert "name" not in sanitized["messages"][2]
    # Verify tool_choice is string "auto"
    assert sanitized["tool_choice"] == "auto"
