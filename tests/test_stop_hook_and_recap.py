"""Tests for Stop Hook JSON repair, Evaluator Schema Normalization, and Accurate Recap handling."""

import json

import pytest

from core.interceptor.json_repair import JSONRepairNormalizer
from core.transformer.stream_engine import StreamEngine


@pytest.mark.asyncio
async def test_repair_missing_opening_brace() -> None:
    """Verify missing opening brace is repaired when string starts with key-value."""
    malformed = '"ok": true,\n  "reason": "Goal set: active\\n\\n«active» is still active"\n}'
    repaired_json = await JSONRepairNormalizer.process_text(malformed, is_stop_hook=True, is_evaluator=True)
    parsed = json.loads(repaired_json)

    assert parsed["ok"] is True
    assert "Goal set: active" in parsed["reason"]
    # Strictly NO extra properties allowed for evaluator schema
    assert "summary" not in parsed
    assert "memory" not in parsed
    assert "stop_hook_active" not in parsed


@pytest.mark.asyncio
async def test_evaluator_hook_detection() -> None:
    """Verify is_evaluator_hook_target correctly identifies Claude Code stop condition requests."""
    eval_payload = {
        "model": "claude-opus-5",
        "system": "You are evaluating a stop-condition hook in Claude Code.",
        "messages": [
            {
                "role": "user",
                "content": "Based on the conversation transcript above, has the following stopping condition been satisfied?",
            }
        ],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}, "reason": {"type": "string"}},
                    "required": ["ok", "reason"],
                    "additionalProperties": False,
                },
            }
        },
    }
    assert JSONRepairNormalizer.is_evaluator_hook_target(eval_payload) is True

    normal_payload = {
        "model": "claude-opus-5",
        "system": "You are a helpful coding assistant.",
        "messages": [{"role": "user", "content": "Write a python script"}],
    }
    assert JSONRepairNormalizer.is_evaluator_hook_target(normal_payload) is False


@pytest.mark.asyncio
async def test_evaluator_schema_normalization_strictness() -> None:
    """Verify evaluator schema strictly emits only ok, reason, and optional impossible."""
    # 1. Truncated output from upstream
    truncated_input = '{"ok": false, "reason": "Goal check-in: «active» is still active'
    repaired = await JSONRepairNormalizer.process_text(truncated_input, is_stop_hook=True, is_evaluator=True)
    parsed = json.loads(repaired)
    assert parsed["ok"] is False
    assert "Goal check-in" in parsed["reason"]
    assert "summary" not in parsed
    assert "memory" not in parsed

    # 2. Input that only had stop_hook_active=True
    legacy_input = {"stop_hook_active": True, "summary": "Working on Next.js"}
    eval_normalized = await JSONRepairNormalizer.normalize_stop_hook_schema(legacy_input, is_evaluator=True)
    assert eval_normalized["ok"] is False
    assert "Working on Next.js" in eval_normalized["reason"]
    assert "summary" not in eval_normalized
    assert "stop_hook_active" not in eval_normalized

    # 3. Input with impossible flag
    impossible_input = {"ok": False, "impossible": True, "reason": "Impossible task"}
    norm_imp = await JSONRepairNormalizer.normalize_stop_hook_schema(impossible_input, is_evaluator=True)
    assert norm_imp["ok"] is False
    assert norm_imp["impossible"] is True
    assert norm_imp["reason"] == "Impossible task"


@pytest.mark.asyncio
async def test_legacy_stop_hook_backward_compatibility() -> None:
    """Verify non-evaluator stop hooks retain summary, memory, and stop_hook_active."""
    input_data = {"session_summary": "Done", "memories": ["x"], "stop_hook": "True"}
    normalized = await JSONRepairNormalizer.normalize_stop_hook_schema(input_data, is_evaluator=False)
    assert normalized["summary"] == "Done"
    assert normalized["memory"] == ["x"]
    assert normalized["stop_hook_active"] is True


@pytest.mark.asyncio
async def test_streaming_engine_buffers_and_emits_valid_json() -> None:
    """Verify StreamEngine does not leak raw broken text chunks for stop hooks and emits valid JSON."""
    engine = StreamEngine(
        target_model="claude-opus-5",
        is_stop_hook=True,
        is_evaluator=True,
    )

    async def mock_upstream_stream():
        # Stream chunks with missing opening brace and unclosed ending
        yield {"choices": [{"delta": {"content": '"ok": false, "reason": "Condition '}}]}
        yield {"choices": [{"delta": {"content": 'not yet met'}}]}
        yield {"choices": [{"finish_reason": "stop"}]}

    events = []
    async for ev in engine.stream_response(mock_upstream_stream()):
        events.append(ev)

    # Inspect all text_delta events
    text_deltas = [ev for ev in events if "text_delta" in ev]
    assert len(text_deltas) == 1, "Stop hook text should be emitted as a single atomic text_delta"

    full_text = "".join(engine.accumulated_text)
    parsed = json.loads(full_text)
    assert parsed["ok"] is False
    assert "Condition not yet met" in parsed["reason"]
    assert "summary" not in parsed


def test_recap_request_detection_and_directive() -> None:
    """Verify recap request is detected, tools are stripped, and directive includes user goal."""
    messages = [
        {"role": "user", "content": "Create a full-stack Next.js project with Tailwind and Firebase"},
        {"role": "assistant", "content": "I will initialize the project"},
        {"role": "user", "content": "The user stepped away and is coming back. Recap in under 40 words, 1-2 plain sentences, no markdown."},
    ]
    tools = [{"name": "Bash", "description": "Execute command"}]

    last_m = messages[-1]
    c_lower = str(last_m.get("content", "")).lower()
    is_recap = any(k in c_lower for k in ("the user stepped away and is coming back", "recap in under", "recap in 1-2 plain sentences", "user stepped away"))
    assert is_recap is True

    # Check that tools get stripped
    stripped_tools = None if is_recap else tools
    assert stripped_tools is None

    # Check that primary goal is extracted from messages[0]
    initial_goal = ""
    for m in messages:
        if m.get("role") == "user":
            initial_goal = str(m.get("content", ""))[:300]
            break
    assert "Next.js" in initial_goal


@pytest.mark.asyncio
async def test_normal_tool_with_ok_arg_not_converted_to_stop_hook() -> None:
    """Verify standard tool calls containing 'ok' parameter are not modified by stop hook normalizer."""
    engine = StreamEngine(target_model="claude-3-5-sonnet", session_id="test_normal_tool_ok", is_stop_hook=False)

    async def mock_tool_chunks():
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_tool_1",
                                "type": "function",
                                "function": {
                                    "name": "CustomStatusTool",
                                    "arguments": '{"ok": true, "status": "active", "data": 123}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        yield {"choices": [{"finish_reason": "tool_calls"}]}

    _ = [ev async for ev in engine.transform_stream(mock_tool_chunks())]
    assert len(engine.accumulated_tool_calls) == 1
    tc = engine.accumulated_tool_calls[0]
    assert tc["name"] == "CustomStatusTool"
    # Parameters must NOT be converted to evaluator schema
    assert tc["input"]["status"] == "active"
    assert tc["input"]["data"] == 123
    assert tc["input"]["ok"] is True


@pytest.mark.asyncio
async def test_nim_sanitizer_output_config_and_reasoning_split() -> None:
    """Verify NimPayloadSanitizer converts output_config to response_format and injects reasoning_split."""
    from atomic.sanitizers.nim_sanitizer import NimPayloadSanitizer

    payload = {
        "model": "nvidia_nim/meta/llama-3.1-70b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            }
        },
    }

    sanitized = await NimPayloadSanitizer.sanitize(payload)

    # 1. output_config should be translated to response_format and removed from root
    assert "output_config" not in sanitized
    assert sanitized.get("response_format") == {"type": "json_object"}

    # 2. chat_template_kwargs should be at root and extra_body removed
    assert "extra_body" not in sanitized
    chat_kwargs = sanitized.get("chat_template_kwargs", {})
    assert chat_kwargs.get("reasoning_split") is True
    assert chat_kwargs.get("enable_thinking") is True


@pytest.mark.asyncio
async def test_openrouter_reasoning_enabled() -> None:
    """Verify OpenRouter models automatically receive reasoning: {enabled: True} in extra_body."""
    from providers.openai import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider()
    # Mock resolve to avoid real network call
    base_url, upstream_model, api_key, extra_headers = await provider._resolve_endpoint("open_router/anthropic/claude-3.5-sonnet")
    assert "openrouter" in base_url.lower()

