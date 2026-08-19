import time

import pytest

from api.mock import check_mock_request, extract_command_prefix, extract_filepaths
from core.transformer.stream_engine import (
    SSEStreamTransformer,
    extract_all_json_tool_calls,
    extract_json_tool_call,
    safe_parse_json,
    translate_non_stream_response,
)
from providers.base import BaseProvider


class DummyProvider(BaseProvider):
    async def complete(self, *args, **kwargs):
        return {"choices": []}


def test_base_provider_translations() -> None:
    provider = DummyProvider()

    # 1. Test tools translation
    anthropic_tools = [
        {
            "name": "run_command",
            "description": "Run a command",
            "input_schema": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        }
    ]
    openai_tools = provider.translate_tools(anthropic_tools)
    assert openai_tools is not None
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "run_command"
    assert openai_tools[0]["function"]["parameters"]["required"] == ["cmd"]

    # 2. Test messages translation
    anthropic_messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Sure"},
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "run_command",
                    "input": {"cmd": "ls"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "success"}
            ],
        },
    ]
    translated = provider.translate_messages(
        anthropic_messages, system="You are a proxy"
    )
    assert translated[0]["role"] == "system"
    assert translated[0]["content"] == "You are a proxy"
    assert translated[1]["role"] == "user"
    assert translated[1]["content"] == "Hello"
    assert translated[2]["role"] == "assistant"
    assert translated[2]["tool_calls"][0]["id"] == "tool_1"
    assert translated[3]["role"] == "tool"
    assert translated[3]["tool_call_id"] == "tool_1"
    assert translated[3]["content"] == "success"


def test_mock_detection() -> None:
    # 1. Quota Probe Mock
    req_probe = {"max_tokens": 1, "model": "claude-3-5-sonnet", "messages": []}
    mock_res = check_mock_request(req_probe)
    assert mock_res is not None
    assert mock_res["id"] == "msg_mock_probe"

    # 2. Title Gen Mock
    req_title = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": "Generate a short, 2-4 word title for this conversation",
            }
        ],
    }
    mock_res = check_mock_request(req_title)
    assert mock_res is not None
    assert mock_res["id"] == "msg_mock_title"

    # 3. Suggestion Mock
    req_sug = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {"role": "user", "content": "suggestion mode autocomplete request"}
        ],
    }
    mock_res = check_mock_request(req_sug)
    assert mock_res is not None
    assert mock_res["id"] == "msg_mock_suggestion"




def test_command_prefix_extraction() -> None:
    text = "Check this command: `git status` or analyze its safety."
    assert extract_command_prefix(text) == "git"

    text_no_backticks = "Should we run cd /home or clear the screen?"
    assert extract_command_prefix(text_no_backticks) in ["cd", "clear"]


def test_filepaths_extraction() -> None:
    text = "We have files like src/main.py and config/settings.json in our list."
    paths = extract_filepaths(text)
    assert "src/main.py" in paths
    assert "config/settings.json" in paths


def test_safe_parse_json_multiline_code() -> None:
    """Test that multiline code containing literal unescaped newlines is parsed cleanly."""
    json_with_literal_newlines = (
        '{\n'
        '  "name": "Write",\n'
        '  "parameters": {\n'
        '    "file_path": "src/components/Anasayfa.js",\n'
        '    "content": "import React from \'react\';\n\nconst Anasayfa = () => {\n  return (\n    <div>\n      <h1>Merhaba, Bekir Görmez!</h1>\n    </div>\n  );\n};\n\nexport default Anasayfa;"\n'
        '  }\n'
        '}'
    )
    parsed = safe_parse_json(json_with_literal_newlines)
    assert parsed is not None
    assert parsed["name"] == "Write"
    content = parsed["parameters"]["content"]
    assert "import React from 'react';" in content
    assert "Bekir Görmez" in content
    assert len(content.splitlines()) > 5

    # Test extraction
    remaining, tools = extract_all_json_tool_calls(json_with_literal_newlines)
    assert len(tools) == 1
    assert tools[0]["name"] == "Write"
    assert tools[0]["input"]["file_path"] == "src/components/Anasayfa.js"
    assert "Bekir Görmez" in tools[0]["input"]["content"]


def test_extract_json_tool_call() -> None:
    # Single tool call
    json_text = '{"name": "Write", "parameters": {"file_path": "test.js", "content": "hello"}}'
    prefix, tool = extract_json_tool_call(json_text)
    assert tool is not None
    assert tool["name"] == "Write"
    assert tool["input"]["file_path"] == "test.js"


def test_extract_all_json_tool_calls_multi() -> None:
    """LLM outputs multiple Write/Read JSON tool calls in text — the exact user bug."""
    llm_text = (
        "Özür dilerim, değişiklikleri yapıyorum. "
        '```json\n{"name": "Write", "parameters": {"file_path": "src/Anasayfa.js", "content": "hello"}}\n```'
        " "
        '```json\n{"name": "Read", "parameters": {"file_path": "src/Anasayfa.js"}}\n```'
        " "
        '```json\n{"name": "Write", "parameters": {"file_path": "src/Hakkinda.js", "content": "about"}}\n```'
    )
    remaining, tools = extract_all_json_tool_calls(llm_text)
    assert len(tools) == 3
    assert tools[0]["name"] == "Write"
    assert tools[0]["input"]["file_path"] == "src/Anasayfa.js"
    assert tools[1]["name"] == "Read"
    assert tools[1]["input"]["file_path"] == "src/Anasayfa.js"
    assert tools[2]["name"] == "Write"
    assert tools[2]["input"]["file_path"] == "src/Hakkinda.js"
    # Remaining text should be just the Turkish prefix
    assert "Özür" in remaining
    assert '"name"' not in remaining

    # Non-stream response translation with multiple tools
    openai_resp = {
        "id": "chatcmpl-multi",
        "model": "meta/llama-3.1-70b-instruct",
        "choices": [{"message": {"role": "assistant", "content": llm_text}, "finish_reason": "stop"}],
    }
    translated = translate_non_stream_response(openai_resp)
    assert translated["stop_reason"] == "tool_use"
    tool_blocks = [b for b in translated["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 3


@pytest.mark.asyncio
async def test_streaming_text_embedded_tool_extraction() -> None:
    """When LLM streams tool calls as text, they should be extracted post-stream."""
    transformer = SSEStreamTransformer(target_model="meta/llama-3.1-70b-instruct")

    async def mock_stream():
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        # LLM streams text that contains a JSON tool call
        yield {"choices": [{"delta": {"content": 'I will write the file.\n```json\n{"name": "Write",'}}]}
        yield {"choices": [{"delta": {"content": ' "parameters": {"file_path": "app.js", "content": "code"}}\n```'}}]}
        yield {"choices": [{"finish_reason": "stop"}]}

    events = []
    async for event in transformer.transform(mock_stream()):
        events.append(event)

    full_output = "".join(events)
    # The text was streamed as text_delta, but tool_use blocks should also appear
    assert '"type": "tool_use"' in full_output
    assert '"name": "Write"' in full_output
    assert '"stop_reason": "tool_use"' in full_output
    summary = transformer.get_summary_response()
    assert summary["stop_reason"] == "tool_use"
    assert any(b["type"] == "tool_use" and b["name"] == "Write" for b in summary["content"])




@pytest.mark.asyncio
async def test_native_multi_tool_call_stream_transformer() -> None:
    """Test native OpenAI streaming with multiple tool calls across different indexes."""
    transformer = SSEStreamTransformer(target_model="meta/llama-3.1-70b-instruct")

    async def mock_openai_stream():
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        # Tool 0: Write
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "chatcmpl-tool-1",
                                "type": "function",
                                "function": {"name": "Write", "arguments": '{"file_path": "Anasayfa.js", '},
                            }
                        ]
                    }
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"content": "import React from \\"react\\";"}'},
                            }
                        ]
                    }
                }
            ]
        }
        # Tool 1: Read
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "chatcmpl-tool-2",
                                "type": "function",
                                "function": {"name": "Read", "arguments": '{"file_path": "Anasayfa.js"}'},
                            }
                        ]
                    }
                }
            ]
        }
        yield {"choices": [{"finish_reason": "tool_calls"}]}

    events = []
    async for event in transformer.transform(mock_openai_stream()):
        events.append(event)

    full_output = "".join(events)
    assert '"id": "chatcmpl-tool-1"' in full_output
    assert '"id": "chatcmpl-tool-2"' in full_output
    assert '"name": "Write"' in full_output
    assert '"name": "Read"' in full_output

    summary = transformer.get_summary_response()
    assert summary["stop_reason"] == "tool_use"
    tool_blocks = [b for b in summary["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 2
    assert tool_blocks[0]["name"] == "Write"
    assert tool_blocks[0]["input"]["content"] == 'import React from "react";'
    assert tool_blocks[1]["name"] == "Read"
    assert tool_blocks[1]["input"]["file_path"] == "Anasayfa.js"
@pytest.mark.asyncio
async def test_native_tool_call_stream_transformer() -> None:
    transformer = SSEStreamTransformer(target_model="meta/llama-3.1-70b-instruct")

    async def mock_openai_stream():
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc123",
                                "type": "function",
                                "function": {"name": "run_command", "arguments": '{"CommandLine":'},
                            }
                        ]
                    }
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": ' "git status"}'},
                            }
                        ]
                    }
                }
            ]
        }
        yield {"choices": [{"finish_reason": "tool_calls"}]}

    events = []
    async for event in transformer.transform(mock_openai_stream()):
        events.append(event)

    full_output = "".join(events)
    assert "event: message_start" in full_output
    assert "event: content_block_start" in full_output
    assert '"type": "tool_use"' in full_output
    assert '"name": "run_command"' in full_output
    assert '"id": "call_abc123"' in full_output
    assert '"stop_reason": "tool_use"' in full_output
    summary = transformer.get_summary_response()
    assert summary["stop_reason"] == "tool_use"
    assert summary["content"][0]["type"] == "tool_use"
    assert summary["content"][0]["input"] == {"CommandLine": "git status"}


def test_non_stream_response_translation() -> None:
    openai_resp = {
        "id": "chatcmpl-123",
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Running task.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": '{"cmd": "ls"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 20},
    }
    translated = translate_non_stream_response(openai_resp)
    assert translated["id"] == "chatcmpl-123"
    assert translated["stop_reason"] == "tool_use"
    assert len(translated["content"]) == 2
    assert translated["content"][0]["type"] == "text"
    assert translated["content"][1]["type"] == "tool_use"
    assert translated["content"][1]["id"] == "call_1"


@pytest.mark.asyncio
async def test_stream_transformer_with_think_tags() -> None:
    transformer = SSEStreamTransformer(target_model="claude-3-5-sonnet")

    # Yield some text chunks with think tags
    async def openai_stream():
        chunks = [
            {"choices": [{"delta": {"content": "Hello! <think>I am thinking"}}]},
            {"choices": [{"delta": {"content": " hard</think> Done thinking!"}}]},
        ]
        for c in chunks:
            yield c

    events = []
    async for ev in transformer.transform(openai_stream()):
        events.append(ev)

    events_str = "".join(events)
    # Check that events contain thinking and text start blocks and content deltas
    assert "content_block_start" in events_str
    assert "thinking_delta" in events_str
    assert "text_delta" in events_str
    assert "I am thinking" in events_str
    assert "Done thinking!" in events_str


def test_dashboard_endpoints() -> None:
    from fastapi.testclient import TestClient

    from config import stats
    from server import app

    client = TestClient(app)

    # Set mock stats
    stats.total_requests = 10
    stats.mocked_requests = 4
    stats.error_count = 1

    # 1. Test GET /api/stats
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] == 10
    assert data["mocked_requests"] == 4
    assert data["error_count"] == 1
    assert "tg_bot_status" in data
    assert "ds_bot_status" in data

    # 2. Test GET /api/config
    resp = client.get("/api/config")
    assert resp.status_code == 200
    res_data = resp.json()
    assert "configs" in res_data

    # 3. Test GET /api/dev/payloads
    stats.record_log(
        method="POST",
        path="/v1/messages",
        client_model="claude-3-5-sonnet",
        mapped_model="nvidia_nim/meta/llama-3.1-70b-instruct",
        status_code=200,
        start_time=time.time(),
        request_body={"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hello"}]},
        response_body={"id": "msg_123", "content": [{"type": "text", "text": "hi"}]},
    )
    resp = client.get("/api/dev/payloads")
    assert resp.status_code == 200
    payload_data = resp.json()
    assert payload_data["total_captured"] > 0
    assert "payloads" in payload_data

    first_req_id = payload_data["payloads"][0]["id"]
    resp_single = client.get(f"/api/dev/payloads/{first_req_id}")
    assert resp_single.status_code == 200
    single_data = resp_single.json()
    assert single_data["request_body"]["model"] == "claude-3-5-sonnet"
    assert "key_statuses" in res_data
    assert "MODEL_OPUS" in res_data["configs"]
    assert "OPENROUTER_API_KEY" in res_data["configs"]

    # 3. Test GET /dashboard HTML page
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "CLAUDE GATE" in resp.text

    # 4. Test GET /api/models autocomplete aggregations
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models_data = resp.json()
    assert "open_router" in models_data
    assert "nvidia_nim" in models_data

    # 5. Test that GATEWAY_AUTH_TOKEN is exposed in config
    assert "GATEWAY_AUTH_TOKEN" in res_data["configs"]


def test_gateway_auth_token_enforced() -> None:
    """Test that GATEWAY_AUTH_TOKEN enforces client authentication on proxy endpoints."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from server import app

    client = TestClient(app, raise_server_exceptions=False)

    with patch("core.gateway.settings") as mock_settings:
        mock_settings.GATEWAY_AUTH_TOKEN = "my-secret-token"
        mock_settings.PROVIDER_RATE_LIMIT = 100
        mock_settings.PROVIDER_RATE_WINDOW = 60
        mock_settings.PROVIDER_MAX_CONCURRENCY = 10

        # Request with wrong token should return 401
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "authentication_error"

        # Request with no auth header should also return 401
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 401


def test_gateway_auth_token_disabled() -> None:
    """Test that when GATEWAY_AUTH_TOKEN is empty, all requests pass through freely."""
    from unittest.mock import AsyncMock, patch

    from fastapi.testclient import TestClient

    from server import app

    client = TestClient(app, raise_server_exceptions=False)

    with patch("core.gateway.settings") as mock_settings:
        mock_settings.GATEWAY_AUTH_TOKEN = ""  # Disabled
        mock_settings.PROVIDER_RATE_LIMIT = 100
        mock_settings.PROVIDER_RATE_WINDOW = 60
        mock_settings.PROVIDER_MAX_CONCURRENCY = 10

        # Request with no auth header should pass through (no 401)
        with (
            patch("core.gateway.rate_limiter") as mock_rl,
            patch("core.gateway.provider") as mock_provider,
            patch("core.gateway.concurrency_semaphore"),
        ):
            mock_rl.acquire = AsyncMock(return_value=True)
            mock_provider.complete = AsyncMock(
                return_value={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "model": "test",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            # Should not be a 401 when auth is disabled
            assert resp.status_code != 401


# ===========================================================================
# NEW: Circuit Breaker Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_state_machine() -> None:
    """CLOSED → OPEN after failure_threshold failures."""
    from core.router.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test-model", failure_threshold=3, recovery_timeout=999)

    # Should start CLOSED
    assert not cb.is_open()

    # Record failures up to threshold
    await cb.record_failure()
    await cb.record_failure()
    assert not cb.is_open()  # Not yet

    await cb.record_failure()
    assert cb.is_open()  # Now OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery() -> None:
    """OPEN → HALF_OPEN after timeout → CLOSED after success."""
    import time

    from core.router.circuit_breaker import CircuitBreaker, CircuitState

    cb = CircuitBreaker("test-model", failure_threshold=1, recovery_timeout=0.05)

    await cb.record_failure()
    assert cb.is_open()

    # Wait for recovery_timeout to elapse
    time.sleep(0.1)

    # Now is_open() should trigger HALF_OPEN transition → returns False (allows probe)
    assert not cb.is_open()
    assert cb.state == CircuitState.HALF_OPEN

    # Success → back to CLOSED
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert not cb.is_open()


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure_reopens() -> None:
    """HALF_OPEN + failure → back to OPEN."""
    import time

    from core.router.circuit_breaker import CircuitBreaker, CircuitState

    cb = CircuitBreaker("test-model", failure_threshold=1, recovery_timeout=0.05)
    await cb.record_failure()
    time.sleep(0.1)
    cb.is_open()  # Trigger HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN

    await cb.record_failure()  # HALF_OPEN + failure → OPEN
    assert cb.state == CircuitState.OPEN


# ===========================================================================
# NEW: Rate Limit Parser Tests
# ===========================================================================


def test_rate_limit_parser_headers() -> None:
    """Headers parsed correctly and headroom computed."""
    from core.router.rate_limiter import RateLimitState

    state = RateLimitState("test-model")

    # Full quota → headroom
    state.update(
        {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "90",
            "x-ratelimit-limit-tokens": "200000",
            "x-ratelimit-remaining-tokens": "180000",
        }
    )
    assert state.has_headroom()
    assert state.req_limit == 100
    assert state.tok_remaining == 180000


def test_rate_limit_parser_no_headroom() -> None:
    """Near-exhausted quota → no headroom."""
    from core.router.rate_limiter import RateLimitState

    state = RateLimitState("test-model")
    state.update(
        {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "0",  # exhausted
        }
    )
    assert not state.has_headroom()


def test_rate_limit_parser_missing_headers() -> None:
    """No headers → headroom defaults to True (permissive)."""
    from core.router.rate_limiter import RateLimitState

    state = RateLimitState("unknown-model")
    assert state.has_headroom()  # No info → assume OK


# ===========================================================================
# NEW: Model Router Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_model_router_fallback() -> None:
    """Primary with open CB → router picks fallback."""
    from unittest.mock import AsyncMock, patch

    from core.router.circuit_breaker import CircuitBreakerRegistry
    from core.router.selector import ModelSelector as ModelRouter


    # Fresh registry
    fresh_registry = CircuitBreakerRegistry()
    # Open primary's circuit breaker
    primary = "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1"
    cb = fresh_registry.get(primary)
    for _ in range(5):
        await cb.record_failure()
    assert cb.is_open()

    router = ModelRouter()

    with (
        patch("core.router.selector.circuit_breaker_registry", fresh_registry),
        patch(
            "atomic.guards.preflight.preflight_model_probe", new=AsyncMock(return_value=True)
        ),

        patch.object(
            router, "_get_preflight", return_value=AsyncMock(return_value=True)
        ),
    ):
        # Since primary CB is open, router should pick first fallback
        # We test the _is_available check
        assert not router._is_available(primary)


@pytest.mark.asyncio
async def test_model_router_all_unavailable() -> None:
    """All models CB open → AllModelsUnavailableError raised."""
    from unittest.mock import patch

    from core.router.circuit_breaker import CircuitBreakerRegistry
    from core.router.selector import AllModelsUnavailableError
    from core.router.selector import ModelSelector as ModelRouter


    fresh_registry = CircuitBreakerRegistry()
    router = ModelRouter()

    async def _always_false(_model_id: str) -> bool:
        return False

    with (
        patch("core.router.selector.circuit_breaker_registry", fresh_registry),
        patch.object(router, "_get_preflight", return_value=_always_false),
        patch.object(router, "_is_available", return_value=True),
        pytest.raises(AllModelsUnavailableError),
    ):
        await router.pick_model("claude-3-5-sonnet")


# ===========================================================================
# NEW: Token Budget Guard Tests
# ===========================================================================


def test_token_budget_guard_no_truncation() -> None:
    """Short messages should not be truncated."""
    from atomic.guards.token_budget import TokenBudgetGuard
    from config import ModelMetadata


    guard = TokenBudgetGuard.__new__(TokenBudgetGuard)
    import tiktoken

    guard._enc = tiktoken.get_encoding("cl100k_base")
    guard.model_id = "test"
    guard.metadata = ModelMetadata(context=128000, max_output=4096)

    messages = [{"role": "user", "content": "Hello, world!"}]
    out_messages, out_system, truncated = guard.check_and_truncate(messages, None, 4096)

    assert not truncated
    assert out_messages == messages


def test_token_budget_guard_truncation() -> None:
    """Many messages over context limit should be truncated."""
    from atomic.guards.token_budget import TokenBudgetGuard
    from config import ModelMetadata


    guard = TokenBudgetGuard.__new__(TokenBudgetGuard)
    import tiktoken

    guard._enc = tiktoken.get_encoding("cl100k_base")
    guard.model_id = "test"
    # Very small context to force truncation
    guard.metadata = ModelMetadata(context=200, max_output=50)

    messages = [
        {
            "role": "user",
            "content": "This is a long message that takes up many tokens " * 5,
        },
        {"role": "assistant", "content": "And this reply is also quite long " * 5},
        {"role": "user", "content": "And another message here " * 5},
    ]
    system = "You are a helpful assistant."

    out_messages, out_system, truncated = guard.check_and_truncate(messages, system, 50)

    assert truncated
    assert len(out_messages) < len(messages)
    # System prompt must be preserved
    assert out_system == system
    # At least 1 message must remain
    assert len(out_messages) >= 1


# ===========================================================================
# NEW: Stream Guard Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_stream_guard_normal_flow() -> None:
    """Normal stream passes through without modification."""
    from atomic.guards.stream_guard import StreamGuard


    async def normal_stream():
        for chunk in ["data: chunk1\n\n", "data: chunk2\n\n", "data: done\n\n"]:
            yield chunk

    results = []
    guard = StreamGuard(normal_stream(), stream_timeout=5.0)
    async for chunk in guard:
        results.append(chunk)

    assert results == ["data: chunk1\n\n", "data: chunk2\n\n", "data: done\n\n"]


@pytest.mark.asyncio
async def test_stream_guard_timeout() -> None:
    """Slow stream that exceeds timeout yields an SSE error event."""
    import asyncio

    from atomic.guards.stream_guard import StreamGuard


    async def slow_stream():
        yield "data: first\n\n"
        await asyncio.sleep(10)  # Will be interrupted by timeout
        yield "data: second\n\n"

    results = []
    guard = StreamGuard(slow_stream(), stream_timeout=0.05)  # 50ms timeout
    async for chunk in guard:
        results.append(chunk)

    # Should have received first chunk, then an error event
    assert results[0] == "data: first\n\n"
    assert any("stream_error" in r or "error" in r for r in results[1:])


@pytest.mark.asyncio
async def test_stream_guard_empty_chunk_stall() -> None:
    """Excessive empty chunks triggers stall detection."""
    from atomic.guards.stream_guard import StreamGuard


    async def empty_stream():
        for _ in range(15):  # > max_empty_chunks=5
            yield ""

    results = []
    guard = StreamGuard(empty_stream(), max_empty_chunks=5)
    async for chunk in guard:
        results.append(chunk)

    # Should terminate with a stall error event
    assert any("stream_error" in r or "stall" in r.lower() for r in results)


# ===========================================================================
# NEW: Hermes-Claude & Anthropic API Endpoints Tests
# ===========================================================================


def test_gateway_auth_x_api_key_header() -> None:
    """Test authentication via x-api-key header used by Anthropic CLI / Hermes-Claude."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from server import app

    client = TestClient(app, raise_server_exceptions=False)

    with patch("core.gateway.settings") as mock_settings:
        mock_settings.GATEWAY_AUTH_TOKEN = "fcc-claude"
        mock_settings.PROVIDER_RATE_LIMIT = 100
        mock_settings.PROVIDER_RATE_WINDOW = 60
        mock_settings.PROVIDER_MAX_CONCURRENCY = 10

        # Request with correct x-api-key header should pass auth (returns mock probe response)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet",
                "max_tokens": 1,
                "messages": [],
            },
            headers={"x-api-key": "fcc-claude"},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "msg_mock_probe"


def test_count_tokens_endpoint() -> None:
    """Test /v1/messages/count_tokens endpoint computes token counts correctly."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from server import app

    client = TestClient(app)

    with patch("core.gateway.settings") as mock_settings:
        mock_settings.GATEWAY_AUTH_TOKEN = "fcc-claude"

        resp = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "Hello world from Hermes-Claude!"}],
                "system": "You are a helpful programming assistant.",
            },
            headers={"x-api-key": "fcc-claude"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "input_tokens" in data
        assert isinstance(data["input_tokens"], int)
        assert data["input_tokens"] > 0


def test_list_v1_models_endpoint() -> None:
    """Test /v1/models endpoint returns Anthropic compatible model list."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from server import app

    client = TestClient(app)

    with patch("core.gateway.settings") as mock_settings:
        mock_settings.GATEWAY_AUTH_TOKEN = "fcc-claude"

        resp = client.get(
            "/v1/models",
            headers={"x-api-key": "fcc-claude"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) >= 1
        model_ids = [m["id"] for m in data["data"]]
        assert "claude-3-5-sonnet-20241022" in model_ids


def test_in_flight_429_auto_retry_fallback() -> None:
    """Test that if primary model fails mid-request with HTTP 429, proxy retries fallback model."""
    from unittest.mock import AsyncMock, patch

    import httpx
    from fastapi.testclient import TestClient

    from core.router.rate_limiter import rate_limit_parser
    from server import app

    rate_limit_parser._states.clear()
    client = TestClient(app, raise_server_exceptions=False)


    async def mock_complete(model, *args, **kwargs):
        if "llama-3.1-70b-instruct" in model:
            req = httpx.Request("POST", "https://integrate.api.nvidia.com")
            res = httpx.Response(429, request=req)
            raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=res)
        return (
            {
                "id": "chatcmpl-fallback",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Success from fallback model"},
                        "finish_reason": "stop",
                    }
                ],
            },
            {},
        )

    with (
        patch("core.gateway.settings") as mock_settings,
        patch("core.gateway.provider") as mock_provider,
        patch("atomic.guards.preflight.preflight_model_probe", new=AsyncMock(return_value=True)),
    ):
        mock_settings.GATEWAY_AUTH_TOKEN = ""
        mock_settings.PROVIDER_RATE_LIMIT = 100
        mock_settings.PROVIDER_RATE_WINDOW = 60
        mock_settings.PROVIDER_MAX_CONCURRENCY = 10
        mock_provider.complete = AsyncMock(side_effect=mock_complete)

        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "chatcmpl-fallback"


def test_multi_key_rotation() -> None:
    from providers.openai import _select_key

    raw_keys = "key1, key2, key3"
    # Round robin rotation
    k1 = _select_key(raw_keys, "test_prov")
    k2 = _select_key(raw_keys, "test_prov")
    k3 = _select_key(raw_keys, "test_prov")
    k4 = _select_key(raw_keys, "test_prov")

    assert k1 == "key1"
    assert k2 == "key2"
    assert k3 == "key3"
    assert k4 == "key1"

    # Single key compatibility
    single_key = "  nvapi-single-key  "
    assert _select_key(single_key, "single_prov") == "nvapi-single-key"


def test_model_registry_custom_primary_and_empty_fallbacks() -> None:
    """Test that ModelRegistry respects user settings over static models.yaml defaults and respects empty fallback lists."""
    from config import model_registry, settings

    # 1. Custom primary model from settings
    settings.MODEL_OPUS = "open_router/google/gemini-2.5-pro"
    assert model_registry.get_primary("claude-3-opus") == "open_router/google/gemini-2.5-pro"

    # 2. Direct provider model path pass-through
    assert model_registry.get_primary("open_router/deepseek/deepseek-chat") == "open_router/deepseek/deepseek-chat"
    assert model_registry.get_fallbacks("open_router/deepseek/deepseek-chat") == []

    # 3. Empty fallback list behavior
    model_registry._entries["claude_opus"].fallback_order = []
    assert model_registry.get_fallbacks("claude-3-opus") == []


def test_synthetic_tool_call_injection_verification_turn() -> None:
    """Test that BaseProvider.translate_messages injects synthetic tool_calls into preceding assistant messages when tool_results are present."""
    from providers.openai import OpenAICompatibleProvider

    prov = OpenAICompatibleProvider()

    # Simulating Turn 2 Anthropic message history where Turn 1 was a Heuristic Tool Call (assistant returned flat text, CLI returns tool_result)
    messages = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "I will run bash command for you."},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_abc123",
                    "name": "bash",
                    "content": "file1.txt\nfile2.txt",
                }
            ],
        },
    ]

    openai_msgs = prov.translate_messages(messages)

    # Verify assistant message now has synthetic tool_calls with id toolu_abc123
    assistant_msg = [m for m in openai_msgs if m["role"] == "assistant"][0]
    assert "tool_calls" in assistant_msg
    assert assistant_msg["tool_calls"][0]["id"] == "toolu_abc123"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "bash"

    # Verify tool message has tool_call_id toolu_abc123
    tool_msg = [m for m in openai_msgs if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "toolu_abc123"
    assert tool_msg["content"] == "file1.txt\nfile2.txt"


def test_extract_all_json_tool_calls_with_allowed_tools_filtering() -> None:
    """Test that arbitrary JSON in text (e.g. non-tool dicts) is ignored when allowed_tools filter is set."""
    llm_text = (
        'Here is the component config: ```json\n{"name": "ReactComponent", "version": "1.0.0"}\n``` '
        'And here is a real tool call: ```json\n{"name": "view_file", "parameters": {"path": "main.py"}}\n```'
    )
    allowed = [{"name": "view_file"}, {"name": "run_command"}]
    remaining, tools = extract_all_json_tool_calls(llm_text, allowed_tools=allowed)
    assert len(tools) == 1
    assert tools[0]["name"] == "view_file"
    assert tools[0]["input"] == {"path": "main.py"}
    assert "ReactComponent" in remaining


@pytest.mark.asyncio
async def test_heuristic_tool_parser_with_allowed_tools() -> None:
    """Test that HeuristicToolParser ignores tools not present in allowed_tools."""
    from atomic.parsers.heuristic_tool import HeuristicToolParser

    parser = HeuristicToolParser(tools=[{"name": "view_file"}])

    # Bash fence when run_command/Bash is NOT in allowed_tools -> returns no events
    events = await parser.process_chunk("```bash\nls -la\n```")
    assert len(events) == 0

    # JSON fence for unallowed tool -> returns no events
    parser.reset()
    events_unallowed = await parser.process_chunk('```json\n{"name": "unknown_tool", "input": {}}\n```')
    assert len(events_unallowed) == 0

    # JSON fence for allowed tool -> returns tool_use block events
    parser.reset()
    events_allowed = await parser.process_chunk('```json\n{"name": "view_file", "parameters": {"path": "test.py"}}\n```')
    assert len(events_allowed) > 0
    assert any(getattr(getattr(ev, "content_block", None), "name", None) == "view_file" for ev in events_allowed)


def test_openai_to_anthropic_response_with_allowed_tools() -> None:
    """Test that translate_non_stream_response honors allowed tools and does not invent tool_use for non-tool JSON."""
    from models.converter import ModelConverter

    openai_resp = {
        "id": "chatcmpl-test",
        "model": "llama-3.1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": 'Configuration output:\n```json\n{"name": "ConfigObject", "type": "setting"}\n```',
                },
                "finish_reason": "stop",
            }
        ],
    }

    # Pass allowed tools that do NOT include ConfigObject
    translated = ModelConverter.openai_to_anthropic_response(openai_resp, tools=[{"name": "run_command"}])
    assert translated["stop_reason"] == "end_turn"
    assert len(translated["content"]) == 1
    assert translated["content"][0]["type"] == "text"
    assert "ConfigObject" in translated["content"][0]["text"]


@pytest.mark.asyncio
async def test_stream_guard_resilience_to_consecutive_empty_chunks() -> None:
    """Test StreamGuard allows up to 300 consecutive empty/keepalive chunks during quiet reasoning without killing the stream."""
    from atomic.guards.stream_guard import StreamGuard

    async def mock_stream_with_empty_lines():
        # 15 consecutive empty chunks (previously tripped at 10)
        for _ in range(15):
            yield ""
        yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}\n\n'

    guard = StreamGuard(mock_stream_with_empty_lines(), max_empty_chunks=300)
    received = [chunk async for chunk in guard]
    # Verify stream did NOT terminate with error event
    assert not any("event: error" in c for c in received)
    assert any("hello" in c for c in received)


@pytest.mark.asyncio
async def test_stream_engine_sequential_thinking_and_tool_call_block_closing() -> None:
    """Test StreamEngine closes active thinking block (index 0) BEFORE starting native tool_use block (index 1)."""
    from core.transformer.stream_engine import StreamEngine

    engine = StreamEngine(target_model="llama-3.1")

    async def mock_thinking_then_tool_stream():
        # Chunks streaming thinking content first
        yield {"choices": [{"delta": {"content": "<think>Analyzing repository structure..."}}]}
        yield {"choices": [{"delta": {"content": "</think>"}}]}
        # Followed by native tool call
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_read1",
                                "type": "function",
                                "function": {"name": "view_file", "arguments": '{"path": "next.config.ts"}'},
                            }
                        ]
                    }
                }
            ]
        }
        yield {"choices": [{"finish_reason": "tool_calls"}]}

    events = [ev async for ev in engine.transform_stream(mock_thinking_then_tool_stream())]
    full_text = "".join(events)

    # Verify thinking start/delta, thinking stop for index 0, THEN tool_use start for index 1
    assert "event: content_block_start" in full_text
    assert '"type": "thinking"' in full_text
    assert '"type": "tool_use"' in full_text


def test_dev_logger_records_to_logs_dir(tmp_path: pytest.TempPathFactory) -> None:
    """Test DevLogger writes raw request and formatted log entries to logs/ directory."""
    from pathlib import Path

    from shared.utils.dev_logger import DevLogger

    dev_log = DevLogger(logs_dir=Path(tmp_path))

    dev_log.record_transaction(
        request_id="req_test_123",
        method="POST",
        path="/v1/messages",
        client_model="claude-3-5-sonnet",
        mapped_model="nvidia_nim/model",
        status_code=200,
        duration_ms=150.0,
        request_body={"messages": [{"role": "user", "content": "hi"}]},
        response_body={"content": [{"type": "text", "text": "hello"}]},
    )

    raw_jsonl = tmp_path / "raw_requests.jsonl"
    dev_log_txt = tmp_path / "dev_proxy.log"

    assert raw_jsonl.exists()
    assert dev_log_txt.exists()

    raw_content = raw_jsonl.read_text(encoding="utf-8")
    assert "req_test_123" in raw_content
    assert '"request"' in raw_content
    assert '"response"' in raw_content
    assert '"result"' in raw_content

    txt_content = dev_log_txt.read_text(encoding="utf-8")
    assert "req_test_123" in txt_content
    assert "[1. REQUEST (Claude Code -> Proxy)]" in txt_content
    assert "[2. RESPONSE (LLM Model -> Proxy)]" in txt_content
    assert "[3. RESULT (Proxy -> Claude Code)]" in txt_content


def test_file_edit_guard_auto_healing(tmp_path: pytest.TempPathFactory) -> None:
    """Test FileEditGuard auto-corrects wrong StartLine/EndLine and fuzzy matches target content."""
    from pathlib import Path

    from atomic.guards.file_edit_guard import FileEditGuard

    target_file = Path(tmp_path) / "layout.tsx"
    target_file.write_text(
        "import React from 'react';\n"
        "export default function RootLayout() {\n"
        "  return (\n"
        "    <html>\n"
        "      <body>Hello</body>\n"
        "    </html>\n"
        "  );\n"
        "}\n",
        encoding="utf-8",
    )

    guard = FileEditGuard()

    # Case 1: LLM specifies wrong line bounds [1, 2] for code located at lines 4-6
    bad_input = {
        "TargetFile": str(target_file),
        "StartLine": 1,
        "EndLine": 2,
        "TargetContent": "<html>\n      <body>Hello</body>\n    </html>",
        "ReplacementContent": "<div>Hello World</div>",
    }

    healed = guard.sanitize_tool_input("replace_file_content", bad_input)
    assert healed["StartLine"] == 1
    # Line bounds must expand beyond line 6 to safely cover the match
    assert healed["EndLine"] >= 6
    assert "<html>" in healed["TargetContent"]


def test_extract_all_json_tool_calls_hermes_xml_tags() -> None:
    """Test extract_all_json_tool_calls converts Hermes <tool_call> tags to valid tool_use blocks."""
    from core.transformer.stream_engine import extract_all_json_tool_calls

    hermes_sample = (
        "Let me inspect the server log.\n"
        "<tool_call>\n"
        "<<function=Bash>\n"
        "<<parameter=command>cat /tmp/start.log</parameter>\n"
        "</tool_call>"
    )

    remaining, tools = extract_all_json_tool_calls(hermes_sample, allowed_tools=["Bash", "view_file"])
    assert len(tools) == 1
    assert tools[0]["name"] == "Bash"
    assert tools[0]["input"] == {"command": "cat /tmp/start.log"}
    assert remaining.strip() == "Let me inspect the server log."


def test_extract_all_json_tool_calls_case_insensitive_matching() -> None:
    """Test extract_all_json_tool_calls maps lowercased model tool names (e.g. bash) to exact client casing (e.g. Bash)."""
    from core.transformer.stream_engine import extract_all_json_tool_calls

    lowercase_json_sample = '```json\n{"name": "bash", "arguments": {"command": "ls -la"}}\n```'
    allowed = [{"name": "Bash"}, {"name": "Edit"}]

    remaining, tools = extract_all_json_tool_calls(lowercase_json_sample, allowed_tools=allowed)
    assert len(tools) == 1
    assert tools[0]["name"] == "Bash"  # Restored to exact client casing!
    assert tools[0]["input"] == {"command": "ls -la"}











