from typing import Any

import pytest

from api.mock import check_mock_request, extract_command_prefix, extract_filepaths
from api.stream_transformer import (
    SSEStreamTransformer,
    parse_heuristic_tool_call,
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

    # 2. Test messages translation with system prompt
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Ok"},
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
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "success",
                    "is_error": False,
                }
            ],
        },
    ]
    system = "You are a proxy"
    translated = provider.translate_messages(messages, system)

    assert len(translated) == 4
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


def test_heuristic_tool_call_parser() -> None:
    # Text with markdown JSON tool call
    text = (
        "Let me run this command.\n"
        "```json\n"
        "{\n"
        '  "name": "default_api:run_command",\n'
        '  "arguments": {\n'
        '    "CommandLine": "git diff"\n'
        "  }\n"
        "}\n"
        "```"
    )
    clean, tool = parse_heuristic_tool_call(text)
    assert tool is not None
    assert clean == "Let me run this command."
    assert tool["name"] == "default_api:run_command"
    assert tool["input"]["CommandLine"] == "git diff"

    # Text with MCP slash command
    mcp_text = (
        "I'll generate the screen using MCP.\n"
        '/mcp__stitch__generate_screen_from_text "Login form with email and password"'
    )
    clean_mcp, tool_mcp = parse_heuristic_tool_call(mcp_text)
    # Text with /graphify slash command
    graphify_text = "Updating design graph.\n/graphify"
    clean_g, tool_g = parse_heuristic_tool_call(graphify_text)
    assert tool_g is not None
    assert clean_g == "Updating design graph."
    assert tool_g["name"] == "graphify"


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
    assert "key_statuses" in res_data
    assert "MODEL_OPUS" in res_data["configs"]
    assert "OPENROUTER_API_KEY" in res_data["configs"]

    # 3. Test GET /dashboard HTML page
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "HERMES GATE" in resp.text

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

    with patch("api.routes.settings") as mock_settings:
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

    with patch("api.routes.settings") as mock_settings:
        mock_settings.GATEWAY_AUTH_TOKEN = ""  # Disabled
        mock_settings.PROVIDER_RATE_LIMIT = 100
        mock_settings.PROVIDER_RATE_WINDOW = 60
        mock_settings.PROVIDER_MAX_CONCURRENCY = 10

        # Request with no auth header should pass through (no 401)
        with (
            patch("api.routes.rate_limiter") as mock_rl,
            patch("api.routes.provider") as mock_provider,
            patch("api.routes.concurrency_semaphore"),
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
    from router.circuit_breaker import CircuitBreaker

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

    from router.circuit_breaker import CircuitBreaker, CircuitState

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

    from router.circuit_breaker import CircuitBreaker, CircuitState

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
    from router.rate_limit_parser import RateLimitState

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
    from router.rate_limit_parser import RateLimitState

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
    from router.rate_limit_parser import RateLimitState

    state = RateLimitState("unknown-model")
    assert state.has_headroom()  # No info → assume OK


# ===========================================================================
# NEW: Model Router Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_model_router_fallback() -> None:
    """Primary with open CB → router picks fallback."""
    from unittest.mock import AsyncMock, patch

    from router.circuit_breaker import CircuitBreakerRegistry
    from router.model_router import ModelRouter

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
        patch("router.model_router.circuit_breaker_registry", fresh_registry),
        patch(
            "guards.preflight.preflight_model_probe", new=AsyncMock(return_value=True)
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

    from router.circuit_breaker import CircuitBreakerRegistry
    from router.model_router import AllModelsUnavailableError, ModelRouter

    fresh_registry = CircuitBreakerRegistry()
    router = ModelRouter()

    async def _always_false(_model_id: str) -> bool:
        return False

    with (
        patch("router.model_router.circuit_breaker_registry", fresh_registry),
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
    from config import ModelMetadata
    from guards.token_budget import TokenBudgetGuard

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
    from config import ModelMetadata
    from guards.token_budget import TokenBudgetGuard

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
    from guards.stream_guard import StreamGuard

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

    from guards.stream_guard import StreamGuard

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
    from guards.stream_guard import StreamGuard

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

    with patch("api.routes.settings") as mock_settings:
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

    with patch("api.routes.settings") as mock_settings:
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

    with patch("api.routes.settings") as mock_settings:
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

    from server import app

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
        patch("api.routes.settings") as mock_settings,
        patch("api.routes.provider") as mock_provider,
        patch("guards.preflight.preflight_model_probe", new=AsyncMock(return_value=True)),
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


