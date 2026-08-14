"""
Unit tests for models package.
"""

from models import (
    AnthropicContentBlock,
    AnthropicMessageResponse,
    AnthropicUsage,
    CircuitState,
    CircuitStatus,
    ClientModelMapping,
    RateLimitStatus,
    RequestLogEntry,
    TelemetrySummary,
    UpstreamModelConfig,
)


def test_request_log_entry_model() -> None:
    entry = RequestLogEntry(
        id="req_1_12345",
        timestamp="12:00:00",
        method="POST",
        path="/v1/messages",
        client_model="claude-3-5-sonnet",
        mapped_model="nvidia_nim/meta/llama-3.1-70b-instruct",
        status_code=200,
        duration_ms=150.5,
        mocked=False,
        fallbacks_used=[],
        request_body={"model": "claude-3-5-sonnet"},
        response_body={"id": "msg_1"},
    )
    assert entry.is_success is True
    assert entry.is_error is False
    dict_full = entry.to_dict(include_payload=True)
    assert "request_body" in dict_full
    assert "response_body" in dict_full

    dict_light = entry.to_dict(include_payload=False)
    assert "request_body" not in dict_light


def test_telemetry_summary_model() -> None:
    summary = TelemetrySummary(
        total_requests=10,
        mocked_requests=3,
        error_count=1,
        active_concurrency=2,
    )
    assert summary.mock_ratio_percent == 30


def test_circuit_models() -> None:
    assert CircuitState.CLOSED == "closed"
    assert CircuitState.OPEN == "open"
    assert CircuitState.HALF_OPEN == "half_open"

    status = CircuitStatus(
        model_id="test_model",
        state=CircuitState.CLOSED,
        failure_count=0,
        failure_threshold=5,
        recovery_timeout_s=60.0,
    )
    assert status.is_available is True


def test_rate_limit_model() -> None:
    rl = RateLimitStatus(
        model_id="test_model",
        req_limit=100,
        req_remaining=50,
        tok_limit=1000,
        tok_remaining=500,
        has_headroom=True,
    )
    d = rl.to_dict()
    assert d["model_id"] == "test_model"
    assert d["has_headroom"] is True


def test_routing_models() -> None:
    mapping = ClientModelMapping(
        client_model="claude-opus-5",
        label="CLAUDE-OPUS-5",
        resolved_target="nvidia_nim/llama",
        is_fallback=False,
        step_name="PRIMARY DIRECT",
        chain=["nvidia_nim/llama"],
    )
    assert mapping.to_dict()["step_name"] == "PRIMARY DIRECT"

    cfg = UpstreamModelConfig(
        model_id="nvidia_nim/llama",
        provider="nvidia_nim",
        max_context_tokens=1_000_000,
    )
    assert cfg.max_context_tokens == 1_000_000


def test_anthropic_payload_models() -> None:
    usage = AnthropicUsage(input_tokens=10, output_tokens=20)
    b_text = AnthropicContentBlock(type="text", text="Hello")
    b_think = AnthropicContentBlock(type="thinking", thinking="Thinking...", signature="sig_1")
    b_tool = AnthropicContentBlock(type="tool_use", id="t1", name="run_cmd", input={"cmd": "ls"})

    resp = AnthropicMessageResponse(
        id="msg_123",
        model="claude-3-5-sonnet",
        content=[b_text, b_think, b_tool],
        usage=usage,
    )
    resp_dict = resp.to_dict()
    assert resp_dict["id"] == "msg_123"
    assert len(resp_dict["content"]) == 3
    assert resp_dict["content"][0]["type"] == "text"
    assert resp_dict["usage"]["output_tokens"] == 20
