"""Mock streaming and request telemetry recording helpers for Core Gateway."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from config import stats


def extract_session_id(request: Any, body: dict[str, Any] | None = None) -> str:
    """Extract session ID reliably from HTTP headers or nested body metadata."""
    if hasattr(request, "headers"):
        header_sid = request.headers.get("x-session-id") or request.headers.get("x-conversation-id")
        if header_sid and str(header_sid).strip():
            return str(header_sid).strip()

    if body and isinstance(body, dict):
        meta = body.get("metadata")
        if isinstance(meta, dict):
            sid = meta.get("session_id") or meta.get("conversation_id")
            if sid and isinstance(sid, str) and sid.strip():
                return sid.strip()
            user_id = meta.get("user_id")
            if isinstance(user_id, str) and "session_id" in user_id:
                try:
                    uid_data = json.loads(user_id)
                    if isinstance(uid_data, dict) and uid_data.get("session_id"):
                        return str(uid_data["session_id"]).strip()
                except Exception:
                    pass
    return "default_session"


def record_request_log(
    method: str,
    path: str,
    client_model: str,
    target_model: str,
    status_code: int,
    start_time: float,
    mocked: bool = False,
    fallbacks_used: list[str] | None = None,
    request_body: dict[str, Any] | None = None,
    response_body: dict[str, Any] | str | None = None,
    headers: dict[str, str] | None = None,
    error_details: dict[str, Any] | None = None,
    attempt_history: list[dict[str, Any]] | None = None,
) -> None:
    """Record request log telemetry to in-memory stats and disk sink."""
    stats.record_log(
        method,
        path,
        client_model,
        target_model,
        status_code,
        start_time,
        mocked,
        fallbacks_used,
        request_body=request_body,
        response_body=response_body,
        headers=headers,
        error_details=error_details,
        attempt_history=attempt_history,
    )


async def log_after_stream(
    gen: AsyncGenerator[str, None],
    method: str,
    path: str,
    client_model: str,
    target_model: str,
    start_time: float,
    mocked: bool,
    request_body: dict[str, Any] | None = None,
    response_body: dict[str, Any] | str | None = None,
) -> AsyncGenerator[str, None]:
    """Wrap SSE stream generator and record request log telemetry in finally block upon completion."""
    try:
        async for chunk in gen:
            yield chunk
    finally:
        record_request_log(
            method,
            path,
            client_model,
            target_model,
            200,
            start_time,
            mocked=mocked,
            request_body=request_body,
            response_body=response_body,
        )


async def stream_mock_response(mock_data: dict[str, Any]) -> AsyncGenerator[str, None]:
    """Yield mock data in Anthropic compatible SSE format."""
    model = mock_data.get("model", "claude-3-5-sonnet-latest")
    msg_id = mock_data.get("id", "msg_mock")
    text_content = mock_data["content"][0]["text"] if mock_data.get("content") else " "

    _msg_start = json.dumps(
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
    )
    _blk_start = json.dumps(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    )
    _blk_delta = json.dumps(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text_content},
        }
    )
    _blk_stop = json.dumps({"type": "content_block_stop", "index": 0})
    _msg_delta = json.dumps(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }
    )

    yield f"event: message_start\ndata: {_msg_start}\n\n"
    yield f"event: content_block_start\ndata: {_blk_start}\n\n"
    yield f"event: content_block_delta\ndata: {_blk_delta}\n\n"
    yield f"event: content_block_stop\ndata: {_blk_stop}\n\n"
    yield f"event: message_delta\ndata: {_msg_delta}\n\n"
    yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'
