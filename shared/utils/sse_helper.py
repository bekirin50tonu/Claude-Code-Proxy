"""SSE (Server-Sent Events) stateless helper utilities."""

import json
from typing import Any


def format_sse_event(event_type: str, data: dict[str, Any] | str) -> str:
    """Format an event type and data payload into standard SSE protocol text format.

    Output format:
    event: {event_type}
    data: {data}

    """
    payload_str = json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event_type}\ndata: {payload_str}\n\n"



def parse_sse_line(line: str) -> tuple[str | None, str | None]:
    """Parse a single SSE stream line into (field_name, field_value).

    Example lines:
    - 'event: content_block_delta' -> ('event', 'content_block_delta')
    - 'data: {"foo": "bar"}' -> ('data', '{"foo": "bar"}')
    """
    clean_line = line.strip()
    if not clean_line or clean_line.startswith(":"):
        return None, None

    if ":" in clean_line:
        field, val = clean_line.split(":", 1)
        return field.strip(), val.strip()
    return clean_line, ""


def create_ping_event() -> str:
    """Generate an SSE ping keep-alive event."""
    return format_sse_event("ping", {"type": "ping"})


class AnthropicSSEFormatter:
    """Stateless static class for generating Anthropic-compatible SSE string events."""

    @staticmethod
    def message_start(
        message_id: str = "msg_default",
        model: str = "claude-3-5-sonnet",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> str:
        """Generate event: message_start SSE string."""
        data = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            },
        }
        return f"event: message_start\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def text_start(index: int = 1) -> str:
        """Generate event: content_block_start for text block SSE string."""
        data = {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        }
        return f"event: content_block_start\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def text_delta(text: str, index: int = 1) -> str:
        """Generate event: content_block_delta for text_delta SSE string."""
        data = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
        return f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def thinking_start(index: int = 0) -> str:
        """Generate event: content_block_start for thinking block SSE string."""
        data = {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": ""},
        }
        return f"event: content_block_start\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def thinking_delta(thinking_text: str, index: int = 0) -> str:
        """Generate event: content_block_delta for thinking_delta SSE string."""
        data = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": thinking_text},
        }
        return f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def tool_use_start(index: int, tool_id: str, name: str) -> str:
        """Generate event: content_block_start for tool_use block SSE string."""
        data = {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        }
        return f"event: content_block_start\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def input_json_delta(partial_json: str, index: int = 1) -> str:
        """Generate event: content_block_delta for input_json_delta SSE string."""
        data = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        }
        return f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def block_stop(index: int = 1) -> str:
        """Generate event: content_block_stop SSE string."""
        data = {
            "type": "content_block_stop",
            "index": index,
        }
        return f"event: content_block_stop\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def message_delta(stop_reason: str = "end_turn", output_tokens: int = 0) -> str:
        """Generate event: message_delta SSE string."""
        data = {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": output_tokens},
        }
        return f"event: message_delta\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def message_stop() -> str:
        """Generate event: message_stop SSE string."""
        data = {"type": "message_stop"}
        return f"event: message_stop\ndata: {json.dumps(data)}\n\n"

