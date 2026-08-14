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
