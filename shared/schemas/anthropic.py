"""Anthropic API Request, Response, and SSE Event Pydantic Models & Dataclasses."""

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class AnthropicUsage:
    """Token consumption statistics for an Anthropic message transaction."""

    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class AnthropicContentBlock:
    """Represents a single content block (text, thinking, or tool_use) in Anthropic API."""

    type: str  # "text", "thinking", "tool_use"
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.type == "text" and self.text is not None:
            data["text"] = self.text
        elif self.type == "thinking" and self.thinking is not None:
            data["thinking"] = self.thinking
            if self.signature:
                data["signature"] = self.signature
        elif self.type == "tool_use":
            if self.id:
                data["id"] = self.id
            if self.name:
                data["name"] = self.name
            data["input"] = self.input or {}
        return data


@dataclass
class AnthropicMessageResponse:
    """Represents a complete Anthropic message response object."""

    id: str
    model: str
    role: str = "assistant"
    type: str = "message"
    content: list[AnthropicContentBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage = field(default_factory=AnthropicUsage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "model": self.model,
            "content": [b.to_dict() for b in self.content],
            "stop_reason": self.stop_reason,
            "stop_sequence": self.stop_sequence,
            "usage": self.usage.to_dict(),
        }


# --- Streaming SSE Models ---

class SSEBaseEvent(BaseModel):
    """Base Pydantic model for Anthropic SSE events."""

    model_config = ConfigDict(populate_by_name=True)

    def to_sse(self) -> str:
        """Serialize event object to SSE formatted string event: <type>\\ndata: <json>\\n\\n."""
        data_dict = self.model_dump(by_alias=True, exclude_none=True)
        event_name = data_dict.pop("type", getattr(self, "event_type", "message"))
        return f"event: {event_name}\ndata: {json.dumps(data_dict)}\n\n"


class UsageInfo(BaseModel):
    """Token usage counters."""

    input_tokens: int = 0
    output_tokens: int = 0


class MessageInfo(BaseModel):
    """Anthropic Message metadata object."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[dict[str, Any]] = Field(default_factory=list)
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: UsageInfo = Field(default_factory=UsageInfo)


class SSEMessageStartEvent(SSEBaseEvent):
    """`event: message_start`"""

    type: Literal["message_start"] = "message_start"
    message: MessageInfo


class ContentBlockSchema(BaseModel):
    """Content block descriptor inside content_block_start."""

    type: Literal["text", "thinking", "tool_use"]
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


class SSEContentBlockStartEvent(SSEBaseEvent):
    """`event: content_block_start`"""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlockSchema


class TextDelta(BaseModel):
    """Delta payload for text content."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(BaseModel):
    """Delta payload for thinking content."""

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class InputJsonDelta(BaseModel):
    """Delta payload for tool input partial JSON."""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class SSEContentBlockDeltaEvent(SSEBaseEvent):
    """`event: content_block_delta`"""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: TextDelta | ThinkingDelta | InputJsonDelta


class SSEContentBlockStopEvent(SSEBaseEvent):
    """`event: content_block_stop`"""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaInfo(BaseModel):
    """Delta details for message_delta."""

    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None


class SSEMessageDeltaEvent(SSEBaseEvent):
    """`event: message_delta`"""

    type: Literal["message_delta"] = "message_delta"
    delta: MessageDeltaInfo
    usage: UsageInfo


class SSEMessageStopEvent(SSEBaseEvent):
    """`event: message_stop`"""

    type: Literal["message_stop"] = "message_stop"
