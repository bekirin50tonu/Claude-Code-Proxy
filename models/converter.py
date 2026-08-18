"""Protocol Model Converter.

Provides static conversion functions between Anthropic Messages API structures
and OpenAI Chat Completions API structures using clean models/dataclasses.
"""

import json
import uuid
from typing import Any

from models.anthropic import (
    AnthropicContentBlock,
    AnthropicMessageResponse,
    AnthropicUsage,
)
from models.sse_schemas import (
    ContentBlock,
    InputJsonDelta,
    MessageDeltaInfo,
    MessageInfo,
    SSEContentBlockDeltaEvent,
    SSEContentBlockStartEvent,
    SSEContentBlockStopEvent,
    SSEMessageDeltaEvent,
    SSEMessageStartEvent,
    SSEMessageStopEvent,
    TextDelta,
    ThinkingDelta,
    UsageInfo,
)


class ModelConverter:
    """Static utility class for bidirectional Anthropic <-> OpenAI format conversions."""

    @staticmethod
    def anthropic_to_openai_tools(
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert Anthropic tools list to OpenAI function calling format."""
        if not tools:
            return None

        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name", "")
            if not name:
                continue
            schema = tool.get("input_schema") or tool.get("parameters") or {}
            if not isinstance(schema, dict):
                schema = {}
            if "type" not in schema:
                schema["type"] = "object"
            if "properties" not in schema:
                schema["properties"] = {}

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": schema,
                    },
                }
            )
        return openai_tools or None

    @staticmethod
    def anthropic_to_openai_messages(
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert Anthropic messages array (with system prompt) to OpenAI messages format."""
        openai_msgs: list[dict[str, Any]] = []

        # Process system prompt
        if system:
            sys_text = ""
            if isinstance(system, str):
                sys_text = system
            elif isinstance(system, list):
                sys_text = "\n".join(
                    b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
                )
            if sys_text.strip():
                openai_msgs.append({"role": "system", "content": sys_text.strip()})

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            raw_content = msg.get("content")

            if isinstance(raw_content, str):
                openai_msgs.append({"role": role, "content": raw_content})
                continue

            if not isinstance(raw_content, list):
                openai_msgs.append({"role": role, "content": ""})
                continue

            # Process structured content block array
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for block in raw_content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "thinking":
                    # Pass thinking context into prompt if relevant
                    think_str = block.get("thinking", "")
                    if think_str:
                        text_parts.append(f"<think>{think_str}</think>")
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", f"toolu_{uuid.uuid4().hex[:10]}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                elif btype == "tool_result":
                    tool_results.append(block)

            # Build assistant message with text and/or tool_calls
            if role == "assistant":
                msg_obj: dict[str, Any] = {"role": "assistant"}
                combined_text = "\n".join(text_parts).strip()
                msg_obj["content"] = combined_text or None
                if tool_calls:
                    msg_obj["tool_calls"] = tool_calls
                openai_msgs.append(msg_obj)

            # Build user tool_result messages
            elif role == "user":
                if tool_results:
                    for tr in tool_results:
                        tr_content = tr.get("content", "")
                        if isinstance(tr_content, list):
                            tr_content = "\n".join(
                                item.get("text", "")
                                for item in tr_content
                                if isinstance(item, dict) and item.get("type") == "text"
                            )
                        openai_msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": tr.get("tool_use_id", ""),
                                "content": str(tr_content),
                            }
                        )
                else:
                    openai_msgs.append({"role": "user", "content": "\n".join(text_parts)})

        return openai_msgs

    @staticmethod
    def openai_to_anthropic_response(
        openai_resp: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Convert standard OpenAI Chat Completions response to Anthropic Message response dict."""
        choices = openai_resp.get("choices", [])
        if not choices:
            return AnthropicMessageResponse(
                id=f"msg_{uuid.uuid4().hex[:10]}",
                model=openai_resp.get("model", ""),
                content=[AnthropicContentBlock(type="text", text="")],
            ).to_dict()

        choice = choices[0]
        message = choice.get("message", {})
        text_content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        tool_calls = message.get("tool_calls") or []

        if not reasoning and text_content:
            import re

            match = re.search(r"<(think|thought)>(.*?)</\1>", text_content, re.DOTALL | re.IGNORECASE)
            if match:
                think_str = match.group(2).strip()
                if think_str:
                    reasoning = think_str
                text_content = (text_content[: match.start()] + text_content[match.end() :]).strip()
            elif text_content.startswith("<think"):
                parts = re.split(r"</?(?:think|thought)>", text_content, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    reasoning = parts[1].strip()
                    text_content = "".join(parts[2:]).strip()

        blocks: list[AnthropicContentBlock] = []

        if reasoning:
            blocks.append(
                AnthropicContentBlock(
                    type="thinking",
                    thinking=reasoning,
                    signature=f"sig_{uuid.uuid4().hex[:8]}",
                )
            )

        has_tool_block = False
        if tool_calls:
            if text_content:
                blocks.append(AnthropicContentBlock(type="text", text=text_content))
            for tc in tool_calls:
                func = tc.get("function") or {}
                raw_args = func.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                blocks.append(
                    AnthropicContentBlock(
                        type="tool_use",
                        id=tc.get("id", f"toolu_{uuid.uuid4().hex[:10]}"),
                        name=func.get("name", ""),
                        input=args,
                    )
                )
            has_tool_block = True
        elif text_content:
            from core.transformer.stream_engine import extract_all_json_tool_calls

            remaining, parsed_tools = extract_all_json_tool_calls(text_content, allowed_tools=tools)

            if parsed_tools:
                if remaining:
                    blocks.append(AnthropicContentBlock(type="text", text=remaining))
                for pt in parsed_tools:
                    blocks.append(
                        AnthropicContentBlock(
                            type="tool_use",
                            id=pt.get("id", f"toolu_{uuid.uuid4().hex[:10]}"),
                            name=pt.get("name", ""),
                            input=pt.get("input", {}),
                        )
                    )
                has_tool_block = True
            else:
                blocks.append(AnthropicContentBlock(type="text", text=text_content))

        if not blocks:
            blocks.append(AnthropicContentBlock(type="text", text=""))

        finish_reason = choice.get("finish_reason", "stop")
        if finish_reason == "tool_calls" or tool_calls or has_tool_block:
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        usage_data = openai_resp.get("usage", {})
        usage = AnthropicUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
        )

        resp = AnthropicMessageResponse(
            id=openai_resp.get("id") or f"msg_{uuid.uuid4().hex[:10]}",
            model=openai_resp.get("model", ""),
            content=blocks,
            stop_reason=stop_reason,
            usage=usage,
        )
        return resp.to_dict()

    @staticmethod
    def build_sse_message_start(message_id: str, model: str) -> SSEMessageStartEvent:
        """Create SSEMessageStartEvent object."""
        return SSEMessageStartEvent(
            message=MessageInfo(
                id=message_id,
                model=model,
                role="assistant",
                content=[],
                usage=UsageInfo(input_tokens=0, output_tokens=0),
            )
        )

    @staticmethod
    def build_sse_block_start(
        index: int, block_type: str, metadata: dict[str, Any] | None = None
    ) -> SSEContentBlockStartEvent:
        """Create SSEContentBlockStartEvent object."""
        meta = metadata or {}
        block = ContentBlock(
            type=block_type,  # type: ignore[arg-type]
            text="" if block_type == "text" else None,
            thinking="" if block_type == "thinking" else None,
            signature=f"sig_{uuid.uuid4().hex[:8]}" if block_type == "thinking" else None,
            id=meta.get("id") if block_type == "tool_use" else None,
            name=meta.get("name") if block_type == "tool_use" else None,
            input={} if block_type == "tool_use" else None,
        )
        return SSEContentBlockStartEvent(index=index, content_block=block)

    @staticmethod
    def build_sse_block_delta(
        index: int, delta_type: str, value: str
    ) -> SSEContentBlockDeltaEvent:
        """Create SSEContentBlockDeltaEvent object."""
        if delta_type == "text_delta":
            delta = TextDelta(text=value)
        elif delta_type == "thinking_delta":
            delta = ThinkingDelta(thinking=value)
        else:
            delta = InputJsonDelta(partial_json=value)
        return SSEContentBlockDeltaEvent(index=index, delta=delta)

    @staticmethod
    def build_sse_block_stop(index: int) -> SSEContentBlockStopEvent:
        """Create SSEContentBlockStopEvent object."""
        return SSEContentBlockStopEvent(index=index)

    @staticmethod
    def build_sse_message_delta(
        stop_reason: str, output_tokens: int
    ) -> SSEMessageDeltaEvent:
        """Create SSEMessageDeltaEvent object."""
        return SSEMessageDeltaEvent(
            delta=MessageDeltaInfo(stop_reason=stop_reason),
            usage=UsageInfo(output_tokens=output_tokens),
        )

    @staticmethod
    def build_sse_message_stop() -> SSEMessageStopEvent:
        """Create SSEMessageStopEvent object."""
        return SSEMessageStopEvent()
