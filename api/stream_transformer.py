import json
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any


def build_heuristic_input(
    tool_name: str, remaining_text: str, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Dynamically construct input dictionary matching the target tool schema."""
    if not remaining_text:
        return {}

    # Try parsing remaining_text as JSON if present
    if remaining_text.startswith("{") and remaining_text.endswith("}"):
        try:
            parsed = json.loads(remaining_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    target_tool = None
    if tools:
        for t in tools:
            if isinstance(t, dict) and t.get("name") == tool_name:
                target_tool = t
                break

    if target_tool and "input_schema" in target_tool:
        schema = target_tool.get("input_schema") or {}
        props = schema.get("properties", {})
        reqs = schema.get("required", [])

        inp: dict[str, Any] = {}
        if reqs:
            for req in reqs:
                inp[req] = remaining_text
            return inp
        if props:
            for key in ("prompt", "description", "input", "text", "query", "command", "name"):
                if key in props:
                    inp[key] = remaining_text
                    return inp
            first_key = next(iter(props.keys()))
            inp[first_key] = remaining_text
            return inp

    return {
        "prompt": remaining_text,
        "description": remaining_text,
        "input": remaining_text,
        "text": remaining_text,
    }


def parse_heuristic_tool_call(
    text: str, tools: list[dict[str, Any]] | None = None
) -> tuple[str, dict[str, Any] | None]:
    """Search text for patterns representing tool calls and parse them.

    Returns (clean_text, tool_use_block_dict) or (original_text, None).
    """
    # 1. Search for JSON block inside markdown, e.g. ```json ... ``` or ``` ... ```
    json_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL)

    json_str = None
    matched_span = None
    if match:
        json_str = match.group(1)
        matched_span = match.span()
    else:
        # 2. Search for any substring that looks like a JSON dictionary containing "name"
        for m in re.finditer(r"(\{.*?\})", text, re.DOTALL):
            candidate = m.group(1)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "name" in parsed:
                    json_str = candidate
                    matched_span = m.span()
                    break
            except json.JSONDecodeError:
                continue

    if json_str and matched_span:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "name" in parsed:
                name = parsed["name"]
                # Extract inputs (input, arguments, or args)
                inputs = (
                    parsed.get("input") or parsed.get("arguments") or parsed.get("args")
                )
                if inputs is None:
                    # Treat all keys except "name" as inputs
                    inputs = {k: v for k, v in parsed.items() if k != "name"}
                if not isinstance(inputs, dict):
                    inputs = {}

                tool_use_id = f"toolu_{uuid.uuid4().hex[:10]}"
                clean_text = text[: matched_span[0]].strip()

                return clean_text, {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": name,
                    "input": inputs,
                }
        except json.JSONDecodeError:
            pass

    # 3. Search for any slash command pattern e.g. /graphify, /mcp__stitch__generate_screen_from_text, /bash
    slash_pattern = r"/([a-zA-Z0-9_:-]+)\b"
    slash_match = re.search(slash_pattern, text)
    if slash_match:
        raw_cmd = slash_match.group(1)
        matched_span = slash_match.span()
        remaining_text = text[matched_span[1] :].strip()
        remaining_text = remaining_text.strip('"').strip("'").strip()

        # Match exact or partial tool name against available tools
        resolved_tool_name = raw_cmd
        if tools:
            for t in tools:
                if isinstance(t, dict):
                    t_name = t.get("name", "")
                    if t_name.lower() == raw_cmd.lower() or t_name.lower().endswith(raw_cmd.lower()):
                        resolved_tool_name = t_name
                        break

        tool_use_id = f"toolu_{uuid.uuid4().hex[:10]}"
        clean_text = text[: matched_span[0]].strip()
        inp = build_heuristic_input(resolved_tool_name, remaining_text, tools)
        return clean_text, {
            "type": "tool_use",
            "id": tool_use_id,
            "name": resolved_tool_name,
            "input": inp,
        }

    # 4. Search for shell command invocation e.g. "pnpm dev", "npm run dev", "git diff"
    cmd_pattern = r"\b(pnpm|npm|yarn|bun|git|pytest|cargo|pip|python)\s+([a-zA-Z0-9_.-]+(?:\s+[a-zA-Z0-9_.-]+)*)\b"
    cmd_match = re.search(cmd_pattern, text)
    if cmd_match:
        run_cmd_tool = None
        if tools:
            for t in tools:
                if isinstance(t, dict) and "run_command" in t.get("name", "").lower():
                    run_cmd_tool = t.get("name")
                    break
        if run_cmd_tool:
            full_cmd = cmd_match.group(0)
            tool_use_id = f"toolu_{uuid.uuid4().hex[:10]}"
            clean_text = text[: cmd_match.span()[0]].strip()
            return clean_text, {
                "type": "tool_use",
                "id": tool_use_id,
                "name": run_cmd_tool,
                "input": {"CommandLine": full_cmd, "Cwd": "."},
            }

    return text, None


def translate_non_stream_response(openai_resp: dict[str, Any]) -> dict[str, Any]:
    """Translate standard OpenAI Chat Completions response to Anthropic Message response."""
    choices = openai_resp.get("choices", [])
    if not choices:
        return {
            "id": f"msg_{uuid.uuid4().hex[:10]}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": openai_resp.get("model", ""),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    choice = choices[0]
    message = choice.get("message", {})
    text_content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    tool_calls = message.get("tool_calls") or []

    content_blocks = []

    # Handle native reasoning
    if reasoning:
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": reasoning,
                "signature": f"sig_{uuid.uuid4().hex[:8]}",
            }
        )

    # Heuristically check text_content for <think> tags if reasoning content was empty
    if not reasoning and "<think>" in text_content and "</think>" in text_content:
        think_match = re.search(r"<think>(.*?)</think>", text_content, re.DOTALL)
        if think_match:
            think_text = think_match.group(1).strip()
            if think_text:
                content_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": think_text,
                        "signature": f"sig_{uuid.uuid4().hex[:8]}",
                    }
                )
            # Remove the <think> block from text content
            text_content = (
                text_content[: think_match.start()] + text_content[think_match.end() :]
            )
            text_content = text_content.strip()

    # Heuristic parsing of text for tool calls
    clean_text, parsed_tool = parse_heuristic_tool_call(text_content)
    if parsed_tool:
        if clean_text:
            content_blocks.append({"type": "text", "text": clean_text})
        content_blocks.append(parsed_tool)
    else:
        # Standard flow
        if text_content:
            content_blocks.append({"type": "text", "text": text_content})

        # Translate API level tool calls
        for tc in tool_calls:
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:10]}"),
                    "name": tc["function"]["name"],
                    "input": args,
                }
            )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    # Map finish reasons
    finish_reason = choice.get("finish_reason", "stop")
    if finish_reason == "stop":
        stop_reason = "end_turn"
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage = openai_resp.get("usage", {})
    return {
        "id": openai_resp.get("id") or f"msg_{uuid.uuid4().hex[:10]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": openai_resp.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


class SSEStreamTransformer:
    def __init__(self, target_model: str, tools: list[dict[str, Any]] | None = None):
        self.target_model = target_model
        self.tools = tools
        self.message_id = f"msg_{uuid.uuid4().hex[:10]}"
        self.sent_start = False
        self.current_block_type: str | None = None  # "thinking", "text", "tool_use"
        self.current_block_index = -1

        # Buffers for tag detection
        self.text_buffer = ""
        self.in_think_tag = False

        # Keep track of active tool details
        self.active_tool_id: str | None = None
        self.active_tool_name: str | None = None

    def _yield_start_event(self) -> dict[str, Any]:
        """Generate message_start event."""
        self.sent_start = True
        return {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.target_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }

    def _yield_block_start(
        self, block_type: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Generate content_block_start event."""
        self.current_block_index += 1
        self.current_block_type = block_type

        content_block: dict[str, Any] = {"type": block_type}
        if block_type == "text":
            content_block["text"] = ""
        elif block_type == "thinking":
            content_block["thinking"] = ""
            content_block["signature"] = f"sig_{uuid.uuid4().hex[:8]}"
        elif block_type == "tool_use":
            content_block["id"] = (
                metadata.get("id") if metadata else f"toolu_{uuid.uuid4().hex[:10]}"
            )
            content_block["name"] = metadata.get("name", "") if metadata else ""
            content_block["input"] = {}

        return {
            "type": "content_block_start",
            "index": self.current_block_index,
            "content_block": content_block,
        }

    def _yield_block_stop(self) -> dict[str, Any]:
        """Generate content_block_stop event."""
        return {"type": "content_block_stop", "index": self.current_block_index}

    def _yield_delta(self, delta_type: str, value: str) -> dict[str, Any]:
        """Generate content_block_delta event."""
        delta: dict[str, Any] = {"type": delta_type}
        if delta_type == "text_delta":
            delta["text"] = value
        elif delta_type == "thinking_delta":
            delta["thinking"] = value
        elif delta_type == "input_json_delta":
            delta["partial_json"] = value

        return {
            "type": "content_block_delta",
            "index": self.current_block_index,
            "delta": delta,
        }

    async def transform(
        self, openai_chunks: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[str, None]:
        """Transform OpenAI Stream events to Anthropic SSE event strings."""
        if not self.sent_start:
            yield f"event: message_start\ndata: {json.dumps(self._yield_start_event())}\n\n"

        input_tokens = 0
        output_tokens = 0
        finish_reason = "stop"

        async for chunk in openai_chunks:
            # Capture usage info if sent
            if "usage" in chunk and chunk["usage"]:
                input_tokens = chunk["usage"].get("prompt_tokens", input_tokens)
                output_tokens = chunk["usage"].get("completion_tokens", output_tokens)

            choices = chunk.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            delta = choice.get("delta", {})

            # 1. Handle native reasoning content
            reasoning = delta.get("reasoning_content") or ""
            if reasoning:
                # Ensure we are inside a thinking block
                if self.current_block_type != "thinking":
                    if self.current_block_type is not None:
                        yield f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                    yield f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('thinking'))}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('thinking_delta', reasoning))}\n\n"
                continue

            # 2. Handle standard content text stream (checking for `<think>` tags)
            content = delta.get("content") or ""
            if content:
                self.text_buffer += content
                yielded_events = []

                # On-the-fly tag processing
                while self.text_buffer:
                    if not self.in_think_tag:
                        # Look for opening tag
                        idx = self.text_buffer.find("<think>")
                        if idx != -1:
                            # Flush text before tag
                            prefix_text = self.text_buffer[:idx]
                            if prefix_text:
                                if self.current_block_type != "text":
                                    if self.current_block_type is not None:
                                        yielded_events.append(
                                            f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                                        )
                                    yielded_events.append(
                                        f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('text'))}\n\n"
                                    )
                                yielded_events.append(
                                    f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('text_delta', prefix_text))}\n\n"
                                )

                            # Start thinking block
                            if self.current_block_type is not None:
                                yielded_events.append(
                                    f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                                )
                            yielded_events.append(
                                f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('thinking'))}\n\n"
                            )
                            self.in_think_tag = True
                            self.text_buffer = self.text_buffer[idx + len("<think>") :]
                        else:
                            # Protect partial matches (like `<th`)
                            partial_idx = self.text_buffer.rfind("<")
                            if partial_idx != -1 and "<think>".startswith(
                                self.text_buffer[partial_idx:]
                            ):
                                text_to_flush = self.text_buffer[:partial_idx]
                                self.text_buffer = self.text_buffer[partial_idx:]
                            else:
                                text_to_flush = self.text_buffer
                                self.text_buffer = ""

                            if text_to_flush:
                                if self.current_block_type != "text":
                                    if self.current_block_type is not None:
                                        yielded_events.append(
                                            f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                                        )
                                    yielded_events.append(
                                        f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('text'))}\n\n"
                                    )
                                yielded_events.append(
                                    f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('text_delta', text_to_flush))}\n\n"
                                )
                            break
                    else:
                        # Look for closing tag
                        idx = self.text_buffer.find("</think>")
                        if idx != -1:
                            # Flush thinking before tag
                            prefix_text = self.text_buffer[:idx]
                            if prefix_text:
                                if self.current_block_type != "thinking":
                                    if self.current_block_type is not None:
                                        yielded_events.append(
                                            f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                                        )
                                    yielded_events.append(
                                        f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('thinking'))}\n\n"
                                    )
                                yielded_events.append(
                                    f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('thinking_delta', prefix_text))}\n\n"
                                )

                            # Start text block again
                            if self.current_block_type is not None:
                                yielded_events.append(
                                    f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                                )
                            yielded_events.append(
                                f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('text'))}\n\n"
                            )
                            self.in_think_tag = False
                            self.text_buffer = self.text_buffer[idx + len("</think>") :]
                        else:
                            # Protect partial matches (like `</th`)
                            partial_idx = self.text_buffer.rfind("<")
                            if partial_idx != -1 and "</think>".startswith(
                                self.text_buffer[partial_idx:]
                            ):
                                text_to_flush = self.text_buffer[:partial_idx]
                                self.text_buffer = self.text_buffer[partial_idx:]
                            else:
                                text_to_flush = self.text_buffer
                                self.text_buffer = ""

                            if text_to_flush:
                                if self.current_block_type != "thinking":
                                    if self.current_block_type is not None:
                                        yielded_events.append(
                                            f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                                        )
                                    yielded_events.append(
                                        f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('thinking'))}\n\n"
                                    )
                                yielded_events.append(
                                    f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('thinking_delta', text_to_flush))}\n\n"
                                )
                            break

                for ev in yielded_events:
                    yield ev
                continue

            # 3. Handle tool calls delta
            tool_calls = delta.get("tool_calls") or []
            if tool_calls:
                tc = tool_calls[0]
                tc_id = tc.get("id")
                tc_func = tc.get("function") or {}
                tc_name = tc_func.get("name")
                tc_args = tc_func.get("arguments") or ""

                if tc_id or tc_name:
                    if self.current_block_type is not None:
                        yield f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                    # Capture tool details
                    self.active_tool_id = (
                        tc_id or self.active_tool_id or f"toolu_{uuid.uuid4().hex[:10]}"
                    )
                    self.active_tool_name = tc_name or self.active_tool_name or ""
                    yield f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('tool_use', {'id': self.active_tool_id, 'name': self.active_tool_name}))}\n\n"

                if tc_args:
                    yield f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('input_json_delta', tc_args))}\n\n"
                continue

        # Flush any remaining text in text_buffer
        if self.text_buffer:
            clean_text, parsed_tool = parse_heuristic_tool_call(self.text_buffer, self.tools)
            if parsed_tool:
                if clean_text:
                    if self.current_block_type is None:
                        yield f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('text'))}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('text_delta', clean_text))}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                elif self.current_block_type is not None:
                    yield f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"

                tool_id = parsed_tool.get("id", f"toolu_{uuid.uuid4().hex[:10]}")
                tool_name = parsed_tool.get("name", "")
                tool_input = json.dumps(parsed_tool.get("input", {}))
                yield f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('tool_use', {'id': tool_id, 'name': tool_name}))}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('input_json_delta', tool_input))}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
                finish_reason = "tool_calls"
            else:
                if self.current_block_type is None:
                    yield f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('text'))}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('text_delta', self.text_buffer))}\n\n"
            self.text_buffer = ""

        # Close any open content block or emit fallback text block if 0 blocks yielded
        if self.current_block_index == -1:
            yield f"event: content_block_start\ndata: {json.dumps(self._yield_block_start('text'))}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps(self._yield_delta('text_delta', ''))}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"
        elif self.current_block_type is not None:
            yield f"event: content_block_stop\ndata: {json.dumps(self._yield_block_stop())}\n\n"

        # Output Message Delta (usage + stop reason)
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'
