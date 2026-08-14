"""Core Stream Engine and Transformer Orchestrator."""

import json
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from atomic.guards.subagent import SubagentGuard
from atomic.parsers.heuristic_tool import HeuristicToolParser
from atomic.parsers.thinking import ThinkingParser
from cli.session import session_manager
from models.converter import ModelConverter


def safe_parse_json(json_str: str | None) -> Any:
    """Parse JSON string robustly, handling unescaped control characters and truncated JSON."""
    if not json_str or not isinstance(json_str, str):
        return None
    stripped = json_str.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped, strict=False)
    except Exception:
        pass

    try:
        fixed = stripped.replace("\r\n", "\n").replace("\r", "\n")
        return json.loads(fixed, strict=False)
    except Exception:
        pass

    for suffix in ['"', '}', '}"', '}}', '"}}', '}]}', '"}]}']:
        try:
            return json.loads(stripped + suffix, strict=False)
        except Exception:
            pass

    return None


def _find_balanced_json_objects(text: str) -> list[tuple[int, int]]:
    """Find (start, end) spans of all top-level balanced ``{}`` blocks."""
    results: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            escape_next = False
            for j in range(i, n):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_str:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        results.append((i, j + 1))
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return results


def _parse_tool_from_json(obj: dict[str, Any]) -> dict[str, Any] | None:
    """If *obj* looks like a tool call dict return a ``tool_use`` block."""
    if "name" not in obj or not isinstance(obj["name"], str):
        return None
    name: str = obj["name"]
    input_data = (
        obj.get("parameters")
        or obj.get("arguments")
        or obj.get("input")
        or obj.get("args")
    )
    if input_data is None:
        input_data = {k: v for k, v in obj.items() if k != "name"}
    if not isinstance(input_data, dict):
        input_data = {}
    return {
        "type": "tool_use",
        "id": f"toolu_{uuid.uuid4().hex[:10]}",
        "name": name,
        "input": input_data,
    }


def extract_all_json_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract all JSON tool-call structures embedded in text."""
    if not text:
        return text, []

    cleaned = re.sub(r"```(?:json|JSON)?\s*\n?", "", text)
    spans = _find_balanced_json_objects(cleaned)
    if not spans:
        return text, []

    tools: list[dict[str, Any]] = []
    remove_ranges: list[tuple[int, int]] = []

    for start, end in spans:
        json_str = cleaned[start:end]
        parsed = safe_parse_json(json_str)
        if not isinstance(parsed, dict):
            continue
        tool_block = _parse_tool_from_json(parsed)
        if tool_block is not None:
            tools.append(tool_block)
            remove_ranges.append((start, end))

    if not tools:
        return text, []

    parts: list[str] = []
    last = 0
    for s, e in sorted(remove_ranges):
        if s > last:
            parts.append(cleaned[last:s])
        last = max(last, e)
    if last < len(cleaned):
        parts.append(cleaned[last:])

    remaining = " ".join(p.strip() for p in parts if p.strip()).strip()
    return remaining, tools


def extract_json_tool_call(text: str) -> tuple[str, dict[str, Any] | None]:
    """Return the first extracted tool call from text."""
    remaining, tools = extract_all_json_tool_calls(text)
    if tools:
        return remaining, tools[0]
    return text, None


def translate_non_stream_response(openai_resp: dict[str, Any]) -> dict[str, Any]:
    """Translate non-stream OpenAI response to Anthropic response structure."""
    return ModelConverter.openai_to_anthropic_response(openai_resp)


class StreamEngine:
    """Core class for streaming Anthropic-compatible SSE events on-the-fly."""

    def __init__(
        self,
        target_model: str,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ):
        self.target_model = target_model
        self.tools = tools
        self.message_id = f"msg_{uuid.uuid4().hex[:10]}"
        self.session = session_manager.get_or_create_session(session_id)

        self.block_index = -1
        self.sent_start = False

        self.thinking_parser = ThinkingParser(block_index_provider=self._next_block_index)
        self.heuristic_tool_parser = HeuristicToolParser(block_index_provider=self._next_block_index)
        self.subagent_guard = SubagentGuard()

        self.accumulated_thinking: list[str] = []
        self.accumulated_text: list[str] = []
        self.accumulated_tool_calls: list[dict[str, Any]] = []
        self.accumulated_tool_args: list[str] = []
        self.final_stop_reason: str = "end_turn"
        self.final_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        self.current_block_type: str | None = None
        self.current_tool_index: int | None = None
        self.active_tool_id: str | None = None
        self.active_tool_name: str | None = None

    def _next_block_index(self) -> int:
        self.block_index += 1
        return self.block_index

    def _get_current_index(self) -> int:
        return max(0, self.block_index)

    def get_summary_response(self) -> dict[str, Any]:
        """Return summary dict of accumulated blocks for dashboard logging."""
        blocks: list[dict[str, Any]] = []
        full_think = "".join(self.accumulated_thinking).strip()
        if full_think:
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": full_think,
                    "signature": f"sig_{uuid.uuid4().hex[:8]}",
                }
            )
        full_text = "".join(self.accumulated_text).strip()
        if full_text:
            blocks.append({"type": "text", "text": full_text})
        for tc in self.accumulated_tool_calls:
            blocks.append(tc)
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.target_model,
            "content": blocks,
            "stop_reason": self.final_stop_reason,
            "stop_sequence": None,
            "usage": self.final_usage,
        }

    async def stream_response(
        self, upstream_stream: AsyncIterable[dict[str, Any] | str]
    ) -> AsyncGenerator[str, None]:
        """Async generator yielding Anthropic SSE event strings on-the-fly."""
        if not self.sent_start:
            self.sent_start = True
            start_event = ModelConverter.build_sse_message_start(self.message_id, self.target_model)
            yield start_event.to_sse()

        input_tokens = 0
        output_tokens = 0
        finish_reason = "stop"

        async for chunk in upstream_stream:
            if isinstance(chunk, bytes):
                try:
                    chunk = chunk.decode("utf-8")
                except Exception:
                    continue

            if isinstance(chunk, str):
                stripped = chunk.strip()
                if stripped.startswith("data:"):
                    raw_data = stripped[5:].strip()
                    if raw_data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(raw_data)
                    except Exception:
                        chunk = {"choices": [{"delta": {"content": chunk}}]}
                else:
                    chunk = {"choices": [{"delta": {"content": chunk}}]}

            if not isinstance(chunk, dict):
                continue

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

            # Reasoning content
            reasoning = delta.get("reasoning_content") or ""
            if reasoning:
                events = await self.thinking_parser.process_chunk({"choices": [{"delta": {"reasoning_content": reasoning}}]})
                for ev in events:
                    if hasattr(ev, "delta") and getattr(ev.delta, "type", None) == "thinking_delta":
                        self.accumulated_thinking.append(getattr(ev.delta, "thinking", ""))
                    yield ev.to_sse()
                continue

            # Native tool_calls delta
            tool_calls_delta = delta.get("tool_calls") or []
            if tool_calls_delta:
                for tc in tool_calls_delta:
                    tc_index = tc.get("index", 0)
                    tc_id = tc.get("id")
                    tc_func = tc.get("function") or {}
                    tc_name = tc_func.get("name")
                    tc_args = tc_func.get("arguments") or ""

                    if self.current_block_type != "tool_use" or self.current_tool_index != tc_index:
                        if self.current_block_type is not None:
                            if self.current_block_type == "tool_use" and self.active_tool_id and self.active_tool_name:
                                full_args_str = "".join(self.accumulated_tool_args)
                                parsed_args = safe_parse_json(full_args_str) or {}
                                if isinstance(parsed_args, dict):
                                    parsed_args = await self.subagent_guard.enforce_tool_call(self.active_tool_name, parsed_args)
                                self.accumulated_tool_calls.append(
                                    {
                                        "type": "tool_use",
                                        "id": self.active_tool_id,
                                        "name": self.active_tool_name,
                                        "input": parsed_args,
                                    }
                                )
                                self.accumulated_tool_args = []

                            stop_ev = ModelConverter.build_sse_block_stop(self._get_current_index())
                            yield stop_ev.to_sse()

                        self.current_tool_index = tc_index
                        self.active_tool_id = tc_id or f"toolu_{uuid.uuid4().hex[:10]}"
                        self.active_tool_name = tc_name or ""
                        self.current_block_type = "tool_use"

                        idx = self._next_block_index()
                        start_ev = ModelConverter.build_sse_block_start(
                            idx, "tool_use", {"id": self.active_tool_id, "name": self.active_tool_name}
                        )
                        yield start_ev.to_sse()
                    else:
                        if tc_id and not self.active_tool_id:
                            self.active_tool_id = tc_id
                        if tc_name and not self.active_tool_name:
                            self.active_tool_name = tc_name

                    if tc_args:
                        self.accumulated_tool_args.append(tc_args)
                        delta_ev = ModelConverter.build_sse_block_delta(
                            self._get_current_index(), "input_json_delta", tc_args
                        )
                        yield delta_ev.to_sse()

                finish_reason = "tool_calls"
                continue

            # Text stream content through atomic parsers
            content = delta.get("content") or ""
            if content:
                think_events = await self.thinking_parser.process_chunk(content)
                if think_events:
                    for ev in think_events:
                        if hasattr(ev, "delta"):
                            dtype = getattr(ev.delta, "type", None)
                            if dtype == "thinking_delta":
                                self.accumulated_thinking.append(getattr(ev.delta, "thinking", ""))
                            elif dtype == "text_delta":
                                self.accumulated_text.append(getattr(ev.delta, "text", ""))
                        yield ev.to_sse()
                else:
                    tool_events = await self.heuristic_tool_parser.process_chunk(content)
                    if tool_events:
                        for ev in tool_events:
                            yield ev.to_sse()
                    else:
                        if self.current_block_type != "text":
                            if self.current_block_type is not None:
                                stop_ev = ModelConverter.build_sse_block_stop(self._get_current_index())
                                yield stop_ev.to_sse()
                            idx = self._next_block_index()
                            start_ev = ModelConverter.build_sse_block_start(idx, "text")
                            self.current_block_type = "text"
                            yield start_ev.to_sse()

                        self.accumulated_text.append(content)
                        delta_ev = ModelConverter.build_sse_block_delta(
                            self._get_current_index(), "text_delta", content
                        )
                        yield delta_ev.to_sse()

        # Flush Atomic Parsers
        for flush_ev in await self.thinking_parser.flush():
            yield flush_ev.to_sse()

        for flush_ev in await self.heuristic_tool_parser.flush():
            yield flush_ev.to_sse()

        if self.current_block_type is not None:
            stop_ev = ModelConverter.build_sse_block_stop(self._get_current_index())
            yield stop_ev.to_sse()
            self.current_block_type = None

        if self.active_tool_id and self.active_tool_name:
            full_args_str = "".join(self.accumulated_tool_args)
            parsed_args = safe_parse_json(full_args_str) or {}

            if isinstance(parsed_args, dict):
                parsed_args = await self.subagent_guard.enforce_tool_call(self.active_tool_name, parsed_args)

            self.accumulated_tool_calls.append(
                {
                    "type": "tool_use",
                    "id": self.active_tool_id,
                    "name": self.active_tool_name,
                    "input": parsed_args,
                }
            )

        if not self.accumulated_tool_calls:
            full_streamed_text = "".join(self.accumulated_text)
            _rem, extracted_tools = extract_all_json_tool_calls(full_streamed_text)
            if extracted_tools:
                for tool in extracted_tools:
                    tool["input"] = await self.subagent_guard.enforce_tool_call(tool["name"], tool.get("input", {}))
                    idx = self._next_block_index()
                    yield ModelConverter.build_sse_block_start(idx, "tool_use", {"id": tool["id"], "name": tool["name"]}).to_sse()
                    yield ModelConverter.build_sse_block_delta(idx, "input_json_delta", json.dumps(tool["input"])).to_sse()
                    yield ModelConverter.build_sse_block_stop(idx).to_sse()
                    self.accumulated_tool_calls.append(tool)
                finish_reason = "tool_calls"

        if finish_reason == "tool_calls" or self.accumulated_tool_calls:
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        self.final_stop_reason = stop_reason
        self.final_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        self.session.record_tokens(input_tokens, output_tokens)

        yield ModelConverter.build_sse_message_delta(stop_reason, output_tokens).to_sse()
        yield ModelConverter.build_sse_message_stop().to_sse()

    async def transform(self, upstream_stream: AsyncIterable[dict[str, Any] | str]) -> AsyncGenerator[str, None]:
        """Backwards compatibility alias for transform()."""
        async for chunk in self.stream_response(upstream_stream):
            yield chunk



# Aliases for backward compatibility
StreamTransformer = StreamEngine
SSEStreamTransformer = StreamEngine

