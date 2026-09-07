"""Core Stream Engine and Transformer Orchestrator."""

import json
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from loguru import logger

from atomic.guards.subagent import SubagentGuard
from atomic.parsers.heuristic_tool import (
    HeuristicToolStatefulParser,
)
from atomic.parsers.thinking import ThinkingStatefulParser
from cli.session import session_manager
from models.converter import ModelConverter
from shared.utils.sse_helper import AnthropicSSEFormatter


def safe_parse_json(json_str: str | None) -> Any:
    """Parse JSON string robustly, handling unescaped control characters, Python literals, single quotes, and truncated JSON."""
    if not json_str or not isinstance(json_str, str):
        return None
    stripped = json_str.strip()
    if not stripped:
        return None

    # 1. Fast path: standard json.loads
    try:
        return json.loads(stripped, strict=False)
    except Exception:
        pass

    # 2. Fix line endings
    try:
        fixed = stripped.replace("\r\n", "\n").replace("\r", "\n")
        return json.loads(fixed, strict=False)
    except Exception:
        pass

    # 3. Auto-close truncated stream tags/quotes
    from atomic.parsers.auto_close_tag import AutoCloseTagParser
    repaired = AutoCloseTagParser.repair_truncated_stream(stripped)

    # Convert single quotes to double quotes if valid Python dict repr
    if "'" in repaired:
        if '"' not in repaired:
            repaired = repaired.replace("'", '"')
        else:
            repaired = re.sub(r"'([a-zA-Z0-9_]+)'\s*:", r'"\1":', repaired)
            repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)

    # Fix Python boolean/None literals: True/False/None -> true/false/null
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)

    # Remove trailing commas
    repaired = re.sub(r",\s*\}", "}", repaired)
    repaired = re.sub(r",\s*\]", "]", repaired)

    try:
        return json.loads(repaired, strict=False)
    except Exception:
        pass

    # Append missing closing quotes and braces
    for suffix in ['"', '}', '}"', '}}', '"}}', '}]}', '"}]}']:
        try:
            return json.loads(repaired + suffix, strict=False)
        except Exception:
            pass

    # Fallback regex key-value extraction for JSON objects
    extracted: dict[str, Any] = {}
    pattern = r'["\']([a-zA-Z0-9_]+)["\']\s*:\s*(["\'](?:[^"\'\\]|\\.)*["\']|true|false|null|\d+(?:\.\d+)?|\[.*?\]|\{.*?\})'
    matches = re.findall(pattern, stripped, re.DOTALL)
    if matches:
        for k, v in matches:
            v_str = v.strip('\'"')
            if v_str.lower() == "true":
                extracted[k] = True
            elif v_str.lower() == "false":
                extracted[k] = False
            elif v_str.lower() == "null":
                extracted[k] = None
            else:
                try:
                    extracted[k] = json.loads(v)
                except Exception:
                    extracted[k] = v_str
        return extracted

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


def _parse_tool_from_json(
    obj: dict[str, Any],
    allowed_names: set[str] | None = None,
    allowed_name_map: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """If *obj* looks like a tool call dict return a ``tool_use`` block."""
    if "name" not in obj or not isinstance(obj["name"], str):
        return None
    name: str = obj["name"]
    resolved_name = name
    if allowed_names is not None:
        if name in allowed_names:
            resolved_name = name
        elif allowed_name_map and name.lower() in allowed_name_map:
            resolved_name = allowed_name_map[name.lower()]
        else:
            return None

    input_data = (
        obj.get("parameters")
        or obj.get("arguments")
        or obj.get("input")
        or obj.get("args")
    )
    if isinstance(input_data, str):
        parsed = safe_parse_json(input_data)
        if isinstance(parsed, dict):
            input_data = parsed
        else:
            input_data = {}
    elif input_data is None:
        input_data = {k: v for k, v in obj.items() if k not in ("name", "id", "type")}
    if not isinstance(input_data, dict):
        input_data = {}
    return {
        "type": "tool_use",
        "id": f"toolu_{uuid.uuid4().hex[:10]}",
        "name": resolved_name,
        "input": input_data,
    }


def extract_all_json_tool_calls(
    text: str,
    allowed_tools: list[dict[str, Any]] | list[str] | set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract all JSON and XML/Hermes tool-call structures embedded in text."""
    if not text:
        return text, []

    from atomic.parsers.auto_close_tag import AutoCloseTagParser
    from core.interceptor.json_repair import JSONRepairNormalizer

    repaired_input = AutoCloseTagParser.repair_truncated_stream(text)
    safe_text = JSONRepairNormalizer.fix_angle_brackets(repaired_input)

    allowed_names: set[str] | None = None
    allowed_name_map: dict[str, str] | None = None
    if allowed_tools is not None:
        allowed_names = set()
        allowed_name_map = {}
        for item in allowed_tools:
            if isinstance(item, str):
                allowed_names.add(item)
                allowed_name_map[item.lower()] = item
            elif isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                tname = item["name"]
                allowed_names.add(tname)
                allowed_name_map[tname.lower()] = tname

    tools: list[dict[str, Any]] = []

    # 1. Parse <tool_call> tags (Hermes / XML / JSON in <tool_call>)
    def replace_tool_call_tag(match: re.Match[str]) -> str:
        block = match.group(1).strip()
        fn_match = re.search(r"<<?function=(.*?)>>?", block)
        if fn_match:
            name = fn_match.group(1).strip()
            resolved_name = name
            if allowed_names is not None:
                if name in allowed_names:
                    resolved_name = name
                elif allowed_name_map and name.lower() in allowed_name_map:
                    resolved_name = allowed_name_map[name.lower()]
                elif name:
                    resolved_name = name

            if resolved_name:
                params: dict[str, Any] = {}
                param_matches = re.findall(r"<<?parameter=(.*?)>>?(.*?)</parameter>", block, re.DOTALL)
                for p_name, p_val in param_matches:
                    params[p_name.strip()] = p_val.strip()
                tools.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:10]}",
                        "name": resolved_name,
                        "input": params,
                    }
                )
                return ""
        parsed = safe_parse_json(block)
        if isinstance(parsed, dict):
            t_block = _parse_tool_from_json(parsed, allowed_names=allowed_names, allowed_name_map=allowed_name_map)
            if t_block:
                tools.append(t_block)
                return ""
        return ""  # Always strip <tool_call> XML noise from text output

    cleaned_text = re.sub(r"<tool_call>(.*?)</tool_call>", replace_tool_call_tag, safe_text, flags=re.DOTALL)

    # 2. Parse [TOOL_CALLS] or [TOOL_CALL] tags
    def replace_tool_calls_bracket(match: re.Match[str]) -> str:
        block = match.group(1).strip()
        parsed = safe_parse_json(block)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    t_block = _parse_tool_from_json(item, allowed_names=allowed_names, allowed_name_map=allowed_name_map)
                    if t_block:
                        tools.append(t_block)
            return ""
        elif isinstance(parsed, dict):
            t_block = _parse_tool_from_json(parsed, allowed_names=allowed_names, allowed_name_map=allowed_name_map)
            if t_block:
                tools.append(t_block)
                return ""
        return match.group(0)

    cleaned_text = re.sub(
        r"\[TOOL_CALLS?\]\s*(\[.*?\]|\{.*?\})",
        replace_tool_calls_bracket,
        cleaned_text,
        flags=re.DOTALL,
    )

    # 3. Parse balanced JSON objects (standard embedded JSON codeblocks or text)
    cleaned = re.sub(r"```(?:json|JSON)?\s*\n?", "", cleaned_text)
    spans = _find_balanced_json_objects(cleaned)
    if spans:
        remove_ranges: list[tuple[int, int]] = []
        for start, end in spans:
            json_str = cleaned[start:end]
            parsed = safe_parse_json(json_str)
            if not isinstance(parsed, dict):
                continue
            tool_block = _parse_tool_from_json(parsed, allowed_names=allowed_names, allowed_name_map=allowed_name_map)
            if tool_block is not None:
                tools.append(tool_block)
                remove_ranges.append((start, end))

        if remove_ranges:
            parts: list[str] = []
            last = 0
            for s, e in sorted(remove_ranges):
                if s > last:
                    parts.append(cleaned[last:s])
                last = max(last, e)
            if last < len(cleaned):
                parts.append(cleaned[last:])
            cleaned_text = " ".join(p.strip() for p in parts if p.strip()).strip()

    cleaned_text = cleaned_text.strip()
    return cleaned_text, tools


def extract_json_tool_call(
    text: str,
    allowed_tools: list[dict[str, Any]] | list[str] | set[str] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Return the first extracted tool call from text."""
    remaining, tools = extract_all_json_tool_calls(text, allowed_tools=allowed_tools)
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
        is_stop_hook: bool = False,
        is_evaluator: bool = False,
    ):
        self.target_model = target_model
        self.tools = tools
        self.session_id = session_id
        self.is_evaluator = is_evaluator
        self.is_stop_hook = is_stop_hook or is_evaluator
        self.message_id = f"msg_{uuid.uuid4().hex[:10]}"
        self.session = session_manager.get_or_create_session(session_id)

        self.block_index = -1
        self.sent_start = False
        self.text_or_tool_emitted = False
        self.preemptive_text_emitted = False

        self.thinking_parser = ThinkingStatefulParser(block_index_provider=self._next_block_index)
        self.heuristic_tool_parser = HeuristicToolStatefulParser(block_index_provider=self._next_block_index, tools=self.tools)
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
        self.tool_state_map: dict[int, dict[str, Any]] = {}
        self.initial_pre_think_buffer: str = ""

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
            blocks.append({"type": "text", "text": " "})

        from atomic.sanitizers.deduplicator import DeDuplicator
        blocks = DeDuplicator.deduplicate(blocks)

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

    async def transform_stream(
        self, upstream_stream: AsyncIterable[dict[str, Any] | str | bytes]
    ) -> AsyncGenerator[str, None]:
        """Asynchronously transform upstream OpenAI/NIM SSE stream into Anthropic SSE stream."""
        if not self.sent_start:
            self.sent_start = True
            yield AnthropicSSEFormatter.message_start(
                message_id=self.message_id,
                model=self.target_model,
                input_tokens=0,
                output_tokens=0,
            )

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

            # 1. Native reasoning_content
            reasoning = delta.get("reasoning_content") or ""
            if reasoning:
                self.initial_pre_think_buffer = ""
                import asyncio

                from bot.live_bridge import live_bridge_manager
                asyncio.create_task(live_bridge_manager.dispatch_thinking_chunk(self.session.session_id, reasoning))
                events = await self.thinking_parser.process_chunk(
                    {"choices": [{"delta": {"reasoning_content": reasoning}}]}
                )
                for ev in events:
                    ev_type = getattr(ev, "type", None)
                    if ev_type == "content_block_start":
                        cb_type = getattr(getattr(ev, "content_block", None), "type", None)
                        self.current_block_type = cb_type
                    elif ev_type == "content_block_stop":
                        self.current_block_type = None

                    if hasattr(ev, "delta"):
                        dtype = getattr(ev.delta, "type", None)
                        if dtype == "thinking_delta":
                            self.accumulated_thinking.append(getattr(ev.delta, "thinking", ""))
                    yield ev.to_sse()
                continue

            # 2. Native tool_calls delta (Isolated state tracking per tc_index for parallel multi-tool execution)
            tool_calls_delta = delta.get("tool_calls") or []
            if tool_calls_delta:
                self.text_or_tool_emitted = True
                for tc in tool_calls_delta:
                    tc_index = tc.get("index", 0)
                    tc_id = tc.get("id")
                    tc_func = tc.get("function") or {}
                    tc_name = tc_func.get("name")
                    tc_args = tc_func.get("arguments") or ""

                    if tc_index not in self.tool_state_map:
                        if self.current_block_type in ("text", "thinking"):
                            yield AnthropicSSEFormatter.block_stop(self._get_current_index())
                            self.current_block_type = None

                        for close_ev in self.thinking_parser.close_active_block():
                            yield close_ev.to_sse()
                        self.current_block_type = None

                        active_id = tc_id or f"toolu_{uuid.uuid4().hex[:10]}"
                        active_name = tc_name or ""
                        blk_idx = self._next_block_index()

                        self.tool_state_map[tc_index] = {
                            "tc_index": tc_index,
                            "block_index": blk_idx,
                            "tool_id": active_id,
                            "tool_name": active_name,
                            "args": [],
                            "started": False,
                            "stopped": False,
                            "streamed": False,
                        }

                    st = self.tool_state_map[tc_index]
                    if tc_id and not st["tool_id"]:
                        st["tool_id"] = tc_id
                    if tc_name and not st["tool_name"]:
                        st["tool_name"] = tc_name

                    if not st["started"] and st["tool_name"]:
                        yield AnthropicSSEFormatter.tool_use_start(st["block_index"], st["tool_id"], st["tool_name"])
                        st["started"] = True

                    if tc_args:
                        st["args"].append(tc_args)

                finish_reason = "tool_calls"
                continue

            # 3. Content deltas (Linear Sequential Pipeline: Thinking -> HeuristicTool -> Text)
            content = delta.get("content") or ""
            if content:
                if self.block_index == -1 and not self.accumulated_thinking:
                    if "<think" in content.lower() or "<thought" in content.lower():
                        self.initial_pre_think_buffer = ""
                    elif content.strip().startswith("<"):
                        self.initial_pre_think_buffer += content
                        if len(self.initial_pre_think_buffer) < 40:
                            if "<think" in self.initial_pre_think_buffer.lower() or "<thought" in self.initial_pre_think_buffer.lower():
                                self.initial_pre_think_buffer = ""
                                continue
                            continue
                        else:
                            content = self.initial_pre_think_buffer
                            self.initial_pre_think_buffer = ""
                    else:
                        if self.initial_pre_think_buffer:
                            content = self.initial_pre_think_buffer + content
                            self.initial_pre_think_buffer = ""

                think_events, clean_text = await self.thinking_parser.process_chunk_pipeline(content)
                if think_events:
                    self.initial_pre_think_buffer = ""
                    for ev in think_events:
                        ev_type = getattr(ev, "type", None)
                        if ev_type == "content_block_start":
                            cb_type = getattr(getattr(ev, "content_block", None), "type", None)
                            self.current_block_type = cb_type
                        elif ev_type == "content_block_stop":
                            self.current_block_type = None

                        if hasattr(ev, "delta"):
                            dtype = getattr(ev.delta, "type", None)
                            if dtype == "thinking_delta":
                                think_val = getattr(ev.delta, "thinking", "")
                                self.accumulated_thinking.append(think_val)
                                if think_val:
                                    import asyncio

                                    from bot.live_bridge import live_bridge_manager

                                    asyncio.create_task(
                                        live_bridge_manager.dispatch_thinking_chunk(self.session.session_id, think_val)
                                    )
                        yield ev.to_sse()

                if clean_text:
                    tool_events, remaining_text = await self.heuristic_tool_parser.process_chunk_pipeline(clean_text)
                    if tool_events:
                        for ev in tool_events:
                            ev_type = getattr(ev, "type", None)
                            if ev_type == "content_block_start":
                                cb_type = getattr(getattr(ev, "content_block", None), "type", None)
                                self.current_block_type = cb_type
                            elif ev_type == "content_block_stop":
                                self.current_block_type = None

                            if hasattr(ev, "content_block") and getattr(ev.content_block, "type", None) == "tool_use":
                                self.text_or_tool_emitted = True
                                cb = ev.content_block
                                self._current_heuristic_tc = {
                                    "type": "tool_use",
                                    "id": getattr(cb, "id", f"toolu_{uuid.uuid4().hex[:10]}"),
                                    "name": getattr(cb, "name", ""),
                                    "input": {},
                                }
                            elif hasattr(ev, "delta") and getattr(ev.delta, "type", None) == "input_json_delta":
                                partial_json = getattr(ev.delta, "partial_json", "")
                                if getattr(self, "_current_heuristic_tc", None):
                                    try:
                                        self._current_heuristic_tc["input"] = json.loads(partial_json)
                                    except Exception:
                                        self._current_heuristic_tc["input"] = {}
                                    if self._current_heuristic_tc["name"]:
                                        self.accumulated_tool_calls.append(self._current_heuristic_tc)
                                    self._current_heuristic_tc = None
                            yield ev.to_sse()

                    if remaining_text:
                        self.accumulated_text.append(remaining_text)
                        if not self.is_stop_hook:
                            if self.current_block_type != "text":
                                if self.current_block_type is not None:
                                    yield AnthropicSSEFormatter.block_stop(self._get_current_index())
                                    self.current_block_type = None

                                for close_ev in self.thinking_parser.close_active_block():
                                    yield close_ev.to_sse()

                                idx = self._next_block_index()
                                yield AnthropicSSEFormatter.text_start(idx)
                                self.current_block_type = "text"

                            self.text_or_tool_emitted = True
                            if remaining_text.strip():
                                import asyncio

                                from bot.live_bridge import live_bridge_manager

                                asyncio.create_task(
                                    live_bridge_manager.dispatch_text_chunk(self.session.session_id, remaining_text)
                                )
                            yield AnthropicSSEFormatter.text_delta(remaining_text, self._get_current_index())
                        else:
                            self.text_or_tool_emitted = True

        if self.initial_pre_think_buffer:
            clean_text = self.initial_pre_think_buffer
            self.initial_pre_think_buffer = ""
            self.accumulated_text.append(clean_text)
            if not self.is_stop_hook:
                if self.current_block_type != "text":
                    idx = self._next_block_index()
                    yield AnthropicSSEFormatter.text_start(idx)
                    self.current_block_type = "text"
                yield AnthropicSSEFormatter.text_delta(clean_text, self._get_current_index())
            else:
                self.text_or_tool_emitted = True

        # Flush Thinking Parser
        for flush_ev in await self.thinking_parser.flush():
            ev_type = getattr(flush_ev, "type", None)
            if ev_type == "content_block_start":
                cb_type = getattr(getattr(flush_ev, "content_block", None), "type", None)
                self.current_block_type = cb_type
            elif ev_type == "content_block_stop":
                self.current_block_type = None

            if hasattr(flush_ev, "delta"):
                dtype = getattr(flush_ev.delta, "type", None)
                if dtype == "thinking_delta":
                    think_val = getattr(flush_ev.delta, "thinking", "")
                    self.accumulated_thinking.append(think_val)
                    if think_val:
                        import asyncio

                        from bot.live_bridge import live_bridge_manager

                        asyncio.create_task(
                            live_bridge_manager.dispatch_thinking_chunk(self.session.session_id, think_val)
                        )
                elif dtype == "text_delta":
                    text_val = getattr(flush_ev.delta, "text", "")
                    self.accumulated_text.append(text_val)
                    if text_val:
                        self.text_or_tool_emitted = True
            yield flush_ev.to_sse()

        # Flush Heuristic Tool Parser
        for flush_ev in await self.heuristic_tool_parser.flush():
            ev_type = getattr(flush_ev, "type", None)
            if ev_type == "content_block_start":
                cb_type = getattr(getattr(flush_ev, "content_block", None), "type", None)
                self.current_block_type = cb_type
            elif ev_type == "content_block_stop":
                self.current_block_type = None

            if hasattr(flush_ev, "content_block") and getattr(flush_ev.content_block, "type", None) == "tool_use":
                self.text_or_tool_emitted = True
                cb = flush_ev.content_block
                self._current_heuristic_tc = {
                    "type": "tool_use",
                    "id": getattr(cb, "id", f"toolu_{uuid.uuid4().hex[:10]}"),
                    "name": getattr(cb, "name", ""),
                    "input": {},
                }
            elif hasattr(flush_ev, "delta") and getattr(flush_ev.delta, "type", None) == "input_json_delta":
                partial_json = getattr(flush_ev.delta, "partial_json", "")
                if getattr(self, "_current_heuristic_tc", None):
                    try:
                        self._current_heuristic_tc["input"] = json.loads(partial_json)
                    except Exception:
                        self._current_heuristic_tc["input"] = {}
                    if self._current_heuristic_tc["name"]:
                        self.accumulated_tool_calls.append(self._current_heuristic_tc)
                    self._current_heuristic_tc = None
            yield flush_ev.to_sse()

        # Stop active text/thinking block if open
        if self.current_block_type is not None:
            yield AnthropicSSEFormatter.block_stop(self._get_current_index())
            self.current_block_type = None

        # Finalize and stop active native tool call blocks per tc_index
        for _tc_idx, st in sorted(self.tool_state_map.items()):
            if not st.get("started"):
                yield AnthropicSSEFormatter.tool_use_start(
                    st["block_index"], st["tool_id"] or f"toolu_{uuid.uuid4().hex[:10]}", st["tool_name"] or "bash"
                )
                st["started"] = True

            full_args_str = "".join(st["args"])
            from atomic.parsers.auto_close_tag import AutoCloseTagParser
            repaired_args_str = AutoCloseTagParser.repair_truncated_stream(full_args_str)
            parsed_args = safe_parse_json(repaired_args_str) or {}

            if isinstance(parsed_args, dict):
                tname_lower = st["tool_name"].lower()
                is_stop_hook_tool = self.is_stop_hook or self.is_evaluator or any(k in tname_lower for k in ("exit_session", "stop_hook", "save_session_summary", "evaluator", "goal_evaluator"))
                if is_stop_hook_tool and ("ok" in parsed_args or any(k in tname_lower for k in ("exit_session", "stop_hook", "evaluator", "goal"))):
                    from core.interceptor.json_repair import JSONRepairNormalizer
                    parsed_args = await JSONRepairNormalizer.normalize_stop_hook_schema(parsed_args, is_evaluator=self.is_evaluator)

                parsed_args = await self.subagent_guard.enforce_tool_call(st["tool_name"], parsed_args)
                from atomic.guards.file_edit_guard import file_edit_guard
                parsed_args = file_edit_guard.sanitize_tool_input(st["tool_name"], parsed_args)

            if not st["stopped"]:
                yield AnthropicSSEFormatter.input_json_delta(json.dumps(parsed_args), st["block_index"])
                yield AnthropicSSEFormatter.block_stop(st["block_index"])
                st["stopped"] = True

            self.accumulated_tool_calls.append(
                {
                    "type": "tool_use",
                    "id": st["tool_id"],
                    "name": st["tool_name"],
                    "input": parsed_args,
                }
            )
            import asyncio

            from bot.live_bridge import live_bridge_manager
            asyncio.create_task(live_bridge_manager.dispatch_tool_call(self.session.session_id, st["tool_name"], parsed_args))
            self.text_or_tool_emitted = True

        # Extract embedded JSON tool calls from accumulated text if no native tool calls were present
        if not self.accumulated_tool_calls:
            full_streamed_text = "".join(self.accumulated_text)
            _rem, extracted_tools = extract_all_json_tool_calls(full_streamed_text, allowed_tools=self.tools)

            if extracted_tools:
                for tool in extracted_tools:
                    tname_lower = tool.get("name", "").lower()
                    if isinstance(tool.get("input"), dict):
                        is_stop_hook_tool = self.is_stop_hook or self.is_evaluator or any(k in tname_lower for k in ("exit_session", "stop_hook", "save_session_summary", "evaluator", "goal_evaluator"))
                        if is_stop_hook_tool and ("ok" in tool["input"] or any(k in tname_lower for k in ("exit_session", "stop_hook", "evaluator", "goal"))):
                            from core.interceptor.json_repair import (
                                JSONRepairNormalizer,
                            )
                            tool["input"] = await JSONRepairNormalizer.normalize_stop_hook_schema(tool["input"], is_evaluator=self.is_evaluator)

                    tool["input"] = await self.subagent_guard.enforce_tool_call(tool["name"], tool.get("input", {}))
                    from atomic.guards.file_edit_guard import file_edit_guard
                    tool["input"] = file_edit_guard.sanitize_tool_input(tool["name"], tool.get("input", {}))
                    idx = self._next_block_index()
                    yield AnthropicSSEFormatter.tool_use_start(idx, tool["id"], tool["name"])
                    yield AnthropicSSEFormatter.input_json_delta(json.dumps(tool["input"]), idx)
                    yield AnthropicSSEFormatter.block_stop(idx)
                    self.accumulated_tool_calls.append(tool)

                    import asyncio

                    from bot.live_bridge import live_bridge_manager
                    asyncio.create_task(live_bridge_manager.dispatch_tool_call(self.session.session_id, tool["name"], tool.get("input", {})))
                finish_reason = "tool_calls"
                self.text_or_tool_emitted = True

        if bool(self.accumulated_tool_calls):
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        # --- ASYNCHRONOUS SAFETY NET (PREEMPTIVE / FALLBACK INJECTION) ---
        # If no text content and no tool calls were emitted throughout the entire stream,
        # check if thinking text contained embedded tool calls, or emit thinking as text so Claude Code CLI gets the result!
        if not self.text_or_tool_emitted:
            full_think_text = "".join(self.accumulated_thinking).strip()
            
            # 1. Attempt extracting embedded JSON or XML tool calls from thinking text
            if full_think_text:
                _rem, extracted_tools = extract_all_json_tool_calls(full_think_text, allowed_tools=self.tools)
                if extracted_tools:
                    for tool in extracted_tools:
                        tname_lower = tool.get("name", "").lower()
                        if isinstance(tool.get("input"), dict) and (
                            "ok" in tool["input"]
                            or any(
                                k in tname_lower
                                for k in ("exit_session", "stop_hook", "save_session_summary", "evaluator", "goal")
                            )
                        ):
                            from core.interceptor.json_repair import (
                                JSONRepairNormalizer,
                            )

                            tool["input"] = await JSONRepairNormalizer.normalize_stop_hook_schema(tool["input"])

                        tool["input"] = await self.subagent_guard.enforce_tool_call(tool["name"], tool.get("input", {}))
                        from atomic.guards.file_edit_guard import file_edit_guard
                        tool["input"] = file_edit_guard.sanitize_tool_input(tool["name"], tool.get("input", {}))
                        
                        idx = self._next_block_index()
                        yield AnthropicSSEFormatter.tool_use_start(idx, tool["id"], tool["name"])
                        yield AnthropicSSEFormatter.input_json_delta(json.dumps(tool["input"]), idx)
                        yield AnthropicSSEFormatter.block_stop(idx)
                        self.accumulated_tool_calls.append(tool)
                    
                    stop_reason = "tool_use"
                    self.text_or_tool_emitted = True

            # 2. If still no text or tool emitted, attempt action continuation from thinking plan!
            if not self.text_or_tool_emitted and full_think_text and not self.is_stop_hook and self.tools:
                logger.info("StreamEngine: Model completed reasoning without action. Triggering Action Continuation...")
                async for event in self._attempt_action_continuation(full_think_text):
                    yield event
                if bool(self.accumulated_tool_calls):
                    stop_reason = "tool_use"
                    self.text_or_tool_emitted = True

            # 3. If still no text or tool emitted, output minimal space (never fake error text!)
            if not self.text_or_tool_emitted and not self.is_stop_hook:
                fallback_text = " "
                idx = self._next_block_index()
                yield AnthropicSSEFormatter.text_start(idx)
                yield AnthropicSSEFormatter.text_delta(fallback_text, idx)
                yield AnthropicSSEFormatter.block_stop(idx)
                self.text_or_tool_emitted = True
                self.accumulated_text.append(fallback_text)

        # Normalize and emit Stop Hook responses atomically if target flag is active
        if self.is_stop_hook:
            full_stop_text = "".join(self.accumulated_text).strip()
            from core.interceptor.json_repair import JSONRepairNormalizer

            if not full_stop_text and self.accumulated_thinking:
                full_think_text = "".join(self.accumulated_thinking).strip()
                sanitized_think = await JSONRepairNormalizer.sanitize_markdown_json(full_think_text)
                repaired_think = await JSONRepairNormalizer.heuristic_repair_json(sanitized_think)
                if isinstance(repaired_think, dict):
                    normalized_dict = await JSONRepairNormalizer.normalize_stop_hook_schema(
                        repaired_think, is_evaluator=self.is_evaluator
                    )
                    full_stop_text = json.dumps(normalized_dict, ensure_ascii=False)
                else:
                    if self.is_evaluator:
                        cond_met = any(kw in full_think_text.lower() for kw in ("condition is satisfied", "condition met", "goal completed", "all operations completed", "ok: true", '"ok": true'))
                        reason = full_think_text[-300:].replace("\n", " ").strip()
                        full_stop_text = json.dumps({"ok": cond_met, "reason": reason})
                    else:
                        full_stop_text = json.dumps({
                            "summary": full_think_text[-250:].replace("\n", " ").strip(),
                            "memory": "",
                            "stop_hook_active": False,
                        })

            if full_stop_text:
                normalized_stop_text = await JSONRepairNormalizer.process_text(
                    full_stop_text, is_stop_hook=True, is_evaluator=self.is_evaluator
                )
            else:
                if self.is_evaluator:
                    fallback_dict = {"ok": False, "reason": "Goal loop ended."}
                else:
                    fallback_dict = {
                        "summary": "Goal completed.",
                        "memory": "",
                        "stop_hook_active": False,
                    }
                normalized_stop_text = json.dumps(fallback_dict, ensure_ascii=False)

            self.accumulated_text = [normalized_stop_text]

            # Emit clean, fully formed JSON as a single text block
            if self.current_block_type != "text":
                if self.current_block_type is not None:
                    yield AnthropicSSEFormatter.block_stop(self._get_current_index())
                    self.current_block_type = None

                for close_ev in self.thinking_parser.close_active_block():
                    yield close_ev.to_sse()

                idx = self._next_block_index()
                yield AnthropicSSEFormatter.text_start(idx)
                self.current_block_type = "text"

            yield AnthropicSSEFormatter.text_delta(normalized_stop_text, self._get_current_index())
            yield AnthropicSSEFormatter.block_stop(self._get_current_index())
            self.current_block_type = None
            self.text_or_tool_emitted = True

        self.final_stop_reason = stop_reason
        self.final_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        self.session.record_tokens(input_tokens, output_tokens)

        yield AnthropicSSEFormatter.message_delta(stop_reason=stop_reason, output_tokens=output_tokens)
        yield AnthropicSSEFormatter.message_stop()

    async def _attempt_action_continuation(self, plan_text: str) -> AsyncGenerator[str, None]:
        """When model completed reasoning without emitting a tool or text,
        automatically execute action continuation using an agile agentic tool caller.
        """
        try:
            from config import settings
            from providers.openai import OpenAICompatibleProvider

            continuation_model = getattr(
                settings, "MODEL_SONNET", "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
            )
            provider = OpenAICompatibleProvider()

            continuation_messages = [
                {
                    "role": "system",
                    "content": "You are Claude Code CLI assistant. You must immediately execute the necessary tool call for the planned actions.",
                },
                {
                    "role": "user",
                    "content": f"You have planned the following actions in your reasoning:\n{plan_text}\n\nExecute the first tool call now without any conversational text.",
                },
            ]

            res = await provider.complete(
                model=continuation_model,
                messages=continuation_messages,
                tools=self.tools,
                stream=False,
                temperature=0.2,
                max_tokens=2048,
            )
            if isinstance(res, tuple):
                res_body = res[0]
            else:
                res_body = res

            choices = res_body.get("choices", [{}])
            msg = choices[0].get("message", {}) if choices else {}
            tc_list = msg.get("tool_calls") or []

            if tc_list:
                for tc in tc_list:
                    func = tc.get("function", {})
                    t_name = func.get("name", "")
                    raw_args = func.get("arguments", "{}")
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        parsed_args = {}

                    from atomic.guards.file_edit_guard import file_edit_guard
                    sanitized_args = file_edit_guard.sanitize_tool_input(t_name, parsed_args)
                    tool_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:10]}"

                    idx = self._next_block_index()
                    yield AnthropicSSEFormatter.tool_use_start(idx, tool_id, t_name)
                    yield AnthropicSSEFormatter.input_json_delta(json.dumps(sanitized_args), idx)
                    yield AnthropicSSEFormatter.block_stop(idx)

                    self.accumulated_tool_calls.append({"id": tool_id, "name": t_name, "input": sanitized_args})

                    import asyncio

                    from bot.live_bridge import live_bridge_manager
                    asyncio.create_task(live_bridge_manager.dispatch_tool_call(self.session.session_id, t_name, sanitized_args))

                self.text_or_tool_emitted = True
            elif msg.get("content"):
                c_text = msg["content"].strip()
                if c_text:
                    idx = self._next_block_index()
                    yield AnthropicSSEFormatter.text_start(idx)
                    yield AnthropicSSEFormatter.text_delta(c_text, idx)
                    yield AnthropicSSEFormatter.block_stop(idx)
                    self.text_or_tool_emitted = True
                    self.accumulated_text.append(c_text)
        except Exception as e:
            logger.warning("StreamEngine: Action continuation attempt failed: %s", e)

    async def stream_response(
        self, upstream_stream: AsyncIterable[dict[str, Any] | str | bytes]
    ) -> AsyncGenerator[str, None]:
        """Async generator yielding Anthropic SSE event strings on-the-fly."""
        try:
            async for chunk in self.transform_stream(upstream_stream):
                yield chunk
        finally:
            from bot.live_bridge import live_bridge_manager
            live_bridge_manager.finalize_session_stream(self.session.session_id)

    async def transform(
        self, upstream_stream: AsyncIterable[dict[str, Any] | str | bytes]
    ) -> AsyncGenerator[str, None]:
        """Backwards compatibility alias for transform_stream()."""
        async for chunk in self.transform_stream(upstream_stream):
            yield chunk


# Aliases for backward compatibility
StreamTransformer = StreamEngine
SSEStreamTransformer = StreamEngine


