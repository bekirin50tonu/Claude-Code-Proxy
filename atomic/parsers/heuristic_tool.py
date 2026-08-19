"""Heuristic tool call parser detecting markdown fences, embedded JSON, and slash commands in streams."""

import json
import re
import uuid
from typing import Any

from atomic.parsers.base import BaseAtomicParser
from models.converter import ModelConverter
from shared.schemas.anthropic import SSEBaseEvent


class HeuristicToolParser(BaseAtomicParser):
    """Parser detecting markdown codeblocks and embedded JSON tool calls in text stream."""

    BASH_FENCE_REGEX = re.compile(r"```(?:bash|sh|shell|zsh)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
    JSON_FENCE_REGEX = re.compile(r"```(?:json|JSON)?\s*\n({.*?})\n```", re.DOTALL)

    def __init__(
        self,
        block_index_provider: Any = None,
        tools: list[dict[str, Any]] | list[str] | set[str] | None = None,
    ):
        self.block_index_provider = block_index_provider
        self.tools = tools
        self.allowed_tool_names: set[str] | None = None
        if tools is not None:
            self.allowed_tool_names = set()
            for item in tools:
                if isinstance(item, str):
                    self.allowed_tool_names.add(item)
                elif isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                    self.allowed_tool_names.add(item["name"])

        self.text_buffer = ""
        self.buffering_tool = False

    def reset(self) -> None:
        self.text_buffer = ""
        self.buffering_tool = False

    def _get_next_index(self) -> int:
        if self.block_index_provider and callable(self.block_index_provider):
            return self.block_index_provider()
        return 0

    def _is_tool_allowed(self, name: str) -> bool:
        if self.allowed_tool_names is None:
            return True
        return name in self.allowed_tool_names

    async def process_chunk(self, chunk: dict[str, Any] | str) -> list[SSEBaseEvent]:
        events: list[SSEBaseEvent] = []
        text = chunk if isinstance(chunk, str) else ""
        if isinstance(chunk, dict):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content") or ""

        if not text:
            return events

        self.text_buffer += text

        # Check for bash code blocks (```bash ... ```)
        bash_match = self.BASH_FENCE_REGEX.search(self.text_buffer)
        if bash_match:
            cmd_text = bash_match.group(1).strip()
            target_tool = "run_command"
            if not self._is_tool_allowed(target_tool):
                # Fallback check if Bash or execute_command is allowed instead
                for alt in ("Bash", "ExecuteCommand", "bash", "execute_command"):
                    if self._is_tool_allowed(alt):
                        target_tool = alt
                        break

            if self._is_tool_allowed(target_tool):
                prefix = self.text_buffer[: bash_match.start()].strip()
                if prefix:
                    idx = self._get_next_index()
                    events.append(ModelConverter.build_sse_block_start(idx, "text"))
                    events.append(ModelConverter.build_sse_block_delta(idx, "text_delta", prefix))
                    events.append(ModelConverter.build_sse_block_stop(idx))

                t_id = f"toolu_{uuid.uuid4().hex[:10]}"
                t_input = {"command": cmd_text} if target_tool in ("Bash", "bash") else {"CommandLine": cmd_text}
                t_idx = self._get_next_index()

                events.append(ModelConverter.build_sse_block_start(t_idx, "tool_use", {"id": t_id, "name": target_tool}))
                events.append(ModelConverter.build_sse_block_delta(t_idx, "input_json_delta", json.dumps(t_input)))
                events.append(ModelConverter.build_sse_block_stop(t_idx))

                self.text_buffer = self.text_buffer[bash_match.end() :]
                return events

        # Check for JSON tool calls embedded in codeblocks or text
        json_match = self.JSON_FENCE_REGEX.search(self.text_buffer)
        if json_match:
            raw_json = json_match.group(1).strip()
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_name = parsed["name"]
                    if self._is_tool_allowed(tool_name):
                        prefix = self.text_buffer[: json_match.start()].strip()
                        if prefix:
                            idx = self._get_next_index()
                            events.append(ModelConverter.build_sse_block_start(idx, "text"))
                            events.append(ModelConverter.build_sse_block_delta(idx, "text_delta", prefix))
                            events.append(ModelConverter.build_sse_block_stop(idx))

                        tool_input = parsed.get("parameters") or parsed.get("arguments") or parsed.get("input") or {}
                        if not isinstance(tool_input, dict):
                            tool_input = {}
                        t_id = f"toolu_{uuid.uuid4().hex[:10]}"
                        t_idx = self._get_next_index()

                        events.append(ModelConverter.build_sse_block_start(t_idx, "tool_use", {"id": t_id, "name": tool_name}))
                        events.append(ModelConverter.build_sse_block_delta(t_idx, "input_json_delta", json.dumps(tool_input)))
                        events.append(ModelConverter.build_sse_block_stop(t_idx))

                        self.text_buffer = self.text_buffer[json_match.end() :]
                        return events
            except Exception:
                pass

        # Stateful partial tag buffering across chunk boundaries
        partial_prefixes = ("<tool_call", "<function=", "<parameter=", "[TOOL_CALL", "```bash", "```json")
        for tag in partial_prefixes:
            for i in range(1, len(tag)):
                sub = tag[:i]
                if self.text_buffer.endswith(sub):
                    from loguru import logger
                    logger.info("🔄 \033[1;34m[HeuristicToolStatefulParser]\033[0m Buffered partial tool tag '{}' across chunk boundary.", sub)
                    return events

        return events

    async def flush(self) -> list[SSEBaseEvent]:
        events: list[SSEBaseEvent] = []
        if self.text_buffer:
            bash_match = self.BASH_FENCE_REGEX.search(self.text_buffer)
            if bash_match:
                cmd_text = bash_match.group(1).strip()
                target_tool = "run_command"
                if not self._is_tool_allowed(target_tool):
                    for alt in ("Bash", "ExecuteCommand", "bash", "execute_command"):
                        if self._is_tool_allowed(alt):
                            target_tool = alt
                            break
                if self._is_tool_allowed(target_tool):
                    t_id = f"toolu_{uuid.uuid4().hex[:10]}"
                    t_input = {"command": cmd_text} if target_tool in ("Bash", "bash") else {"CommandLine": cmd_text}
                    t_idx = self._get_next_index()
                    events.append(ModelConverter.build_sse_block_start(t_idx, "tool_use", {"id": t_id, "name": target_tool}))
                    events.append(ModelConverter.build_sse_block_delta(t_idx, "input_json_delta", json.dumps(t_input)))
                    events.append(ModelConverter.build_sse_block_stop(t_idx))
            self.text_buffer = ""
        return events



HeuristicToolStatefulParser = HeuristicToolParser

