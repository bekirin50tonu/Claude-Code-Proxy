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
    XML_TOOL_CALL_REGEX = re.compile(
        r"<tool_call>\s*(?:<<|#<|<)?function=([\w_]+)>>?\s*(.*?)</tool_call>",
        re.DOTALL | re.IGNORECASE,
    )

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

    @classmethod
    def mask_code_generics(cls, text: str) -> str:
        """Mask double angle brackets ('<<' / '>>') inside programming codeblocks to avoid false XML tag matches."""
        if not text or "```" not in text:
            return text

        def replace_codeblock(match: re.Match[str]) -> str:
            lang = match.group(1).lower() if match.group(1) else ""
            body = match.group(2)
            if lang in ("bash", "sh", "shell", "zsh", "json"):
                return match.group(0)
            # Mask << and >> in programming code blocks
            masked_body = body.replace("<<", "«").replace(">>", "»")
            return f"```{lang}\n{masked_body}\n```"

        return re.sub(r"```([\w_]*)\n(.*?)\n```", replace_codeblock, text, flags=re.DOTALL)

    @classmethod
    def unmask_code_generics(cls, text: str) -> str:
        """Restore double angle brackets in codeblocks."""
        return text.replace("«", "<<").replace("»", ">>")

    async def process_chunk_pipeline(self, chunk: dict[str, Any] | str) -> tuple[list[SSEBaseEvent], str]:
        """
        Linear pipeline step. Returns (events, remaining_unemitted_text).
        Yields tool_use SSE events if tool matched, or returns plain text fragment if no tool matched.
        """
        events: list[SSEBaseEvent] = []
        text = chunk if isinstance(chunk, str) else ""
        if isinstance(chunk, dict):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content") or ""

        if not text:
            return events, ""

        self.text_buffer += text
        masked_buffer = self.mask_code_generics(self.text_buffer)

        # 1. Check for XML tool call structure: <tool_call>...<<function=...>>...</tool_call>
        xml_match = self.XML_TOOL_CALL_REGEX.search(masked_buffer)
        if xml_match:
            func_name = xml_match.group(1).strip()
            body = xml_match.group(2)
            func_name = self.unmask_code_generics(func_name)

            params: dict[str, Any] = {}
            param_matches = re.findall(r"(?:<<|#<|<)?parameter=([\w_]+)>>?(.*?)(?=(?:(?:<<|#<|<)?parameter=|\Z))", body, re.DOTALL)
            for p_name, p_val in param_matches:
                clean_val = re.sub(r"</parameter>.*$", "", p_val, flags=re.DOTALL).strip()
                params[p_name.strip()] = self.unmask_code_generics(clean_val)

            # Fallback if body contains JSON
            if not params and "{" in body:
                try:
                    parsed_body = json.loads(body.strip(), strict=False)
                    if isinstance(parsed_body, dict):
                        params = parsed_body.get("parameters") or parsed_body.get("arguments") or parsed_body.get("input") or {}
                except Exception:
                    pass

            target_tool = func_name
            if not self._is_tool_allowed(target_tool):
                for alt in ("Bash", "ExecuteCommand", "run_command", "Edit", "Write", "View"):
                    if self._is_tool_allowed(alt) and alt.lower() == func_name.lower():
                        target_tool = alt
                        break

            prefix = self.unmask_code_generics(self.text_buffer[: xml_match.start()].strip())
            if prefix:
                idx = self._get_next_index()
                events.append(ModelConverter.build_sse_block_start(idx, "text"))
                events.append(ModelConverter.build_sse_block_delta(idx, "text_delta", prefix))
                events.append(ModelConverter.build_sse_block_stop(idx))

            t_id = f"toolu_{uuid.uuid4().hex[:10]}"
            t_idx = self._get_next_index()
            events.append(ModelConverter.build_sse_block_start(t_idx, "tool_use", {"id": t_id, "name": target_tool}))
            events.append(ModelConverter.build_sse_block_delta(t_idx, "input_json_delta", json.dumps(params)))
            events.append(ModelConverter.build_sse_block_stop(t_idx))

            self.text_buffer = self.unmask_code_generics(self.text_buffer[xml_match.end() :])
            return events, ""

        # 2. Check for bash code blocks (```bash ... ```)
        bash_match = self.BASH_FENCE_REGEX.search(masked_buffer)
        if bash_match:
            cmd_text = self.unmask_code_generics(bash_match.group(1).strip())
            target_tool = "run_command"
            if not self._is_tool_allowed(target_tool):
                for alt in ("Bash", "ExecuteCommand", "bash", "execute_command"):
                    if self._is_tool_allowed(alt):
                        target_tool = alt
                        break

            if self._is_tool_allowed(target_tool):
                prefix = self.unmask_code_generics(self.text_buffer[: bash_match.start()].strip())
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

                self.text_buffer = self.unmask_code_generics(self.text_buffer[bash_match.end() :])
                return events, ""

        # 3. Check for JSON tool calls embedded in codeblocks or text
        json_match = self.JSON_FENCE_REGEX.search(masked_buffer)
        if json_match:
            raw_json = self.unmask_code_generics(json_match.group(1).strip())
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_name = parsed["name"]
                    if self._is_tool_allowed(tool_name):
                        prefix = self.unmask_code_generics(self.text_buffer[: json_match.start()].strip())
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

                        self.text_buffer = self.unmask_code_generics(self.text_buffer[json_match.end() :])
                        return events, ""
            except Exception:
                pass

        # 4. Check for open unclosed tool block or codeblock across chunk boundaries
        tool_openers = (r"```\s*[\w_]*", r"<tool_call>", r"\[TOOL_CALLS?\]", r"<(?:function|parameter)=", r"<<(?:function|parameter)=")
        for pattern in tool_openers:
            open_match = re.search(pattern, masked_buffer, re.IGNORECASE)
            if open_match:
                prefix = self.unmask_code_generics(self.text_buffer[: open_match.start()])
                if prefix.strip():
                    idx = self._get_next_index()
                    events.append(ModelConverter.build_sse_block_start(idx, "text"))
                    events.append(ModelConverter.build_sse_block_delta(idx, "text_delta", prefix))
                    events.append(ModelConverter.build_sse_block_stop(idx))
                
                self.text_buffer = self.unmask_code_generics(self.text_buffer[open_match.start() :])
                return events, ""

        # 5. Check for partial tag or fence prefix at very end of buffer across chunk boundaries
        if re.search(r"(`{1,3}[\w_]*|<[\w_=#<]*|\[[\w_]*)$", self.text_buffer, re.IGNORECASE):
            return events, ""

        # Plain text remaining in buffer
        remaining_text = self.text_buffer
        self.text_buffer = ""
        return events, remaining_text

    async def process_chunk(self, chunk: dict[str, Any] | str) -> list[SSEBaseEvent]:
        events, remaining = await self.process_chunk_pipeline(chunk)
        if remaining:
            idx = self._get_next_index()
            events.append(ModelConverter.build_sse_block_start(idx, "text"))
            events.append(ModelConverter.build_sse_block_delta(idx, "text_delta", remaining))
            events.append(ModelConverter.build_sse_block_stop(idx))
        return events

    async def flush(self) -> list[SSEBaseEvent]:
        events: list[SSEBaseEvent] = []
        if self.text_buffer:
            masked_buffer = self.mask_code_generics(self.text_buffer)
            xml_match = self.XML_TOOL_CALL_REGEX.search(masked_buffer)
            if xml_match:
                func_name = self.unmask_code_generics(xml_match.group(1).strip())
                body = xml_match.group(2)
                params: dict[str, Any] = {}
                param_matches = re.findall(r"(?:<<|#<|<)?parameter=([\w_]+)>>?(.*?)(?=(?:(?:<<|#<|<)?parameter=|\Z))", body, re.DOTALL)
                for p_name, p_val in param_matches:
                    clean_val = re.sub(r"</parameter>.*$", "", p_val, flags=re.DOTALL).strip()
                    params[p_name.strip()] = self.unmask_code_generics(clean_val)
                t_id = f"toolu_{uuid.uuid4().hex[:10]}"
                t_idx = self._get_next_index()
                events.append(ModelConverter.build_sse_block_start(t_idx, "tool_use", {"id": t_id, "name": func_name}))
                events.append(ModelConverter.build_sse_block_delta(t_idx, "input_json_delta", json.dumps(params)))
                events.append(ModelConverter.build_sse_block_stop(t_idx))
            else:
                json_match = self.JSON_FENCE_REGEX.search(masked_buffer)
                if json_match:
                    raw_json = self.unmask_code_generics(json_match.group(1).strip())
                    try:
                        parsed = json.loads(raw_json)
                        if isinstance(parsed, dict) and "name" in parsed:
                            tool_name = parsed["name"]
                            if self._is_tool_allowed(tool_name):
                                tool_input = parsed.get("parameters") or parsed.get("arguments") or parsed.get("input") or {}
                                if not isinstance(tool_input, dict):
                                    tool_input = {}
                                t_id = f"toolu_{uuid.uuid4().hex[:10]}"
                                t_idx = self._get_next_index()
                                events.append(ModelConverter.build_sse_block_start(t_idx, "tool_use", {"id": t_id, "name": tool_name}))
                                events.append(ModelConverter.build_sse_block_delta(t_idx, "input_json_delta", json.dumps(tool_input)))
                                events.append(ModelConverter.build_sse_block_stop(t_idx))
                    except Exception:
                        pass
                else:
                    bash_match = self.BASH_FENCE_REGEX.search(masked_buffer)
                    if bash_match:
                        cmd_text = self.unmask_code_generics(bash_match.group(1).strip())
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

