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
    TAG_TOOL_REGEX = re.compile(
        r"<(tool_code|tool_call)>\s*(.*?)\s*</\1>",
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
        if not name:
            return False
        if self.allowed_tool_names is None:
            return True
        return self._resolve_tool_name(name) is not None

    def _resolve_tool_name(self, name: str | None) -> str | None:
        """Resolve tool name case-insensitively and through common aliases."""
        if not name:
            return None
        if self.allowed_tool_names is None:
            return name
        if name in self.allowed_tool_names:
            return name

        lower_map = {t.lower(): t for t in self.allowed_tool_names}
        if name.lower() in lower_map:
            return lower_map[name.lower()]

        aliases: dict[str, list[str]] = {
            "bash": ["Bash", "run_command", "ExecuteCommand", "execute_command", "bash"],
            "run_command": ["Bash", "bash", "ExecuteCommand", "execute_command"],
            "read": ["Read", "View", "view_file", "view", "read_file", "read"],
            "view": ["Read", "View", "view_file", "read"],
            "write": ["Write", "write_to_file", "write_file", "write"],
            "edit": ["Edit", "replace_file_content", "edit_file", "edit"],
            "grep": ["Grep", "grep_search", "grep"],
            "glob": ["Glob", "list_dir", "file_search", "glob"],
        }
        name_lower = name.lower()
        if name_lower in aliases:
            for cand in aliases[name_lower]:
                if cand in self.allowed_tool_names:
                    return cand
                if cand.lower() in lower_map:
                    return lower_map[cand.lower()]

        return None

    def _parse_tag_tool_body(self, body: str) -> tuple[str | None, dict[str, Any]]:
        """Extract tool name and parameters from <tool_code> or <tool_call> inner body."""
        body = body.strip()
        if not body:
            return None, {}

        # 1. Try JSON extraction
        if body.startswith("{") or "{" in body:
            json_str = body
            fence_match = re.search(r"```(?:json)?\s*({.*?})\s*```", body, re.DOTALL)
            if fence_match:
                json_str = fence_match.group(1).strip()
            elif not body.startswith("{"):
                brace_match = re.search(r"({.*})", body, re.DOTALL)
                if brace_match:
                    json_str = brace_match.group(1).strip()

            from core.transformer.stream_engine import safe_parse_json
            parsed = safe_parse_json(json_str)
            if isinstance(parsed, dict):
                name = parsed.get("name")
                input_params = parsed.get("parameters") or parsed.get("arguments") or parsed.get("input")
                if name and isinstance(name, str):
                    if not isinstance(input_params, dict):
                        input_params = {k: v for k, v in parsed.items() if k not in ("name", "parameters", "arguments", "input")}
                    return self._resolve_tool_name(name) or name, input_params

                if "command" in parsed or "cmd" in parsed:
                    cmd = parsed.get("command") or parsed.get("cmd")
                    resolved_bash = self._resolve_tool_name("Bash") or "Bash"
                    p_key = "CommandLine" if resolved_bash == "run_command" else "command"
                    return resolved_bash, {p_key: cmd}

                if "file_path" in parsed or "path" in parsed:
                    path = parsed.get("file_path") or parsed.get("path")
                    if "content" in parsed:
                        resolved_write = self._resolve_tool_name("Write") or "Write"
                        return resolved_write, {"path": path, "content": parsed.get("content")}
                    resolved_read = self._resolve_tool_name("Read") or "Read"
                    return resolved_read, {"file_path": path}

                if input_params and isinstance(input_params, dict):
                    resolved_bash = self._resolve_tool_name("Bash") or "Bash"
                    return resolved_bash, input_params

        # 2. Try XML structures: <name>...</name> or function=...
        name_match = re.search(r"<(?:name|function)>([\w_]+)</(?:name|function)>", body, re.IGNORECASE)
        if not name_match:
            name_match = re.search(r"(?:<<|#<|<)?function=([\w_]+)>>?", body, re.IGNORECASE)

        raw_name = name_match.group(1).strip() if name_match else None
        resolved_name = self._resolve_tool_name(raw_name) if raw_name else None

        params: dict[str, Any] = {}
        param_block_match = re.search(r"<parameters>(.*?)</parameters>", body, re.DOTALL | re.IGNORECASE)
        param_content = param_block_match.group(1).strip() if param_block_match else body

        kv_matches = re.findall(r"<([a-zA-Z0-9_]+)>(.*?)</\1>", param_content, re.DOTALL)
        if kv_matches:
            for k, v in kv_matches:
                if k.lower() not in ("name", "function", "parameters"):
                    params[k] = v.strip()
        else:
            param_matches = re.findall(
                r"(?:<<|#<|<)?parameter=([\w_]+)>>?(.*?)(?=(?:(?:<<|#<|<)?parameter=|\Z))",
                param_content,
                re.DOTALL,
            )
            for p_name, p_val in param_matches:
                clean_val = re.sub(r"</parameter>.*$", "", p_val, flags=re.DOTALL).strip()
                params[p_name.strip()] = clean_val

        if not params and "{" in param_content:
            from core.transformer.stream_engine import safe_parse_json
            parsed_p = safe_parse_json(param_content)
            if isinstance(parsed_p, dict):
                params = parsed_p

        # Fallback tool name if omitted but parameters identify intent
        if not resolved_name:
            if "command" in params or "cmd" in params:
                resolved_name = self._resolve_tool_name("Bash") or "Bash"
            elif "file_path" in params or "path" in params:
                resolved_name = self._resolve_tool_name("Read") or "Read"

        return resolved_name or raw_name, params

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

        # 1. Check for <tool_code>...</tool_code> or <tool_call>...</tool_call>
        tag_match = self.TAG_TOOL_REGEX.search(masked_buffer)
        if tag_match:
            body = self.unmask_code_generics(tag_match.group(2))
            func_name, params = self._parse_tag_tool_body(body)
            target_tool = self._resolve_tool_name(func_name) or func_name

            if target_tool and self._is_tool_allowed(target_tool):
                prefix = self.unmask_code_generics(self.text_buffer[: tag_match.start()].strip())
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

                self.text_buffer = self.unmask_code_generics(self.text_buffer[tag_match.end() :])
                return events, ""

        # 2. Check for XML tool call structure: <tool_call>...<<function=...>>...</tool_call>
        xml_match = self.XML_TOOL_CALL_REGEX.search(masked_buffer)
        if xml_match:
            func_name = xml_match.group(1).strip()
            body = xml_match.group(2)
            func_name = self.unmask_code_generics(func_name)

            params = {}
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

            target_tool = self._resolve_tool_name(func_name) or func_name
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
        tool_openers = (
            r"```\s*[\w_]*",
            r"<tool_code>",
            r"<tool_call>",
            r"\[TOOL_CALLS?\]",
            r"<(?:function|parameter)=",
            r"<<(?:function|parameter)=",
        )
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
            tag_match = self.TAG_TOOL_REGEX.search(masked_buffer)
            if tag_match:
                body = self.unmask_code_generics(tag_match.group(2))
                func_name, params = self._parse_tag_tool_body(body)
                target_tool = self._resolve_tool_name(func_name) or func_name
                if target_tool and self._is_tool_allowed(target_tool):
                    t_id = f"toolu_{uuid.uuid4().hex[:10]}"
                    t_idx = self._get_next_index()
                    events.append(ModelConverter.build_sse_block_start(t_idx, "tool_use", {"id": t_id, "name": target_tool}))
                    events.append(ModelConverter.build_sse_block_delta(t_idx, "input_json_delta", json.dumps(params)))
                    events.append(ModelConverter.build_sse_block_stop(t_idx))
            else:
                xml_match = self.XML_TOOL_CALL_REGEX.search(masked_buffer)
                if xml_match:
                    func_name = self.unmask_code_generics(xml_match.group(1).strip())
                    body = xml_match.group(2)
                    params = {}
                    param_matches = re.findall(r"(?:<<|#<|<)?parameter=([\w_]+)>>?(.*?)(?=(?:(?:<<|#<|<)?parameter=|\Z))", body, re.DOTALL)
                    for p_name, p_val in param_matches:
                        clean_val = re.sub(r"</parameter>.*$", "", p_val, flags=re.DOTALL).strip()
                        params[p_name.strip()] = self.unmask_code_generics(clean_val)
                    target_tool = self._resolve_tool_name(func_name) or func_name
                    if target_tool and self._is_tool_allowed(target_tool):
                        t_id = f"toolu_{uuid.uuid4().hex[:10]}"
                        t_idx = self._get_next_index()
                        events.append(ModelConverter.build_sse_block_start(t_idx, "tool_use", {"id": t_id, "name": target_tool}))
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
                                resolved = self._resolve_tool_name(tool_name) or tool_name
                                if self._is_tool_allowed(resolved):
                                    tool_input = parsed.get("parameters") or parsed.get("arguments") or parsed.get("input") or {}
                                    if not isinstance(tool_input, dict):
                                        tool_input = {}
                                    t_id = f"toolu_{uuid.uuid4().hex[:10]}"
                                    t_idx = self._get_next_index()
                                    events.append(ModelConverter.build_sse_block_start(t_idx, "tool_use", {"id": t_id, "name": resolved}))
                                    events.append(ModelConverter.build_sse_block_delta(t_idx, "input_json_delta", json.dumps(tool_input)))
                                    events.append(ModelConverter.build_sse_block_stop(t_idx))
                        except Exception:
                            pass
                    else:
                        bash_match = self.BASH_FENCE_REGEX.search(masked_buffer)
                        if bash_match:
                            cmd_text = self.unmask_code_generics(bash_match.group(1).strip())
                            target_tool = self._resolve_tool_name("Bash") or "run_command"
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

