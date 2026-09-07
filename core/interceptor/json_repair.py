"""JSON Repair, De-duplicator & Schema Normalizer Middleware/Interceptor for Claude Code Proxy.

Solves 'Stop hook error: JSON validation failed' by intercepting responses,
cleaning markdown noise, performing heuristic JSON repair, preventing angle bracket traps,
deduplicating tool calls, and normalizing Claude Code Stop Hook schemas.
"""

import json
import re
from typing import Any

from loguru import logger
from starlette.types import ASGIApp, Receive, Scope, Send


class JSONRepairNormalizer:
    """High-performance asynchronous JSON sanitizer, repairer, deduplicator, and schema normalizer."""

    TARGET_KEYWORDS: set[str] = {
        "stop hook",
        "stop_hook",
        "stop_hook_active",
        "session summary",
        "session_summary",
        "session_memory",
        "save_session_summary",
        "exit_session",
        "session_end",
    }

    KEY_ALIASES: dict[str, str] = {
        "memories": "memory",
        "session_memory": "memory",
        "mem": "memory",
        "remember": "memory",
        "session_summary": "summary",
        "overview": "summary",
        "description": "summary",
        "stop_hook": "stop_hook_active",
        "active": "stop_hook_active",
        "should_stop": "stop_hook_active",
        "is_active": "stop_hook_active",
    }

    @classmethod
    def is_stop_hook_target(cls, payload: dict[str, Any] | None) -> bool:
        """Detect if request payload involves a Stop Hook or Goal Evaluator execution request."""
        if not payload or not isinstance(payload, dict):
            return False

        # 1. Inspect tools definition
        tools = payload.get("tools")
        has_stop_hook_tool = False
        has_coding_tools = False
        if isinstance(tools, list):
            coding_tool_names = {"bash", "read", "write", "edit", "glob", "grep", "notebookcell", "task"}
            for t in tools:
                if isinstance(t, dict):
                    name = str(t.get("name", "")).lower()
                    if name in coding_tool_names or any(c in name for c in ("bash", "read", "write", "edit")):
                        has_coding_tools = True
                    if any(kw in name for kw in ("exit_session", "stop_hook", "save_session_summary", "evaluator", "goal")):
                        has_stop_hook_tool = True

        # 2. Inspect user messages and system prompt
        messages = payload.get("messages", [])
        last_user_content = ""
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    c = msg.get("content")
                    if isinstance(c, str):
                        last_user_content = c
                    elif isinstance(c, list):
                        last_user_content = " ".join(
                            str(b.get("text", "")) for b in c if isinstance(b, dict) and b.get("type") == "text"
                        )
                    break

        user_text_lower = last_user_content.lower()

        stop_hook_patterns = (
            "stop_hook",
            "stop hook",
            "stop_hook_active",
            "save_session_summary",
            "exit_session",
            "session_summary",
            "return the stop hook",
            "a session-scoped stop hook is now active",
            "goal evaluator",
            "evaluate goal",
            "evaluating a stop-condition hook",
            "stopping condition",
            "has the following stopping condition been satisfied",
        )
        has_user_stop_hook_msg = any(pat in user_text_lower for pat in stop_hook_patterns)

        if has_coding_tools:
            return has_user_stop_hook_msg

        return has_stop_hook_tool or has_user_stop_hook_msg

    @classmethod
    def is_evaluator_hook_target(cls, payload: dict[str, Any] | None) -> bool:
        """Detect specifically if request is a Claude Code stop-condition evaluator hook."""
        if not payload or not isinstance(payload, dict):
            return False

        # 1. Inspect system prompt
        system = payload.get("system", "")
        system_str = ""
        if isinstance(system, str):
            system_str = system
        elif isinstance(system, list):
            system_str = " ".join(str(b.get("text", "")) for b in system if isinstance(b, dict))

        eval_keywords = (
            "evaluating a stop-condition hook",
            "stopping condition",
            "has the following stopping condition been satisfied",
        )
        if any(kw in system_str.lower() for kw in eval_keywords):
            return True

        # 2. Inspect user messages
        messages = payload.get("messages", [])
        if isinstance(messages, list):
            for msg in reversed(messages[-3:]):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    c = msg.get("content")
                    c_str = ""
                    if isinstance(c, str):
                        c_str = c
                    elif isinstance(c, list):
                        c_str = " ".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
                    if any(kw in c_str.lower() for kw in eval_keywords):
                        return True

        # 3. Check output_config for Claude Code evaluator schema
        output_config = payload.get("output_config")
        if isinstance(output_config, dict):
            props = output_config.get("format", {}).get("schema", {}).get("properties", {})
            if "ok" in props and "reason" in props:
                return True

        return False

    @classmethod
    def fix_angle_brackets(cls, text: str) -> str:
        """Safely escape or protect erroneous double angle brackets ('<<') and code generics that confuse XML/regex parsers."""
        if not text or not isinstance(text, str):
            return ""
        from atomic.sanitizers.angle_bracket_escape import AngleBracketEscape
        return AngleBracketEscape.sanitize(text)

    @classmethod
    async def sanitize_markdown_json(cls, text: str) -> str:
        """Clean markdown backticks, leading/trailing whitespace, XML tool_call tags, and extract JSON bounds."""
        if not text or not isinstance(text, str):
            return ""

        text = cls.fix_angle_brackets(text)
        cleaned = text.strip()

        # Strip Hermes XML tool call tags if present
        cleaned = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", cleaned)
        cleaned = re.sub(r"\[TOOL_CALLS?\][\s\S]*?(?:\[/TOOL_CALLS?\]|\Z)", "", cleaned).strip()

        # Extract content inside markdown ```json ... ``` or ``` ... ```
        codeblock_match = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", cleaned)
        if codeblock_match:
            cleaned = codeblock_match.group(1).strip()

        # Strip remaining backticks or leading/trailing noise
        if cleaned.startswith("```") or cleaned.endswith("```"):
            cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Auto-repair missing opening brace `{` if text starts directly with JSON key
        trimmed = cleaned.strip()
        if not trimmed.startswith("{") and not trimmed.startswith("["):
            if trimmed.endswith("}") or re.match(r'^["\']?(?:ok|reason|summary|stop_hook|memory|impossible)["\']?\s*:', trimmed, re.IGNORECASE):
                cleaned = "{" + cleaned
                if not cleaned.endswith("}"):
                    cleaned = cleaned + "}"

        # Isolate JSON object bounds {...} or array bounds [...]
        start_obj = cleaned.find("{")
        end_obj = cleaned.rfind("}")
        start_arr = cleaned.find("[")
        end_arr = cleaned.rfind("]")

        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            if start_arr != -1 and start_arr < start_obj and end_arr > end_obj:
                cleaned = cleaned[start_arr : end_arr + 1].strip()
            else:
                cleaned = cleaned[start_obj : end_obj + 1].strip()
        elif start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            cleaned = cleaned[start_arr : end_arr + 1].strip()

        if cleaned != text.strip():
            logger.success("🧹 [JSONRepair] Stripped markdown noise and extracted JSON bounds.")

        return cleaned

    @classmethod
    async def heuristic_repair_json(cls, json_str: str) -> dict[str, Any] | list[Any] | None:
        """Local heuristic JSON repair for missing trailing commas, Python literals, quotes, etc."""
        if not json_str:
            return None

        # 1. Fast path: standard json.loads
        try:
            return json.loads(json_str, strict=False)
        except Exception:
            pass

        logger.warning("⚠️ [JSONRepair] Invalid JSON format encountered. Initiating heuristic repair sequence...")
        from atomic.parsers.auto_close_tag import AutoCloseTagParser
        repaired = AutoCloseTagParser.repair_truncated_stream(json_str)

        # 2. Fix Python boolean/None literals: True/False/None -> true/false/null
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)

        # 3. Remove trailing commas in objects and arrays: ,} -> } and ,] -> ]
        repaired = re.sub(r",\s*\}", "}", repaired)
        repaired = re.sub(r",\s*\]", "]", repaired)

        # 4. Convert single quotes to double quotes for keys and string values if applicable
        if "'" in repaired and '"' not in repaired:
            repaired = repaired.replace("'", '"')
        else:
            repaired = re.sub(r"'([a-zA-Z0-9_]+)'\s*:", r'"\1":', repaired)

        # 5. Fix unclosed quotes at end of string value before closing bracket
        repaired = re.sub(r'(":\s*"[^"]*)\s*$', r'\1"', repaired)

        try:
            res = json.loads(repaired, strict=False)
            logger.success("🛠️ [JSONRepair] Successfully repaired malformed JSON (literals/trailing commas).")
            return res
        except Exception:
            pass

        # 6. Fix unclosed braces or brackets
        open_curly = repaired.count("{") - repaired.count("}")
        open_bracket = repaired.count("[") - repaired.count("]")

        if open_bracket > 0:
            repaired += "]" * open_bracket
        if open_curly > 0:
            repaired += "}" * open_curly

        try:
            res = json.loads(repaired, strict=False)
            logger.success("🛠️ [JSONRepair] Successfully repaired unclosed braces/brackets.")
            return res
        except Exception:
            pass

        # 7. Fallback regex key-value extraction for objects
        extracted: dict[str, Any] = {}
        pattern = r'"([a-zA-Z0-9_]+)"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|\d+(?:\.\d+)?|\[.*?\]|\{.*?\})'
        matches = re.findall(pattern, json_str, re.DOTALL)
        if matches:
            for k, v in matches:
                try:
                    extracted[k] = json.loads(v)
                except Exception:
                    extracted[k] = v.strip('"')
            logger.success("🛠️ [JSONRepair] Fallback regex key-value extraction succeeded.")
            return extracted

        logger.error("❌ [JSONRepair] All heuristic repair methods exhausted for payload: {}", json_str[:100])
        return None

    @classmethod
    async def normalize_stop_hook_schema(cls, data: Any, is_evaluator: bool = False) -> dict[str, Any]:
        """Normalize JSON schema for Claude Code Stop Hook canonical requirements."""
        result_dict: dict[str, Any] = {}

        is_eval = is_evaluator
        if isinstance(data, dict):
            if "ok" in data or "impossible" in data:
                is_eval = True
            elif is_evaluator or any(k in data for k in ("should_stop", "stop_reason", "condition_met")):
                is_eval = is_evaluator

        if is_eval:
            # Claude Code Evaluator Hook schema strictly requires:
            # {"ok": bool, "reason": str} and optional {"impossible": bool}.
            # No forbidden additional properties allowed (additionalProperties: false).
            ok_val = False
            reason_val = ""
            impossible_val = None

            if isinstance(data, dict):
                if "ok" in data:
                    raw_ok = data["ok"]
                    ok_val = raw_ok if isinstance(raw_ok, bool) else (str(raw_ok).lower() in ("true", "1", "yes"))
                elif "stop_hook_active" in data:
                    # If goal is still active, stopping condition is NOT satisfied -> ok=False
                    ok_val = not (data["stop_hook_active"] is True or str(data["stop_hook_active"]).lower() in ("true", "1", "yes"))
                elif "should_stop" in data:
                    raw_s = data["should_stop"]
                    ok_val = raw_s if isinstance(raw_s, bool) else (str(raw_s).lower() in ("true", "1", "yes"))

                if "reason" in data and data["reason"]:
                    reason_val = str(data["reason"])
                elif "summary" in data and data["summary"]:
                    reason_val = str(data["summary"])
                else:
                    reason_val = "Stopping condition evaluated."

                if "impossible" in data:
                    raw_imp = data["impossible"]
                    impossible_val = raw_imp if isinstance(raw_imp, bool) else (str(raw_imp).lower() in ("true", "1", "yes"))
            elif isinstance(data, str):
                reason_val = data
                ok_val = False
            else:
                reason_val = "Condition evaluated."
                ok_val = False

            res: dict[str, Any] = {
                "ok": ok_val,
                "reason": reason_val,
            }
            if impossible_val is not None:
                res["impossible"] = impossible_val
            return res

        # Traditional / Legacy Stop Hook Schema {"summary": ..., "memory": ..., "stop_hook_active": ...}
        if isinstance(data, dict):
            for k, v in data.items():
                canonical_key = cls.KEY_ALIASES.get(k.lower(), k)
                result_dict[canonical_key] = v
        elif isinstance(data, str):
            result_dict["summary"] = data
        elif isinstance(data, list):
            result_dict["memory"] = data
            result_dict["summary"] = "Session summary completed."
        else:
            result_dict["summary"] = ""

        # Enforce compulsory Stop Hook schema keys
        if "summary" not in result_dict or result_dict["summary"] is None:
            result_dict["summary"] = ""
        elif not isinstance(result_dict["summary"], str):
            result_dict["summary"] = str(result_dict["summary"])

        if "memory" not in result_dict or result_dict["memory"] is None:
            result_dict["memory"] = ""

        if "stop_hook_active" not in result_dict or result_dict["stop_hook_active"] is None:
            result_dict["stop_hook_active"] = True
        elif isinstance(result_dict["stop_hook_active"], str):
            result_dict["stop_hook_active"] = result_dict["stop_hook_active"].lower() in ("true", "1", "yes")

        logger.info(
            "📋 [JSONRepair] Normalized Stop Hook Schema: summary='{}', memory={}, stop_hook_active={}",
            result_dict["summary"][:50],
            result_dict["memory"],
            result_dict["stop_hook_active"],
        )
        return result_dict

    @classmethod
    def deduplicate_tool_calls(cls, content_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate tool calls and strip text XML noise when official tool_use blocks exist."""
        if not isinstance(content_list, list):
            return content_list
        from atomic.sanitizers.deduplicator import DeDuplicator
        return DeDuplicator.deduplicate(content_list)

    @classmethod
    async def process_text(cls, text: str, is_stop_hook: bool = False, is_evaluator: bool = False) -> str:
        """Complete pipeline: sanitize markdown -> repair JSON -> normalize schema -> serialize."""
        if not text or not isinstance(text, str):
            return text

        sanitized = await cls.sanitize_markdown_json(text)
        repaired_data = await cls.heuristic_repair_json(sanitized)

        is_evaluator_or_stop_hook = is_stop_hook or is_evaluator
        if isinstance(repaired_data, dict):
            if "ok" in repaired_data or any(k.lower() in ("summary", "memory", "memories", "session_summary", "stop_hook_active", "should_stop") for k in repaired_data):
                is_evaluator_or_stop_hook = True
                if "ok" in repaired_data:
                    is_evaluator = True

        if is_evaluator_or_stop_hook:
            target_input = repaired_data if repaired_data is not None else text
            normalized_dict = await cls.normalize_stop_hook_schema(target_input, is_evaluator=is_evaluator)
            return json.dumps(normalized_dict, ensure_ascii=False)

        return text

    @classmethod
    async def process_response_dict(
        cls,
        data: dict[str, Any],
        subagents_enabled: bool | None = None,
        is_stop_hook: bool = False,
        is_evaluator: bool = False,
    ) -> dict[str, Any]:
        """Process Anthropic message response dict for Stop Hook repair & tool deduplication."""
        if not isinstance(data, dict):
            return data

        if subagents_enabled is None:
            try:
                from api.dashboard import SUBAGENTS_ENABLED
                subagents_enabled = SUBAGENTS_ENABLED
            except Exception:
                subagents_enabled = True

        content_list = data.get("content", [])
        if isinstance(content_list, list):
            content_list = cls.deduplicate_tool_calls(content_list)

            for block in content_list:
                if isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        raw_text = block["text"]
                        if is_stop_hook or is_evaluator:
                            block["text"] = await cls.process_text(raw_text, is_stop_hook=is_stop_hook, is_evaluator=is_evaluator)
                        elif "```json" in raw_text.lower():
                            block["text"] = await cls.process_text(raw_text, is_stop_hook=False, is_evaluator=False)
                    elif block.get("type") == "tool_use":
                        tname = block.get("name", "")
                        tname_lower = tname.lower()
                        if "input" in block and isinstance(block["input"], dict):
                            inp = block["input"]
                            if is_stop_hook or is_evaluator or "ok" in inp or any(k in tname_lower for k in ("exit_session", "stop_hook", "save_session_summary", "evaluator", "goal")):
                                inp = await cls.normalize_stop_hook_schema(inp, is_evaluator=is_evaluator)

                            from atomic.guards.subagent import subagent_guard
                            block["input"] = await subagent_guard.enforce_tool_call(tname, inp, enabled=subagents_enabled)

            data["content"] = content_list

        return data


class JSONRepairMiddleware:
    """Pure ASGI Middleware for zero-overhead JSON repair and deduplication on non-streaming responses."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/v1/messages":
            await self.app(scope, receive, send)
            return

        response_headers: list[tuple[bytes, bytes]] = []
        response_body_chunks: list[bytes] = []
        status_code: int = 200

        async def wrapped_send(message: Any) -> None:
            nonlocal status_code, response_headers

            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = message.get("headers", [])
                is_json = any(
                    k.lower() == b"content-type" and b"application/json" in v.lower()
                    for k, v in response_headers
                )
                is_stream = any(
                    k.lower() == b"content-type" and b"text/event-stream" in v.lower()
                    for k, v in response_headers
                )
                if not is_json or is_stream or status_code != 200:
                    await send(message)
                    return
            elif message["type"] == "http.response.body":
                body_chunk = message.get("body", b"")
                is_json = any(
                    k.lower() == b"content-type" and b"application/json" in v.lower()
                    for k, v in response_headers
                )
                is_stream = any(
                    k.lower() == b"content-type" and b"text/event-stream" in v.lower()
                    for k, v in response_headers
                )

                if not is_json or is_stream or status_code != 200:
                    await send(message)
                    return

                response_body_chunks.append(body_chunk)
                if not message.get("more_body", False):
                    full_body = b"".join(response_body_chunks)
                    try:
                        data = json.loads(full_body.decode("utf-8"))
                        if isinstance(data, dict):
                            repaired = await JSONRepairNormalizer.process_response_dict(data)
                            new_body = json.dumps(repaired, ensure_ascii=False).encode("utf-8")

                            new_headers = [
                                (k, v) for k, v in response_headers if k.lower() != b"content-length"
                            ]
                            new_headers.append((b"content-length", str(len(new_body)).encode("utf-8")))

                            await send({"type": "http.response.start", "status": status_code, "headers": new_headers})
                            await send({"type": "http.response.body", "body": new_body, "more_body": False})
                            return
                    except Exception:
                        pass

                    await send({"type": "http.response.start", "status": status_code, "headers": response_headers})
                    await send({"type": "http.response.body", "body": full_body, "more_body": False})
                    return
            else:
                await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        except Exception as exc:
            logger.debug("JSONRepairMiddleware pass through error: {}", exc)
            raise exc
