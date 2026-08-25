"""Gemini Payload Sanitizer & Message Converter Module.

Asynchronously sterilizes OpenAI Chat Completions payload schemas to strictly comply
with Google Gemini OpenAI Compatibility Layer specifications
(https://generativelanguage.googleapis.com/v1beta/openai/chat/completions).

Solves the 3 primary structural causes of Gemini HTTP 400 Bad Request:
1. Tool Call/Result Mismatch: Enforces synthetic dummy `tool_calls` on the immediately preceding
   assistant message for any `tool` role message in turn history.
2. Consecutive Roles Violation: Merges consecutive `user`, `assistant`, or `system` messages.
3. Incompatible Parameters: Clamps `max_tokens` (<= 8192), cleans `stop` sequences, clamps `temperature`.
"""

import copy
from typing import Any

from loguru import logger

GEMINI_MAX_TOKENS_LIMIT = 8192


class GeminiPayloadSanitizer:
    """Asynchronous payload sanitizer for Google Gemini OpenAI compatibility endpoint."""

    @classmethod
    async def sanitize(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Asynchronously sanitize request payload before dispatch to Gemini API.

        Guarantees schema compliance and handles exception safety.
        """
        try:
            sanitized = copy.deepcopy(payload)
            sanitized = cls._sanitize_parameters(sanitized)

            raw_messages = sanitized.get("messages", [])
            if isinstance(raw_messages, list) and raw_messages:
                # Step 1: Normalize & extract system messages
                normalized = cls._extract_and_normalize_system(raw_messages)

                # Step 2: Inject synthetic tool calls for tool result messages
                with_tool_calls = cls._inject_dummy_tool_calls(normalized)

                # Step 3: Merge consecutive identical roles & fix role sequence
                final_messages = cls._merge_consecutive_roles(with_tool_calls)

                sanitized["messages"] = final_messages

            return sanitized
        except Exception as exc:
            logger.error("Error during Gemini payload sanitization: {exc}", exc=exc)
            return payload

    @classmethod
    def _sanitize_parameters(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Clamp max_tokens, sanitize stop sequences, clamp temperature, and remove unsupported keys."""
        # 1. max_tokens clamping
        if "max_tokens" in payload and payload["max_tokens"] is not None:
            try:
                mt = int(payload["max_tokens"])
                if mt > GEMINI_MAX_TOKENS_LIMIT:
                    logger.debug(
                        "Clamping Gemini max_tokens from {orig} to {limit}",
                        orig=mt,
                        limit=GEMINI_MAX_TOKENS_LIMIT,
                    )
                    payload["max_tokens"] = GEMINI_MAX_TOKENS_LIMIT
                elif mt <= 0:
                    payload["max_tokens"] = GEMINI_MAX_TOKENS_LIMIT
            except (ValueError, TypeError):
                payload["max_tokens"] = GEMINI_MAX_TOKENS_LIMIT

        # 2. temperature clamping (0.0 to 2.0)
        if "temperature" in payload and payload["temperature"] is not None:
            try:
                temp = float(payload["temperature"])
                payload["temperature"] = max(0.0, min(temp, 2.0))
            except (ValueError, TypeError):
                payload["temperature"] = 1.0

        # 3. stop sequences sanitization
        if "stop" in payload:
            stop_val = payload["stop"]
            if stop_val is None or stop_val == "" or stop_val == []:
                payload.pop("stop", None)
            elif isinstance(stop_val, str):
                payload["stop"] = [stop_val]
            elif isinstance(stop_val, list):
                cleaned_stop = [
                    str(s)
                    for s in stop_val
                    if s is not None and str(s).strip() != ""
                ]
                if cleaned_stop:
                    payload["stop"] = cleaned_stop[:4]
                else:
                    payload.pop("stop", None)
            else:
                payload.pop("stop", None)

        # 4. Remove unsupported / provider-incompatible top-level keys
        unsupported_keys = ["thinking", "anthropic_version", "top_k", "metadata"]
        for key in unsupported_keys:
            payload.pop(key, None)

        # 5. Convert native tools to prompt instructions for Gemini (prevents 400 thought_signature error)
        tools = payload.pop("tools", None)
        payload.pop("tool_choice", None)
        if tools and isinstance(tools, list):
            tool_descriptions = []
            for t in tools:
                if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
                    fn = t["function"]
                    name = fn.get("name", "")
                    desc = fn.get("description", "")
                    params = fn.get("parameters", {})
                    tool_descriptions.append(f"- Name: {name}\n  Description: {desc}\n  Parameters: {params}")

            if tool_descriptions:
                xml_instructions = (
                    "\n\n[AVAILABLE TOOLS]\n"
                    + "\n".join(tool_descriptions)
                    + "\n\nTo call a tool, format your response as:\n"
                    + "<tool_call><name>TOOL_NAME</name><parameters>JSON_PARAMETERS</parameters></tool_call>\n"
                )
                messages = payload.get("messages", [])
                if messages and isinstance(messages, list):
                    first_msg = messages[0]
                    if isinstance(first_msg, dict) and first_msg.get("role") == "system":
                        first_msg["content"] = str(first_msg.get("content", "")) + xml_instructions
                    else:
                        messages.insert(0, {"role": "system", "content": "You are a helpful coding assistant." + xml_instructions})

        return payload

    @classmethod
    def _extract_and_normalize_system(
        cls, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract all system role messages and combine them into a single system message at index 0."""
        system_contents: list[str] = []
        non_system_msgs: list[dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            if role == "system":
                c = msg.get("content")
                if c:
                    if isinstance(c, str):
                        system_contents.append(c.strip())
                    elif isinstance(c, list):
                        parts = [
                            b.get("text", "")
                            for b in c
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        system_contents.append("\n".join(parts).strip())
            else:
                non_system_msgs.append(msg)

        res: list[dict[str, Any]] = []
        if system_contents:
            combined_system = "\n\n".join(filter(None, system_contents)).strip()
            if combined_system:
                res.append({"role": "system", "content": combined_system})

        res.extend(non_system_msgs)
        return res

    @classmethod
    def _inject_dummy_tool_calls(
        cls, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Flatten native OpenAI tool_calls and `role: tool` messages into plain Hermes XML text for Gemini."""
        processed: list[dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")

            if role == "tool":
                # Convert `role: tool` to `role: user` with clear Tool Result tag
                t_name = msg.get("name") or "tool"
                content = cls._stringify_content(msg.get("content"))
                processed.append({
                    "role": "user",
                    "content": f"[Tool Output for {t_name}]:\n{content}"
                })
            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant native `tool_calls` array into Hermes XML text
                existing_text = cls._stringify_content(msg.get("content"))
                xml_calls = []
                for tc in msg.get("tool_calls", []):
                    if isinstance(tc, dict) and "function" in tc:
                        fn = tc["function"]
                        n = fn.get("name", "")
                        args = fn.get("arguments", "{}")
                        xml_calls.append(f"<tool_call><name>{n}</name><parameters>{args}</parameters></tool_call>")
                
                full_content = (existing_text + "\n" + "\n".join(xml_calls)).strip()
                processed.append({
                    "role": "assistant",
                    "content": full_content
                })
            else:
                processed.append(msg)

        return processed

    @classmethod
    def _merge_consecutive_roles(
        cls, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge consecutive messages of identical roles and enforce alternating order."""
        if not messages:
            return []

        merged: list[dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            if not merged:
                merged.append(msg)
                continue

            prev = merged[-1]
            prev_role = prev.get("role")
            curr_role = msg.get("role")

            # Merge consecutive `user` messages
            if prev_role == "user" and curr_role == "user":
                c1 = cls._stringify_content(prev.get("content"))
                c2 = cls._stringify_content(msg.get("content"))
                combined = (c1 + "\n\n" + c2).strip()
                prev["content"] = combined or " "
                continue

            # Merge consecutive `assistant` messages
            if prev_role == "assistant" and curr_role == "assistant":
                c1 = cls._stringify_content(prev.get("content"))
                c2 = cls._stringify_content(msg.get("content"))
                combined = (c1 + "\n\n" + c2).strip()
                prev["content"] = combined if combined else None

                tc1 = prev.get("tool_calls") or []
                tc2 = msg.get("tool_calls") or []
                if isinstance(tc1, list) and isinstance(tc2, list) and tc2:
                    prev["tool_calls"] = tc1 + tc2
                continue

            merged.append(msg)

        # Ensure conversation starts with user if system is absent or index 0
        non_system_start_idx = 1 if (merged and merged[0].get("role") == "system") else 0
        if len(merged) > non_system_start_idx:
            first_non_sys_role = merged[non_system_start_idx].get("role")
            if first_non_sys_role in ("assistant", "tool"):
                dummy_user = {"role": "user", "content": "Initialize session."}
                merged.insert(non_system_start_idx, dummy_user)

        return merged

    @staticmethod
    def _stringify_content(content: Any) -> str:
        """Helper to safely convert content blocks or strings into flat text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict):
                    parts.append(str(b.get("text", "")))
                else:
                    parts.append(str(b))
            return "\n".join(parts)
        return str(content)
