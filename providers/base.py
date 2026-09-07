import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class BaseProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Send chat completions request to upstream provider."""
        pass

    def translate_tools(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Translate Anthropic tools schema to OpenAI tool call format."""
        if not tools:
            return None
        openai_tools = []
        for tool in tools:
            params = dict(tool.get("input_schema", {"type": "object", "properties": {}}))
            params.pop("$schema", None)
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": params,
                    },
                }
            )
        return openai_tools

    def translate_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Translate Anthropic messages and system prompt to OpenAI chat message list.

        Includes Heuristic Synthetic Tool Call Injection:
        If a user message contains tool_result blocks, checks preceding assistant message.
        If preceding assistant message lacks matching tool_calls, injects synthetic tool_call
        to satisfy upstream provider (NVIDIA NIM / OpenRouter / DeepSeek) schema validations.
        """
        raw_openai_messages = []

        # Handle system prompt mapping
        if system:
            if isinstance(system, list):
                system_text = "\n".join(
                    [
                        block.get("text", "")
                        for block in system
                        if block.get("type") == "text"
                    ]
                )
            else:
                system_text = str(system)
            if system_text:
                raw_openai_messages.append({"role": "system", "content": system_text})

        # Pre-pass: map tool_use_id to tool name from history
        tool_id_to_name: dict[str, str] = {}
        for m in messages:
            m_content = m.get("content")
            if isinstance(m_content, list):
                for b in m_content:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tid = b.get("id")
                        tname = b.get("name")
                        if tid and tname:
                            tool_id_to_name[tid] = tname

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            role_str = str(role or "user")

            if isinstance(content, str):
                raw_openai_messages.append({"role": role_str, "content": content})
                continue

            if not isinstance(content, list):
                raw_openai_messages.append({"role": role_str, "content": ""})
                continue

            text_parts = []
            tool_calls = []
            tool_results = []

            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:
                        text_parts.append(f"<think>\n{thinking}\n</think>")
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                elif block_type == "tool_result":
                    res_content = block.get("content")
                    if isinstance(res_content, list):
                        res_text = "\n".join(
                            [
                                b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else str(b)
                                for b in res_content
                            ]
                        )
                    else:
                        res_text = str(res_content or "")

                    t_id = block.get("tool_use_id") or f"toolu_{uuid.uuid4().hex[:8]}"
                    tool_res_dict: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": t_id,
                        "content": res_text,
                    }
                    # Preserve tool name from tool_result block for synthetic injection
                    if "name" in block:
                        tool_res_dict["name"] = block["name"]
                    tool_results.append(tool_res_dict)

            content_str = "\n".join(text_parts) if text_parts else None

            if role_str == "assistant":
                msg_obj: dict[str, Any] = {"role": "assistant"}
                if content_str:
                    msg_obj["content"] = content_str
                elif not tool_calls:
                    msg_obj["content"] = " "
                else:
                    msg_obj["content"] = None

                if tool_calls:
                    msg_obj["tool_calls"] = tool_calls
                raw_openai_messages.append(msg_obj)
            elif role_str == "user":
                # Crucial for OpenAI spec: tool_results must immediately follow assistant tool_calls
                if tool_results:
                    raw_openai_messages.extend(tool_results)
                if content_str:
                    raw_openai_messages.append({"role": "user", "content": content_str})
            elif role_str == "system":
                if content_str:
                    # In OpenAI Chat Completions API, system role should not appear after dialogue begins.
                    # Convert intermediate hook/reminder system messages to <system-reminder> on a user turn
                    # so the model maintains valid alternation and always responds to the active user prompt.
                    has_prior_dialogue = any(m.get("role") in ("user", "assistant", "tool") for m in raw_openai_messages)
                    if not has_prior_dialogue:
                        raw_openai_messages.append({"role": "system", "content": content_str})
                    elif raw_openai_messages and raw_openai_messages[-1].get("role") == "user":
                        prev_content = raw_openai_messages[-1].get("content") or ""
                        raw_openai_messages[-1]["content"] = (
                            f"{prev_content}\n\n<system-reminder>\n{content_str}\n</system-reminder>".strip()
                        )
                    else:
                        raw_openai_messages.append({
                            "role": "user",
                            "content": f"<system-reminder>\n{content_str}\n</system-reminder>",
                        })

        # ---------------------------------------------------------------------
        # Post-Processing Pass: Synthetic Tool Call Injection & Schema Fixing
        # ---------------------------------------------------------------------
        final_messages: list[dict[str, Any]] = []

        for msg in raw_openai_messages:
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id") or f"toolu_{uuid.uuid4().hex[:8]}"
                tool_name = msg.pop("name", None) or tool_id_to_name.get(tool_call_id) or "custom_tool"
                msg["tool_call_id"] = tool_call_id

                # Find preceding assistant message
                preceding_assistant = None
                for prev in reversed(final_messages):
                    if prev.get("role") == "assistant":
                        preceding_assistant = prev
                        break

                if preceding_assistant is not None:
                    existing_calls = preceding_assistant.get("tool_calls", [])
                    has_match = any(
                        isinstance(c, dict) and c.get("id") == tool_call_id
                        for c in existing_calls
                    )
                    if not has_match:
                        synthetic_call = {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": "{}",
                            },
                        }
                        if "tool_calls" not in preceding_assistant:
                            preceding_assistant["tool_calls"] = []
                        preceding_assistant["tool_calls"].append(synthetic_call)
                else:
                    synthetic_assistant = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                    final_messages.append(synthetic_assistant)

                if not msg.get("content"):
                    msg["content"] = "Tool execution completed."

                final_messages.append(msg)
            else:
                final_messages.append(msg)

        # Clean up empty content fields on assistant messages with tool_calls
        for msg in final_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls") and msg.get("content") == "":
                msg["content"] = None


        return final_messages

