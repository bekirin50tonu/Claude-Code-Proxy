"""NVIDIA NIM Payload Sanitizer.

Normalizes payload parameters and schema structures for NVIDIA NIM endpoints to prevent 400 Bad Request errors:
1. Clamps `max_tokens` to model-allowed maximums (e.g. max 8192 for LLaMA 3.1 models).
2. Strips Anthropic-specific non-standard root fields (`thinking`, `context_management`, `output_config`, `cache_control`, `metadata`).
3. Normalizes `tool_choice` to valid OpenAI specification formats.
4. Cleans up non-standard message fields (such as `name` from `role: tool` messages).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Maximum output token limit enforced by NVIDIA NIM for LLaMA & Nemotron models
NVIDIA_NIM_MAX_OUTPUT_CAP = 32768


class NimPayloadSanitizer:
    @classmethod
    async def sanitize(cls, payload: dict[str, Any], max_output_override: int | None = None) -> dict[str, Any]:
        """Sanitize request payload before sending to NVIDIA NIM endpoint."""
        clean_payload = dict(payload)

        # 1. Clamp max_tokens to NIM output limit (default 32768 cap if higher)
        max_output_limit = max_output_override or NVIDIA_NIM_MAX_OUTPUT_CAP
        if "max_tokens" in clean_payload:
            original_max_tokens = clean_payload["max_tokens"]
            if isinstance(original_max_tokens, int) and original_max_tokens > max_output_limit:
                clean_payload["max_tokens"] = max_output_limit
                logger.debug(
                    "NimPayloadSanitizer: Clamped max_tokens from %d to %d for NVIDIA NIM",
                    original_max_tokens,
                    max_output_limit,
                )

        # 2. Extract and translate structured output if output_config is present
        output_config = clean_payload.pop("output_config", None)
        if output_config and isinstance(output_config, dict):
            fmt = output_config.get("format", {})
            if fmt.get("type") == "json_schema":
                clean_payload.setdefault("response_format", {"type": "json_object"})

        # Remove non-standard root parameters that cause 400 Bad Request on OpenAI/NIM API
        unsupported_root_params = [
            "thinking",
            "context_management",
            "cache_control",
            "metadata",
        ]
        for param in unsupported_root_params:
            clean_payload.pop(param, None)

        # 2b. Inject NVIDIA NIM native reasoning split & chat template kwargs directly into payload root
        clean_payload.pop("extra_body", None)
        clean_payload.setdefault(
            "chat_template_kwargs",
            {
                "thinking": True,
                "enable_thinking": True,
                "reasoning_split": True,
                "clear_thinking": False,
            },
        )

        # 3. Normalize tool_choice
        if "tool_choice" in clean_payload:
            tool_choice = clean_payload["tool_choice"]
            if isinstance(tool_choice, dict):
                tc_type = tool_choice.get("type")
                if tc_type in ("auto", "none", "required"):
                    clean_payload["tool_choice"] = tc_type
                elif tc_type == "function" and "function" in tool_choice:
                    clean_payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice["function"].get("name", "")},
                    }
                else:
                    clean_payload["tool_choice"] = "auto"

        # 4. Clean up messages
        if "messages" in clean_payload and isinstance(clean_payload["messages"], list):
            clean_messages = []
            for msg in clean_payload["messages"]:
                if not isinstance(msg, dict):
                    clean_messages.append(msg)
                    continue

                msg_copy = dict(msg)

                # Remove non-standard 'name' field from role=tool messages
                if msg_copy.get("role") == "tool":
                    msg_copy.pop("name", None)

                # Normalize empty string content to None when assistant has tool_calls
                if msg_copy.get("role") == "assistant" and msg_copy.get("tool_calls") and msg_copy.get("content") == "":
                    msg_copy["content"] = None

                clean_messages.append(msg_copy)

            clean_payload["messages"] = clean_messages

        return clean_payload
