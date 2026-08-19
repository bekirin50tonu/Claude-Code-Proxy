"""Token Budget Guard — context-window-aware message truncation in Atomic layer."""

import functools
from typing import Any

import tiktoken
from loguru import logger

from config import ModelMetadata, model_registry

SAFETY_BUFFER = 256
_O200K_FAMILIES = ("llama", "mistral", "qwen", "glm", "gemma", "deepseek", "phi")


@functools.lru_cache(maxsize=16)
def _get_encoding(model_id: str) -> tiktoken.Encoding:
    """Select the best tiktoken encoding for the upstream model (cached in LRU)."""
    model_lower = model_id.lower()
    if any(fam in model_lower for fam in _O200K_FAMILIES):
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            pass
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text, disallowed_special=()))


def _extract_block_text(block: Any) -> str:
    """Extract all text, tool inputs, and tool outputs from a message content block."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        b_type = block.get("type")
        if b_type == "text":
            return block.get("text", "")
        elif b_type == "thinking":
            return block.get("thinking", "")
        elif b_type == "tool_use":
            import json
            return f"{block.get('name', '')} {json.dumps(block.get('input', {}))}"
        elif b_type == "tool_result":
            res_content = block.get("content", "")
            if isinstance(res_content, list):
                return " ".join(_extract_block_text(sub) for sub in res_content)
            return str(res_content)
        elif "text" in block:
            return str(block["text"])
        else:
            import json
            return json.dumps(block)
    return str(block)


def _message_tokens(msg: dict[str, Any], enc: tiktoken.Encoding) -> int:
    """Approximate token count for a single message dict."""
    content = msg.get("content") or ""
    if isinstance(content, list):
        text = " ".join(_extract_block_text(block) for block in content)
    else:
        text = str(content)

    if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
        import json
        text += " " + json.dumps(msg["tool_calls"])

    return _count_tokens(text, enc) + 4


def _system_tokens(system: str | list[dict[str, Any]] | None, enc: tiktoken.Encoding) -> int:
    if system is None:
        return 0
    if isinstance(system, list):
        text = " ".join(_extract_block_text(block) for block in system)
    else:
        text = str(system)
    return _count_tokens(text, enc) + 4


class TokenBudgetGuard:
    """Check and optionally truncate messages to fit model context window."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.metadata: ModelMetadata = model_registry.get_metadata(model_id)
        if "/" in model_id:
            provider_part = model_id.split("/", 1)[0]
            from config import settings
            p_config = settings.get_provider_config(provider_part)
            if p_config:
                if p_config.get("context"):
                    self.metadata.context = p_config["context"]
                if p_config.get("max_output"):
                    self.metadata.max_output = p_config["max_output"]
        self._enc = _get_encoding(model_id)

    def count_prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        import json
        total = _system_tokens(system, self._enc)
        for msg in messages:
            total += _message_tokens(msg, self._enc)
        if tools:
            total += _count_tokens(json.dumps(tools), self._enc)
        return max(total, 1)

    def count_total_tokens(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> int:
        total = _system_tokens(system, self._enc)
        for msg in messages:
            total += _message_tokens(msg, self._enc)
        total += max_tokens
        total += SAFETY_BUFFER
        return total

    def clamp_max_tokens(self, requested_max_tokens: int) -> int:
        if self.metadata.max_output and requested_max_tokens > self.metadata.max_output:
            return self.metadata.max_output
        return requested_max_tokens

    def check_and_truncate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> tuple[list[dict[str, Any]], str | list[dict[str, Any]] | None, bool]:
        max_tokens = self.clamp_max_tokens(max_tokens)
        context_limit = self.metadata.context
        total = self.count_total_tokens(messages, system, max_tokens)

        if total <= context_limit:
            return messages, system, False

        logger.warning("TokenBudget: %s tokens > context %s for model '%s'. Truncating.", total, context_limit, self.model_id)

        truncated = list(messages)
        was_truncated = False

        while len(truncated) > 1:
            total = self.count_total_tokens(truncated, system, max_tokens)
            if total <= context_limit:
                break
            removed = truncated.pop(0)
            was_truncated = True
            logger.debug("TokenBudget: removed '%s' message", removed.get("role"))

        total = self.count_total_tokens(truncated, system, max_tokens)
        if total > context_limit and truncated:
            overhead = _system_tokens(system, self._enc) + max_tokens + SAFETY_BUFFER
            allowed_prompt_tokens = max(500, context_limit - overhead)

            last_msg = dict(truncated[-1])
            content_val = last_msg.get("content", "")
            if isinstance(content_val, str):
                encoded = self._enc.encode(content_val, disallowed_special=())
                if len(encoded) > allowed_prompt_tokens:
                    clipped_tokens = encoded[-allowed_prompt_tokens:]
                    last_msg["content"] = self._enc.decode(clipped_tokens)
                    truncated[-1] = last_msg
                    was_truncated = True

        return truncated, system, was_truncated
