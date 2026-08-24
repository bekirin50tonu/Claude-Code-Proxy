"""Telegram remote prompt sanitization and request payload injection helper."""

import html
from typing import Any

from loguru import logger

from core.interceptor.prompt_queue import prompt_queue_manager


def sanitize_telegram_prompt(prompt: str, max_length: int = 2000) -> str:
    """Sanitize remote Telegram prompts via HTML escape, length capping, and control token filtering."""
    if not prompt or not isinstance(prompt, str):
        return ""
    forbidden = ["</think>", "<|im_end|>", "<|endoftext|>", "[INST]", "[/INST]"]
    cleaned = prompt
    for token in forbidden:
        cleaned = cleaned.replace(token, "")

    escaped = html.escape(cleaned.strip())
    if len(escaped) > max_length:
        escaped = escaped[:max_length] + " [truncated]"
    return escaped


def inject_telegram_prompts(body: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Pop pending remote prompts for session_id and inject into request messages payload."""
    messages = body.get("messages")
    pending_prompts = prompt_queue_manager.pop_all_prompts(session_id)
    if not pending_prompts and session_id != "default_session":
        pending_prompts = prompt_queue_manager.pop_all_prompts("default_session")

    if pending_prompts:
        sanitized_prompts = [sanitize_telegram_prompt(p) for p in pending_prompts if p]
        sanitized_prompts = [p for p in sanitized_prompts if p]
        if sanitized_prompts:
            injection_text = "\n\n".join([f"📌 [Remote User Instruction via Telegram]: {p}" for p in sanitized_prompts])
            if messages and isinstance(messages, list):
                if messages[-1].get("role") == "user":
                    last_content = messages[-1].get("content")
                    if isinstance(last_content, str):
                        messages[-1]["content"] = last_content + f"\n\n{injection_text}"
                    elif isinstance(last_content, list):
                        messages[-1]["content"].append({"type": "text", "text": injection_text})
                else:
                    messages.append({"role": "user", "content": injection_text})
            logger.info(f"Injected {len(sanitized_prompts)} sanitized Telegram prompt(s) into session '{session_id}' request payload.")
    return body
