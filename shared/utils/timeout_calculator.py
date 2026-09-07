"""Dynamic HTTP Read Timeout Calculator for Claude Code Proxy.

Calculates intelligent, workload-aware HTTP read timeouts based on payload complexity,
thinking/reasoning mode, tool execution status, input tokens, and requested output size.
"""

from typing import Any

from loguru import logger

from config import settings


def calculate_dynamic_timeout(
    model: str,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    system: str | list[dict[str, Any]] | None = None,
) -> float:
    """Calculate an intelligent, workload-aware HTTP read timeout for LLM requests.

    Factors considered:
    1. Base provider/settings HTTP_READ_TIMEOUT (default 180s).
    2. Reasoning / Thinking models (+120s to +180s).
    3. Tool usage (e.g. bash, write, edit, mcp, agentic workloads) (+120s).
    4. Input prompt payload size (+15s per 10k tokens/chars).
    5. Requested max output tokens (+20s per 4k output tokens).

    Clamped between MIN_DYNAMIC_READ_TIMEOUT (180s) and MAX_DYNAMIC_READ_TIMEOUT (600s).
    """
    if not getattr(settings, "ENABLE_DYNAMIC_TIMEOUT", True):
        return float(getattr(settings, "HTTP_READ_TIMEOUT", 180))

    provider_part = model.split("/", 1)[0] if "/" in model else ""
    p_cfg = settings.get_provider_config(provider_part) if provider_part else {}
    raw_base = p_cfg.get("http_read_timeout") or getattr(settings, "HTTP_READ_TIMEOUT", 60)
    base_t = max(15.0, float(raw_base))

    extra_timeout = 0.0

    # 1. Inspect Thinking / Reasoning mode
    has_thinking = False
    model_lower = model.lower()
    if any(k in model_lower for k in ("nemotron", "r1", "reasoning", "opus", "sonnet-3-7", "sonnet-3.7", "deepseek-r1")):
        has_thinking = True

    if messages:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        has_thinking = True
                        break
            elif isinstance(content, str) and "<think" in content.lower():
                has_thinking = True

    if has_thinking:
        extra_timeout += 180.0  # Add 3 minutes for deep reasoning models

    # 2. Inspect Tool usage / Agentic tasks / Bash & Download commands
    has_tools = False
    if tools and isinstance(tools, list) and len(tools) > 0:
        has_tools = True

    if messages:
        for msg in reversed(messages[-5:]):
            if isinstance(msg, dict):
                txt = str(msg.get("content", "")).lower()
                if any(kw in txt for kw in ("wget", "curl", "download", "scrape", "bash", "create page", "write file", "edit file", "router", "subagent", "mcp", "stitch")):
                    has_tools = True
                    extra_timeout += 180.0
                    break

    if has_tools:
        extra_timeout += 180.0  # Add 3 minutes for heavy tool / page generation tasks

    # 3. Payload length (input chars approximation)
    payload_char_count = 0
    if system:
        payload_char_count += len(str(system))
    if messages:
        for m in messages:
            payload_char_count += len(str(m.get("content", "")))

    if payload_char_count > 20000:  # ~5k tokens
        extra_timeout += min(300.0, (payload_char_count / 20000) * 30.0)

    # 4. Requested Max Tokens
    if max_tokens > 4096:
        extra_timeout += min(300.0, (max_tokens / 4096) * 30.0)

    calculated = base_t + extra_timeout

    min_t = float(getattr(settings, "MIN_DYNAMIC_READ_TIMEOUT", 45.0))
    max_t = float(getattr(settings, "MAX_DYNAMIC_READ_TIMEOUT", 1800.0))

    final_t = max(min_t, min(max_t, calculated))

    logger.info(
        "Dynamic Timeout: model='{}', base={:.0f}s, thinking={}, tools={}, final={:.0f}s",
        model,
        base_t,
        has_thinking,
        has_tools,
        final_t,
    )
    return final_t
