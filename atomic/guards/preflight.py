"""Preflight Probe — lightweight pre-request availability check in Atomic layer."""

import httpx
from loguru import logger

from config import settings
from core.router.circuit_breaker import circuit_breaker_registry

# Timeout for the probe (connect + read combined)
PROBE_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)

_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]
_PROBE_MAX_TOKENS = 1


def _resolve_url_and_headers(model_id: str) -> tuple[str, dict[str, str]] | None:
    """Return (url, headers) for a probe request, or None if model_id is unknown."""
    extra: dict[str, str] = {"Content-Type": "application/json"}

    if "/" not in model_id:
        url = f"{settings.LM_STUDIO_BASE_URL.rstrip('/')}/chat/completions"
        return url, extra

    provider_part, _ = model_id.split("/", 1)

    if provider_part == "nvidia_nim":
        url = f"{settings.NVIDIA_NIM_BASE_URL.rstrip('/')}/chat/completions"
        if settings.NVIDIA_NIM_API_KEY:
            extra["Authorization"] = f"Bearer {settings.NVIDIA_NIM_API_KEY}"
    elif provider_part == "open_router":
        url = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
        if settings.OPENROUTER_API_KEY:
            extra["Authorization"] = f"Bearer {settings.OPENROUTER_API_KEY}"
        extra["HTTP-Referer"] = "https://github.com/Alishahryar1/free-claude-code"
        extra["X-Title"] = "Claude Code Proxy"
    elif provider_part == "groq":
        url = f"{settings.GROQ_BASE_URL.rstrip('/')}/chat/completions"
        if settings.GROQ_API_KEY:
            extra["Authorization"] = f"Bearer {settings.GROQ_API_KEY}"
    elif provider_part == "deepseek":
        url = f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
        if settings.DEEPSEEK_API_KEY:
            extra["Authorization"] = f"Bearer {settings.DEEPSEEK_API_KEY}"
    elif provider_part == "mistral":
        url = f"{settings.MISTRAL_BASE_URL.rstrip('/')}/chat/completions"
        if settings.MISTRAL_API_KEY:
            extra["Authorization"] = f"Bearer {settings.MISTRAL_API_KEY}"
    elif provider_part == "cerebras":
        url = f"{settings.CEREBRAS_BASE_URL.rstrip('/')}/chat/completions"
        if settings.CEREBRAS_API_KEY:
            extra["Authorization"] = f"Bearer {settings.CEREBRAS_API_KEY}"
    elif provider_part == "fireworks":
        url = f"{settings.FIREWORKS_BASE_URL.rstrip('/')}/chat/completions"
        if settings.FIREWORKS_API_KEY:
            extra["Authorization"] = f"Bearer {settings.FIREWORKS_API_KEY}"
    elif provider_part == "kimi":
        url = "https://api.moonshot.cn/v1/chat/completions"
        if settings.KIMI_API_KEY:
            extra["Authorization"] = f"Bearer {settings.KIMI_API_KEY}"
    elif provider_part == "gemini":
        url = f"{settings.GEMINI_BASE_URL.rstrip('/')}/openai/chat/completions"
        if settings.GEMINI_API_KEY:
            extra["Authorization"] = f"Bearer {settings.GEMINI_API_KEY}"
    elif provider_part == "lmstudio":
        url = f"{settings.LM_STUDIO_BASE_URL.rstrip('/')}/chat/completions"
    elif provider_part == "ollama":
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    elif provider_part == "llama_cpp":
        url = f"{settings.LLAMA_CPP_BASE_URL.rstrip('/')}/chat/completions"
    else:
        url = f"{settings.LM_STUDIO_BASE_URL.rstrip('/')}/chat/completions"

    return url, extra


def _extract_model_name(model_id: str) -> str:
    """Strip provider prefix from model_id."""
    if "/" in model_id:
        _, name = model_id.split("/", 1)
        return name
    return model_id


async def preflight_model_probe(model_id: str) -> bool:
    """Probe model_id with a minimal request. Updates circuit breaker state."""
    resolved = _resolve_url_and_headers(model_id)
    if resolved is None:
        logger.warning("Preflight: cannot resolve endpoint for '%s'", model_id)
        return False

    url, headers = resolved
    payload = {
        "model": _extract_model_name(model_id),
        "messages": _PROBE_MESSAGES,
        "max_tokens": _PROBE_MAX_TOKENS,
        "stream": False,
    }

    cb = circuit_breaker_registry.get(model_id)
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code in (200, 201, 202):
            await cb.record_success()
            logger.debug("Preflight OK: '%s' → %d", model_id, response.status_code)
            return True

        logger.warning("Preflight FAIL: '%s' → %d", model_id, response.status_code)
        await cb.record_failure()
        return False

    except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
        logger.warning("Preflight FAIL: '%s' → %s: %s", model_id, type(exc).__name__, exc)
        await cb.record_failure()
        return False
    except Exception as exc:
        logger.error("Preflight ERROR: '%s' → unexpected: %s", model_id, exc)
        return False
