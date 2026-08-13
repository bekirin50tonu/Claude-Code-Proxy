import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from config import settings
from providers.base import BaseProvider

logger = logging.getLogger("proxy.providers.openai")

_key_counters: dict[str, int] = {}


def _select_key(raw_key: str, provider_part: str) -> str:
    """Select single key or rotate through comma-separated keys round-robin."""
    if not raw_key:
        return ""
    if "," in raw_key:
        keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        if keys:
            idx = _key_counters.get(provider_part, 0)
            _key_counters[provider_part] = (idx + 1) % len(keys)
            return keys[idx]
    return raw_key.strip()


class OpenAICompatibleProvider(BaseProvider):
    def _resolve_endpoint(
        self, mapped_model: str
    ) -> tuple[str, str, str, dict[str, str]]:
        """Resolve base URL, actual model name, API key, and extra headers."""
        extra_headers = {}
        if "/" in mapped_model:
            provider_part, model_name = mapped_model.split("/", 1)
        else:
            provider_part = "lmstudio"
            model_name = mapped_model

        if provider_part == "nvidia_nim":
            base_url = settings.NVIDIA_NIM_BASE_URL
            api_key = _select_key(settings.NVIDIA_NIM_API_KEY, "nvidia_nim")
        elif provider_part == "open_router":
            base_url = settings.OPENROUTER_BASE_URL
            api_key = _select_key(settings.OPENROUTER_API_KEY, "open_router")
            extra_headers["HTTP-Referer"] = (
                "https://github.com/Alishahryar1/free-claude-code"
            )
            extra_headers["X-Title"] = "Claude Code Proxy"
        elif provider_part == "groq":
            base_url = settings.GROQ_BASE_URL
            api_key = _select_key(settings.GROQ_API_KEY, "groq")
        elif provider_part == "deepseek":
            base_url = settings.DEEPSEEK_BASE_URL
            api_key = _select_key(settings.DEEPSEEK_API_KEY, "deepseek")
        elif provider_part == "mistral":
            base_url = settings.MISTRAL_BASE_URL
            api_key = _select_key(settings.MISTRAL_API_KEY, "mistral")
        elif provider_part == "cerebras":
            base_url = settings.CEREBRAS_BASE_URL
            api_key = _select_key(settings.CEREBRAS_API_KEY, "cerebras")
        elif provider_part == "fireworks":
            base_url = settings.FIREWORKS_BASE_URL
            api_key = _select_key(settings.FIREWORKS_API_KEY, "fireworks")
        elif provider_part == "kimi":
            base_url = "https://api.moonshot.cn/v1"
            api_key = _select_key(settings.KIMI_API_KEY, "kimi")
        elif provider_part == "gemini":
            base_url = settings.GEMINI_BASE_URL.rstrip("/") + "/openai"
            api_key = _select_key(settings.GEMINI_API_KEY, "gemini")
        elif provider_part == "lmstudio":
            base_url = settings.LM_STUDIO_BASE_URL
            api_key = ""
        elif provider_part == "ollama":
            base_url = settings.OLLAMA_BASE_URL
            api_key = ""
        elif provider_part == "llama_cpp":
            base_url = settings.LLAMA_CPP_BASE_URL
            api_key = ""
        else:
            # Unknown provider — fall back to LM Studio local
            base_url = settings.LM_STUDIO_BASE_URL
            api_key = ""
            model_name = mapped_model

        return base_url, model_name, api_key, extra_headers

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
        """Send request to OpenAI-compatible chat completions endpoint.

        Non-stream: returns (response_dict, response_headers_dict) tuple.
        Stream:     returns AsyncGenerator that yields OpenAI chunk dicts;
                    the response headers are stored on the generator object
                    as `response_headers` after the first chunk is yielded.
        """
        base_url, upstream_model, api_key, extra_headers = self._resolve_endpoint(model)

        openai_messages = self.translate_messages(messages, system)
        openai_tools = self.translate_tools(tools)

        headers = {"Content-Type": "application/json", **extra_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": upstream_model,
            "messages": openai_messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        timeout = httpx.Timeout(
            connect=settings.HTTP_CONNECT_TIMEOUT,
            read=settings.HTTP_READ_TIMEOUT,
            write=settings.HTTP_WRITE_TIMEOUT,
            pool=None,
        )

        client = httpx.AsyncClient(timeout=timeout)

        try:
            if stream:
                req = client.build_request(
                    "POST",
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response = await client.send(req, stream=True)
                if response.status_code >= 400:
                    error_bytes = await response.aread()
                    await response.aclose()
                    error_text = error_bytes.decode("utf-8", errors="replace")
                    logger.error(
                        "Upstream Stream HTTP %d Error from %s: %s",
                        response.status_code,
                        f"{base_url.rstrip('/')}/chat/completions",
                        error_text,
                    )
                    response.raise_for_status()

                self._last_stream_headers = dict(response.headers)
                return self._stream_response_generator(client, response)
            else:
                return await self._non_stream_request(
                    client, f"{base_url.rstrip('/')}/chat/completions", headers, payload
                )
        except Exception:
            await client.aclose()
            raise

    async def _stream_response_generator(
        self, client: httpx.AsyncClient, response: httpx.Response
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield OpenAI SSE chunks from an established stream response."""
        try:
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
        finally:
            await response.aclose()
            await client.aclose()

    async def _non_stream_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Returns (response_body, response_headers)."""
        async with client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code >= 400:
                logger.error(
                    "Upstream HTTP %d Error from %s: %s",
                    response.status_code,
                    url,
                    response.text,
                )
            response.raise_for_status()
            resp_headers = dict(response.headers)
            return response.json(), resp_headers
