import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from loguru import logger

from atomic.sanitizers.gemini_sanitizer import GeminiPayloadSanitizer
from atomic.sanitizers.nim_sanitizer import NimPayloadSanitizer
from config import settings
from config.providers_registry import PROVIDER_REGISTRY, UnknownProviderError
from core.key_manager import universal_key_manager
from providers.base import BaseProvider


async def _select_key(raw_key: str, provider_part: str) -> str:
    """Backward compatibility helper wrapping universal_key_manager.select_key."""
    return await universal_key_manager.select_key(raw_key, provider_part)


class OpenAICompatibleProvider(BaseProvider):
    async def _resolve_endpoint(
        self, mapped_model: str
    ) -> tuple[str, str, str, dict[str, str]]:
        """Resolve base URL, actual model name, API key, and extra headers."""
        if "/" in mapped_model:
            provider_part, model_name = mapped_model.split("/", 1)
        else:
            provider_part = "lmstudio"
            model_name = mapped_model

        spec = PROVIDER_REGISTRY.get(provider_part)
        if not spec:
            mode = getattr(settings, "UNKNOWN_PROVIDER_MODE", "raise").lower()
            if mode == "raise":
                raise UnknownProviderError(provider_part, list(PROVIDER_REGISTRY.keys()))
            elif mode == "log":
                logger.error(
                    "Unknown provider '{}'. Falling back to LM Studio local.", provider_part
                )
                spec = PROVIDER_REGISTRY["lmstudio"]
            else:
                spec = PROVIDER_REGISTRY["lmstudio"]

        base_url = getattr(settings, spec.base_url_attr, spec.default_base_url) or spec.default_base_url
        if spec.url_path_suffix:
            base_url = base_url.rstrip("/") + spec.url_path_suffix

        raw_key = getattr(settings, spec.api_key_attr, "") if spec.api_key_attr else ""
        if provider_part == "nvidia_nim" and not raw_key:
            raw_key = getattr(settings, "NVIDIA_NIM_API_KEY", "")

        api_key = await _select_key(raw_key, provider_part)
        extra_headers = dict(spec.extra_headers)

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
        """Send request to OpenAI-compatible chat completions endpoint."""
        base_url, upstream_model, api_key, extra_headers = await self._resolve_endpoint(model)

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
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]
        elif kwargs.get("output_config"):
            payload["output_config"] = kwargs["output_config"]
        if kwargs.get("extra_body"):
            payload["extra_body"] = kwargs["extra_body"]

        provider_part_check = model.split("/", 1)[0] if "/" in model else ""

        p_cfg = settings.get_provider_config(provider_part_check) if provider_part_check else {}

        if provider_part_check == "gemini" or "gemini" in model.lower():
            payload = await GeminiPayloadSanitizer.sanitize(payload)
        elif provider_part_check == "nvidia_nim" or "nvidia_nim" in model.lower():
            max_out = p_cfg.get("max_output")
            payload = await NimPayloadSanitizer.sanitize(payload, max_output_override=max_out)
        elif provider_part_check in ("open_router", "openrouter") or "openrouter" in model.lower():
            extra_body = dict(payload.get("extra_body") or {})
            extra_body.setdefault("reasoning", {"enabled": True})
            payload["extra_body"] = extra_body
        from shared.utils.timeout_calculator import calculate_dynamic_timeout

        connect_t = p_cfg.get("http_connect_timeout") or settings.HTTP_CONNECT_TIMEOUT
        read_t = calculate_dynamic_timeout(model, messages=messages, tools=tools, max_tokens=max_tokens, system=system)
        write_t = p_cfg.get("http_write_timeout") or settings.HTTP_WRITE_TIMEOUT

        timeout = httpx.Timeout(
            connect=connect_t,
            read=read_t,
            write=write_t,
            pool=None,
        )

        candidate_keys = []
        if provider_part_check == "nvidia_nim":
            from core.key_manager import nim_key_manager
            candidate_keys = await nim_key_manager.get_active_candidate_keys()
        if not candidate_keys:
            candidate_keys = [api_key]

        last_exc: Exception | None = None
        for key_idx, current_key in enumerate(candidate_keys):
            if current_key:
                headers["Authorization"] = f"Bearer {current_key}"
            else:
                headers.pop("Authorization", None)

            client = httpx.AsyncClient(timeout=timeout)
            try:
                if stream:
                    req = client.build_request(
                        "POST",
                        f"{base_url.rstrip('/')}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    try:
                        response = await asyncio.wait_for(client.send(req, stream=True), timeout=read_t)
                    except TimeoutError as t_err:
                        await client.aclose()
                        raise httpx.ReadTimeout(f"Upstream HTTP response headers timeout (>{read_t}s) from {base_url}") from t_err

                    if response.status_code in (429, 401) and provider_part_check == "nvidia_nim" and len(candidate_keys) > 1:
                        error_bytes = await response.aread()
                        await response.aclose()
                        await client.aclose()
                        from core.key_manager import nim_key_manager
                        await nim_key_manager.mark_passive(current_key)
                        logger.warning(
                            "NVIDIA NIM Stream HTTP {} on key #{}. Silent failover to next key in pool...",
                            response.status_code,
                            key_idx + 1,
                        )
                        continue

                    if response.status_code >= 400:
                        error_bytes = await response.aread()
                        await response.aclose()
                        error_text = error_bytes.decode("utf-8", errors="replace")
                        logger.error(
                            "Upstream Stream HTTP {} Error from {}: {}",
                            response.status_code,
                            f"{base_url.rstrip('/')}/chat/completions",
                            error_text,
                        )
                        response.raise_for_status()

                    self._last_stream_headers = dict(response.headers)
                    return self._stream_response_generator(client, response, chunk_timeout=read_t)
                else:
                    try:
                        res_body, res_headers = await self._non_stream_request(
                            client, f"{base_url.rstrip('/')}/chat/completions", headers, payload
                        )
                        return res_body, res_headers
                    except httpx.HTTPStatusError as status_err:
                        if status_err.response.status_code in (429, 401) and provider_part_check == "nvidia_nim" and len(candidate_keys) > 1:
                            from core.key_manager import nim_key_manager
                            await nim_key_manager.mark_passive(current_key)
                            logger.warning(
                                "NVIDIA NIM Non-Stream HTTP {} on key #{}. Silent failover to next key in pool...",
                                status_err.response.status_code,
                                key_idx + 1,
                            )
                            continue
                        raise
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout, TimeoutError) as timeout_err:
                await client.aclose()
                last_exc = timeout_err
                if provider_part_check == "nvidia_nim" and len(candidate_keys) > 1:
                    from core.key_manager import nim_key_manager
                    await nim_key_manager.mark_passive(current_key)
                    logger.warning(
                        "NVIDIA NIM Network/Timeout (%s) on key #%d. Silent failover to next key in pool...",
                        type(timeout_err).__name__,
                        key_idx + 1,
                    )
                    continue
                raise
            except Exception:
                await client.aclose()
                raise

        if last_exc:
            raise last_exc

    async def _stream_response_generator(
        self, client: httpx.AsyncClient, response: httpx.Response, chunk_timeout: float = 120.0
    ) -> AsyncGenerator[dict[str, Any], None]:

        """Yield OpenAI SSE chunks from an established stream response with an active idle timeout."""
        try:
            line_iter = response.aiter_lines().__aiter__()
            while True:
                try:
                    line = await asyncio.wait_for(line_iter.__anext__(), timeout=chunk_timeout)
                except StopAsyncIteration:
                    break
                except TimeoutError as err:
                    logger.warning("Upstream SSE stream stalled (no data for %.1fs)", chunk_timeout)
                    raise httpx.ReadTimeout(f"Upstream stream stalled for {chunk_timeout}s") from err

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
                    "Upstream HTTP {} Error from {}: {}",
                    response.status_code,
                    url,
                    response.text,
                )
            response.raise_for_status()
            resp_headers = dict(response.headers)
            return response.json(), resp_headers
