"""Upstream model execution and candidate failover loop for Core Gateway."""

from collections.abc import AsyncGenerator
from typing import Any
from fastapi import Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from atomic.guards.nim_guard import nim_throttle_guard
from atomic.guards.stream_guard import guarded
from atomic.guards.token_budget import TokenBudgetGuard
from config import settings, stats
from core.interceptor.json_repair import JSONRepairNormalizer
from core.router.selector import model_selector
from core.transformer.stream_engine import StreamEngine, translate_non_stream_response
from providers.openai import OpenAICompatibleProvider
from shared.exceptions import NimQueueTimeoutError
from core.gateway.stream_handler import record_request_log

import sys
provider = OpenAICompatibleProvider()


def _get_provider():
    gw = sys.modules.get("core.gateway")
    if gw and hasattr(gw, "_get_provider") and gw._get_provider is not _get_provider:
        res = gw._get_provider
        return res() if callable(res) else res
    if gw and hasattr(gw, "provider"):
        return gw.provider
    r = sys.modules.get("api.routes")
    if r and hasattr(r, "provider"):
        return r.provider
    return provider


def _get_settings():
    gw = sys.modules.get("core.gateway")
    if gw and hasattr(gw, "settings"):
        return gw.settings
    r = sys.modules.get("api.routes")
    if r and hasattr(r, "settings"):
        return r.settings
    return settings


async def try_models(
    candidates: list[str],
    body: dict[str, Any],
    request: Request,
    start_time: float,
    system_prompt: str | list[dict[str, Any]] | None = None,
) -> Any:
    """Iterate through candidate models, executing requests with NIM concurrency control, token budget clamping, and automatic circuit-breaker fallback."""
    client_model = body.get("model", "unknown")
    stream = body.get("stream", False)
    messages = body.get("messages", [])
    system = system_prompt if system_prompt is not None else body.get("system")

    tools = body.get("tools")
    if tools and isinstance(tools, list):
        tools = [t for t in tools if isinstance(t, dict) and t.get("name") not in ("Artifact", "artifact")]
    temperature = body.get("temperature", 1.0)
    max_tokens = body.get("max_tokens", 4096)

    last_error: Exception | None = None
    tried_models: list[str] = []
    attempt_history: list[dict[str, Any]] = []

    is_stop_hook = JSONRepairNormalizer.is_stop_hook_target(body)

    for mapped_model in candidates:
        if not await model_selector._is_available(mapped_model):
            tried_models.append(mapped_model)
            attempt_history.append({
                "model": mapped_model,
                "error_type": "CircuitBreakerOpen",
                "error_message": "Model unavailable / circuit breaker open",
                "status_code": 503,
                "failure_reason": "Circuit Breaker Open",
            })
            continue

        token_guard = TokenBudgetGuard(mapped_model)
        cur_messages, cur_system, _ = token_guard.check_and_truncate(messages, system, max_tokens)
        clamped_max_tokens = token_guard.clamp_max_tokens(max_tokens)
        clamped_temp = min(temperature, 0.6) if "llama" in mapped_model.lower() else temperature

        is_nim = mapped_model.startswith("nvidia_nim/")
        nim_cm = nim_throttle_guard.acquire(mapped_model) if is_nim else None
        nim_acquired = False

        stats.active_concurrency += 1
        try:
            if nim_cm:
                await nim_cm.__aenter__()
                nim_acquired = True

            from core.gateway.rate_limit import concurrency_semaphore
            async with concurrency_semaphore:
                upstream_res = await _get_provider().complete(
                    model=mapped_model,
                    messages=cur_messages,
                    system=cur_system,
                    tools=tools,
                    stream=stream,
                    temperature=clamped_temp,
                    max_tokens=clamped_max_tokens,
                )

                if stream:
                    assert hasattr(upstream_res, "__aiter__")

                    session_id = request.headers.get("x-session-id") or request.headers.get("x-conversation-id")
                    engine = StreamEngine(target_model=client_model, tools=tools, session_id=session_id, is_stop_hook=is_stop_hook)
                    guarded_stream = guarded(
                        engine.stream_response(upstream_res),
                        stream_timeout=_get_settings().HTTP_READ_TIMEOUT,
                    )

                    active_nim_cm = nim_cm if nim_acquired else None
                    nim_acquired = False  # Ownership passed to generator finally block

                    async def _record_after_stream(
                        target_stream=guarded_stream,
                        target_model=mapped_model,
                        target_engine=engine,
                        nim_ctx=active_nim_cm,
                    ) -> AsyncGenerator[str, None]:
                        try:
                            async for chunk in target_stream:
                                yield chunk
                            resp_headers = getattr(provider, "_last_stream_headers", {})
                            await model_selector.record_outcome(
                                target_model, success=True, headers=resp_headers
                            )
                        except Exception as exc:
                            await model_selector.record_outcome(
                                target_model, success=False, headers={}
                            )
                            logger.error("Stream error for '%s': %s", target_model, exc)
                        finally:
                            if nim_ctx:
                                await nim_ctx.__aexit__(None, None, None)
                            record_request_log(
                                "POST",
                                "/v1/messages",
                                client_model,
                                target_model,
                                200,
                                start_time,
                                mocked=False,
                                fallbacks_used=tried_models,
                                request_body=body,
                                response_body=target_engine.get_summary_response(),
                                attempt_history=attempt_history,
                            )

                    return StreamingResponse(
                        _record_after_stream(),
                        media_type="text/event-stream",
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )

                else:
                    assert isinstance(upstream_res, tuple)
                    resp_body, resp_headers = upstream_res
                    await model_selector.record_outcome(mapped_model, success=True, headers=resp_headers)

                    translated = translate_non_stream_response(resp_body)
                    translated["model"] = client_model

                    if is_stop_hook:
                        translated = await JSONRepairNormalizer.process_response_dict(translated)

                    record_request_log(
                        "POST",
                        "/v1/messages",
                        client_model,
                        mapped_model,
                        200,
                        start_time,
                        fallbacks_used=tried_models,
                        request_body=body,
                        response_body=translated,
                        attempt_history=attempt_history,
                    )
                    return JSONResponse(content=translated)

        except NimQueueTimeoutError as exc:
            last_error = exc
            tried_models.append(mapped_model)
            stats.error_count += 1
            attempt_history.append({
                "model": mapped_model,
                "error_type": "NimQueueTimeoutError",
                "error_message": str(exc),
                "status_code": 429,
                "failure_reason": "NVIDIA NIM Queue Timeout (>30s)",
            })
            logger.warning(
                "NVIDIA NIM queue timeout for model '%s': %s. Redirecting request to next fallback model.",
                mapped_model,
                exc,
            )
            await model_selector.record_outcome(
                mapped_model, success=False, headers={}, reason="NVIDIA NIM Queue Timeout (>30s)"
            )
            continue
        except Exception as e:
            last_error = e
            tried_models.append(mapped_model)
            stats.error_count += 1

            err_headers: dict[str, str] = {}
            failure_reason = str(e)
            status_code = getattr(getattr(e, "response", None), "status_code", 500) if hasattr(e, "response") else 500
            if hasattr(e, "response") and hasattr(e.response, "headers"):
                err_headers = dict(e.response.headers)
                if status_code == 429:
                    failure_reason = "HTTP 429 (Rate Limit Exceeded)"
                    from core.router.rate_limiter import rate_limit_parser
                    rl_state = rate_limit_parser.get(mapped_model)
                    rl_state.req_remaining = 0
                elif status_code in (404, 410):
                    failure_reason = f"HTTP {status_code} (Model Not Found / EOL)"
                    from core.router.circuit_breaker import circuit_breaker_registry
                    cb = circuit_breaker_registry.get(mapped_model)
                    await cb.force_open(reason=failure_reason)

            attempt_history.append({
                "model": mapped_model,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "status_code": status_code,
                "failure_reason": failure_reason,
                "headers": err_headers,
            })

            await model_selector.record_outcome(mapped_model, success=False, headers=err_headers, reason=failure_reason)
            logger.warning("Model '%s' failed (%s). Retrying next fallback model...", mapped_model, failure_reason)
            continue
        finally:
            if nim_acquired and nim_cm:
                await nim_cm.__aexit__(None, None, None)
            stats.active_concurrency -= 1

    stats.error_count += 1
    err_details = {
        "last_error": str(last_error),
        "last_error_type": type(last_error).__name__ if last_error else "Unknown",
        "tried_models": tried_models,
        "attempts": attempt_history,
    }
    err_content = {
        "type": "error",
        "error": {
            "type": "overloaded_error",
            "message": f"All upstream models are currently unavailable. Tried: {tried_models}. Last error: {last_error}",
            "details": err_details,
        },
    }
    record_request_log(
        "POST",
        "/v1/messages",
        client_model,
        "none",
        503,
        start_time,
        fallbacks_used=tried_models,
        request_body=body,
        response_body=err_content,
        error_details=err_details,
        attempt_history=attempt_history,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=err_content,
    )
