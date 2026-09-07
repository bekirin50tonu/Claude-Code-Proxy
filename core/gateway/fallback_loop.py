"""Upstream model execution and candidate failover loop for Core Gateway."""

import asyncio
import sys
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from atomic.guards.nim_guard import nim_throttle_guard
from atomic.guards.stream_guard import guarded
from atomic.guards.token_budget import TokenBudgetGuard
from config import settings, stats
from core.gateway.stream_handler import extract_session_id, record_request_log
from core.interceptor.json_repair import JSONRepairNormalizer
from core.router.selector import model_selector
from core.transformer.stream_engine import StreamEngine, translate_non_stream_response
from providers.openai import OpenAICompatibleProvider
from shared.exceptions import NimQueueTimeoutError

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
    is_evaluator = JSONRepairNormalizer.is_evaluator_hook_target(body)

    # Detect Recap / Away Summary requests
    is_recap = False
    if isinstance(messages, list) and messages:
        last_m = messages[-1]
        if isinstance(last_m, dict) and last_m.get("role") == "user":
            c = last_m.get("content", "")
            if isinstance(c, list):
                c = " ".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
            c_lower = str(c).lower()
            if any(k in c_lower for k in ("the user stepped away and is coming back", "recap in under", "recap in 1-2 plain sentences", "user stepped away")):
                is_recap = True

    if is_recap:
        # 1. Strip tools from recap requests to save ~43.5k tokens of schema budget
        tools = None
        # 2. Inject clean anti-hallucination guidance strictly based on actual recent messages
        recap_guidance = (
            "\n\n[RECAP DIRECTIVE]\n"
            "- Summarize the ACTUAL current state strictly based on the most recent conversation messages.\n"
            "- Accurately state what was just completed or what error occurred.\n"
            "- If an operation was in progress or interrupted, state that clearly.\n"
            "- Do NOT assume completed actions or invent future steps that have not been executed.\n"
            "- Keep the response concise, strictly under 40 words in plain text without markdown."
        )
        if isinstance(system, list):
            system = list(system) + [{"type": "text", "text": recap_guidance}]
        elif isinstance(system, str):
            system = system + recap_guidance
        else:
            system = recap_guidance

    if is_evaluator or is_stop_hook:
        evaluator_guidance = (
            "\n\n[GOAL EVALUATION DIRECTIVE]\n"
            "- Be critical and rigorous. Do NOT mark a goal satisfied prematurely based on passive inspection alone.\n"
            "- Informational queries (such as list_screens, ls, grep, find, reading files) are exploratory actions, NOT goal fulfillment.\n"
            "- If the goal specifies creating, adding, editing, or implementing code/UI/features, verify that actual file modifications or creations (e.g. Write, Edit, NotebookEdit) were executed in recent turns.\n"
            "- If the requested implementation work has not yet been executed on disk, you must mark stop_hook_active: false (or ok: false) and specify in the reason what remaining work needs to be done."
        )
        if isinstance(system, list):
            system = list(system) + [{"type": "text", "text": evaluator_guidance}]
        elif isinstance(system, str):
            system = system + evaluator_guidance
        else:
            system = evaluator_guidance

    for mapped_model in candidates:
        concurrency_decremented = False
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
        cur_messages, cur_system, _ = token_guard.check_and_truncate(messages, system, max_tokens, tools=tools)
        prompt_tokens_est = token_guard.count_prompt_tokens(cur_messages, cur_system, tools)
        headroom = max(1024, token_guard.metadata.context - prompt_tokens_est - 100)
        clamped_max_tokens = min(token_guard.clamp_max_tokens(max_tokens), headroom)
        clamped_temp = min(temperature, 0.6) if "llama" in mapped_model.lower() else temperature

        is_nim = mapped_model.startswith("nvidia_nim/")
        nim_cm = nim_throttle_guard.acquire(mapped_model) if is_nim else None
        nim_acquired = False

        stats.active_concurrency += 1
        try:
            from shared.utils.dev_logger import dev_logger
            dev_logger.record_transaction_start(
                request_id=f"req_{int(time.time()*1000)}",
                method="POST",
                path="/v1/messages",
                client_model=client_model,
                mapped_model=mapped_model,
                request_body=body,
            )

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
                    output_config=body.get("output_config"),
                    extra_body=body.get("extra_body"),
                )

                if stream and not isinstance(upstream_res, tuple):
                    session_id = extract_session_id(request, body)
                    engine = StreamEngine(
                        target_model=client_model,
                        tools=tools,
                        session_id=session_id,
                        is_stop_hook=is_stop_hook,
                        is_evaluator=is_evaluator,
                    )
                    from shared.utils.timeout_calculator import (
                        calculate_dynamic_timeout,
                    )
                    dynamic_stream_timeout = calculate_dynamic_timeout(
                        client_model,
                        messages=cur_messages,
                        tools=tools,
                        max_tokens=clamped_max_tokens,
                        system=cur_system,
                    )
                    # Peek first chunk to verify stream validity before returning 200 StreamingResponse
                    first_chunk = None
                    try:
                        first_chunk = await asyncio.wait_for(
                            upstream_res.__anext__(),
                            timeout=min(dynamic_stream_timeout, 45.0),
                        )
                    except StopAsyncIteration:
                        raise RuntimeError(f"Upstream model '{mapped_model}' returned an empty stream (0 chunks).")
                    except Exception as first_chunk_err:
                        raise first_chunk_err

                    async def _chained_upstream():
                        if first_chunk is not None:
                            yield first_chunk
                        async for c in upstream_res:
                            yield c

                    guarded_stream = guarded(
                        engine.stream_response(_chained_upstream()),
                        stream_timeout=dynamic_stream_timeout,
                    )

                    active_nim_cm = nim_cm if nim_acquired else None
                    nim_acquired = False  # Ownership passed to generator finally block

                    concurrency_decremented = False

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
                            yield f'event: error\ndata: {{"type": "error", "error": {{"type": "api_error", "message": "Stream error: {exc}"}}}}\n\n'
                        finally:
                            stats.active_concurrency -= 1
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

                    concurrency_decremented = True
                    return StreamingResponse(
                        _record_after_stream(),
                        media_type="text/event-stream",
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )

                else:
                    assert isinstance(upstream_res, tuple)
                    resp_body, resp_headers = upstream_res
                    await model_selector.record_outcome(mapped_model, success=True, headers=resp_headers)

                    translated = translate_non_stream_response(resp_body)
                    translated["model"] = client_model

                    if is_stop_hook or is_evaluator:
                        translated = await JSONRepairNormalizer.process_response_dict(
                            translated, is_stop_hook=is_stop_hook, is_evaluator=is_evaluator
                        )

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
            if not concurrency_decremented:
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
