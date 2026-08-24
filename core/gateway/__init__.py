"""Core Gateway package containing modularized auth, rate limiting, model selection, stream handling, and endpoint handlers."""

import time
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from api.mock import check_mock_request
from atomic.guards.token_budget import TokenBudgetGuard
from config import settings, stats
from core.gateway.auth import auth_error_response as _auth_error_response
from core.gateway.auth import check_auth as _check_auth
from core.gateway.fallback_loop import _get_provider, provider, try_models
from core.gateway.model_select import pick_model_with_fallbacks
from core.gateway.prompt_inject import inject_telegram_prompts
from core.gateway.prompt_inject import (
    sanitize_telegram_prompt as _sanitize_telegram_prompt,
)
from core.gateway.rate_limit import (
    SlidingWindowRateLimiter,
    concurrency_semaphore,
    gateway_rate_limiter,
    rate_limiter,
)
from core.gateway.stream_handler import (
    log_after_stream,
    record_request_log,
    stream_mock_response,
)
from core.router.selector import AllModelsUnavailableError, model_selector
from core.transformer.stream_engine import translate_non_stream_response
from providers.openai import OpenAICompatibleProvider

model_router = model_selector
router = APIRouter()


@router.post("/v1/messages")
async def messages_endpoint(request: Request) -> Any:
    """Anthropic /v1/messages API Gateway endpoint."""
    start_time = time.time()
    stats.total_requests += 1

    if not _check_auth(request):
        stats.error_count += 1
        record_request_log("POST", "/v1/messages", "unknown", "unknown", 401, start_time)
        return _auth_error_response()

    if not await gateway_rate_limiter.acquire():
        stats.error_count += 1
        record_request_log("POST", "/v1/messages", "unknown", "unknown", 429, start_time)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limit exceeded. Too many requests to the proxy.",
                },
            },
        )

    try:
        body = await request.json()
    except Exception:
        stats.error_count += 1
        record_request_log("POST", "/v1/messages", "unknown", "unknown", 400, start_time)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Failed to parse JSON request body.",
                },
            },
        )

    # Sanitize payload for Subagents Emergency Switch policy
    from api.dashboard import SUBAGENTS_ENABLED
    from atomic.guards.subagent import subagent_guard
    body = subagent_guard.sanitize_payload(body, SUBAGENTS_ENABLED)

    session_id = request.headers.get("x-session-id") or request.headers.get("x-conversation-id") or "default_session"
    body = inject_telegram_prompts(body, session_id)

    client_model = body.get("model", "unknown")
    stream = body.get("stream", False)

    mock_resp = check_mock_request(body)
    if mock_resp is not None:
        stats.mocked_requests += 1
        if stream:
            return StreamingResponse(
                log_after_stream(
                    stream_mock_response(mock_resp),
                    "POST",
                    "/v1/messages",
                    client_model,
                    "local_mock",
                    start_time,
                    mocked=True,
                    request_body=body,
                    response_body=mock_resp,
                ),
                media_type="text/event-stream",
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        record_request_log(
            "POST",
            "/v1/messages",
            client_model,
            "local_mock",
            200,
            start_time,
            mocked=True,
            request_body=body,
            response_body=mock_resp,
        )
        return JSONResponse(content=mock_resp)

    candidates, system_prompt = pick_model_with_fallbacks(client_model, body)
    return await try_models(candidates, body, request, start_time, system_prompt=system_prompt)


@router.post("/v1/complete")
async def legacy_complete_endpoint(request: Request) -> Any:
    """Fallback legacy Text Completion endpoint."""
    start_time = time.time()
    stats.total_requests += 1

    if not _check_auth(request):
        stats.error_count += 1
        record_request_log("POST", "/v1/complete", "unknown", "unknown", 401, start_time)
        return _auth_error_response()

    try:
        body = await request.json()
    except Exception:
        stats.error_count += 1
        record_request_log("POST", "/v1/complete", "unknown", "unknown", 400, start_time)
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    prompt = body.get("prompt", "")
    client_model = body.get("model", "")
    messages = [{"role": "user", "content": prompt}]

    try:
        mapped_model = await model_selector.pick_model(client_model)
    except AllModelsUnavailableError as e:
        stats.error_count += 1
        record_request_log("POST", "/v1/complete", client_model, "none", 503, start_time)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"type": "error", "error": {"type": "overloaded_error", "message": str(e)}},
        )

    stats.active_concurrency += 1
    try:
        async with concurrency_semaphore:
            upstream_res = await provider.complete(
                model=mapped_model,
                messages=messages,
                stream=False,
                temperature=body.get("temperature", 1.0),
                max_tokens=body.get("max_tokens", 4096),
            )
            assert isinstance(upstream_res, tuple)
            resp_body, resp_headers = upstream_res
            await model_selector.record_outcome(mapped_model, success=True, headers=resp_headers)

            translated = translate_non_stream_response(resp_body)
            text_result = "".join(
                b.get("text", "") for b in translated.get("content", []) if b.get("type") == "text"
            )

            comp_content = {
                "completion": text_result,
                "stop_reason": "stop_sequence",
                "model": client_model,
            }
            record_request_log("POST", "/v1/complete", client_model, mapped_model, 200, start_time, request_body=body, response_body=comp_content)
            return JSONResponse(content=comp_content)
    except Exception as e:
        stats.error_count += 1
        err_headers: dict[str, str] = {}
        if hasattr(e, "response") and hasattr(e.response, "headers"):
            err_headers = dict(e.response.headers)
        await model_selector.record_outcome(mapped_model, success=False, headers=err_headers)
        record_request_log("POST", "/v1/complete", client_model, mapped_model, 500, start_time)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"type": "error", "error": {"type": "api_error", "message": f"Legacy complete upstream error: {e}"}},
        )
    finally:
        stats.active_concurrency -= 1


@router.post("/v1/messages/count_tokens")
@router.post("/v1/messages/tokens/count")
async def count_tokens_endpoint(request: Request) -> Any:
    """Anthropic token counting endpoint used by Claude Code CLI."""
    if not _check_auth(request):
        return _auth_error_response()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": "Failed to parse JSON request body."}},
        )

    client_model = body.get("model", "claude-3-5-sonnet")
    messages = body.get("messages", [])
    system = body.get("system")
    tools = body.get("tools")

    guard = TokenBudgetGuard(client_model)
    input_tokens = guard.count_prompt_tokens(messages, system, tools)

    return JSONResponse(content={"input_tokens": input_tokens})


@router.get("/v1/models")
async def list_v1_models(request: Request) -> Any:
    """Anthropic /v1/models endpoint."""
    if not _check_auth(request):
        return _auth_error_response()

    models_data = [
        {"type": "model", "id": "claude-opus-5", "display_name": "Claude Opus 5 (1M context)", "created_at": "2025-02-19T00:00:00Z"},
        {"type": "model", "id": "claude-sonnet-5", "display_name": "Claude Sonnet 5", "created_at": "2025-02-19T00:00:00Z"},
        {"type": "model", "id": "claude-haiku-4.5", "display_name": "Claude Haiku 4.5", "created_at": "2025-02-19T00:00:00Z"},
        {"type": "model", "id": "claude-3-7-sonnet-20250219", "display_name": "Claude 3.7 Sonnet", "created_at": "2025-02-19T00:00:00Z"},
        {"type": "model", "id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet", "created_at": "2024-10-22T00:00:00Z"},
        {"type": "model", "id": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku", "created_at": "2024-10-22T00:00:00Z"},
        {"type": "model", "id": "claude-3-opus-20240229", "display_name": "Claude 3 Opus", "created_at": "2024-02-29T00:00:00Z"},
    ]
    return JSONResponse(content={"data": models_data, "has_more": False})
