import asyncio
import json
import sys
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from api.mock import check_mock_request
from atomic.guards.stream_guard import guarded
from atomic.guards.token_budget import TokenBudgetGuard
from config import settings, stats
from core.router.selector import AllModelsUnavailableError, model_selector
from core.transformer.stream_engine import StreamEngine, translate_non_stream_response
from providers.openai import OpenAICompatibleProvider

model_router = model_selector
router = APIRouter()
provider = OpenAICompatibleProvider()



def record_request_log(
    method: str,
    path: str,
    client_model: str,
    target_model: str,
    status_code: int,
    start_time: float,
    mocked: bool = False,
    fallbacks_used: list[str] | None = None,
    request_body: dict[str, Any] | None = None,
    response_body: dict[str, Any] | str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    stats.record_log(
        method,
        path,
        client_model,
        target_model,
        status_code,
        start_time,
        mocked,
        fallbacks_used,
        request_body=request_body,
        response_body=response_body,
        headers=headers,
    )


async def log_after_stream(
    gen: AsyncGenerator[str, None],
    method: str,
    path: str,
    client_model: str,
    target_model: str,
    start_time: float,
    mocked: bool,
    request_body: dict[str, Any] | None = None,
    response_body: dict[str, Any] | str | None = None,
) -> AsyncGenerator[str, None]:
    try:
        async for chunk in gen:
            yield chunk
    finally:
        record_request_log(
            method,
            path,
            client_model,
            target_model,
            200,
            start_time,
            mocked=mocked,
            request_body=request_body,
            response_body=response_body,
        )


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.requests: list[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self.lock:
            now = time.time()
            self.requests = [r for r in self.requests if now - r < self.window]
            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            return False


gateway_rate_limiter = SlidingWindowRateLimiter(
    settings.PROVIDER_RATE_LIMIT, settings.PROVIDER_RATE_WINDOW
)
rate_limiter = gateway_rate_limiter
concurrency_semaphore = asyncio.Semaphore(settings.PROVIDER_MAX_CONCURRENCY)



async def stream_mock_response(mock_data: dict[str, Any]) -> AsyncGenerator[str, None]:
    """Yield mock data in Anthropic compatible SSE format."""
    model = mock_data.get("model", "claude-3-5-sonnet-latest")
    msg_id = mock_data.get("id", "msg_mock")
    text_content = mock_data["content"][0]["text"] if mock_data.get("content") else " "

    _msg_start = json.dumps(
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
    )
    _blk_start = json.dumps(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    )
    _blk_delta = json.dumps(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text_content},
        }
    )
    _blk_stop = json.dumps({"type": "content_block_stop", "index": 0})
    _msg_delta = json.dumps(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }
    )

    yield f"event: message_start\ndata: {_msg_start}\n\n"
    yield f"event: content_block_start\ndata: {_blk_start}\n\n"
    yield f"event: content_block_delta\ndata: {_blk_delta}\n\n"
    yield f"event: content_block_stop\ndata: {_blk_stop}\n\n"
    yield f"event: message_delta\ndata: {_msg_delta}\n\n"
    yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


def _get_settings():
    r = sys.modules.get("api.routes")
    if r and hasattr(r, "settings"):
        return r.settings
    return settings



def _get_provider():
    r = sys.modules.get("api.routes")
    if r and hasattr(r, "provider"):
        return r.provider
    return provider



def _check_auth(request: Request) -> bool:
    s = _get_settings()
    if not s.GATEWAY_AUTH_TOKEN:
        return True
    token = s.GATEWAY_AUTH_TOKEN
    auth_header = request.headers.get("authorization", "")
    x_api_key = request.headers.get("x-api-key", "")
    return (
        auth_header == f"Bearer {token}"
        or auth_header == token
        or x_api_key == token
    )



def _auth_error_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid or missing ANTHROPIC_AUTH_TOKEN on local proxy gateway.",
            },
        },
    )


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

    client_model = body.get("model", "unknown")
    stream = body.get("stream", False)
    messages = body.get("messages", [])
    system = body.get("system")

    model_thinking_mode = settings.get_thinking_mode(client_model)
    if model_thinking_mode == "open":
        thinking_prompt = "\n\nImportant: Analyze the task and write your step-by-step reasoning inside <think>...</think> tags before providing your answer or executing tools."
        if system:
            if isinstance(system, str):
                system = system + thinking_prompt
            elif isinstance(system, list):
                system = list(system) + [{"type": "text", "text": thinking_prompt}]
        else:
            system = thinking_prompt

    tools = body.get("tools")
    if tools and isinstance(tools, list):
        tools = [t for t in tools if isinstance(t, dict) and t.get("name") not in ("Artifact", "artifact")]
    temperature = body.get("temperature", 1.0)
    max_tokens = body.get("max_tokens", 4096)

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

    from config import model_registry

    primary = model_registry.get_primary(client_model)
    fallbacks = model_registry.get_fallbacks(client_model)
    candidates = [c for c in ([primary] + fallbacks) if c]

    last_error: Exception | None = None
    tried_models: list[str] = []

    for mapped_model in candidates:
        if not model_selector._is_available(mapped_model):
            tried_models.append(mapped_model)
            continue

        token_guard = TokenBudgetGuard(mapped_model)
        cur_messages, cur_system, _ = token_guard.check_and_truncate(messages, system, max_tokens)
        clamped_max_tokens = token_guard.clamp_max_tokens(max_tokens)
        clamped_temp = min(temperature, 0.6) if "llama" in mapped_model.lower() else temperature

        stats.active_concurrency += 1
        try:
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
                    assert isinstance(upstream_res, AsyncGenerator)
                    session_id = request.headers.get("x-session-id") or request.headers.get("x-conversation-id")
                    engine = StreamEngine(target_model=client_model, tools=tools, session_id=session_id)
                    guarded_stream = guarded(
                        engine.stream_response(upstream_res),
                        stream_timeout=settings.HTTP_READ_TIMEOUT,
                    )

                    async def _record_after_stream(
                        target_stream=guarded_stream,
                        target_model=mapped_model,
                        target_engine=engine,
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
                    )
                    return JSONResponse(content=translated)

        except Exception as e:
            last_error = e
            tried_models.append(mapped_model)
            stats.error_count += 1

            err_headers: dict[str, str] = {}
            failure_reason = str(e)
            if hasattr(e, "response") and hasattr(e.response, "headers"):
                err_headers = dict(e.response.headers)
                status_code = getattr(e.response, "status_code", None)
                if status_code == 429:
                    failure_reason = "HTTP 429 (Rate Limit Exceeded)"
                    from core.router.rate_limiter import rate_limit_parser
                    rl_state = rate_limit_parser.get(mapped_model)
                    rl_state.req_remaining = 0
                elif status_code in (404, 410):
                    failure_reason = f"HTTP {status_code} (Model Not Found / EOL)"
                    from core.router.circuit_breaker import circuit_breaker_registry
                    cb = circuit_breaker_registry.get(mapped_model)
                    cb.force_open(reason=failure_reason)

            await model_selector.record_outcome(mapped_model, success=False, headers=err_headers, reason=failure_reason)
            logger.warning("Model '%s' failed (%s). Retrying next fallback model...", mapped_model, failure_reason)
            continue
        finally:
            stats.active_concurrency -= 1

    stats.error_count += 1
    err_content = {
        "type": "error",
        "error": {
            "type": "overloaded_error",
            "message": f"All upstream models are currently unavailable. Tried: {tried_models}. Last error: {last_error}",
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
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=err_content,
    )


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
            upstream_res = await _get_provider().complete(
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
