"""BaseHandler abstract class defining the gateway request lifecycle."""

from abc import ABC, abstractmethod
import time
from typing import Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from config import stats
from core.gateway.auth import check_auth, auth_error_response
from core.gateway.rate_limit import gateway_rate_limiter
from core.gateway.stream_handler import record_request_log


class BaseHandler(ABC):
    """Abstract base handler managing common auth, rate limiting, and execution logging."""

    def __init__(self, endpoint_path: str):
        self.endpoint_path = endpoint_path

    async def handle_request(self, request: Request) -> Any:
        """Execute request pipeline: auth check -> rate limit check -> JSON parse -> process implementation."""
        start_time = time.time()
        stats.total_requests += 1

        if not check_auth(request):
            stats.error_count += 1
            record_request_log("POST", self.endpoint_path, "unknown", "unknown", 401, start_time)
            return auth_error_response()

        if not await gateway_rate_limiter.acquire():
            stats.error_count += 1
            record_request_log("POST", self.endpoint_path, "unknown", "unknown", 429, start_time)
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
            record_request_log("POST", self.endpoint_path, "unknown", "unknown", 400, start_time)
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

        return await self.process(request, body, start_time)

    @abstractmethod
    async def process(self, request: Request, body: dict[str, Any], start_time: float) -> Any:
        """Concrete endpoint processing implementation."""
        pass
