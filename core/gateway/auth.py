"""Gateway authentication verification and error response helpers."""

import sys

from fastapi import Request, status
from fastapi.responses import JSONResponse

from config import settings


def _get_settings():
    gw = sys.modules.get("core.gateway")
    if gw and hasattr(gw, "settings"):
        return gw.settings
    r = sys.modules.get("api.routes")
    if r and hasattr(r, "settings"):
        return r.settings
    return settings


def check_auth(request: Request) -> bool:
    """Verify authorization token in request headers."""
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


def auth_error_response() -> JSONResponse:
    """Return standard 401 Unauthorized JSON response."""
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
