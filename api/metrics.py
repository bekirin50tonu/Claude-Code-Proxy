"""Prometheus Metrics & Operational Health Probe Router."""

import time
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from config import stats
from core.router.circuit_breaker import circuit_breaker_registry
from core.router.rate_limiter import rate_limit_parser

metrics_router = APIRouter()


@metrics_router.get("/healthz")
async def healthz_probe() -> JSONResponse:
    """Kubernetes / container liveness probe."""
    return JSONResponse(content={"status": "ok", "timestamp": time.time()})


@metrics_router.get("/readyz")
async def readyz_probe() -> JSONResponse:
    """Kubernetes / container readiness probe."""
    return JSONResponse(content={"status": "ready", "timestamp": time.time()})


@metrics_router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Return Prometheus text-formatted metrics."""
    lines: list[str] = []

    # Total requests
    lines.append("# HELP proxy_requests_total Total number of LLM proxy requests processed.")
    lines.append("# TYPE proxy_requests_total counter")
    lines.append(f"proxy_requests_total {stats.total_requests}")

    # Active concurrency
    lines.append("# HELP proxy_active_concurrency Current active concurrency count.")
    lines.append("# TYPE proxy_active_concurrency gauge")
    lines.append(f"proxy_active_concurrency {stats.active_concurrency}")

    # Error count
    lines.append("# HELP proxy_error_total Total error count.")
    lines.append("# TYPE proxy_error_total counter")
    lines.append(f"proxy_error_total {stats.error_count}")

    # Circuit Breakers State
    lines.append("# HELP proxy_circuit_state Circuit breaker state per model (0=closed, 1=half_open, 2=open).")
    lines.append("# TYPE proxy_circuit_state gauge")
    cb_statuses = circuit_breaker_registry.all_statuses()
    state_map = {"closed": 0, "half_open": 1, "open": 2}
    for model_id, st in cb_statuses.items():
        st_val = state_map.get(str(st.get("state", "closed")), 0)
        lines.append(f'proxy_circuit_state{{model="{model_id}"}} {st_val}')

    # Rate Limiter Headroom
    lines.append("# HELP proxy_rate_limit_headroom Rate limit headroom boolean per model (1=has headroom, 0=exhausted).")
    lines.append("# TYPE proxy_rate_limit_headroom gauge")
    rl_statuses = rate_limit_parser.all_statuses()
    for model_id, st in rl_statuses.items():
        hr_val = 1 if st.get("has_headroom", True) else 0
        lines.append(f'proxy_rate_limit_headroom{{model="{model_id}"}} {hr_val}')

    content = "\n".join(lines) + "\n"
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
