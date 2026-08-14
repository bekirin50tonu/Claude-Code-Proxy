import sys

from loguru import logger

from config import model_registry
from core.router.circuit_breaker import circuit_breaker_registry
from core.router.rate_limiter import rate_limit_parser
from shared.exceptions import ProxyBaseError


class AllModelsUnavailableError(ProxyBaseError):
    """Raised when every model in the candidate chain is circuit-open or rate-limited."""

    def __init__(self, client_model: str, tried: list[str]) -> None:
        self.client_model = client_model
        self.tried = tried
        super().__init__(
            message=f"All models unavailable for '{client_model}'. Tried: {tried}",
            status_code=503,
            details={"client_model": client_model, "tried": tried},
        )




def _get_cb_registry():
    mr = sys.modules.get("router.model_router")
    if mr and hasattr(mr, "circuit_breaker_registry"):
        return mr.circuit_breaker_registry
    return circuit_breaker_registry


class ModelSelector:
    """Central model selection and fallback management in Core layer."""

    def __init__(self) -> None:
        self._preflight_fn = None

    def _get_preflight(self):
        if self._preflight_fn is None:
            from atomic.guards.preflight import preflight_model_probe
            self._preflight_fn = preflight_model_probe
        return self._preflight_fn

    def _is_available(self, model_id: str) -> bool:
        """Fast sync check — Circuit Breaker and Rate Limiter only."""
        cb = _get_cb_registry().get(model_id)
        if cb.is_open():
            return False
        return rate_limit_parser.has_headroom(model_id)


    async def pick_model(self, client_model: str) -> str:
        """Select the best available model for this request."""
        preflight = self._get_preflight()

        primary = model_registry.get_primary(client_model)
        fallbacks = model_registry.get_fallbacks(client_model)
        candidates = [primary] + fallbacks

        tried: list[str] = []
        for model_id in candidates:
            if not model_id:
                continue

            if not self._is_available(model_id):
                logger.info("Selector: %s unavailable (CB/RL), skipping", model_id)
                tried.append(model_id)
                continue

            ok = await preflight(model_id)
            if ok:
                if model_id != primary:
                    logger.warning("Selector: primary '%s' unavailable, routing to fallback '%s'", primary, model_id)
                else:
                    logger.debug("Selector: selected primary '%s'", model_id)
                return model_id
            else:
                logger.warning("Selector: preflight failed for '%s'", model_id)
                tried.append(model_id)

        raise AllModelsUnavailableError(client_model, tried)

    async def record_outcome(
        self,
        model_id: str,
        *,
        success: bool,
        headers: dict[str, str] | None = None,
        reason: str = "",
    ) -> None:
        """Update circuit breaker and rate limiter state after request execution."""
        cb = circuit_breaker_registry.get(model_id)
        if success:
            await cb.record_success()
        else:
            await cb.record_failure(reason=reason or "Upstream request failure")
            logger.warning("Selector: failure recorded for '%s' (CB failures: %d, reason: %s)", model_id, cb._failure_count, reason)

        if headers:
            rate_limit_parser.update_from_headers(model_id, headers)

    def get_status(self) -> dict[str, object]:
        """Return combined CB + RL status for dashboard telemetry."""
        cb_statuses = circuit_breaker_registry.all_statuses()
        rl_statuses = rate_limit_parser.all_statuses()

        all_ids = set(cb_statuses.keys()) | set(rl_statuses.keys())
        result: dict[str, object] = {}
        for mid in sorted(all_ids):
            result[mid] = {
                "circuit_breaker": cb_statuses.get(mid, {"state": "closed", "failure_count": 0}),
                "rate_limit": rl_statuses.get(mid, {"has_headroom": True}),
            }
        return result


model_selector = ModelSelector()
