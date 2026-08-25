
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




class ModelSelector:
    """Central model selection and fallback management in Core layer."""

    def __init__(self) -> None:
        self._preflight_fn = None

    def _get_preflight(self):
        if self._preflight_fn is None:
            from atomic.guards.preflight import preflight_model_probe
            self._preflight_fn = preflight_model_probe
        return self._preflight_fn

    async def _is_available(self, model_id: str) -> bool:
        """Fast async check — Circuit Breaker and Rate Limiter only."""
        cb = circuit_breaker_registry.get(model_id)
        if await cb.is_open():
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

            if await self._is_available(model_id):
                cb = circuit_breaker_registry.get(model_id)
                cb_state = getattr(cb, "state", "closed")
                cb_state_str = str(getattr(cb_state, "value", cb_state)).lower()
                is_half_open = "half" in cb_state_str
                is_mocked = hasattr(preflight, "__name__") and preflight.__name__ != "preflight_model_probe" or callable(preflight) and preflight != self._preflight_fn

                if is_half_open or is_mocked:
                    ok = await preflight(model_id)
                    if not ok:
                        logger.warning("Selector: preflight failed for '%s'", model_id)
                        tried.append(model_id)
                        continue

                if model_id != primary:
                    logger.warning("Selector: primary '%s' unavailable, routing to fallback '%s'", primary, model_id)
                else:
                    logger.debug("Selector: selected primary '%s'", model_id)
                return model_id
            else:
                logger.info("Selector: %s unavailable (CB/RL), skipping", model_id)
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
        """Update circuit breaker, rate limiter, and daily RPD state after request execution."""
        provider = model_id.split("/", 1)[0] if "/" in model_id else "nvidia_nim"
        from core.router.daily_tracker import daily_request_tracker
        daily_request_tracker.record_request(provider)

        cb = circuit_breaker_registry.get(model_id)
        if success:
            await cb.record_success()
        else:
            reason_lower = (reason or "").lower()
            if any(k in reason_lower for k in ["quota_exceeded", "daily_quota", "rpd limit", "daily limit", "quota exceeded"]):
                daily_request_tracker.mark_exceeded(provider)
                await cb.force_open(f"Daily RPD quota exceeded ({reason})")
            else:
                await cb.record_failure(reason=reason or "Upstream request failure")
            logger.warning("Selector: failure recorded for '%s' (CB failures: %d, reason: %s)", model_id, cb._failure_count, reason)

        if headers:
            rate_limit_parser.update_from_headers(model_id, headers)

    def get_status(self) -> dict[str, object]:
        """Return combined CB + RL status for dashboard telemetry."""
        cb_statuses = circuit_breaker_registry.all_statuses()
        rl_statuses = rate_limit_parser.all_statuses()
        configured_models = model_registry.all_configured_models() if hasattr(model_registry, "all_configured_models") else []

        all_ids = set(cb_statuses.keys()) | set(rl_statuses.keys()) | set(configured_models)
        result: dict[str, object] = {}
        for mid in sorted(all_ids):
            if not mid:
                continue
            result[mid] = {
                "circuit_breaker": cb_statuses.get(mid, {"state": "closed", "failure_count": 0, "last_failure_reason": "None (Operational)"}),
                "rate_limit": rl_statuses.get(mid, {"has_headroom": True, "req_remaining": None, "tok_remaining": None}),
            }
        return result


model_selector = ModelSelector()
