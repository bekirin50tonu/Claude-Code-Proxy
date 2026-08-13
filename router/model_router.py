"""
Model Router — intelligent routing with circuit-breaker and rate-limit awareness.

pick_model(client_model) decision flow:
  1. Resolve primary model from ModelRegistry
  2. Run preflight probe on primary (3s timeout)
  3. If primary CB is OPEN or has no headroom → try fallback_order
  4. Each fallback is also preflight-probed before selection
  5. If ALL candidates fail → raise AllModelsUnavailableError

record_outcome() is called after each upstream request to update CB and RL state.
"""

from loguru import logger

from config import model_registry
from router.circuit_breaker import circuit_breaker_registry
from router.rate_limit_parser import rate_limit_parser


class AllModelsUnavailableError(Exception):
    """Raised when every model in the chain is circuit-open or rate-limited."""

    def __init__(self, client_model: str, tried: list[str]) -> None:
        self.client_model = client_model
        self.tried = tried
        super().__init__(f"All models unavailable for '{client_model}'. Tried: {tried}")


class ModelRouter:
    """Central routing decision maker.

    Usage
    -----
    router = ModelRouter()

    # In request handler:
    model_id = await router.pick_model(client_model)
    try:
        result, headers = await provider.complete(model_id, ...)
        await router.record_outcome(model_id, success=True, headers=headers)
    except UpstreamError as e:
        await router.record_outcome(model_id, success=False, headers=e.headers)
        raise
    """

    def __init__(self) -> None:
        # Lazy import to avoid circular dependency at module load time
        self._preflight_fn = None

    def _get_preflight(self):
        if self._preflight_fn is None:
            from guards.preflight import preflight_model_probe

            self._preflight_fn = preflight_model_probe
        return self._preflight_fn

    def _is_available(self, model_id: str) -> bool:
        """Quick sync check — CB and RL only (no network)."""
        cb = circuit_breaker_registry.get(model_id)
        if cb.is_open():
            return False
        return rate_limit_parser.has_headroom(model_id)

    async def pick_model(self, client_model: str) -> str:
        """Select the best available model for this request.

        Runs a preflight probe for each candidate before confirming selection.
        """
        preflight = self._get_preflight()

        primary = model_registry.get_primary(client_model)
        fallbacks = model_registry.get_fallbacks(client_model)
        candidates = [primary] + fallbacks

        tried: list[str] = []
        for model_id in candidates:
            if not model_id:
                continue

            # Fast sync check first (avoids network if CB is already open)
            if not self._is_available(model_id):
                logger.info("Router: %s unavailable (CB/RL), skipping", model_id)
                tried.append(model_id)
                continue

            # Preflight probe (network, 3s timeout)
            ok = await preflight(model_id)
            if ok:
                if model_id != primary:
                    logger.warning(
                        "Router: primary '%s' unavailable, routing to fallback '%s'",
                        primary,
                        model_id,
                    )
                else:
                    logger.debug("Router: selected primary '%s'", model_id)
                return model_id
            else:
                logger.warning("Router: preflight failed for '%s'", model_id)
                tried.append(model_id)

        raise AllModelsUnavailableError(client_model, tried)

    async def record_outcome(
        self,
        model_id: str,
        *,
        success: bool,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Update circuit breaker and rate limiter after a request."""
        cb = circuit_breaker_registry.get(model_id)
        if success:
            await cb.record_success()
        else:
            await cb.record_failure()
            logger.warning(
                "Router: failure recorded for '%s' (CB failures: %d)",
                model_id,
                cb._failure_count,
            )

        if headers:
            rate_limit_parser.update_from_headers(model_id, headers)

    def get_status(self) -> dict[str, object]:
        """Return combined CB + RL status for all known models (dashboard use)."""
        cb_statuses = circuit_breaker_registry.all_statuses()
        rl_statuses = rate_limit_parser.all_statuses()

        all_ids = set(cb_statuses.keys()) | set(rl_statuses.keys())
        result: dict[str, object] = {}
        for mid in sorted(all_ids):
            result[mid] = {
                "circuit_breaker": cb_statuses.get(
                    mid, {"state": "closed", "failure_count": 0}
                ),
                "rate_limit": rl_statuses.get(mid, {"has_headroom": True}),
            }
        return result


# Module-level singleton
model_router = ModelRouter()
