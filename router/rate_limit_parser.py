"""
Rate Limit Parser — reads upstream x-ratelimit-* HTTP response headers
and maintains per-model rate limit state.

Supported header formats (NVIDIA NIM & OpenRouter both use this schema):
  x-ratelimit-limit-requests      → rpm_limit (requests per minute)
  x-ratelimit-remaining-requests  → remaining requests in current window
  x-ratelimit-limit-tokens        → tpm_limit (tokens per minute)
  x-ratelimit-remaining-tokens    → remaining tokens in current window
  x-ratelimit-reset-requests      → ISO-8601 or seconds until reset
  x-ratelimit-reset-tokens        → ISO-8601 or seconds until reset

Headroom policy:
  A model is considered to have headroom when remaining ≥ 10% of limit.
  If the header is absent (provider doesn't send it), headroom = True.
"""

import time


class RateLimitState:
    # Fraction of capacity that must remain for "headroom" to be True
    HEADROOM_FRACTION = 0.10

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

        self.req_limit: int | None = None
        self.req_remaining: int | None = None
        self.req_reset_at: float | None = None  # monotonic time

        self.tok_limit: int | None = None
        self.tok_remaining: int | None = None
        self.tok_reset_at: float | None = None

        self._last_updated: float = 0.0

    def update(self, headers: dict[str, str]) -> None:
        """Parse rate-limit headers and update state."""
        now = time.monotonic()
        self._last_updated = now

        def _int(key: str) -> int | None:
            val = headers.get(key) or headers.get(key.lower())
            if val is None:
                return None
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return None

        def _reset_monotonic(key: str) -> float | None:
            val = headers.get(key) or headers.get(key.lower())
            if val is None:
                return None
            try:
                # Value can be seconds (e.g. "60") or ms (e.g. "60000")
                secs = float(val)
                if secs > 1_000_000:  # looks like milliseconds
                    secs = secs / 1000.0
                return now + secs
            except (ValueError, TypeError):
                return None

        req_limit = _int("x-ratelimit-limit-requests")
        req_remaining = _int("x-ratelimit-remaining-requests")
        tok_limit = _int("x-ratelimit-limit-tokens")
        tok_remaining = _int("x-ratelimit-remaining-tokens")
        req_reset = _reset_monotonic("x-ratelimit-reset-requests")
        tok_reset = _reset_monotonic("x-ratelimit-reset-tokens")

        if req_limit is not None:
            self.req_limit = req_limit
        if req_remaining is not None:
            self.req_remaining = req_remaining
        if tok_limit is not None:
            self.tok_limit = tok_limit
        if tok_remaining is not None:
            self.tok_remaining = tok_remaining
        if req_reset is not None:
            self.req_reset_at = req_reset
        if tok_reset is not None:
            self.tok_reset_at = tok_reset

    def has_headroom(self) -> bool:
        """Return True if the model has sufficient remaining quota."""
        now = time.monotonic()

        # If reset time has passed, assume quota is restored
        if self.req_reset_at and now >= self.req_reset_at:
            self.req_remaining = self.req_limit
            self.req_reset_at = None
        if self.tok_reset_at and now >= self.tok_reset_at:
            self.tok_remaining = self.tok_limit
            self.tok_reset_at = None

        req_low = (
            self.req_limit is not None
            and self.req_remaining is not None
            and self.req_remaining
            < max(1, int(self.req_limit * self.HEADROOM_FRACTION))
        )
        tok_low = (
            self.tok_limit is not None
            and self.tok_remaining is not None
            and self.tok_remaining
            < max(100, int(self.tok_limit * self.HEADROOM_FRACTION))
        )
        return not req_low and not tok_low

    def status_dict(self) -> dict[str, object]:
        return {
            "req_limit": self.req_limit,
            "req_remaining": self.req_remaining,
            "tok_limit": self.tok_limit,
            "tok_remaining": self.tok_remaining,
            "has_headroom": self.has_headroom(),
            "last_updated_ago_s": round(time.monotonic() - self._last_updated, 1)
            if self._last_updated
            else None,
        }


class RateLimitParser:
    """Registry of per-model RateLimitState instances."""

    def __init__(self) -> None:
        self._states: dict[str, RateLimitState] = {}

    def get(self, model_id: str) -> RateLimitState:
        if model_id not in self._states:
            self._states[model_id] = RateLimitState(model_id)
        return self._states[model_id]

    def update_from_headers(self, model_id: str, headers: dict[str, str]) -> None:
        self.get(model_id).update(headers)

    def has_headroom(self, model_id: str) -> bool:
        return self.get(model_id).has_headroom()

    def all_statuses(self) -> dict[str, dict[str, object]]:
        return {mid: state.status_dict() for mid, state in self._states.items()}


# Module-level singleton
rate_limit_parser = RateLimitParser()
