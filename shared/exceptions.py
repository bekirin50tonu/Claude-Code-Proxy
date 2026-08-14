"""Custom exception hierarchy for the Claude Code Proxy Gateway."""

from typing import Any


class ProxyBaseError(Exception):
    """Base exception class for all Proxy Gateway errors."""

    def __init__(self, message: str, status_code: int = 500, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class UpstreamProviderError(ProxyBaseError):
    """Raised when an upstream LLM provider (NIM, OpenRouter, etc.) returns an HTTP 4xx or 5xx error."""

    def __init__(self, provider: str, status_code: int, message: str, raw_response: str | None = None):
        super().__init__(
            message=f"Upstream provider [{provider}] error ({status_code}): {message}",
            status_code=status_code,
            details={"provider": provider, "raw_response": raw_response},
        )
        self.provider = provider


class CircuitOpenError(ProxyBaseError):
    """Raised when an upstream model circuit breaker is in the OPEN state."""

    def __init__(self, model_name: str, recovery_time_remaining: float = 0.0):
        super().__init__(
            message=f"Circuit breaker for model '{model_name}' is OPEN. Retry in {recovery_time_remaining:.1f}s",
            status_code=503,
            details={"model_name": model_name, "recovery_time_remaining": recovery_time_remaining},
        )
        self.model_name = model_name
        self.recovery_time_remaining = recovery_time_remaining


class RateLimitExceededError(ProxyBaseError):
    """Raised when an upstream or local rate limit is exceeded."""

    def __init__(self, model_name: str, retry_after: float = 0.0):
        super().__init__(
            message=f"Rate limit exceeded for model '{model_name}'. Retry after {retry_after:.1f}s",
            status_code=429,
            details={"model_name": model_name, "retry_after": retry_after},
        )
        self.model_name = model_name
        self.retry_after = retry_after


class ContextOverflowError(ProxyBaseError):
    """Raised when request tokens exceed the maximum context limit of the target model."""

    def __init__(self, model_name: str, total_tokens: int, max_tokens: int):
        super().__init__(
            message=f"Context overflow for model '{model_name}': {total_tokens} tokens exceeds limit of {max_tokens}",
            status_code=400,
            details={"model_name": model_name, "total_tokens": total_tokens, "max_tokens": max_tokens},
        )
        self.model_name = model_name
        self.total_tokens = total_tokens
        self.max_tokens = max_tokens


class SubagentPolicyViolationError(ProxyBaseError):
    """Raised when a subagent call violates proxy safety or recursion policies."""

    def __init__(self, subagent_id: str, reason: str):
        super().__init__(
            message=f"Subagent policy violation for '{subagent_id}': {reason}",
            status_code=403,
            details={"subagent_id": subagent_id, "reason": reason},
        )
        self.subagent_id = subagent_id
        self.reason = reason
