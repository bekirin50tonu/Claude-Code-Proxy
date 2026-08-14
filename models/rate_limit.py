"""
Rate Limit Data Models.

Defines RateLimitStatus dataclass for tracking RPM and TPM headers from upstreams.
"""

from dataclasses import dataclass


@dataclass
class RateLimitStatus:
    """Snapshot representation of an upstream model's Rate Limit state."""

    model_id: str
    req_limit: int | None = None
    req_remaining: int | None = None
    tok_limit: int | None = None
    tok_remaining: int | None = None
    has_headroom: bool = True
    last_updated_ago_s: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "req_limit": self.req_limit,
            "req_remaining": self.req_remaining,
            "tok_limit": self.tok_limit,
            "tok_remaining": self.tok_remaining,
            "has_headroom": self.has_headroom,
            "last_updated_ago_s": self.last_updated_ago_s,
        }
