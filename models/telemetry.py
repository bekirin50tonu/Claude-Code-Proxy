"""
Telemetry Data Models.

Defines structured dataclasses for request/response logging,
performance metrics, and system-wide traffic telemetry.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestLogEntry:
    """Represents a single HTTP transaction log entry processed by the proxy."""

    id: str
    timestamp: str
    method: str
    path: str
    client_model: str
    mapped_model: str
    status_code: int
    duration_ms: float
    mocked: bool = False
    fallbacks_used: list[str] = field(default_factory=list)
    request_body: dict[str, Any] | None = None
    response_body: dict[str, Any] | str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    def to_dict(self, include_payload: bool = True) -> dict[str, Any]:
        """Serialize transaction record into dictionary format."""
        data: dict[str, Any] = {
            "id": self.id,
            "timestamp": self.timestamp,
            "method": self.method,
            "path": self.path,
            "client_model": self.client_model,
            "mapped_model": self.mapped_model,
            "status_code": self.status_code,
            "duration_ms": self.duration_ms,
            "mocked": self.mocked,
            "fallbacks_used": self.fallbacks_used,
        }
        if include_payload:
            data["request_body"] = self.request_body
            data["response_body"] = self.response_body
            data["headers"] = self.headers
        return data


@dataclass
class TelemetrySummary:
    """Summary of overall proxy traffic and bot connection states."""

    total_requests: int
    mocked_requests: int
    error_count: int
    active_concurrency: int
    tg_bot_status: str = "Offline"
    ds_bot_status: str = "Offline"

    @property
    def mock_ratio_percent(self) -> int:
        if self.total_requests == 0:
            return 0
        return round((self.mocked_requests / self.total_requests) * 100)
