"""
Telemetry Data Models.

Defines structured dataclasses for request/response logging,
performance metrics, and system-wide traffic telemetry.
"""

import time
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
    input_tokens: int = 0
    output_tokens: int = 0
    error_details: dict[str, Any] | None = None
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

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
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "error_details": self.error_details,
            "attempt_history": self.attempt_history,
        }
        if include_payload:
            data["request_body"] = self.request_body
            data["response_body"] = self.response_body
            data["headers"] = self.headers
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequestLogEntry":
        """Deserialize transaction record from dictionary format."""
        return cls(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", ""),
            method=data.get("method", "POST"),
            path=data.get("path", "/v1/messages"),
            client_model=data.get("client_model", "unknown"),
            mapped_model=data.get("mapped_model", "unknown"),
            status_code=int(data.get("status_code", 200)),
            duration_ms=float(data.get("duration_ms", 0.0)),
            mocked=bool(data.get("mocked", False)),
            fallbacks_used=data.get("fallbacks_used") or [],
            request_body=data.get("request_body"),
            response_body=data.get("response_body"),
            headers=data.get("headers") or {},
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            error_details=data.get("error_details"),
            attempt_history=data.get("attempt_history") or [],
        )




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
