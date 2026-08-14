"""
Routing & Model Configuration Data Models.

Defines ClientModelMapping and UpstreamModelConfig dataclasses for model routing logic.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientModelMapping:
    """Represents a client request model mapping and its fallback resolution chain."""

    client_model: str
    label: str
    resolved_target: str
    is_fallback: bool
    step_name: str
    chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_model": self.client_model,
            "label": self.label,
            "resolved_target": self.resolved_target,
            "is_fallback": self.is_fallback,
            "step_name": self.step_name,
            "chain": self.chain,
        }


@dataclass
class UpstreamModelConfig:
    """Represents configuration parameters for an upstream AI model target."""

    model_id: str
    provider: str  # e.g. "nvidia_nim", "openrouter", "deepseek"
    max_context_tokens: int = 200_000
    max_output_tokens: int = 4096
    supports_vision: bool = False
    supports_tools: bool = True
