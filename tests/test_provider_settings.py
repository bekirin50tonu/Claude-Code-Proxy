"""Unit tests for Per-Provider Throttle, Timeout, and Token Budget Matrix settings."""

from fastapi.testclient import TestClient

from atomic.guards.token_budget import TokenBudgetGuard
from config import PROVIDER_DEFAULTS, settings
from server import app

client = TestClient(app)


def test_provider_defaults_integrity() -> None:
    """Verify all 12 providers have complete 9-field default configurations."""
    expected_providers = [
        "nvidia_nim",
        "open_router",
        "gemini",
        "groq",
        "deepseek",
        "mistral",
        "cerebras",
        "fireworks",
        "kimi",
        "lmstudio",
        "ollama",
        "llama_cpp",
    ]
    expected_fields = [
        "rpm",
        "tpm",
        "rate_window",
        "max_concurrency",
        "context",
        "max_output",
        "http_read_timeout",
        "http_write_timeout",
        "http_connect_timeout",
    ]

    for p in expected_providers:
        assert p in PROVIDER_DEFAULTS, f"Provider {p} missing from PROVIDER_DEFAULTS"
        p_cfg = PROVIDER_DEFAULTS[p]
        for field in expected_fields:
            assert field in p_cfg, f"Field {field} missing from provider {p}"


def test_settings_get_provider_config() -> None:
    """Test retrieving per-provider settings dynamically."""
    gemini_cfg = settings.get_provider_config("gemini")
    assert gemini_cfg["max_output"] == 8192
    assert gemini_cfg["context"] == 1000000

    nim_cfg = settings.get_provider_config("nvidia_nim")
    assert nim_cfg["rpm"] == settings.NVIDIA_NIM_SAFE_RPM
    assert nim_cfg["context"] == 1000000


def test_token_budget_guard_provider_override() -> None:
    """Verify TokenBudgetGuard applies provider-specific context and max_output limits."""
    guard = TokenBudgetGuard("gemini/models/gemini-3.7-flash")
    assert guard.metadata.context == 1000000
    assert guard.metadata.max_output == 8192


def test_api_config_returns_provider_keys() -> None:
    """Verify GET /api/config includes provider settings keys in config_data."""
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    configs = data["configs"]
    assert "PROVIDER_GEMINI_RPM" in configs
    assert "PROVIDER_NVIDIA_NIM_MAX_OUTPUT" in configs
    assert "PROVIDER_DEEPSEEK_HTTP_READ_TIMEOUT" in configs
