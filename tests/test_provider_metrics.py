"""Unit tests for Per-Provider RPM & TPM Metrics Telemetry."""

import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_get_dev_metrics_provider_metrics(client: TestClient) -> None:
    response = client.get("/api/dev/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "global_metrics" in data
    assert "nim_telemetry" in data
    assert "provider_metrics" in data
    assert "model_metrics" in data

    provider_metrics = data["provider_metrics"]
    assert isinstance(provider_metrics, list)
    assert len(provider_metrics) >= 12

    providers_found = {p["provider"] for p in provider_metrics}
    expected_providers = {
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
    }
    assert expected_providers.issubset(providers_found)

    for pm in provider_metrics:
        assert "rpm_60s" in pm
        assert "rpm_limit" in pm
        assert "rpm_usage_pct" in pm
        assert "tpm_60s" in pm
        assert "tpm_limit" in pm
        assert "tpm_usage_pct" in pm
        assert "rpd_count" in pm
        assert "rpd_limit" in pm
        assert "http_read_timeout" in pm
        assert "max_concurrency" in pm
        assert "status" in pm
