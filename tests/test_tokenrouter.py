"""Unit tests for TokenRouter provider integration."""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from config import settings
from providers.openai import OpenAICompatibleProvider
from atomic.guards.preflight import _resolve_url_and_headers


def test_tokenrouter_settings_defaults():
    """Verify TokenRouter settings default configurations."""
    assert hasattr(settings, "TOKENROUTER_API_KEY")
    assert hasattr(settings, "TOKENROUTER_BASE_URL")
    assert settings.TOKENROUTER_BASE_URL == "https://api.tokenrouter.com/v1"


@pytest.mark.asyncio
async def test_tokenrouter_provider_resolution():
    """Verify OpenAICompatibleProvider resolves tokenrouter models to TOKENROUTER_BASE_URL."""
    provider = OpenAICompatibleProvider()
    with patch("providers.openai._select_key", new=AsyncMock(return_value="tr-secret-key")):
        base_url, model_name, api_key, headers = await provider._resolve_endpoint("tokenrouter/claude-3-5-sonnet")
        assert base_url == "https://api.tokenrouter.com/v1"
        assert model_name == "claude-3-5-sonnet"
        assert api_key == "tr-secret-key"


def test_tokenrouter_preflight_resolution():
    """Verify preflight probe resolves tokenrouter URL and authorization header."""
    url, headers = _resolve_url_and_headers("tokenrouter/claude-3-5-sonnet")
    assert url == "https://api.tokenrouter.com/v1/chat/completions"
    assert "Content-Type" in headers
