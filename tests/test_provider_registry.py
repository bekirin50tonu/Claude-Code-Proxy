"""Unit tests for Provider Registry, UniversalKeyManager, and UnknownProvider handling."""

import pytest

from config.providers_registry import (
    PROVIDER_REGISTRY,
    ProviderSpec,
    UnknownProviderError,
)
from core.key_manager import universal_key_manager
from providers.openai import OpenAICompatibleProvider


def test_provider_registry_contains_default_providers() -> None:
    """Verify registry contains expected core providers."""
    assert "nvidia_nim" in PROVIDER_REGISTRY
    assert "open_router" in PROVIDER_REGISTRY
    assert "gemini" in PROVIDER_REGISTRY
    assert "groq" in PROVIDER_REGISTRY
    assert "lmstudio" in PROVIDER_REGISTRY

    spec = PROVIDER_REGISTRY["open_router"]
    assert isinstance(spec, ProviderSpec)
    assert spec.api_key_attr == "OPENROUTER_API_KEY"
    assert "HTTP-Referer" in spec.extra_headers


@pytest.mark.asyncio
async def test_universal_key_manager_rotation() -> None:
    """Verify UniversalKeyManager rotates round-robin through comma-separated key pool."""
    universal_key_manager.reset()
    keys = "key-1, key-2, key-3"

    k1 = await universal_key_manager.select_key(keys, "test_provider")
    k2 = await universal_key_manager.select_key(keys, "test_provider")
    k3 = await universal_key_manager.select_key(keys, "test_provider")
    k4 = await universal_key_manager.select_key(keys, "test_provider")

    assert k1 == "key-1"
    assert k2 == "key-2"
    assert k3 == "key-3"
    assert k4 == "key-1"


@pytest.mark.asyncio
async def test_unknown_provider_raises_exception() -> None:
    """Verify resolving unknown provider in strict mode raises UnknownProviderError."""
    provider = OpenAICompatibleProvider()
    with pytest.raises(UnknownProviderError) as exc_info:
        await provider._resolve_endpoint("invalid_provider/some-model")

    assert "invalid_provider" in str(exc_info.value)
    assert "nvidia_nim" in str(exc_info.value)
