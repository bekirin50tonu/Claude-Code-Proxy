"""Unit tests for NimKeyManager round-robin rotation, passive cooldown tracking, and silent failover."""

import pytest
import time
from core.key_manager import NimKeyManager
from config import settings


@pytest.fixture
def key_manager():
    km = NimKeyManager(default_cooldown=10.0)
    km.reset()
    yield km
    km.reset()


@pytest.mark.asyncio
async def test_get_configured_keys(monkeypatch, key_manager):
    monkeypatch.setattr(settings, "NVIDIA_NIM_API_KEYS", "nvapi-key1, nvapi-key2, nvapi-key3")
    keys = key_manager.get_configured_keys()
    assert keys == ["nvapi-key1", "nvapi-key2", "nvapi-key3"]


@pytest.mark.asyncio
async def test_round_robin_rotation(monkeypatch, key_manager):
    monkeypatch.setattr(settings, "NVIDIA_NIM_API_KEYS", "nvapi-key1, nvapi-key2, nvapi-key3")

    k1 = await key_manager.get_next_key()
    k2 = await key_manager.get_next_key()
    k3 = await key_manager.get_next_key()
    k4 = await key_manager.get_next_key()

    assert k1 == "nvapi-key1"
    assert k2 == "nvapi-key2"
    assert k3 == "nvapi-key3"
    assert k4 == "nvapi-key1"


@pytest.mark.asyncio
async def test_passive_cooldown(monkeypatch, key_manager):
    monkeypatch.setattr(settings, "NVIDIA_NIM_API_KEYS", "nvapi-key1, nvapi-key2, nvapi-key3")

    # Mark key1 as passive
    await key_manager.mark_passive("nvapi-key1", cooldown_seconds=60.0)

    # Next keys should skip key1 and rotate between key2 and key3
    k1 = await key_manager.get_next_key()
    k2 = await key_manager.get_next_key()
    k3 = await key_manager.get_next_key()

    assert k1 == "nvapi-key2"
    assert k2 == "nvapi-key3"
    assert k3 == "nvapi-key2"


@pytest.mark.asyncio
async def test_all_keys_passive_fallback(monkeypatch, key_manager):
    monkeypatch.setattr(settings, "NVIDIA_NIM_API_KEYS", "nvapi-key1, nvapi-key2")

    # Mark both keys passive with different expiration times
    await key_manager.mark_passive("nvapi-key1", cooldown_seconds=10.0)
    await key_manager.mark_passive("nvapi-key2", cooldown_seconds=30.0)

    # Should pick key1 because its cooldown expires earlier
    next_key = await key_manager.get_next_key()
    assert next_key == "nvapi-key1"


@pytest.mark.asyncio
async def test_get_active_candidate_keys(monkeypatch, key_manager):
    monkeypatch.setattr(settings, "NVIDIA_NIM_API_KEYS", "nvapi-key1, nvapi-key2, nvapi-key3")

    # Mark key2 passive
    await key_manager.mark_passive("nvapi-key2", cooldown_seconds=60.0)

    candidates = await key_manager.get_active_candidate_keys()
    # Active keys (key1, key3) first, passive key2 last
    assert candidates[0] in ("nvapi-key1", "nvapi-key3")
    assert candidates[-1] == "nvapi-key2"
