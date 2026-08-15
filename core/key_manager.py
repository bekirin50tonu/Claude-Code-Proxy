"""Thread-Safe Asynchronous API Key Pool and Rotation Manager for NVIDIA NIM."""

import asyncio
import time
from typing import Any
from loguru import logger

from config import settings


class NimKeyManager:
    """Thread-safe asynchronous key manager for NVIDIA NIM API Key pool.

    Features:
    1. Multi API Key Parsing: Supports comma-separated keys from NVIDIA_NIM_API_KEYS or NVIDIA_NIM_API_KEY.
    2. Round-Robin Rotation: Thread-safe selection of keys using asyncio.Lock.
    3. Passive Cooldown Tracking: Temporarily marks failed keys (429/401/timeout) as passive.
    4. Silent Failover: Provides active keys list for retry attempts up to key pool capacity.
    """

    def __init__(self, default_cooldown: float = 60.0) -> None:
        self.default_cooldown = default_cooldown
        self._counter: int = 0
        self._lock = asyncio.Lock()
        self._passive_until: dict[str, float] = {}

    def get_configured_keys(self) -> list[str]:
        """Extract and clean non-empty keys from Settings (NVIDIA_NIM_API_KEYS or fallback NVIDIA_NIM_API_KEY)."""
        raw_keys = getattr(settings, "NVIDIA_NIM_API_KEYS", "") or ""
        if not raw_keys.strip():
            raw_keys = getattr(settings, "NVIDIA_NIM_API_KEY", "") or ""

        if not raw_keys.strip():
            return []

        keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        return keys

    async def get_next_key(self) -> str:
        """Asynchronously return the next active API key using round-robin rotation.

        If all keys are currently in passive cooldown, returns the key whose cooldown expires earliest.
        """
        keys = self.get_configured_keys()
        if not keys:
            return ""
        if len(keys) == 1:
            return keys[0]

        now = time.time()

        async with self._lock:
            # Filter active keys
            active_keys = [k for k in keys if self._passive_until.get(k, 0.0) <= now]

            if not active_keys:
                # All keys in cooldown — log warning and pick the key with earliest expiration
                logger.warning("NimKeyManager: All NVIDIA NIM keys are in passive cooldown. Re-using earliest expiring key.")
                sorted_by_cooldown = sorted(keys, key=lambda k: self._passive_until.get(k, 0.0))
                best_key = sorted_by_cooldown[0]
                return best_key

            idx = self._counter % len(active_keys)
            self._counter += 1
            selected_key = active_keys[idx]
            return selected_key

    async def mark_passive(self, key: str, cooldown_seconds: float | None = None) -> None:
        """Temporarily passivate a key for cooldown_seconds (default 60s) due to 429/401/timeout."""
        if not key:
            return
        cooldown = cooldown_seconds if cooldown_seconds is not None else self.default_cooldown
        until = time.time() + cooldown
        async with self._lock:
            self._passive_until[key] = until

        masked_key = key[:7] + "..." + key[-4:] if len(key) > 12 else key
        logger.warning(
            "NimKeyManager: Key '{}' passivated for {:.0f}s due to upstream failure (429/401/timeout).",
            masked_key,
            cooldown,
        )

    async def get_active_candidate_keys(self) -> list[str]:
        """Return an ordered list of keys starting from the current round-robin index for failover retries."""
        keys = self.get_configured_keys()
        if not keys:
            return []
        now = time.time()
        async with self._lock:
            # Reorder keys based on current counter position
            start_idx = self._counter % len(keys)
            rotated = keys[start_idx:] + keys[:start_idx]
            self._counter += 1

        # Separate non-cooldowned vs cooldowned
        active = [k for k in rotated if self._passive_until.get(k, 0.0) <= now]
        passive = [k for k in rotated if self._passive_until.get(k, 0.0) > now]

        # Return active keys first, followed by passive keys as last resort
        return active + passive

    def reset(self) -> None:
        """Reset internal state (useful for tests)."""
        self._counter = 0
        self._passive_until.clear()


# Singleton instance
nim_key_manager = NimKeyManager()
