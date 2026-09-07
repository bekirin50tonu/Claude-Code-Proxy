"""Thread-Safe Asynchronous API Key Pool and Rotation Manager for NVIDIA NIM."""

import asyncio
import time

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

        # Deduplicate keys while preserving configured order
        seen = set()
        keys = []
        for k in raw_keys.split(","):
            cleaned = k.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                keys.append(cleaned)
        return keys

    async def get_next_key(self) -> str:
        """Asynchronously return the next active API key based on strategy ('first' or 'round_robin')."""
        keys = self.get_configured_keys()
        if not keys:
            return ""
        if len(keys) == 1:
            return keys[0]

        now = time.time()
        strategy = getattr(settings, "NVIDIA_NIM_KEY_STRATEGY", "round_robin").strip().lower()

        async with self._lock:
            active_keys = [k for k in keys if self._passive_until.get(k, 0.0) <= now]

            if not active_keys:
                logger.warning("NimKeyManager: All NVIDIA NIM keys are in passive cooldown. Re-using earliest expiring key.")
                sorted_by_cooldown = sorted(keys, key=lambda k: self._passive_until.get(k, 0.0))
                return sorted_by_cooldown[0]

            if strategy == "first":
                return active_keys[0]

            idx = self._counter % len(active_keys)
            self._counter += 1
            return active_keys[idx]

    async def mark_passive(self, key: str, cooldown_seconds: float | None = None) -> None:
        """Temporarily passivate a key for cooldown_seconds (default 20s) due to 429/401/timeout."""
        if not key:
            return
        configured_default = getattr(settings, "NVIDIA_NIM_KEY_COOLDOWN_SECONDS", 20.0)
        cooldown = cooldown_seconds if cooldown_seconds is not None else configured_default
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
        """Return an ordered candidate keys list according to configured strategy ('first' or 'round_robin')."""
        keys = self.get_configured_keys()
        if not keys:
            return []
        now = time.time()
        strategy = getattr(settings, "NVIDIA_NIM_KEY_STRATEGY", "round_robin").strip().lower()

        async with self._lock:
            if strategy == "first":
                ordered = list(keys)
            else:
                start_idx = self._counter % len(keys)
                ordered = keys[start_idx:] + keys[:start_idx]
                self._counter += 1

        active = [k for k in ordered if self._passive_until.get(k, 0.0) <= now]
        passive = [k for k in ordered if self._passive_until.get(k, 0.0) > now]

        return active + passive

    def reset(self) -> None:
        """Reset internal state (useful for tests)."""
        self._counter = 0
        self._passive_until.clear()


class UniversalKeyManager:
    """Thread-safe asynchronous multi-provider key pool manager.

    Manages key rotation and passive cooldowns across all LLM providers,
    replacing ad-hoc module-level global dictionary counters.
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._passive_until: dict[str, float] = {}

    async def select_key(self, raw_key: str, provider_name: str) -> str:
        """Thread-safely select a single API key or rotate through comma-separated keys."""
        if not raw_key:
            return ""
        if "," not in raw_key:
            return raw_key.strip()

        keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        if not keys:
            return ""
        if len(keys) == 1:
            return keys[0]

        now = time.time()
        async with self._lock:
            active_keys = [k for k in keys if self._passive_until.get(f"{provider_name}:{k}", 0.0) <= now]
            if not active_keys:
                active_keys = keys

            idx = self._counters.get(provider_name, 0) % len(active_keys)
            self._counters[provider_name] = (idx + 1) % len(active_keys)
            return active_keys[idx]

    async def mark_passive(self, provider_name: str, key: str, cooldown_seconds: float = 20.0) -> None:
        """Temporarily mark a key for cooldown due to 429/401/timeout."""
        if not key:
            return
        until = time.time() + cooldown_seconds
        async with self._lock:
            self._passive_until[f"{provider_name}:{key}"] = until
        masked = key[:7] + "..." + key[-4:] if len(key) > 12 else key
        logger.warning(
            "UniversalKeyManager [{}]: Key '{}' passivated for {:.0f}s due to upstream error.",
            provider_name,
            masked,
            cooldown_seconds,
        )

    def reset(self) -> None:
        """Reset internal state (useful for testing)."""
        self._counters.clear()
        self._passive_until.clear()


# Singleton instances
nim_key_manager = NimKeyManager()
universal_key_manager = UniversalKeyManager()

