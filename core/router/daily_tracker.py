"""Daily Request Tracker (RPD) — tracks daily request budgets per provider in Core layer."""

from datetime import date

from loguru import logger

from config import settings


class DailyRequestTracker:
    """Track daily request counts per LLM provider and handle RPD circuit trips."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._current_date: str = date.today().isoformat()
        self._force_exceeded: set[str] = set()

    def _check_day_reset(self) -> None:
        today = date.today().isoformat()
        if today != self._current_date:
            self._counts.clear()
            self._force_exceeded.clear()
            self._current_date = today
            logger.info("DailyRequestTracker: Reset daily request counters for new day %s", today)

    def record_request(self, provider: str) -> int:
        """Increment daily request count for the given provider."""
        self._check_day_reset()
        p = provider.lower().strip()
        self._counts[p] = self._counts.get(p, 0) + 1
        return self._counts[p]

    def get_count(self, provider: str) -> int:
        """Get current daily request count for the given provider."""
        self._check_day_reset()
        p = provider.lower().strip()
        return self._counts.get(p, 0)

    def mark_exceeded(self, provider: str) -> None:
        """Force mark a provider as having exceeded its daily quota (e.g. on upstream 429)."""
        self._check_day_reset()
        p = provider.lower().strip()
        self._force_exceeded.add(p)

    def is_exceeded(self, provider: str) -> tuple[bool, int, int]:
        """Check if provider has exceeded its RPD limit. Return (is_exceeded, count, limit)."""
        self._check_day_reset()
        p = provider.lower().strip()
        p_cfg = settings.get_provider_config(p)
        rpd_limit = p_cfg.get("rpd", 100000)
        current_count = self._counts.get(p, 0)
        exceeded = (p in self._force_exceeded) or (rpd_limit > 0 and current_count >= rpd_limit)
        return exceeded, current_count, rpd_limit

    def reset_provider(self, provider: str) -> None:
        """Reset daily count and force_exceeded flag for a provider."""
        p = provider.lower().strip()
        self._counts[p] = 0
        self._force_exceeded.discard(p)

    def all_statuses(self) -> dict[str, dict[str, object]]:
        """Return daily RPD tracking statuses for all providers."""
        self._check_day_reset()
        from config import PROVIDER_DEFAULTS
        res: dict[str, dict[str, object]] = {}
        for p in PROVIDER_DEFAULTS:
            exceeded, count, limit = self.is_exceeded(p)
            res[p] = {
                "count": count,
                "rpd_limit": limit,
                "exceeded": exceeded,
                "date": self._current_date,
            }
        return res


daily_request_tracker = DailyRequestTracker()
