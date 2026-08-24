"""Async In-Memory Queue Manager for Remote Prompt Interception & Payload Injection."""

import asyncio

from loguru import logger


class PromptQueueManager:
    """Session-based thread-safe async queue manager for pending remote prompts."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[str]] = {}
        self._lock = asyncio.Lock()

    def _get_queue(self, session_id: str) -> asyncio.Queue[str]:
        sid = session_id or "default_session"
        if sid not in self._queues:
            self._queues[sid] = asyncio.Queue()
        return self._queues[sid]

    async def push_prompt(self, prompt: str, session_id: str | None = None) -> int:
        """Enqueue a new remote prompt for a session."""
        sid = session_id or "default_session"
        q = self._get_queue(sid)
        await q.put(prompt.strip())
        size = q.qsize()
        logger.info(f"PromptQueueManager: Enqueued prompt for '{sid}' (Queue size: {size})")
        return size

    def pop_all_prompts(self, session_id: str | None = None) -> list[str]:
        """Drain and return all pending prompts for a session without blocking."""
        sid = session_id or "default_session"
        q = self._get_queue(sid)
        prompts: list[str] = []
        while True:
            try:
                prompts.append(q.get_nowait())
                q.task_done()
            except asyncio.QueueEmpty:
                break
        if prompts:
            logger.info(f"PromptQueueManager: Extracted {len(prompts)} pending prompt(s) for '{sid}'")
        return prompts

    def peek_prompts(self, session_id: str | None = None) -> list[str]:
        """Inspect pending prompts without removing them from queue."""
        sid = session_id or "default_session"
        q = self._get_queue(sid)
        return list(q._queue)  # type: ignore[attr-defined]

    async def inject_prompt_at(self, prompt: str, index: int = 0, session_id: str | None = None) -> int:
        """Inject prompt at a specific index in the session queue."""
        sid = session_id or "default_session"
        q = self._get_queue(sid)
        items = list(q._queue)  # type: ignore[attr-defined]
        idx = max(0, min(index, len(items)))
        items.insert(idx, prompt.strip())
        self._queues[sid] = asyncio.Queue()
        for item in items:
            await self._queues[sid].put(item)
        return self._queues[sid].qsize()

    async def replace_prompt(self, index: int, new_prompt: str, session_id: str | None = None) -> bool:
        """Replace prompt content at a specific index in the session queue."""
        sid = session_id or "default_session"
        q = self._get_queue(sid)
        items = list(q._queue)  # type: ignore[attr-defined]
        if 0 <= index < len(items):
            items[index] = new_prompt.strip()
            self._queues[sid] = asyncio.Queue()
            for item in items:
                await self._queues[sid].put(item)
            return True
        return False

    def clear_queue(self, session_id: str | None = None) -> int:
        """Clear all pending prompts for a session."""
        sid = session_id or "default_session"
        q = self._get_queue(sid)
        count = q.qsize()
        self._queues[sid] = asyncio.Queue()
        logger.info(f"PromptQueueManager: Cleared {count} prompt(s) for session '{sid}'")
        return count

    def has_pending(self, session_id: str | None = None) -> bool:
        """Check if session has pending prompts queued."""
        sid = session_id or "default_session"
        return not self._get_queue(sid).empty()


# Singleton instance
prompt_queue_manager = PromptQueueManager()
