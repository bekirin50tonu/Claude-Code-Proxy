"""Async In-Memory Queue Manager for Remote Prompt Interception & Payload Injection."""

import asyncio
from typing import Any
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
        while not q.empty():
            try:
                prompts.append(q.get_nowait())
                q.task_done()
            except asyncio.QueueEmpty:
                break
        if prompts:
            logger.info(f"PromptQueueManager: Extracted {len(prompts)} pending prompt(s) for '{sid}'")
        return prompts

    def has_pending(self, session_id: str | None = None) -> bool:
        """Check if session has pending prompts queued."""
        sid = session_id or "default_session"
        return not self._get_queue(sid).empty()


# Singleton instance
prompt_queue_manager = PromptQueueManager()
