"""Subagent Guard — enforcing foreground subagent execution and policy isolation."""

from typing import Any

from loguru import logger

from atomic.parsers.base import BaseAtomicParser
from shared.schemas.anthropic import SSEBaseEvent


class SubagentGuard(BaseAtomicParser):
    """Guard enforcing run_in_background=False for task/subagent tool requests in CLI sessions."""

    TASK_TOOL_NAMES = {"Task", "subagent", "TaskManager", "TaskWorker", "agent_task", "run_task"}

    def __init__(self):
        self.enforcements_count = 0

    def reset(self) -> None:
        self.enforcements_count = 0

    async def enforce_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Inspect tool_input and enforce run_in_background=False policy."""
        if (tool_name in self.TASK_TOOL_NAMES or "task" in tool_name.lower()) and tool_input.get("run_in_background") is not False:
            logger.warning("SubagentGuard: Enforcing run_in_background=False on subagent tool '{}'", tool_name)
            tool_input["run_in_background"] = False
            self.enforcements_count += 1

        return tool_input

    async def process_chunk(self, chunk: dict[str, Any] | str) -> list[SSEBaseEvent]:
        return []

    async def flush(self) -> list[SSEBaseEvent]:
        return []
