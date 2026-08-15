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

    async def enforce_tool_call(self, tool_name: str, tool_input: dict[str, Any], enabled: bool | None = None) -> dict[str, Any]:
        """Inspect tool_input and enforce run_in_background=False policy when SUBAGENTS_ENABLED is False (OFF Bypass).
        
        If enabled is True (ON):
            Leave tool_input untouched so run_in_background=True reaches output stream intact.
        If enabled is False (OFF Bypass):
            Force run_in_background=False to bypass background processes.
        """
        if enabled is None:
            try:
                from api.dashboard import SUBAGENTS_ENABLED
                enabled = SUBAGENTS_ENABLED
            except Exception:
                enabled = True

        if not enabled:
            if (tool_name in self.TASK_TOOL_NAMES or "task" in tool_name.lower()) and tool_input.get("run_in_background") is not False:
                logger.warning("SubagentGuard (OFF Bypass): Enforcing run_in_background=False on subagent tool '{}'", tool_name)
                tool_input["run_in_background"] = False
                self.enforcements_count += 1

        return tool_input

    def sanitize_payload(self, body: dict[str, Any], enabled: bool) -> dict[str, Any]:
        """Inspect incoming /v1/messages request payload.

        If enabled is True (ON):
            Leave payload untouched so run_in_background=True reaches upstream.
        If enabled is False (OFF Bypass):
            Force run_in_background=False in messages, tool calls, and payload fields to prevent background subagents.
        """
        if enabled:
            # Switch is ON: Do not touch request body
            return body

        # Switch is OFF: Enforce foreground execution / disable background subagents
        if "run_in_background" in body:
            body["run_in_background"] = False

        messages = body.get("messages", [])
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_input = block.get("input")
                            tool_name = block.get("name", "")
                            if isinstance(tool_input, dict):
                                if "run_in_background" in tool_input or tool_name in self.TASK_TOOL_NAMES or "task" in tool_name.lower():
                                    if tool_input.get("run_in_background") is not False:
                                        logger.warning("SubagentGuard (OFF Bypass): Overriding run_in_background=False for tool '{}'", tool_name)
                                        tool_input["run_in_background"] = False
                                        self.enforcements_count += 1

        return body

    async def process_chunk(self, chunk: dict[str, Any] | str) -> list[SSEBaseEvent]:
        return []

    async def flush(self) -> list[SSEBaseEvent]:
        return []


# Singleton instance
subagent_guard = SubagentGuard()

