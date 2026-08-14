"""CLI Session Management System in CLI layer."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class CLISession:
    """Represents an active Claude Code CLI / VSCode client session."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    subagent_depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update last active timestamp and increment turn count."""
        self.last_active = time.time()
        self.turn_count += 1

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token consumption for this session."""
        self.total_input_tokens += max(0, input_tokens)
        self.total_output_tokens += max(0, output_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Convert session object to dictionary representation."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "turn_count": self.turn_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "subagent_depth": self.subagent_depth,
            "metadata": self.metadata,
        }


class CLISessionManager:
    """Manager for active CLI sessions."""

    def __init__(self, default_ttl: float = 86400.0):
        self._sessions: dict[str, CLISession] = {}
        self.default_ttl = default_ttl

    def get_or_create_session(
        self, session_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> CLISession:
        """Retrieve existing session or instantiate a new one."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        if sid not in self._sessions:
            session = CLISession(session_id=sid, metadata=metadata or {})
            self._sessions[sid] = session
            logger.info("Created new CLI Session: '{}'", sid)
        else:
            session = self._sessions[sid]
            session.touch()
        return session

    def get_session(self, session_id: str) -> CLISession | None:
        """Get session by ID if existing."""
        return self._sessions.get(session_id)

    def record_turn(
        self, session_id: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """Record completed turn and tokens for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.touch()
            session.record_tokens(input_tokens, output_tokens)

    def active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._sessions)

    def cleanup_stale_sessions(self, max_idle_seconds: float | None = None) -> int:
        """Purge sessions that have exceeded idle timeout."""
        ttl = max_idle_seconds or self.default_ttl
        now = time.time()
        stale_keys = [sid for sid, s in self._sessions.items() if (now - s.last_active) > ttl]
        for sid in stale_keys:
            del self._sessions[sid]
        if stale_keys:
            logger.info("Cleaned up {} stale CLI sessions", len(stale_keys))
        return len(stale_keys)


session_manager = CLISessionManager()
