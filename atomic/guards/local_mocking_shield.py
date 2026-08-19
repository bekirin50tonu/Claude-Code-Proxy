"""Local Mocking & Housekeeping Bypass Shield.

Intercepts background CLI housekeeping requests (suggestions, autocomplete,
title generation, ping tests) before hitting upstream models, returning
in-memory mock responses with valid Pydantic schemas.
"""

from typing import Any

from loguru import logger


class LocalMockingShield:
    """Shield to intercept and bypass CLI housekeeping requests locally."""

    HOUSEKEEPING_KEYWORDS: set[str] = {
        # Suggestions / Autocomplete
        "suggestion mode",
        "suggest the next command",
        "autocomplete suggestion",
        "predict next input",
        "suggestion for the next prompt",
        "autocomplete",
        "suggestions",
        # Title Generation
        "generate a short, 2-4 word title",
        "create a title for this conversation",
        "summarize this conversation into a title",
        "provide a short title",
        "title_generation",
        # Network / Ping / Discovery Probes
        "network-probe",
        "network probe",
        "ping check",
        "test connection",
        "ping",
        "discovery",
    }

    @classmethod
    def get_all_text(cls, payload: dict[str, Any]) -> str:
        """Extract and aggregate all system and message text from payload."""
        parts: list[str] = []
        system = payload.get("system")
        if system:
            if isinstance(system, list):
                parts.extend(
                    [str(b.get("text", "")) for b in system if isinstance(b, dict) and b.get("type") == "text"]
                )
            else:
                parts.append(str(system))

        messages = payload.get("messages", [])
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                parts.append(str(block.get("text", "")))
        return "\n".join(parts)

    @classmethod
    def is_housekeeping_request(cls, payload: dict[str, Any]) -> tuple[bool, str]:
        """
        Check if request is a housekeeping request (autocomplete, title, ping, discovery).
        Returns (is_housekeeping, reason_kind).
        """
        if not isinstance(payload, dict):
            return False, ""

        all_text = cls.get_all_text(payload).lower()

        # Stop Hook evaluation & schema requests from Claude CLI MUST NOT be intercepted!
        if "evaluating a stop-condition hook" in all_text or "hook_event_name" in all_text or "output_config" in payload:
            return False, ""

        max_tokens = payload.get("max_tokens", 4096)

        # 1. Check max_tokens <= 2 (Network probe / Ping)
        if isinstance(max_tokens, int) and max_tokens <= 2:
            return True, "ping_probe"

        # 2. Check explicitly matching keywords
        for kw in cls.HOUSEKEEPING_KEYWORDS:
            if kw in all_text:
                if kw in ("network-probe", "network probe", "ping check", "test connection", "ping", "discovery"):
                    return True, "ping_probe"
                elif kw in ("title_generation", "generate a short, 2-4 word title", "create a title for this conversation", "summarize this conversation into a title", "provide a short title"):
                    return True, "title_generation"
                else:
                    return True, "autocomplete_suggestion"

        return False, ""

    @classmethod
    def generate_mock_response(cls, payload: dict[str, Any], kind: str = "housekeeping") -> dict[str, Any]:
        """
        Generate Pydantic-compliant mock Anthropic Message response dict.
        Contains ' ' single space to satisfy min_length=1 text validation.
        """
        client_model = payload.get("model", "claude-3-5-sonnet-latest")
        logger.info(
            f"🛡️ \033[1;32m[LocalMockingShield]\033[0m Intercepted background housekeeping request (kind={kind}). "
            "Bypassed upstream model to preserve RPM quota!"
        )

        mock_id_map = {
            "ping_probe": "msg_mock_probe",
            "title_generation": "msg_mock_title",
            "autocomplete_suggestion": "msg_mock_suggestion",
        }
        mock_id = mock_id_map.get(kind, "msg_mock_housekeeping")

        return {
            "id": mock_id,
            "type": "message",
            "role": "assistant",
            "model": client_model,
            "content": [{"type": "text", "text": " "}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


local_mocking_shield = LocalMockingShield()
