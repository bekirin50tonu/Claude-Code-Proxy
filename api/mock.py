import re
from typing import Any

from config import settings


def get_all_text(request_body: dict[str, Any]) -> str:
    """Extract and aggregate all system and user text from the request body."""
    parts = []
    system = request_body.get("system")
    if system:
        if isinstance(system, list):
            parts.extend(
                [str(b.get("text", "")) for b in system if b.get("type") == "text"]
            )
        else:
            parts.append(str(system))

    messages = request_body.get("messages", [])
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def extract_command_prefix(text: str) -> str:
    """Extract the first word of a command (prefix) from the request text."""
    # Look for command inside backticks first, e.g. `git commit`
    matches = re.findall(r"`([^`]+)`", text)
    if matches:
        cmd = matches[-1].strip()  # take the most recent one
        parts = cmd.split()
        if parts:
            return parts[0]

    # Look for common CLI tools in lines
    common_tools = [
        "git",
        "npm",
        "python",
        "pip",
        "uv",
        "node",
        "docker",
        "ls",
        "cd",
        "cat",
        "mkdir",
        "rm",
        "mv",
        "cp",
        "grep",
        "cargo",
        "go",
        "make",
        "clear",
        "echo",
    ]
    for tool in common_tools:
        if re.search(rf"\b{tool}\b", text):
            return tool

    return "git"  # fallback default prefix


def extract_filepaths(text: str) -> list[str]:
    """Find potential file paths or names in text using heuristic regex."""
    # Matches patterns like src/main.py, config/settings.json, README.md, etc.
    pattern = r"\b[a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9_\-]{1,6}\b"
    candidates = re.findall(pattern, text)
    unique_paths = []
    seen = set()
    for path in candidates:
        # Filter out numbers, URLs, or invalid path pieces
        if "http://" in path or "https://" in path or path.startswith("."):
            continue
        if "/" not in path and "." not in path:
            continue
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def check_mock_request(request_body: dict[str, Any]) -> dict[str, Any] | None:
    """
    Check if the request is a mock candidate.
    Returns a mocked Anthropic Message response dict if matched, or None.
    """
    if not isinstance(request_body, dict):
        return None

    all_text = get_all_text(request_body)
    all_text_lower = all_text.lower()
    model = request_body.get("model", "claude-3-5-sonnet-latest")

    # Fast-path for Title Generation (free-claude-code style optimization)
    # Detects when CLI names a coding session, which carries output_config json_schema
    if settings.ENABLE_TITLE_GENERATION_SKIP and not request_body.get("tools"):
        if (
            "naming a coding session" in all_text_lower
            or "new conversation topic" in all_text_lower
            or ("<session>" in all_text and "title" in all_text_lower)
            or any(kw in all_text for kw in [
                "Generate a short, 2-4 word title",
                "Create a title for this conversation",
                "summarize this conversation into a title",
                "Provide a short title",
            ])
        ):
            output_config = request_body.get("output_config", {})
            format_type = output_config.get("format", {}).get("type") if isinstance(output_config, dict) else None
            if format_type == "json_schema" or "return json" in all_text_lower:
                text_content = '{"title": "Claude Code Session"}'
            else:
                text_content = "Claude Code Session"

            return {
                "id": "msg_mock_title",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text_content}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 50, "output_tokens": 5},
            }

    # Fast-path for Suggestion Mode Requests (free-claude-code style optimization)
    # Intercepts CLI background requests that predict what the user might type next, returning instantly at 0ms.
    if "[suggestion mode:" in all_text_lower:
        import uuid
        return {
            "id": f"msg_mock_suggestion_{uuid.uuid4().hex[:8]}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 1},
        }

    # Safety Guard: Never intercept requests containing tool definitions, output configs, thinking, or stop hooks
    if (
        request_body.get("tools")
        or "output_config" in request_body
        or "thinking" in request_body
        or "evaluating a stop-condition hook" in all_text_lower
        or "hook_event_name" in all_text_lower
    ):
        return None

    from atomic.guards.local_mocking_shield import LocalMockingShield
    is_hk, hk_kind = LocalMockingShield.is_housekeeping_request(request_body)
    if is_hk:
        return LocalMockingShield.generate_mock_response(request_body, kind=hk_kind)

    model = request_body.get("model", "claude-3-5-sonnet-latest")
    max_tokens = request_body.get("max_tokens", 4096)
    text_len = len(all_text)

    # 1. Network Probe / Quota Probe Mocking (only for short requests < 500 chars)
    if settings.ENABLE_NETWORK_PROBE_MOCK and text_len < 500:
        is_probe = False
        if max_tokens <= 2 or any(
            kw in all_text.lower()
            for kw in [
                "network-probe",
                "network probe",
                "ping check",
                "test connection",
            ]
        ):
            is_probe = True

        if is_probe:
            return {
                "id": "msg_mock_probe",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": " "}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }

    # 2. Title Generation Skip (only for short requests < 1000 chars)
    if settings.ENABLE_TITLE_GENERATION_SKIP and text_len < 1000:
        is_title_req = any(
            kw in all_text
            for kw in [
                "Generate a short, 2-4 word title",
                "Create a title for this conversation",
                "summarize this conversation into a title",
                "Provide a short title",
            ]
        )
        if is_title_req:
            return {
                "id": "msg_mock_title",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Claude Code Session"}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 3},
            }

    # 3. Suggestion Mode Skip (only for short requests < 1000 chars)
    if settings.ENABLE_SUGGESTION_MODE_SKIP and text_len < 1000:
        is_suggestion = any(
            kw in all_text
            for kw in [
                "suggestion mode",
                "suggest the next command",
                "autocomplete suggestion",
                "predict next input",
                "suggestion for the next prompt",
            ]
        )
        if is_suggestion:
            return {
                "id": "msg_mock_suggestion",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": " "}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }

    # 4. Command-Prefix Detection (Fast Prefix, only for short requests < 1000 chars)
    if settings.FAST_PREFIX_DETECTION and text_len < 1000:
        is_prefix_req = any(
            kw in all_text
            for kw in [
                "extract the prefix",
                "detect command prefix",
                "identify prefix of bash command",
                "prefix detection of bash command",
                "bash command prefix",
            ]
        )
        if is_prefix_req:
            prefix = extract_command_prefix(all_text)
            return {
                "id": "msg_mock_prefix",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": prefix}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 15, "output_tokens": 1},
            }

    # 5. Filepath Extraction Mock (only for short requests < 1000 chars)
    if settings.ENABLE_FILEPATH_EXTRACTION_MOCK and text_len < 1000:
        is_filepath_req = any(
            kw in all_text
            for kw in ["extract file paths", "filepath extraction", "list file paths"]
        )
        if is_filepath_req:
            paths = extract_filepaths(all_text)
            paths_str = "\n".join(paths) if paths else " "
            return {
                "id": "msg_mock_filepath",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": paths_str}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": len(paths) + 1},
            }

    return None
