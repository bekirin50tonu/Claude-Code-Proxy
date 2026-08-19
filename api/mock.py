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
    from atomic.guards.local_mocking_shield import LocalMockingShield
    is_hk, hk_kind = LocalMockingShield.is_housekeeping_request(request_body)
    if is_hk:
        return LocalMockingShield.generate_mock_response(request_body, kind=hk_kind)

    model = request_body.get("model", "claude-3-5-sonnet-latest")
    all_text = get_all_text(request_body)
    max_tokens = request_body.get("max_tokens", 4096)

    # 1. Network Probe / Quota Probe Mocking
    if settings.ENABLE_NETWORK_PROBE_MOCK:
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

    # 2. Title Generation Skip
    if settings.ENABLE_TITLE_GENERATION_SKIP:
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

    # 3. Suggestion Mode Skip
    if settings.ENABLE_SUGGESTION_MODE_SKIP:
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

    # 4. Command-Prefix Detection (Fast Prefix)
    if settings.FAST_PREFIX_DETECTION:
        is_prefix_req = any(
            kw in all_text
            for kw in [
                "extract the prefix",
                "detect command prefix",
                "identify prefix of bash command",
                "prefix detection",
                "policy spec",
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

    # 5. Filepath Extraction Mock
    if settings.ENABLE_FILEPATH_EXTRACTION_MOCK:
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
