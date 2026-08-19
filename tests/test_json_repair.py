"""Unit tests for JSONRepairNormalizer and JSONRepairMiddleware."""

import json

import pytest

from core.interceptor.json_repair import JSONRepairNormalizer


@pytest.mark.asyncio
async def test_is_stop_hook_target_detection() -> None:
    # 1. Target in user message
    payload_msg = {
        "messages": [
            {"role": "user", "content": "Please perform Stop hook and save session summary."}
        ]
    }
    assert JSONRepairNormalizer.is_stop_hook_target(payload_msg) is True

    # 2. Target in system prompt
    payload_sys = {
        "messages": [{"role": "user", "content": "Hello"}],
        "system": "System instructions: Handle stop_hook requests carefully.",
    }
    assert JSONRepairNormalizer.is_stop_hook_target(payload_sys) is True

    # 3. Target in tools definition
    payload_tool = {
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [{"name": "exit_session", "description": "Exit session and save memory"}],
    }
    assert JSONRepairNormalizer.is_stop_hook_target(payload_tool) is True

    # 4. Sensitive edit operations (standard edit request should NOT force stop hook JSON schema)
    payload_edit = {
        "messages": [{"role": "user", "content": "Please Edit file src/main.ts"}],
    }
    assert JSONRepairNormalizer.is_stop_hook_target(payload_edit) is False

    # 5. Non-target payload
    payload_normal = {
        "messages": [{"role": "user", "content": "Write a python function to add two numbers."}]
    }
    assert JSONRepairNormalizer.is_stop_hook_target(payload_normal) is False


def test_angle_bracket_trap_fixer() -> None:
    # Code generics and bitwise shift operators inside code blocks
    code_text = "const map: Record<Locale, string> = {}; std::cout << 'test' << std::endl;"
    fixed = JSONRepairNormalizer.fix_angle_brackets(code_text)
    assert "Record<Locale, string>" in fixed
    assert "std::cout <<" in fixed


@pytest.mark.asyncio
async def test_sanitize_markdown_json() -> None:
    # 1. Markdown codeblock ```json ... ```
    raw_1 = "```json\n{\n  \"summary\": \"Saved work\",\n  \"memory\": \"Remember tests\"\n}\n```"
    sanitized_1 = await JSONRepairNormalizer.sanitize_markdown_json(raw_1)
    assert sanitized_1 == '{\n  "summary": "Saved work",\n  "memory": "Remember tests"\n}'

    # 2. Text prefix/suffix surrounding JSON object
    raw_2 = "Here is the summary JSON:\n{\"summary\": \"Done\"}\nHope this helps!"
    sanitized_2 = await JSONRepairNormalizer.sanitize_markdown_json(raw_2)
    assert sanitized_2 == '{"summary": "Done"}'


@pytest.mark.asyncio
async def test_heuristic_repair_json() -> None:
    # 1. Trailing commas & Python True/False/None
    raw_broken = '{"summary": "Finished task", "stop_hook_active": True, "memory": None,}'
    repaired = await JSONRepairNormalizer.heuristic_repair_json(raw_broken)
    assert isinstance(repaired, dict)
    assert repaired["summary"] == "Finished task"
    assert repaired["stop_hook_active"] is True
    assert repaired["memory"] is None

    # 2. Single quotes repair
    raw_single_quotes = "{'summary': 'Session ended', 'memories': ['a', 'b']}"
    repaired_sq = await JSONRepairNormalizer.heuristic_repair_json(raw_single_quotes)
    assert isinstance(repaired_sq, dict)
    assert repaired_sq["summary"] == "Session ended"

    # 3. Unclosed curly brace
    raw_unclosed = '{"summary": "Unclosed object", "stop_hook_active": false'
    repaired_unclosed = await JSONRepairNormalizer.heuristic_repair_json(raw_unclosed)
    assert isinstance(repaired_unclosed, dict)
    assert repaired_unclosed["summary"] == "Unclosed object"
    assert repaired_unclosed["stop_hook_active"] is False


@pytest.mark.asyncio
async def test_normalize_stop_hook_schema() -> None:
    # 1. Key alias mapping & default injection
    input_data = {
        "session_summary": "Created new feature",
        "memories": ["test_1", "test_2"],
        "stop_hook": "True",
    }
    normalized = await JSONRepairNormalizer.normalize_stop_hook_schema(input_data)
    assert normalized["summary"] == "Created new feature"
    assert normalized["memory"] == ["test_1", "test_2"]
    assert normalized["stop_hook_active"] is True

    # 2. String input converted to dict
    string_input = "Simple summary text"
    norm_str = await JSONRepairNormalizer.normalize_stop_hook_schema(string_input)
    assert norm_str["summary"] == "Simple summary text"
    assert norm_str["memory"] == ""
    assert norm_str["stop_hook_active"] is True

    # 3. Missing keys injected with canonical defaults
    empty_dict = {}
    norm_empty = await JSONRepairNormalizer.normalize_stop_hook_schema(empty_dict)
    assert norm_empty["summary"] == ""
    assert norm_empty["memory"] == ""
    assert norm_empty["stop_hook_active"] is True


def test_double_tool_call_deduplication() -> None:
    content = [
        {
            "type": "text",
            "text": "I will execute tool <tool_call><<function=Bash>><<parameter=command>>ls</parameter>></function>></tool_call>",
        },
        {
            "type": "tool_use",
            "id": "tool_1",
            "name": "Bash",
            "input": {"command": "ls"},
        },
        {
            "type": "tool_use",
            "id": "tool_2",
            "name": "Bash",
            "input": {"command": "ls"},
        },
    ]

    deduped = JSONRepairNormalizer.deduplicate_tool_calls(content)
    # 1. XML in text stripped
    assert "<tool_call>" not in deduped[0]["text"]
    # 2. Duplicate tool_use block removed (only 1 tool_use block remains)
    tool_blocks = [b for b in deduped if b.get("type") == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["input"]["command"] == "ls"


@pytest.mark.asyncio
async def test_subagents_switch_integration() -> None:
    raw_response = {
        "id": "msg_subagent_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tool_sub",
                "name": "Agent",
                "input": {"description": "Subtask", "run_in_background": True},
            }
        ],
    }

    # When subagents_enabled=False, run_in_background should be overridden to False
    processed = await JSONRepairNormalizer.process_response_dict(raw_response, subagents_enabled=False)
    tool_input = processed["content"][0]["input"]
    assert tool_input["run_in_background"] is False


@pytest.mark.asyncio
async def test_process_text_pipeline() -> None:
    raw_input = (
        "```json\n"
        "{\n"
        "  'session_summary': 'Refactored code',\n"
        "  'memories': ['Remember to test'],\n"
        "  'stop_hook': True,\n"
        "}\n"
        "```"
    )
    processed_json = await JSONRepairNormalizer.process_text(raw_input)
    parsed = json.loads(processed_json)
    assert parsed["summary"] == "Refactored code"
    assert parsed["memory"] == ["Remember to test"]
    assert parsed["stop_hook_active"] is True


@pytest.mark.asyncio
async def test_json_repair_response_dict_integration() -> None:
    malformed = (
        "```json\n"
        "{\n"
        "  \"session_summary\": \"All tasks finished\",\n"
        "  \"memories\": \"Updated config\",\n"
        "  \"active\": True,\n"
        "}\n"
        "```"
    )
    raw_response = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": malformed}],
        "stop_reason": "end_turn",
    }

    repaired_resp = await JSONRepairNormalizer.process_response_dict(raw_response)
    text_content = repaired_resp["content"][0]["text"]

    repaired_dict = json.loads(text_content)
    assert repaired_dict["summary"] == "All tasks finished"
    assert repaired_dict["memory"] == "Updated config"
    assert repaired_dict["stop_hook_active"] is True


def test_hermes_xml_tool_call_extraction() -> None:
    from core.transformer.stream_engine import extract_all_json_tool_calls

    xml_text = (
        "<tool_call>\n"
        "<<function=Bash>>\n"
        "<<parameter=command>>\n"
        "ls -la /src/i18n\n"
        "</parameter>>\n"
        "</function>>\n"
        "</tool_call>"
    )
    cleaned, tools = extract_all_json_tool_calls(xml_text)
    assert len(tools) == 1
    assert tools[0]["name"] == "Bash"
    assert tools[0]["input"]["command"] == "ls -la /src/i18n"
    assert "<tool_call>" not in cleaned
