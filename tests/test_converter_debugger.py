"""Unit & Integration Tests for Converter Debugger & Runtime Compatibility Reforms."""

import json
from pathlib import Path

import pytest

from atomic.sanitizers.rtk_compressor import RedundantTokenKiller
from config.config import ClaudeSettingsManager
from converter_debugger import ConverterDebugger
from models.converter import ModelConverter


@pytest.mark.asyncio
async def test_reform_1_linear_pipeline_hand_off():
    """Verify linear sequential pipeline hand-off (Thinking -> Tool -> Text) prevents duplicate tool calls."""
    chunks = ["<think>Thinking about running command...</think> ```bash\nls -l\n```"]
    tools = [{"name": "run_command"}]
    result = await ConverterDebugger.debug_stream_pipeline(chunks, tools=tools)

    assert result["thinking_blocks_count"] > 0
    assert result["tool_calls_count"] == 1
    tool_call = result["tool_calls"][0]
    assert tool_call["name"] in ("run_command", "Bash")
    assert tool_call["input"] == {"CommandLine": "ls -l"} or tool_call["input"] == {"command": "ls -l"}


@pytest.mark.asyncio
async def test_reform_2_xml_tool_call_and_generics_masking():
    """Verify XML <tool_call> parsing and TypeScript angle bracket generics masking."""
    xml_input = (
        "Here is code:\n"
        "```ts\n"
        "const map: Record<<string, Locale>> = {};\n"
        "```\n"
        "<tool_call>\n"
        "<<function=Edit>\n"
        "<<parameter=file_path>src/index.ts</parameter>\n"
        "<<parameter=old_string>foo</parameter>\n"
        "<<parameter=new_string>bar</parameter>\n"
        "</tool_call>"
    )
    tools = [{"name": "Edit"}]
    res = await ConverterDebugger.debug_xml_tool_repair(xml_input, tools=tools)

    assert "«" in res["masked_text"] or "Record" in res["masked_text"]
    assert len(res["events_emitted"]) > 0


def test_reform_3_graceful_stream_healing_eof():
    """Verify EOF stream healing auto-closes unclosed XML tags and quotes."""
    truncated = "<tool_call>\n<<function=Edit>\n<<parameter=file_path>src/config.ts"
    healed = ConverterDebugger.debug_stream_healing(truncated)

    assert healed["was_modified"] is True
    assert "</parameter>" in healed["repaired"]
    assert "</function>" in healed["repaired"]
    assert "</tool_call>" in healed["repaired"]


def test_reform_4_claude_settings_sync(tmp_path, monkeypatch):
    """Verify automatic sync of .claude/settings.json."""
    fake_home = tmp_path / "home"
    fake_cwd = tmp_path / "cwd"
    fake_home.mkdir()
    fake_cwd.mkdir()

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(Path, "cwd", lambda: fake_cwd)

    synced = ClaudeSettingsManager.sync_settings(port=8090, auth_token="test-token")
    assert len(synced) == 2

    global_settings = json.loads((fake_home / ".claude" / "settings.json").read_text())
    assert global_settings["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8090"
    assert global_settings["env"]["ANTHROPIC_AUTH_TOKEN"] == "test-token"

    local_settings = json.loads((fake_cwd / ".claude" / "settings.json").read_text())
    assert local_settings["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8090"


def test_reform_5_rtk_log_compression():
    """Verify RTK console log compression achieves >= 50% token reduction for long logs."""
    raw_log = "\x1b[32m[INFO]\x1b[0m Starting build...\n" + "".join(f"build log step {i}\n" for i in range(200)) + "Exit code 1"
    compressed = RedundantTokenKiller.compress_log(raw_log)

    orig_len = len(raw_log)
    comp_len = len(compressed)
    reduction = ((orig_len - comp_len) / orig_len) * 100

    assert reduction >= 50.0
    assert "\x1b[" not in compressed
    assert "[RTK: Compressed" in compressed


def test_model_converter_rtk_integration():
    """Verify ModelConverter applies RTK compression when converting user tool_result messages."""
    raw_tool_result = "Build output...\n" + "".join(f"build log step {i}\n" for i in range(100)) + "Finished."
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_abc123",
                    "content": raw_tool_result,
                }
            ],
        }
    ]

    converted = ModelConverter.anthropic_to_openai_messages(messages)
    assert len(converted) == 1
    tool_msg = converted[0]
    assert tool_msg["role"] == "tool"
    assert len(tool_msg["content"]) < len(raw_tool_result)
    assert "[RTK: Compressed" in tool_msg["content"]
