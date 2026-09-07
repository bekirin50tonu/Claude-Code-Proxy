"""Unit tests for Atomic Parsers, ModelConverter, StreamTransformer, and CLISessionManager."""

import pytest

from atomic.guards.subagent import SubagentGuard
from atomic.parsers.heuristic_tool import HeuristicToolParser
from atomic.parsers.thinking import ThinkingParser
from cli.session import CLISessionManager
from core.transformer.stream_engine import StreamEngine
from shared.schemas.anthropic import (
    SSEContentBlockDeltaEvent,
    SSEContentBlockStartEvent,
    SSEContentBlockStopEvent,
)


@pytest.mark.asyncio
async def test_thinking_parser_split_tags() -> None:
    """Test ThinkingParser handling <think> tags split across streaming chunks."""
    parser = ThinkingParser()

    # Chunk 1: Starts with text and partial "<th"
    events1 = await parser.process_chunk("Hello! <th")
    assert len(events1) >= 2
    assert any(isinstance(e, SSEContentBlockStartEvent) for e in events1)

    # Chunk 2: Completes "ink>Reasoning content</th"
    events2 = await parser.process_chunk("ink>Reasoning content</th")
    assert any(isinstance(e, SSEContentBlockDeltaEvent) for e in events2)

    # Chunk 3: Completes "ink> Final response"
    events3 = await parser.process_chunk("ink> Final response")
    assert any(isinstance(e, SSEContentBlockStopEvent) for e in events3)

    flush_events = await parser.flush()
    assert flush_events or events3


@pytest.mark.asyncio
async def test_thinking_parser_pre_think_noise_filtering() -> None:
    """Verify ThinkingParser suppresses pre-think noise (like Priority Order) or 9`)) before <think> tags."""
    parser = ThinkingParser()
    events, clean_text = await parser.process_chunk_pipeline("Priority Order)**\n\n<think>Internal thought</think>Final answer")
    assert clean_text == "Final answer"
    assert any(hasattr(e, "delta") and getattr(e.delta, "thinking", "") == "Internal thought" for e in events)


@pytest.mark.asyncio
async def test_heuristic_tool_parser_bash() -> None:
    """Test HeuristicToolParser extracting ```bash command blocks."""
    parser = HeuristicToolParser()

    chunk = "Here is the solution:\n```bash\npnpm dev\n```\nDone."
    events = await parser.process_chunk(chunk)

    tool_start = [e for e in events if isinstance(e, SSEContentBlockStartEvent) and getattr(e.content_block, "type", None) == "tool_use"]
    assert len(tool_start) == 1
    assert tool_start[0].content_block.name == "run_command"


@pytest.mark.asyncio
async def test_heuristic_tool_parser_tool_code_json() -> None:
    """Test HeuristicToolParser extracting <tool_code> blocks containing JSON command."""
    parser = HeuristicToolParser(tools=["Bash", "Read", "Write"])

    chunk = (
        "Elbette, .claude/ dizininize bakalım:\n"
        "<tool_code>\n"
        '{"command":"ls /media/bekir/HDDStorage/PROJECTS/MY_SITE/website/.claude","description":"Lists files"}\n'
        "</tool_code>"
    )
    events = await parser.process_chunk(chunk)

    tool_start = [e for e in events if isinstance(e, SSEContentBlockStartEvent) and getattr(e.content_block, "type", None) == "tool_use"]
    assert len(tool_start) == 1
    assert tool_start[0].content_block.name == "Bash"

    delta = [e for e in events if isinstance(e, SSEContentBlockDeltaEvent) and getattr(e.delta, "type", None) == "input_json_delta"]
    assert len(delta) == 1
    import json
    parsed_input = json.loads(delta[0].delta.partial_json)
    assert parsed_input["command"] == "ls /media/bekir/HDDStorage/PROJECTS/MY_SITE/website/.claude"


@pytest.mark.asyncio
async def test_heuristic_tool_parser_tool_code_xml() -> None:
    """Test HeuristicToolParser extracting <tool_code> blocks containing XML <name> and <parameters>."""
    parser = HeuristicToolParser(tools=["Bash", "Read", "Write"])

    chunk = (
        "<tool_code>\n"
        "<name>Read</name>\n"
        "<parameters>\n"
        "    <file_path>/media/bekir/test.txt</file_path>\n"
        "</parameters>\n"
        "</tool_code>"
    )
    events = await parser.process_chunk(chunk)

    tool_start = [e for e in events if isinstance(e, SSEContentBlockStartEvent) and getattr(e.content_block, "type", None) == "tool_use"]
    assert len(tool_start) == 1
    assert tool_start[0].content_block.name == "Read"

    delta = [e for e in events if isinstance(e, SSEContentBlockDeltaEvent) and getattr(e.delta, "type", None) == "input_json_delta"]
    assert len(delta) == 1
    import json
    parsed_input = json.loads(delta[0].delta.partial_json)
    assert parsed_input["file_path"] == "/media/bekir/test.txt"


@pytest.mark.asyncio
async def test_subagent_guard_enforcement() -> None:
    """Test SubagentGuard enforcing run_in_background=False on Task tool calls in OFF bypass mode."""
    guard = SubagentGuard()

    input_data = {"prompt": "Analyze repo", "run_in_background": True}
    policed = await guard.enforce_tool_call("Task", input_data, enabled=False)

    assert policed["run_in_background"] is False
    assert guard.enforcements_count == 1


@pytest.mark.asyncio
async def test_cli_session_manager() -> None:
    """Test CLISessionManager tracking session metrics."""
    mgr = CLISessionManager()
    session = mgr.get_or_create_session("test_sess_100")

    assert session.session_id == "test_sess_100"
    assert session.turn_count == 0

    mgr.record_turn("test_sess_100", input_tokens=150, output_tokens=50)
    assert session.turn_count == 1
    assert session.total_input_tokens == 150
    assert session.total_output_tokens == 50


@pytest.mark.asyncio
async def test_stream_transformer_end_to_end() -> None:
    """Test StreamEngine streaming pipeline with OpenAI chunk generator."""
    engine = StreamEngine(target_model="claude-3-5-sonnet", session_id="test_stream_sess")

    async def mock_chunks():
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        yield {"choices": [{"delta": {"content": "Hello! <think>Let me analyze</think> Here is the answer."}}]}
        yield {"choices": [{"finish_reason": "stop"}]}

    events = []
    async for sse_str in engine.stream_response(mock_chunks()):
        events.append(sse_str)

    full_sse = "".join(events)
    assert "event: message_start" in full_sse
    assert "event: content_block_start" in full_sse
    assert "thinking_delta" in full_sse
    assert "text_delta" in full_sse
    assert "event: message_stop" in full_sse

    summary = engine.get_summary_response()
    assert summary["model"] == "claude-3-5-sonnet"
    assert len(summary["content"]) >= 2


def test_anthropic_sse_formatter() -> None:
    """Test AnthropicSSEFormatter stateless static helper class."""
    from shared.utils.sse_helper import AnthropicSSEFormatter

    msg_start = AnthropicSSEFormatter.message_start("msg_test", "claude-3-5-sonnet")
    assert "event: message_start" in msg_start
    assert '"id": "msg_test"' in msg_start

    txt_start = AnthropicSSEFormatter.text_start(1)
    assert "event: content_block_start" in txt_start
    assert '"type": "text"' in txt_start

    txt_delta = AnthropicSSEFormatter.text_delta("Hello world", 1)
    assert "event: content_block_delta" in txt_delta
    assert '"text": "Hello world"' in txt_delta

    think_delta = AnthropicSSEFormatter.thinking_delta("Reasoning...", 0)
    assert "event: content_block_delta" in think_delta
    assert '"thinking": "Reasoning..."' in think_delta

    blk_stop = AnthropicSSEFormatter.block_stop(1)
    assert "event: content_block_stop" in blk_stop

    msg_stop = AnthropicSSEFormatter.message_stop()
    assert "event: message_stop" in msg_stop


@pytest.mark.asyncio
async def test_stream_transformer_thinking_only_fallback_safety_net() -> None:
    """Test safety net when model yields ONLY thinking (no text/tools), ensuring fallback text delta space is injected."""
    engine = StreamEngine(target_model="claude-3-5-sonnet", session_id="test_safety_net_1")

    async def mock_thinking_only_chunks():
        yield {"choices": [{"delta": {"reasoning_content": "Deep thinking process without text..."}}]}
        yield {"choices": [{"finish_reason": "stop"}]}

    events = []
    async for sse_str in engine.transform_stream(mock_thinking_only_chunks()):
        events.append(sse_str)

    full_sse = "".join(events)
    assert "event: message_start" in full_sse
    assert "thinking_delta" in full_sse
    # Crucial assertion: safety net injects text_delta with space " " so message is not empty
    assert "text_delta" in full_sse
    assert "event: message_stop" in full_sse
    assert engine.text_or_tool_emitted is True


@pytest.mark.asyncio
async def test_stream_transformer_empty_stream_fallback_safety_net() -> None:
    """Test safety net when stream is completely empty (0 chunks)."""
    engine = StreamEngine(target_model="claude-3-5-sonnet", session_id="test_safety_net_2")

    async def empty_chunks():
        if False:
            yield {}

    events = []
    async for sse_str in engine.transform_stream(empty_chunks()):
        events.append(sse_str)

    full_sse = "".join(events)
    assert "event: message_start" in full_sse
    assert "text_delta" in full_sse
    assert "event: message_stop" in full_sse
    assert engine.text_or_tool_emitted is True


def test_safe_parse_json_robustness() -> None:
    """Test safe_parse_json repairing python literals, trailing commas, single quotes, and stringified JSON tool input."""
    from core.transformer.stream_engine import _parse_tool_from_json, safe_parse_json

    res1 = safe_parse_json("{'file_path': 'src/app.py', 'overwrite': True,}")
    assert res1 == {"file_path": "src/app.py", "overwrite": True}

    obj = {"name": "View", "arguments": '{"file_path": "src/index.ts"}'}
    parsed = _parse_tool_from_json(obj)
    assert parsed is not None
    assert parsed["name"] == "View"
    assert parsed["input"] == {"file_path": "src/index.ts"}


@pytest.mark.asyncio
async def test_stream_transformer_thinking_only_fallback_does_not_echo_private_thinking() -> None:
    """Ensure thinking-only stream emits a space text delta, never echoing private thinking text."""
    engine = StreamEngine(target_model="claude-3-5-sonnet", session_id="test_no_echo_sess")

    async def mock_thinking_chunks():
        yield {"choices": [{"delta": {"reasoning_content": "Private internal reasoning that should not leak to user"}}]}
        yield {"choices": [{"finish_reason": "stop"}]}

    events = [ev async for ev in engine.transform_stream(mock_thinking_chunks())]
    full_sse = "".join(events)

    assert "event: message_start" in full_sse
    assert "Private internal reasoning" in full_sse  # In thinking_delta
    # Crucial: text_delta must be " ", not the private thinking text
    assert '{"type": "text_delta", "text": " "}' in full_sse
    assert '"text": "Private internal reasoning' not in full_sse


@pytest.mark.asyncio
async def test_stream_transformer_no_duplicate_content_block_stop() -> None:
    """Verify stream never emits consecutive duplicate content_block_stop events for the same block index."""
    engine = StreamEngine(target_model="claude-3-5-sonnet", session_id="test_no_dup_stop")

    async def mock_chunks():
        yield {"choices": [{"delta": {"content": "<think>Thinking chunk</think>"}}]}
        yield {"choices": [{"finish_reason": "stop"}]}

    events = [ev async for ev in engine.transform_stream(mock_chunks())]
    
    # Count content_block_stop for index 0
    stop_0_count = sum(1 for ev in events if 'content_block_stop' in ev and '"index": 0' in ev)
    assert stop_0_count == 1, f"Expected exactly 1 content_block_stop for index 0, got {stop_0_count}"


@pytest.mark.asyncio
async def test_native_tool_call_streaming_sanitizes_input() -> None:
    """Verify native tool call arguments are sanitized before being emitted to the SSE stream."""
    engine = StreamEngine(target_model="claude-3-5-sonnet", session_id="test_tool_sanitize")

    async def mock_tool_chunks():
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_read_1",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"description": "Read file", "file_path": "AGENTS.md"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        yield {"choices": [{"finish_reason": "tool_calls"}]}

    events = [ev async for ev in engine.transform_stream(mock_tool_chunks())]
    full_sse = "".join(events)

    assert "event: content_block_start" in full_sse
    assert '"name": "Read"' in full_sse
    # Crucial: The emitted input_json_delta must NOT contain the illegal 'description' parameter
    assert "description" not in full_sse
    assert "AGENTS.md" in full_sse
    assert "event: content_block_stop" in full_sse


def test_mock_suggestion_mode_intercepted() -> None:
    """Verify [SUGGESTION MODE: requests are intercepted at 0ms by api/mock.py."""
    from api.mock import check_mock_request

    req_body = {
        "model": "claude-opus-5",
        "messages": [
            {
                "role": "user",
                "content": "[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code.]",
            }
        ],
        "tools": [{"name": "Bash", "description": "Run bash"}],
    }

    mock_resp = check_mock_request(req_body)
    assert mock_resp is not None
    assert mock_resp["role"] == "assistant"
    assert mock_resp["content"] == [{"type": "text", "text": ""}]
    assert mock_resp["stop_reason"] == "end_turn"


def test_translate_messages_preserves_assistant_thinking() -> None:
    """Verify assistant messages with thinking blocks are translated into <think> tags."""
    from providers.openai import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider()
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me examine AGENTS.md first"},
                {"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "ls"}},
            ],
        }
    ]

    translated = provider.translate_messages(messages)
    assert len(translated) == 1
    assert translated[0]["role"] == "assistant"
    assert "<think>\nLet me examine AGENTS.md first\n</think>" in translated[0]["content"]
    assert translated[0]["tool_calls"][0]["function"]["name"] == "Bash"


