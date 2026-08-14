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
async def test_heuristic_tool_parser_bash() -> None:
    """Test HeuristicToolParser extracting ```bash command blocks."""
    parser = HeuristicToolParser()

    chunk = "Here is the solution:\n```bash\npnpm dev\n```\nDone."
    events = await parser.process_chunk(chunk)

    tool_start = [e for e in events if isinstance(e, SSEContentBlockStartEvent) and getattr(e.content_block, "type", None) == "tool_use"]
    assert len(tool_start) == 1
    assert tool_start[0].content_block.name == "run_command"


@pytest.mark.asyncio
async def test_subagent_guard_enforcement() -> None:
    """Test SubagentGuard enforcing run_in_background=False on Task tool calls."""
    guard = SubagentGuard()

    input_data = {"prompt": "Analyze repo", "run_in_background": True}
    policed = await guard.enforce_tool_call("Task", input_data)

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
