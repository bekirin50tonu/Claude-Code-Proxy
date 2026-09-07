"""Comprehensive Stress Testing Suite for Thinking / Reasoning Stream Engine.

Tests:
1. Native reasoning_content -> text transition SSE block ordering.
2. Native reasoning_content -> tool_calls transition SSE block ordering.
3. Fragmented <think>...</think> tags split across 1-character SSE chunks.
4. Thinking-only fallback safety net (zero text response scenario).
5. High concurrency (50 parallel stream engine instances).
"""

import asyncio
import contextlib
import json

import pytest

from core.transformer.stream_engine import StreamEngine


@pytest.mark.asyncio
async def test_native_reasoning_to_text_block_sequence():
    """Verify that reasoning_content block (0) is STOPPED before text block (1) STARTS."""
    async def mock_upstream():
        yield {"choices": [{"delta": {"reasoning_content": "Thinking step 1..."}}]}
        yield {"choices": [{"delta": {"reasoning_content": "Thinking step 2..."}}]}
        yield {"choices": [{"delta": {"content": "Here is the final answer."}}]}

    engine = StreamEngine(target_model="claude-sonnet-5")
    events = []
    async for sse in engine.transform_stream(mock_upstream()):
        events.append(sse.strip())

    parsed_events = []
    for sse in events:
        lines = sse.split("\n")
        event_type = None
        data = {}
        for line in lines:
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                with contextlib.suppress(Exception):
                    data = json.loads(line[5:].strip())
        parsed_events.append({"event": event_type, "data": data})

    # Find indices of key events
    stop_block_0_idx = None
    start_block_1_idx = None

    for i, pe in enumerate(parsed_events):
        ev = pe["event"]
        d = pe["data"]
        if ev == "content_block_stop" and d.get("index") == 0:
            stop_block_0_idx = i
        elif ev == "content_block_start" and d.get("index") == 1:
            start_block_1_idx = i

    assert stop_block_0_idx is not None, "Block 0 (thinking) was not stopped"
    assert start_block_1_idx is not None, "Block 1 (text) was not started"
    assert stop_block_0_idx < start_block_1_idx, f"Block 0 stop ({stop_block_0_idx}) must precede Block 1 start ({start_block_1_idx})"


@pytest.mark.asyncio
async def test_native_reasoning_to_tool_calls_sequence():
    """Verify that reasoning_content block (0) is STOPPED before tool_use block (1) STARTS."""
    async def mock_upstream():
        yield {"choices": [{"delta": {"reasoning_content": "Analyzing request..."}}]}
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_123",
                                "function": {"name": "run_command", "arguments": '{"CommandLine": "ls"}'},
                            }
                        ]
                    }
                }
            ]
        }

    engine = StreamEngine(target_model="claude-sonnet-5")
    events = []
    async for sse in engine.transform_stream(mock_upstream()):
        events.append(sse.strip())

    stop_block_0_idx = None
    start_tool_idx = None

    for i, sse in enumerate(events):
        if "content_block_stop" in sse and '"index": 0' in sse:
            stop_block_0_idx = i
        if "content_block_start" in sse and '"type": "tool_use"' in sse:
            start_tool_idx = i

    assert stop_block_0_idx is not None, "Thinking block 0 was not stopped"
    assert start_tool_idx is not None, "Tool use block was not started"
    assert stop_block_0_idx < start_tool_idx, "Thinking block must stop before tool use starts"


@pytest.mark.asyncio
async def test_split_think_tags_across_chunks():
    """Verify <think>...</think> tags split across 1-char chunks parse without errors."""
    full_text = "<think>Step 1. Step 2.</think>Final Result Code"

    async def mock_upstream():
        for ch in full_text:
            yield {"choices": [{"delta": {"content": ch}}]}

    engine = StreamEngine(target_model="claude-sonnet-5")
    events = []
    async for sse in engine.transform_stream(mock_upstream()):
        events.append(sse.strip())

    accumulated_text = "".join(engine.accumulated_text)
    accumulated_thinking = "".join(engine.accumulated_thinking)

    assert accumulated_thinking == "Step 1. Step 2."
    assert accumulated_text == "Final Result Code"


@pytest.mark.asyncio
async def test_thinking_only_fallback():
    """Verify stream with thinking but ZERO text triggers safety fallback injection."""
    async def mock_upstream():
        yield {"choices": [{"delta": {"reasoning_content": "Calculated answer in head."}}]}

    engine = StreamEngine(target_model="claude-sonnet-5")
    events = []
    async for sse in engine.transform_stream(mock_upstream()):
        events.append(sse.strip())

    text_emitted = engine.text_or_tool_emitted
    accumulated_text = "".join(engine.accumulated_text)
    accumulated_thinking = "".join(engine.accumulated_thinking)

    assert text_emitted is True
    assert "Calculated answer in head." in accumulated_thinking
    assert " " in accumulated_text
    assert "Calculated answer in head." not in accumulated_text


@pytest.mark.asyncio
async def test_high_concurrency_thinking_streams():
    """Simulate 50 concurrent streaming pipelines under load."""
    async def run_single_stream(idx: int):
        async def mock_upstream():
            yield {"choices": [{"delta": {"reasoning_content": f"Thinking worker {idx}..."}}]}
            await asyncio.sleep(0.001)
            yield {"choices": [{"delta": {"content": f"Answer worker {idx}."}}]}

        engine = StreamEngine(target_model="claude-sonnet-5")
        count = 0
        async for _ in engine.transform_stream(mock_upstream()):
            count += 1
        return count

    tasks = [run_single_stream(i) for i in range(50)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 50
    for r in results:
        assert r > 5, "Each stream should emit valid SSE protocol event sequence"
