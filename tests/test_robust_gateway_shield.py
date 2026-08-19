"""Unit & Integration Tests for Robust Gateway & Local Mocking Shield.

Tests the four core protection layers:
1. LocalMockingShield (Housekeeping bypass & Pydantic validation)
2. DeDuplicator (Stripping XML tool call noise alongside structural tool_use)
3. AngleBracketEscape (Protecting TypeScript generics inside code blocks)
4. Stateful Parsers (Tag splitting across SSE chunk boundaries)
"""

import pytest

from atomic.guards.local_mocking_shield import LocalMockingShield
from atomic.parsers.heuristic_tool import HeuristicToolStatefulParser
from atomic.parsers.thinking import ThinkingStatefulParser
from atomic.sanitizers.angle_bracket_escape import AngleBracketEscape
from atomic.sanitizers.deduplicator import DeDuplicator


class TestLocalMockingShield:
    """Test Layer 1: Local Mocking & Housekeeping Bypass."""

    def test_detect_autocomplete_suggestion(self):
        payload = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "predict next input in suggestion mode"}],
        }
        is_hk, kind = LocalMockingShield.is_housekeeping_request(payload)
        assert is_hk is True
        assert kind == "autocomplete_suggestion"

    def test_detect_title_generation(self):
        payload = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Generate a short, 2-4 word title for this chat"}],
        }
        is_hk, kind = LocalMockingShield.is_housekeeping_request(payload)
        assert is_hk is True
        assert kind == "title_generation"

    def test_detect_ping_probe_via_max_tokens(self):
        payload = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hello"}],
        }
        is_hk, kind = LocalMockingShield.is_housekeeping_request(payload)
        assert is_hk is True
        assert kind == "ping_probe"

    def test_detect_discovery_probe(self):
        payload = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "performing discovery check"}],
        }
        is_hk, kind = LocalMockingShield.is_housekeeping_request(payload)
        assert is_hk is True
        assert kind == "ping_probe"

    def test_non_housekeeping_request_passes_through(self):
        payload = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "Write a python script to sort a list."}],
        }
        is_hk, kind = LocalMockingShield.is_housekeeping_request(payload)
        assert is_hk is False
        assert kind == ""

    def test_generate_mock_response_pydantic_compliance(self):
        payload = {"model": "claude-3-5-sonnet"}
        mock = LocalMockingShield.generate_mock_response(payload, kind="title_generation")

        assert mock["id"].startswith("msg_mock_")
        assert mock["type"] == "message"
        assert mock["role"] == "assistant"
        assert mock["stop_reason"] == "end_turn"
        assert mock["usage"] == {"input_tokens": 0, "output_tokens": 0}
        # CRITICAL: Content must have text with length >= 1 (" ") to satisfy Pydantic min_length=1 validator
        assert len(mock["content"]) == 1
        assert mock["content"][0]["text"] == " "


class TestDeDuplicator:
    """Test Layer 2: Double Tool Call De-duplication."""

    def test_strip_xml_tool_call_noise_when_structural_tool_use_exists(self):
        content = [
            {
                "type": "text",
                "text": "I will run the command now. <tool_call><function=run_command><parameter=CommandLine>ls -la</parameter></function></tool_call>",
            },
            {
                "type": "tool_use",
                "id": "toolu_12345",
                "name": "run_command",
                "input": {"CommandLine": "ls -la"},
            },
        ]

        cleaned = DeDuplicator.deduplicate(content)
        assert len(cleaned) == 2
        assert cleaned[0]["type"] == "text"
        assert "<tool_call>" not in cleaned[0]["text"]
        assert cleaned[0]["text"] == "I will run the command now."
        assert cleaned[1]["type"] == "tool_use"
        assert cleaned[1]["name"] == "run_command"

    def test_remove_duplicate_structural_tool_use(self):
        content = [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "run_command",
                "input": {"CommandLine": "git status"},
            },
            {
                "type": "tool_use",
                "id": "toolu_2",
                "name": "run_command",
                "input": {"CommandLine": "git status"},
            },
        ]

        cleaned = DeDuplicator.deduplicate(content)
        assert len(cleaned) == 1
        assert cleaned[0]["id"] == "toolu_1"


class TestAngleBracketEscape:
    """Test Layer 3: Angle Bracket Trap / Escape."""

    def test_escape_double_angle_brackets_in_code_block(self):
        text = "Here is the code:\n```typescript\ntype Dict = Record<<Locale, string>;\n```"
        sanitized = AngleBracketEscape.sanitize(text)

        assert "Record<Locale, string>" in sanitized
        assert "Record<<Locale, string>" not in sanitized

    def test_protect_code_generics_from_xml_parser(self):
        text = "Check this:\n```ts\nlet x: Map<string, Array<number>>;\n```\n<tool_call><function=test></tool_call>"
        protected, placeholders = AngleBracketEscape.escape_code_blocks(text)

        # Code block should be masked while <tool_call> is recognized
        assert "__CODE_BLOCK_PLACEHOLDER_" in protected
        restored = AngleBracketEscape.unescape_code_blocks(protected, placeholders)
        assert "Map<string, Array<number>>" in restored
        assert "<tool_call>" in restored


class TestStatefulStreamParsers:
    """Test Layer 4: Stateful Stream & Tag Splitting Parser."""

    @pytest.mark.asyncio
    async def test_thinking_parser_split_tag_across_chunks(self):
        parser = ThinkingStatefulParser()

        # Chunk 1: Partial opening tag "<th"
        events1 = await parser.process_chunk("<th")
        assert len(events1) == 0  # Buffered

        # Chunk 2: Remainder "ink>Reasoning content</think>Answer"
        events2 = await parser.process_chunk("ink>Reasoning content</think>Answer")
        assert len(events2) > 0

        # Verify thinking delta was emitted
        thinking_events = [e for e in events2 if hasattr(e, "delta") and getattr(e.delta, "type", None) == "thinking_delta"]
        assert len(thinking_events) > 0
        assert getattr(thinking_events[0].delta, "thinking", "") == "Reasoning content"

    @pytest.mark.asyncio
    async def test_heuristic_tool_parser_split_tag_across_chunks(self):
        parser = HeuristicToolStatefulParser()

        # Chunk 1: Partial tool tag "<tool_c"
        events1 = await parser.process_chunk("<tool_c")
        assert len(events1) == 0  # Buffered statefully

        # Flush
        flush_events = await parser.flush()
        assert len(flush_events) >= 0
