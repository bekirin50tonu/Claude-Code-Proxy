"""State machine parser for native reasoning_content and split <think>...</think> / <thought>...</thought> tags."""

import re
from typing import Any

from atomic.parsers.base import BaseAtomicParser
from models.converter import ModelConverter
from shared.schemas.anthropic import SSEBaseEvent


class ThinkingParser(BaseAtomicParser):
    """State machine parser for reasoning content and split <think>...</think> tags."""

    def __init__(self, block_index_provider: Any = None):
        self.block_index_provider = block_index_provider
        self.in_think_tag = False
        self.buffer = ""
        self.active_block_type: str | None = None  # "thinking" or "text"
        self.current_block_index: int | None = None

    def reset(self) -> None:
        self.in_think_tag = False
        self.buffer = ""
        self.active_block_type = None
        self.current_block_index = None

    def _get_next_index(self) -> int:
        if self.block_index_provider and callable(self.block_index_provider):
            return self.block_index_provider()
        return 0

    def _ensure_block(self, block_type: str, events: list[SSEBaseEvent]) -> int:
        if self.active_block_type != block_type:
            if self.active_block_type is not None and self.current_block_index is not None:
                events.append(ModelConverter.build_sse_block_stop(self.current_block_index))
            self.current_block_index = self._get_next_index()
            events.append(ModelConverter.build_sse_block_start(self.current_block_index, block_type))
            self.active_block_type = block_type
        return self.current_block_index if self.current_block_index is not None else 0

    def _close_block(self, events: list[SSEBaseEvent]) -> None:
        if self.active_block_type is not None and self.current_block_index is not None:
            events.append(ModelConverter.build_sse_block_stop(self.current_block_index))
            self.active_block_type = None
            self.current_block_index = None

    def close_active_block(self) -> list[SSEBaseEvent]:
        """Close any currently active thinking or text block explicitly."""
        events: list[SSEBaseEvent] = []
        self._close_block(events)
        return events


    async def process_chunk(self, chunk: dict[str, Any] | str) -> list[SSEBaseEvent]:
        events: list[SSEBaseEvent] = []

        # Handle native reasoning_content field in OpenAI deltas
        if isinstance(chunk, dict):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            reasoning = delta.get("reasoning_content") or ""
            if reasoning:
                idx = self._ensure_block("thinking", events)
                events.append(ModelConverter.build_sse_block_delta(idx, "thinking_delta", reasoning))
                return events

            content = delta.get("content") or ""
            text = content
        else:
            text = chunk

        if not text:
            return events

        self.buffer += text

        # Clean malformed tags like "<think</think" or "<think></think>"
        if self.buffer.startswith("<think</think"):
            self.buffer = self.buffer[len("<think</think") :]

        while self.buffer:
            if not self.in_think_tag:
                # Look for opening <think> or <thought> tag
                match = re.search(r"<(think|thought)>", self.buffer, re.IGNORECASE)
                if match:
                    think_start = match.start()
                    think_end_tag = match.end()
                    prefix = self.buffer[:think_start]
                    if prefix:
                        idx = self._ensure_block("text", events)
                        events.append(ModelConverter.build_sse_block_delta(idx, "text_delta", prefix))

                    # Switch to thinking
                    idx = self._ensure_block("thinking", events)
                    self.in_think_tag = True
                    self.buffer = self.buffer[think_end_tag:]
                    continue

                # Check for partial opening tag at the end of buffer (e.g. "<th", "<think")
                partial_idx = self.buffer.rfind("<")
                if partial_idx != -1 and any(["<think>".startswith(self.buffer[partial_idx:].lower()), "<thought>".startswith(self.buffer[partial_idx:].lower())]):
                    text_to_flush = self.buffer[:partial_idx]
                    self.buffer = self.buffer[partial_idx:]
                else:
                    text_to_flush = self.buffer
                    self.buffer = ""

                if text_to_flush:
                    idx = self._ensure_block("text", events)
                    events.append(ModelConverter.build_sse_block_delta(idx, "text_delta", text_to_flush))
                break

            else:
                # In think tag: look for closing </think> or </thought>
                match = re.search(r"</(think|thought)>", self.buffer, re.IGNORECASE)
                if match:
                    think_end_start = match.start()
                    think_end_finish = match.end()
                    thinking_text = self.buffer[:think_end_start]
                    if thinking_text:
                        idx = self._ensure_block("thinking", events)
                        events.append(ModelConverter.build_sse_block_delta(idx, "thinking_delta", thinking_text))

                    self._close_block(events)
                    self.in_think_tag = False
                    self.buffer = self.buffer[think_end_finish:]
                    continue

                # Check for partial closing tag at end of buffer (e.g. "</th", "</think")
                partial_idx = self.buffer.rfind("<")
                if partial_idx != -1 and any(["</think>".startswith(self.buffer[partial_idx:].lower()), "</thought>".startswith(self.buffer[partial_idx:].lower())]):
                    thinking_to_flush = self.buffer[:partial_idx]
                    self.buffer = self.buffer[partial_idx:]
                else:
                    thinking_to_flush = self.buffer
                    self.buffer = ""


                if thinking_to_flush:
                    idx = self._ensure_block("thinking", events)
                    events.append(ModelConverter.build_sse_block_delta(idx, "thinking_delta", thinking_to_flush))
                break

        return events

    async def flush(self) -> list[SSEBaseEvent]:
        events: list[SSEBaseEvent] = []
        if self.buffer:
            block_type = "thinking" if self.in_think_tag else "text"
            delta_type = "thinking_delta" if self.in_think_tag else "text_delta"
            idx = self._ensure_block(block_type, events)
            events.append(ModelConverter.build_sse_block_delta(idx, delta_type, self.buffer))
            self.buffer = ""
        self._close_block(events)
        return events


ThinkingStatefulParser = ThinkingParser

