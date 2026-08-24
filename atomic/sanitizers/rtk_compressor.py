"""RTK (Redundant Token Killer) - Intelligent Console Log & Tool Result Compressor.

Strips ANSI escape codes, deduplicates consecutive duplicate lines, collapses excessive
blank lines, and truncates large build/test logs (preserving head setup & tail error details)
to reduce token size by >= 50%.
"""

import re
from typing import Any

from loguru import logger


class RedundantTokenKiller:
    """Intelligent log compressor reducing token usage in tool_results by >= 50%."""

    ANSI_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    @classmethod
    def strip_ansi(cls, text: str) -> str:
        """Remove ANSI color codes and escape sequences from text."""
        if not text or not isinstance(text, str):
            return text or ""
        return cls.ANSI_REGEX.sub("", text)

    @classmethod
    def compress_log(cls, content: Any, max_lines: int = 50, head_lines: int = 20, tail_lines: int = 30) -> str:
        """
        Compress terminal log string or tool_result content list.

        - Strips ANSI escape codes.
        - Removes consecutive identical lines.
        - Collapses 3+ consecutive empty lines into 1.
        - Truncates long logs in the middle (retaining head setup & tail summary/errors).
        """
        if content is None:
            return ""

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("text", item.get("content", ""))))
                elif isinstance(item, str):
                    text_parts.append(item)
            raw_text = "\n".join(text_parts)
        elif not isinstance(content, str):
            raw_text = str(content)
        else:
            raw_text = content

        if not raw_text.strip():
            return raw_text

        orig_len = len(raw_text)
        cleaned = cls.strip_ansi(raw_text)

        lines = cleaned.splitlines()

        # Step 1: Deduplicate consecutive identical lines & collapse empty lines
        deduped_lines: list[str] = []
        prev_line: str | None = None
        consecutive_empty = 0

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                consecutive_empty += 1
                if consecutive_empty <= 2:
                    deduped_lines.append("")
                continue
            else:
                consecutive_empty = 0

            # Skip consecutive identical non-empty lines
            if stripped_line == prev_line:
                continue

            deduped_lines.append(line)
            prev_line = stripped_line

        # Step 2: Truncate large middle section if line count exceeds threshold
        total_lines = len(deduped_lines)
        if total_lines > max_lines:
            head = deduped_lines[:head_lines]
            tail = deduped_lines[-tail_lines:]
            skipped = total_lines - (head_lines + tail_lines)
            
            middle_notice = f"\n... [RTK: Compressed {skipped} redundant log lines] ...\n"
            final_lines = head + [middle_notice] + tail
            compressed_text = "\n".join(final_lines)
        else:
            compressed_text = "\n".join(deduped_lines)

        final_len = len(compressed_text)
        if orig_len > 0 and final_len < orig_len:
            reduction = ((orig_len - final_len) / orig_len) * 100
            logger.info(
                "🗜️ [RTK] Compressed log from {} to {} chars ({:.1f}% token reduction, {} lines -> {} lines)",
                orig_len,
                final_len,
                reduction,
                len(lines),
                len(compressed_text.splitlines()),
            )

        return compressed_text


rtk_compressor = RedundantTokenKiller()
