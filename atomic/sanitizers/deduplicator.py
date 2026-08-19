"""Double Tool Call De-duplication Sanitizer.

Prevents Claude Code CLI locks caused by upstream LLMs (NVIDIA NIM, Gemini)
outputting both embedded XML <tool_call> tags in text content AND official
structural tool_use JSON blocks in the same response.
"""

import json
import re
from typing import Any

from loguru import logger


class DeDuplicator:
    """Sanitizer for deduplicating tool calls and stripping XML noise when structural tool_use exists."""

    XML_TOOL_CALL_REGEX = re.compile(
        r"(<tool_call>[\s\S]*?</tool_call>|<<?function=[\s\S]*?(?:</function>>?|\Z)|\[TOOL_CALLS?\][\s\S]*?(?:\[/TOOL_CALLS?\]|\Z))",
        re.IGNORECASE,
    )

    @classmethod
    def clean_xml_tool_calls_from_text(cls, text: str) -> tuple[str, bool]:
        """Strip XML/Hermes tool_call tags from text. Returns (cleaned_text, was_modified)."""
        if not text or not isinstance(text, str):
            return text, False

        cleaned, count = cls.XML_TOOL_CALL_REGEX.subn("", text)
        cleaned = cleaned.strip()
        return cleaned, count > 0

    @classmethod
    def deduplicate(cls, content_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Inspect content blocks. If structural tool_use blocks exist, strip XML <tool_call>
        remnants from text blocks. Also remove duplicate tool_use blocks with identical name and input.
        """
        if not isinstance(content_list, list):
            return content_list

        has_structural_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in content_list
        )

        seen_tool_signatures: set[str] = set()
        new_content: list[dict[str, Any]] = []
        modified_count = 0

        for block in content_list:
            if not isinstance(block, dict):
                new_content.append(block)
                continue

            btype = block.get("type")

            if btype == "text":
                text_val = block.get("text", "")
                if isinstance(text_val, str) and (
                    has_structural_tool_use or "<tool_call>" in text_val or "function=" in text_val or "[TOOL_CALL" in text_val
                ):
                    cleaned_text, stripped = cls.clean_xml_tool_calls_from_text(text_val)
                    if stripped:
                        modified_count += 1
                        logger.info(
                            "🧼 \033[1;36m[DeDuplicator]\033[0m Detected XML <tool_call> noise in text block alongside structural tool_use. "
                            "Stripped XML noise from text content."
                        )
                    block_copy = dict(block)
                    block_copy["text"] = cleaned_text if cleaned_text else " "
                    new_content.append(block_copy)
                else:
                    new_content.append(block)

            elif btype == "tool_use":
                tname = block.get("name", "")
                tinput = json.dumps(block.get("input", {}), sort_keys=True)
                sig = f"{tname}:{tinput}"

                if sig not in seen_tool_signatures:
                    seen_tool_signatures.add(sig)
                    new_content.append(block)
                else:
                    modified_count += 1
                    logger.info(
                        "🧼 \033[1;36m[DeDuplicator]\033[0m Stripped duplicate structural tool_use block for tool '{}'.",
                        tname,
                    )
            else:
                new_content.append(block)

        return new_content


deduplicator = DeDuplicator()
