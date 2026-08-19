"""FileEditGuard — auto-healing file editing tool calls to eliminate 'Error editing file' failures in CLI sessions."""

import os
from pathlib import Path
from typing import Any

from loguru import logger

from atomic.parsers.base import BaseAtomicParser
from shared.schemas.anthropic import SSEBaseEvent


class FileEditGuard(BaseAtomicParser):
    """Guard auto-correcting StartLine, EndLine, line endings, and target content matching for file edits."""

    EDIT_TOOL_NAMES = {"replace_file_content", "multi_replace_file_content", "Edit", "Update", "replace"}

    def reset(self) -> None:
        pass


    def _heal_single_edit(
        self,
        file_path: str,
        file_text: str,
        file_lines: list[str],
        target_content: str,
        start_line: int | None,
        end_line: int | None,
    ) -> tuple[str, int, int]:
        """Auto-heal target_content, start_line, and end_line for a single edit block."""
        total_lines = len(file_lines)
        if total_lines == 0:
            return target_content, start_line or 1, end_line or 1

        # 1. Normalize newlines
        file_ending = "\r\n" if "\r\n" in file_text else "\n"
        norm_target = target_content.replace("\r\n", "\n").replace("\r", "\n")
        if file_ending == "\r\n":
            norm_target = norm_target.replace("\n", "\r\n")

        # 2. Check exact match in file
        if norm_target in file_text:
            # Find line numbers of exact match
            match_index = file_text.find(norm_target)
            line_start = file_text[:match_index].count("\n") + 1
            line_end = line_start + norm_target.count("\n")

            # Expand bounds to safety range (e.g. 10 lines padding or file bounds)
            new_start = max(1, line_start - 10)
            new_end = min(total_lines, line_end + 10)
            return norm_target, new_start, new_end

        # 3. Fuzzy match by line-by-line whitespace-trimmed search
        target_split = [line.strip() for line in norm_target.splitlines() if line.strip()]
        if not target_split:
            return norm_target, 1, total_lines

        first_target_line = target_split[0]
        matching_start_lines: list[int] = []

        for idx, line in enumerate(file_lines):
            if first_target_line in line or line.strip() == first_target_line:
                matching_start_lines.append(idx + 1)

        if matching_start_lines:
            best_start = matching_start_lines[0]
            num_target_lines = len(norm_target.splitlines())
            best_end = min(total_lines, best_start + num_target_lines + 5)

            # Extract exact substring from file if line count matches
            slice_lines = file_lines[best_start - 1 : min(total_lines, best_start + num_target_lines)]
            extracted_target = file_ending.join(slice_lines)

            logger.info(
                "FileEditGuard: Fuzzy matched target content in '{}' at lines {}-{}",
                file_path,
                best_start,
                best_end,
            )
            return extracted_target, max(1, best_start - 5), min(total_lines, best_end + 5)

        # 4. Fallback: expand line bounds to full file bounds so Claude Code searches the entire file
        logger.info(
            "FileEditGuard: Expanding line bounds to full file [1, {}] for '{}'",
            total_lines,
            file_path,
        )
        return norm_target, 1, total_lines

    def sanitize_tool_input(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Inspect and auto-heal file editing tool call parameters."""
        if tool_name not in self.EDIT_TOOL_NAMES or not isinstance(tool_input, dict):
            return tool_input

        file_path = tool_input.get("TargetFile") or tool_input.get("file_path") or tool_input.get("path")
        if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
            return tool_input

        try:
            p = Path(file_path)
            if not p.is_file():
                return tool_input

            file_text = p.read_text(encoding="utf-8", errors="replace")
            file_lines = file_text.splitlines()

            # Handle replace_file_content / Edit / Update
            if tool_name in ("replace_file_content", "Edit", "Update", "replace"):
                target_content = tool_input.get("TargetContent")
                if isinstance(target_content, str):
                    start_line = tool_input.get("StartLine")
                    end_line = tool_input.get("EndLine")
                    start_val = int(start_line) if isinstance(start_line, (int, str)) and str(start_line).isdigit() else None
                    end_val = int(end_line) if isinstance(end_line, (int, str)) and str(end_line).isdigit() else None

                    healed_target, healed_start, healed_end = self._heal_single_edit(
                        file_path, file_text, file_lines, target_content, start_val, end_val
                    )
                    tool_input["TargetContent"] = healed_target
                    tool_input["StartLine"] = healed_start
                    tool_input["EndLine"] = healed_end

            # Handle multi_replace_file_content
            elif tool_name == "multi_replace_file_content":
                chunks = tool_input.get("ReplacementChunks")
                if isinstance(chunks, list):
                    for chunk in chunks:
                        if isinstance(chunk, dict) and "TargetContent" in chunk:
                            target_c = chunk["TargetContent"]
                            if isinstance(target_c, str):
                                s_val = chunk.get("StartLine")
                                e_val = chunk.get("EndLine")
                                s_int = int(s_val) if isinstance(s_val, (int, str)) and str(s_val).isdigit() else None
                                e_int = int(e_val) if isinstance(e_val, (int, str)) and str(e_val).isdigit() else None

                                h_target, h_start, h_end = self._heal_single_edit(
                                    file_path, file_text, file_lines, target_c, s_int, e_int
                                )
                                chunk["TargetContent"] = h_target
                                chunk["StartLine"] = h_start
                                chunk["EndLine"] = h_end

        except Exception as exc:
            logger.warning("FileEditGuard: Error while auto-healing file edit tool call: %s", exc)

        return tool_input

    async def process_chunk(self, chunk: dict[str, Any] | str) -> list[SSEBaseEvent]:
        return []

    async def flush(self) -> list[SSEBaseEvent]:
        return []


file_edit_guard = FileEditGuard()
