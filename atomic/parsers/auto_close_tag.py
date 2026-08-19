"""Asynchronous Tag & JSON Auto-Closer for Truncated LLM Streams.

Fixes EOF stream cutoffs where alternative upstream models (NVIDIA NIM, Gemini)
leave open quotes, unclosed parameter/function/tool_call tags, or unclosed
JSON brackets, preventing Claude CLI 'JSON validation failed' Stop hook crashes.
"""

import re

from loguru import logger


class AutoCloseTagParser:
    """Asynchronous parser and sanitizer to repair truncated streams, tags, and JSON structures."""

    @classmethod
    def auto_close_quotes(cls, text: str) -> str:
        """Close an unclosed double quote at EOF if an odd number of unescaped quotes exist."""
        if not text or not isinstance(text, str):
            return text

        # Count unescaped double quotes
        quote_count = 0
        escaped = False
        for char in text:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                quote_count += 1

        if quote_count % 2 != 0:
            return text + '"'
        return text

    @classmethod
    def auto_close_xml_tags(cls, text: str) -> str:
        """
        Auto-close unclosed Hermes/XML tags (<<parameter=..., <<function=..., <think>, <tool_call>).
        Ensures hierarchical closing order: </parameter> -> </function> -> </tool_call>.
        """
        if not text or not isinstance(text, str):
            return text

        suffix_tags: list[str] = []

        # 1. Check open <think>
        think_open = len(re.findall(r"<think>", text, re.IGNORECASE))
        think_close = len(re.findall(r"</think>", text, re.IGNORECASE))
        if think_open > think_close:
            suffix_tags.append("</think>")

        # 2. Check open <<parameter=... or <parameter
        param_open = len(re.findall(r"<<?parameter=[^>]*>>?|<parameter>", text, re.IGNORECASE))
        param_close = len(re.findall(r"</parameter>>?", text, re.IGNORECASE))
        if param_open > param_close:
            suffix_tags.append("</parameter>")

        # 3. Check open <<function=... or <function
        func_open = len(re.findall(r"<<?function=[^>]*>>?|<function>", text, re.IGNORECASE))
        func_close = len(re.findall(r"</function>>?", text, re.IGNORECASE))
        if func_open > func_close:
            suffix_tags.append("</function>")

        # 4. Check open <tool_call>
        tool_open = len(re.findall(r"<tool_call>", text, re.IGNORECASE))
        tool_close = len(re.findall(r"</tool_call>", text, re.IGNORECASE))
        if tool_open > tool_close:
            suffix_tags.append("</tool_call>")

        if suffix_tags:
            added = "".join(suffix_tags)
            stripped_text = text.rstrip()
            if stripped_text.endswith("</tool_call>"):
                prefix_text = stripped_text[:-12].rstrip()
                trailing_ws = text[len(stripped_text):]
                return prefix_text + "\n" + added + "\n</tool_call>" + trailing_ws
            return text + added
        return text

    @classmethod
    def auto_close_json_brackets(cls, text: str) -> str:
        """
        Scan text for unclosed JSON brackets ('{' and '[') outside string literals
        and close them in LIFO (last-in, first-out) order.
        """
        if not text or not isinstance(text, str):
            return text

        stack: list[str] = []
        in_string = False
        escaped = False

        for char in text:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char in ("{", "["):
                    stack.append(char)
                elif char == "}" and stack and stack[-1] == "{" or char == "]" and stack and stack[-1] == "[":
                    stack.pop()

        if not stack:
            return text

        # Close unclosed brackets in LIFO order
        closers = []
        for open_bracket in reversed(stack):
            if open_bracket == "{":
                closers.append("}")
            elif open_bracket == "[":
                closers.append("]")

        return text + "".join(closers)

    @classmethod
    def repair_truncated_stream(cls, text: str) -> str:
        """
        Full async stream repair pipeline:
        1. Auto-close unclosed string quotes
        2. Auto-close unclosed XML/Hermes tags (</parameter>, </function>, </tool_call>, </think>)
        3. Auto-close unclosed JSON brackets ([ and {)
        """
        if not text or not isinstance(text, str):
            return text

        original = text

        # Step 1: Auto-close unclosed string quote
        repaired = cls.auto_close_quotes(original)

        # Step 2: Auto-close unclosed JSON brackets
        repaired = cls.auto_close_json_brackets(repaired)

        # Step 3: Auto-close unclosed XML/Hermes tags
        repaired = cls.auto_close_xml_tags(repaired)

        if repaired != original:
            added = repaired[len(original) :]
            logger.info(
                "🩹 \033[1;33m[AutoCloseTagParser]\033[0m Repaired truncated stream cutoff. Added suffix: {}",
                repr(added),
            )

        return repaired


auto_close_tag_parser = AutoCloseTagParser()
