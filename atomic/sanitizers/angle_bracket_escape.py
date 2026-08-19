"""Angle Bracket Trap & Code Block Escape Module.

Protects TypeScript/C++/Java generic types (e.g. Record<Locale, string>, Record<<Locale, string>)
inside code blocks from confusing XML tag parsers and causing JSON validation failures.
"""

import re

from loguru import logger


class AngleBracketEscape:
    """Sanitizer and mask provider for angle brackets inside code blocks."""

    # Matches code blocks (```lang ... ``` or ``` ... ```) and inline code (`...`)
    CODEBLOCK_REGEX = re.compile(
        r"(```[a-zA-Z0-9_\-]*\n[\s\S]*?\n```|`[^`\n]+`)",
        re.MULTILINE,
    )

    VALID_XML_TAG_REGEX = re.compile(
        r"(</?tool_call>|<<?function=[^>]+>>?|<<?parameter=[^>]+>>?|</parameter>>?|</function>>?|</?think>|</?thought>)",
        re.IGNORECASE,
    )

    @classmethod
    def escape_code_blocks(cls, text: str) -> tuple[str, dict[str, str]]:
        """
        Mask code blocks and protect internal angle brackets so XML/regex parsers
        are not confused by generics like Record<K, V> or Record<<K, V>>.
        Returns (protected_text, placeholder_map).
        """
        if not text or not isinstance(text, str):
            return text, {}

        placeholders: dict[str, str] = {}
        idx = 0
        protected_count = 0

        def mask_codeblock(match: re.Match[str]) -> str:
            nonlocal idx, protected_count
            code = match.group(0)
            key = f"__CODE_BLOCK_PLACEHOLDER_{idx}__"
            idx += 1

            # Fix accidental double opening angle brackets in generics (e.g. Record<<Locale -> Record<Locale)
            fixed_code = re.sub(r"([A-Za-z0-9_]+)<<\s*([A-Z][A-Za-z0-9_]*)", r"\1<\2", code)

            if fixed_code != code:
                protected_count += 1

            placeholders[key] = fixed_code
            return key

        protected_text = cls.CODEBLOCK_REGEX.sub(mask_codeblock, text)

        # Outside code blocks, protect valid XML/Hermes tags
        def mask_valid_xml(match: re.Match[str]) -> str:
            nonlocal idx
            k = f"__XML_TAG_PLACEHOLDER_{idx}__"
            idx += 1
            placeholders[k] = match.group(0)
            return k

        protected_text = cls.VALID_XML_TAG_REGEX.sub(mask_valid_xml, protected_text)

        if protected_count > 0:
            logger.info(
                "🩹 \033[1;35m[AngleBracketEscape]\033[0m Protected code block angle brackets and fixed stray '<<' traps."
            )

        return protected_text, placeholders

    @classmethod
    def unescape_code_blocks(cls, text: str, placeholders: dict[str, str]) -> str:
        """Restore masked code blocks and valid XML tags."""
        if not text or not placeholders:
            return text

        restored = text
        for key, original in placeholders.items():
            restored = restored.replace(key, original)
        return restored

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Single-pass angle bracket protection and cleanup."""
        protected, placeholders = cls.escape_code_blocks(text)
        return cls.unescape_code_blocks(protected, placeholders)


angle_bracket_escape = AngleBracketEscape()
