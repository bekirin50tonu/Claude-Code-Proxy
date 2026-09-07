"""Language detection + system-prompt injection.

Injects a one-line instruction into the request's system prompt so the model
mirrors the user's language. The first user message is scanned for the
strongest hint (Turkish, English, German, French, Spanish, Italian,
Portuguese, Russian, Arabic, Chinese, Japanese, Korean); an existing system
prompt is preserved and the new line is appended.
"""

import re

# Weighted per-character markers (no need for full Unicode tables here).
# ponytail: very narrow heuristic, add a proper langdetect later if needed.
_LANG_MARKERS = {
    # Turkish: ONLY Turkish-unique letters (ğ, ş, ç, ı, İ) and the most
    # distinctive words. NOTE: avoid plain 'I' (U+0049) which would also
    # match Latin 'i' used in English/German/Spanish.
    "tr": r"[ğşçıĞŞÇİ]|\b(bir|bu|için|ile|ama|şu|çok|şimdi|nasıl|neden|ancak|aslında|herhangi|merhaba|selam|türkçe|teşekkür|sağol)\b",
    # English: distinctive words.
    "en": r"\b(the|are|how|why|hello|please|thanks|hi|hey|okay|write)\b",
    # German: umlauts (ü/ö/ä) and unambiguous words.
    "de": r"[äöüßÄÖÜ]|\b(und|nicht|für|warum|guten|schreibe|gedicht|morgen|abend)\b",
    "fr": r"[àâçéèêëîïôûùüÿÀÂÇÉÈÊËÎÏÔÛÙÜŸ]|\b(avec|mais|comment|pourquoi|bonjour|merci|écrivez)\b",
    "es": r"[áéíóúñ¿¡ÁÉÍÓÚÑ]|\b(pero|como|hola|gracias|buenos)\b",
    "it": r"[àèéìòóùÀÈÉÌÒÓÙ]|\b(ciao|grazie|buongiorno)\b",
    "pt": r"[ãõáéíóúçÃÕÁÉÍÓÚÇ]|\b(como|olá|obrigado)\b",
    "ru": r"[а-яА-ЯёЁ]",
    "ar": r"[\u0600-\u06FF]",
    "zh": r"[\u4e00-\u9fff]",
    "ja": r"[\u3040-\u30ff\u31f0-\u31ff]",
    "ko": r"[\uac00-\ud7af]",
}

_UNIQUE_CHARS = {
    "tr": r"[ğşçıĞŞÇI]",
    "de": r"[äöüßÄÖÜ]",
    "fr": r"[àâçéèêëîïôûùüÿÀÂÇÉÈÊËÎÏÔÛÙÜŸ]",
    "es": r"[ñ¿¡Ñ]",
    "it": r"[àèéìòóùÀÈÉÌÒÓÙ]",
    "pt": r"[ãõÃÕ]",
}

_LANG_NAME = {
    "tr": "Turkish",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
}

# Common programming tokens/identifiers to ignore before language scoring.
_NON_NATURAL = re.compile(
    r"^\s*(```|/{2,}|\$ |curl\s|import\s|from\s|def\s|class\s|function\s|var\s|let\s|const\s|return\s|if\s|else\s|for\s|while\s)",
    re.IGNORECASE | re.MULTILINE,
)

# Default: if no marker matches, don't inject anything.
_FALLBACK_LANG = "en"


def detect_language(messages: list[dict]) -> str:
    """Pick a language code from the most recent user message text."""
    if not messages:
        return _FALLBACK_LANG
    # Walk from the end to find the most recent 'user' turn
    user_text_parts: list[str] = []
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            user_text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    user_text_parts.append(block.get("text", ""))
        break
    text = "\n".join(user_text_parts)
    if not text or _NON_NATURAL.match(text):
        return _FALLBACK_LANG

    # Tie-breaker: each language may declare a *unique* character class whose
    # matches outvote the keyword-based score (e.g. Turkish `ğ/ş/ç/ı` is not
    # found in any other language; German `ß` is uniquely German).

    scores: dict[str, int] = {}
    lowered = text.lower()
    for code, pattern in _LANG_MARKERS.items():
        # Do NOT use re.IGNORECASE: Python's case folding for Turkish dotted
        # I (U+0130) matches Latin 'i' (U+0069), which inflates the TR score
        # for non-Turkish text. Patterns are written with both upper and
        # lower forms so we can rely on .lower() alone.
        if re.search(pattern, lowered):
            scores[code] = sum(1 for m in re.finditer(pattern, lowered) if m.group())
    if not scores:
        return _FALLBACK_LANG

    # Boost languages that have a unique-character match.
    best = max(scores.values())
    unique_hits = {
        code: sum(1 for m in re.finditer(unique, lowered) if m.group())
        for code, unique in _UNIQUE_CHARS.items()
        if code in scores
    }
    # If a language has ANY unique character hit, it owns this text. Even a
    # single 'ş' or 'ñ' is a strong enough signal to override keyword ties
    # between languages that share other characters (e.g. Turkish 'ü' and
    # German 'ü' are the same code point).
    for code, hits in unique_hits.items():
        if hits:
            scores[code] += 1000  # ponytail: hard tie-breaker
    best = max(scores.values())
    # Return the highest-scoring language; ties broken by declaration order
    # in _LANG_MARKERS (English first, then regional languages).
    for code in _LANG_MARKERS:  # preserves declaration order in Python 3.7+
        if scores.get(code) == best:
            return code
    return _FALLBACK_LANG


def inject_language_directive(body: dict) -> None:
    """Append a one-line language directive to the request's system prompt.

    Mutates `body` in place. No-op if language is English (the model's default)
    or if the system prompt already contains a language directive.
    """
    messages = body.get("messages") or []
    lang = detect_language(messages)
    if lang == "en":
        return

    directive = f"Reply in {_LANG_NAME.get(lang, lang)}."

    system = body.get("system")
    if system is None:
        body["system"] = directive
        return
    if isinstance(system, str):
        if directive in system:
            return
        body["system"] = f"{system.rstrip()}\n\n{directive}"
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and directive in str(block.get("text", "")):
                return
        body["system"] = list(system) + [{"type": "text", "text": directive}]
