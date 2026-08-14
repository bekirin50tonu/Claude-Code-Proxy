"""English text and embed formatters for Telegram and Discord bot platforms."""

import re
from typing import Any

from config import model_registry, settings
from core.router.circuit_breaker import circuit_breaker_registry


def escape_markdown_v2(text: str, is_code_block: bool = False) -> str:
    """Escape special characters for Telegram MarkdownV2 format."""
    if not text:
        return ""

    if is_code_block:
        return text.replace("\\", "\\\\").replace("`", "\\`")

    reserved = r"([_*\[\]()~`>#+\-=|{}.!\\])"
    return re.sub(reserved, r"\\\1", str(text))


def format_circuit_breaker_alert_tg(
    model_id: str,
    reason: str = "Upstream error / timeout",
    fallback_model: str | None = None,
) -> str:
    """Generate English Markdown V2 proactive Circuit Breaker alert message for Telegram."""
    esc_model = escape_markdown_v2(model_id, is_code_block=True)
    esc_reason = escape_markdown_v2(reason, is_code_block=True)

    if not fallback_model:
        fallbacks = model_registry.get_fallbacks(model_id)
        fallback_model = fallbacks[0] if fallbacks else "None (Automatic Failover Active)"
    esc_fallback = escape_markdown_v2(fallback_model, is_code_block=True)

    msg = (
        "🚨 *Circuit Breaker Tripped\\!*\n\n"
        f"🔌 *Provider/Model:* `{esc_model}`\n"
        "📊 *Status:* `OPEN` \\(Requests Blocked\\)\n"
        f"⏱️ *Fallback Route:* Traffic automatically routed to `{esc_fallback}`\\.\n"
        f"📝 *Reason:* `{esc_reason}`\n\n"
        "💡 _Click the button below to reset the circuit breaker back to CLOSED\\._"
    )
    return msg


def format_circuit_breaker_alert_discord(
    model_id: str,
    reason: str = "Upstream error / timeout",
    fallback_model: str | None = None,
) -> dict[str, Any]:
    """Generate English Discord Embed payload for Circuit Breaker trip notification."""
    if not fallback_model:
        fallbacks = model_registry.get_fallbacks(model_id)
        fallback_model = fallbacks[0] if fallbacks else "None (Automatic Failover Active)"

    return {
        "title": "🚨 Circuit Breaker Tripped!",
        "description": "An upstream provider or model experienced errors and was isolated by Circuit Breaker.",
        "color": 0xE74C3C,  # Red color
        "fields": [
            {"name": "🔌 Provider / Model", "value": f"`{model_id}`", "inline": False},
            {"name": "📊 Circuit Status", "value": "`OPEN` (Requests Blocked)", "inline": True},
            {"name": "⏱️ Fallback Route", "value": f"`{fallback_model}`", "inline": True},
            {"name": "📝 Failure Reason", "value": f"`{reason}`", "inline": False},
        ],
        "footer": {"text": "Claude Code Proxy Gateway • Resiliency System"},
    }


def format_status_overview_tg() -> str:
    """Generate English Telegram Markdown V2 status overview of proxy settings and circuit breakers."""
    opus = escape_markdown_v2(settings.MODEL_OPUS, is_code_block=True)
    sonnet = escape_markdown_v2(settings.MODEL_SONNET, is_code_block=True)
    haiku = escape_markdown_v2(settings.MODEL_HAIKU, is_code_block=True)
    default_m = escape_markdown_v2(settings.MODEL, is_code_block=True)

    rate_limit = settings.PROVIDER_RATE_LIMIT
    rate_win = settings.PROVIDER_RATE_WINDOW
    concurrency = settings.PROVIDER_MAX_CONCURRENCY

    statuses = circuit_breaker_registry.all_statuses()
    cb_lines = []

    if not statuses:
        cb_lines.append("• _No active circuit breaker records yet\\._")
    else:
        for mid, st in statuses.items():
            state = str(st.get("state", "closed")).upper()
            icon = "🟢" if state == "CLOSED" else ("🔴" if state == "OPEN" else "🟡")
            esc_mid = escape_markdown_v2(mid, is_code_block=True)
            esc_state = escape_markdown_v2(state)
            cb_lines.append(f"• {icon} `{esc_mid}`: *{esc_state}*")

    cb_summary = "\n".join(cb_lines)

    text = (
        "⚙️ *Claude Code Proxy Gateway Status*\n\n"
        "🎯 *Model Mappings:*\n"
        f"• *OPUS:* `{opus}`\n"
        f"• *SONNET:* `{sonnet}`\n"
        f"• *HAIKU:* `{haiku}`\n"
        f"• *DEFAULT:* `{default_m}`\n\n"
        "⚡ *Rate & Concurrency Controls:*\n"
        f"• *Rate Limit:* `{rate_limit}` req / `{rate_win}`s\n"
        f"• *Max Concurrency:* `{concurrency}`\n\n"
        "🛡️ *Circuit Breaker States:*\n"
        f"{cb_summary}"
    )
    return text


def format_status_overview_discord() -> str:
    """Generate English Discord status overview message."""
    statuses = circuit_breaker_registry.all_statuses()
    cb_lines = []
    if not statuses:
        cb_lines.append("• *No active circuit breaker records yet.*")
    else:
        for mid, st in statuses.items():
            state = str(st.get("state", "closed")).upper()
            icon = "🟢" if state == "CLOSED" else ("🔴" if state == "OPEN" else "🟡")
            cb_lines.append(f"• {icon} `{mid}`: **{state}**")

    cb_summary = "\n".join(cb_lines)

    return (
        "⚙️ **Claude Code Proxy Gateway Status**\n\n"
        "🎯 **Model Mappings:**\n"
        f"• **OPUS:** `{settings.MODEL_OPUS}`\n"
        f"• **SONNET:** `{settings.MODEL_SONNET}`\n"
        f"• **HAIKU:** `{settings.MODEL_HAIKU}`\n"
        f"• **DEFAULT:** `{settings.MODEL}`\n\n"
        "⚡ **Rate & Concurrency Controls:**\n"
        f"• **Rate Limit:** `{settings.PROVIDER_RATE_LIMIT}` req / `{settings.PROVIDER_RATE_WINDOW}`s\n"
        f"• **Max Concurrency:** `{settings.PROVIDER_MAX_CONCURRENCY}`\n\n"
        "🛡️ **Circuit Breaker States:**\n"
        f"{cb_summary}"
    )
