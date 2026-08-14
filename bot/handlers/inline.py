"""Hermes Agent Inline Query Autocomplete Handler for Telegram command previews."""

from uuid import uuid4

from loguru import logger
from telegram import InlineQueryResultArticle, InputTextMessageContent, Update
from telegram.ext import ContextTypes

from core.router.circuit_breaker import circuit_breaker_registry

COMMON_RUN_COMMANDS = [
    ("pnpm dev", "Start local development server"),
    ("pnpm build", "Build project production bundle"),
    ("git status", "View git repository working directory status"),
    ("pnpm test", "Execute project frontend/backend unit tests"),
    ("pytest tests/", "Run Claude Code Proxy Gateway test suite"),
]


async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram inline queries (@bot /command) for Hermes Agent command previews."""
    query_obj = update.inline_query
    if not query_obj:
        return

    raw_query = (query_obj.query or "").strip().lower()
    results: list[InlineQueryResultArticle] = []

    # 1. Autocomplete for /run command
    if raw_query.startswith("/run") or raw_query.startswith("run"):
        sub = raw_query.replace("/run", "").replace("run", "").strip()
        for cmd_str, desc in COMMON_RUN_COMMANDS:
            if not sub or sub in cmd_str.lower():
                results.append(
                    InlineQueryResultArticle(
                        id=str(uuid4()),
                        title=f"⚡ /run {cmd_str}",
                        description=desc,
                        input_message_content=InputTextMessageContent(f"/run {cmd_str}"),
                    )
                )

    # 2. Autocomplete for /reset_circuit command with live breaker status
    elif raw_query.startswith("/reset_circuit") or raw_query.startswith("reset_circuit"):
        statuses = circuit_breaker_registry.all_statuses()
        if not statuses:
            results.append(
                InlineQueryResultArticle(
                    id=str(uuid4()),
                    title="🔌 Reset Circuit Breaker",
                    description="No active circuit breaker records. Usage: /reset_circuit <model_id>",
                    input_message_content=InputTextMessageContent("/reset_circuit nvidia_nim"),
                )
            )
        else:
            for mid, st in statuses.items():
                state = str(st.get("state", "closed")).upper()
                icon = "🟢 CLOSED (Active)" if state == "CLOSED" else "🔴 OPEN (Blocked)"
                reason = str(st.get("last_failure_reason", "None"))
                results.append(
                    InlineQueryResultArticle(
                        id=str(uuid4()),
                        title=f"🔌 /reset_circuit {mid}",
                        description=f"Status: {icon} | Reason: {reason}",
                        input_message_content=InputTextMessageContent(f"/reset_circuit {mid}"),
                    )
                )

    # 3. General Command List Previews
    else:
        commands_info = [
            ("/ask", "Ask AI (Claude Code) a prompt/coding question", "/ask Debug the error in Blog.tsx"),
            ("/status", "View Proxy configuration and Circuit Breaker states", "/status"),
            ("/live on", "Enable real-time reasoning and tool call stream tracking", "/live on"),
            ("/live off", "Disable real-time stream tracking", "/live off"),
            ("/reset_circuit", "Reset a Circuit Breaker back to CLOSED state", "/reset_circuit nvidia_nim"),
            ("/run", "Execute bash command in workspace", "/run git status"),
            ("/stats", "Display live RPM/TPM token budget & request stats", "/stats"),
            ("/stop", "Cancel active task execution", "/stop"),
            ("/clear", "Clear current session context", "/clear"),
            ("/help", "Display available bot commands and usage guide", "/help"),
        ]

        for title, desc, msg in commands_info:
            if not raw_query or raw_query in title.lower() or raw_query in desc.lower():
                results.append(
                    InlineQueryResultArticle(
                        id=str(uuid4()),
                        title=f"🛠️ {title}",
                        description=desc,
                        input_message_content=InputTextMessageContent(msg),
                    )
                )

    try:
        await query_obj.answer(results=results[:10], cache_time=1)
    except Exception as e:
        logger.debug(f"Inline query answer error: {e}")
