"""/ask Command Handler — AI prompt execution via /v1/messages SSE stream with live debounced Telegram updates."""

import asyncio
import contextlib
import json
import re

import httpx
from loguru import logger
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from bot.auth import is_authorized_telegram
from bot.formatters import escape_markdown_v2
from config import settings

SHELL_COMMAND_PATTERNS = [
    r"^pnpm\s+",
    r"^npm\s+",
    r"^yarn\s+",
    r"^git\s+",
    r"^pytest",
    r"^python\d*\s+",
    r"^pip\d*\s+",
    r"^cargo\s+",
    r"^rm\s+-",
    r"^cd\s+",
    r"^ls\b",
    r"^cat\s+",
    r"^mkdir\s+",
    r"^touch\s+",
    r"^sudo\s+",
]


def is_shell_command(text: str) -> bool:
    """Check if prompt matches common executable terminal command patterns."""
    text_lower = text.strip().lower()
    return any(re.search(pat, text_lower) for pat in SHELL_COMMAND_PATTERNS)


async def handle_ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ask <prompt> command by streaming response from /v1/messages endpoint."""
    if not is_authorized_telegram(update) or not update.message:
        return

    if update.effective_chat:
        try:
            await update.effective_chat.send_action(ChatAction.TYPING)
        except Exception as e:
            logger.debug(f"Failed to send typing indicator: {e}")

    # 1. Extract prompt text
    args = context.args or []
    prompt = " ".join(args).strip()

    if not prompt:
        empty_warning = (
            "⚠️ *Please specify your prompt or coding task\\.*\n\n"
            "Example: `/ask Debug the error in Blog.tsx`"
        )
        await update.message.reply_text(text=empty_warning, parse_mode=ParseMode.MARKDOWN_V2)
        return

    # 2. Command Conflict Prevention Check
    if is_shell_command(prompt):
        esc_p = escape_markdown_v2(prompt, is_code_block=True)
        shell_warning = (
            "💡 *Notice:* It looks like you entered a terminal command\\.\n"
            "`/ask` is for AI prompt queries\\. To execute bash commands on the server directly, "
            "please use `/run` instead\\.\n\n"
            f"*Example:* `/run {esc_p}`"
        )
        await update.message.reply_text(text=shell_warning, parse_mode=ParseMode.MARKDOWN_V2)
        return

    # Send initial status message
    status_msg = await update.message.reply_text(
        "🧠 *Claude is thinking...*",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    stop_typing = asyncio.Event()

    async def keep_typing_loop() -> None:
        while not stop_typing.is_set():
            if update.effective_chat:
                with contextlib.suppress(Exception):
                    await update.effective_chat.send_action(ChatAction.TYPING)
            try:
                await asyncio.sleep(4.0)
            except asyncio.CancelledError:
                break

    typing_task = asyncio.create_task(keep_typing_loop())

    # State variables for stream accumulation
    thinking_chunks: list[str] = []
    answer_chunks: list[str] = []
    last_edit_time: float = 0.0
    min_edit_interval = 0.5  # Max 2 edits per second

    async def update_telegram_message(force: bool = False) -> None:
        nonlocal last_edit_time
        now = asyncio.get_running_loop().time()
        if not force and (now - last_edit_time) < min_edit_interval:
            return

        last_edit_time = now
        full_thinking = "".join(thinking_chunks).strip()
        full_answer = "".join(answer_chunks).strip()

        msg_parts = []
        if full_thinking:
            trunc_think = full_thinking[:2000] + ("..." if len(full_thinking) > 2000 else "")
            esc_think = escape_markdown_v2(trunc_think, is_code_block=False)
            msg_parts.append(f"🧠 *Claude is thinking...*\n\n_{esc_think}_")
        else:
            msg_parts.append("🧠 *Claude is thinking...*")

        if full_answer:
            trunc_ans = full_answer[:3000] + ("..." if len(full_answer) > 3000 else "")
            esc_ans = escape_markdown_v2(trunc_ans, is_code_block=False)
            msg_parts.append(f"\n💬 *Response:*\n\n{esc_ans}")

        text = "\n".join(msg_parts)

        try:
            await status_msg.edit_text(text=text, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception as e:
            if "Message is not modified" in str(e):
                pass
            else:
                logger.warning(f"MarkdownV2 edit error in /ask: {e}. Falling back to plain text.")
                try:
                    plain_text = re.sub(r"\\(.)", r"\1", text)
                    await status_msg.edit_text(text=plain_text, parse_mode=None)
                except Exception as inner_e:
                    if "Message is not modified" not in str(inner_e):
                        logger.error(f"/ask fallback edit_text failed: {inner_e}")

    try:
        url = f"http://127.0.0.1:{settings.PORT}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": settings.ANTHROPIC_API_KEY or "dummy-key",
        }
        payload = {
            "model": settings.MODEL_SONNET or "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream("POST", url, json=payload, headers=headers) as resp,
        ):
            if resp.status_code != 200:
                err_text = await resp.aread()
                esc_err = escape_markdown_v2(err_text.decode("utf-8", errors="ignore")[:500], is_code_block=True)
                await status_msg.edit_text(
                    f"❌ *API Error ({resp.status_code}):*\n`{esc_err}`",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                return

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data_obj = json.loads(data_str)
                        delta = data_obj.get("delta") or {}
                        dtype = delta.get("type")

                        # Capture thinking deltas
                        if dtype == "thinking_delta":
                            think_text = delta.get("thinking") or ""
                            if think_text:
                                thinking_chunks.append(think_text)
                                await update_telegram_message(force=False)

                        # Capture text deltas
                        elif dtype == "text_delta":
                            text_val = delta.get("text") or ""
                            if text_val:
                                answer_chunks.append(text_val)
                                await update_telegram_message(force=False)

                    except json.JSONDecodeError:
                        pass

        # Final forced edit update
        await update_telegram_message(force=True)

    except Exception as e:
        logger.error(f"Error executing /ask command: {e}")
        esc_err = escape_markdown_v2(str(e), is_code_block=True)
        await status_msg.edit_text(
            f"❌ *Execution Error:*\n`{esc_err}`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    finally:
        stop_typing.set()
        typing_task.cancel()
