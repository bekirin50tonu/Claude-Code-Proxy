"""Live Stream Bridge — Real-time reasoning stream interceptor and tool call reporter for Telegram Bot."""

import asyncio
import os
import time
from typing import Any

import yaml
from loguru import logger

from bot.formatters import escape_markdown_v2
from config import settings

STATE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".data", "telegram_state.yaml")
)


class SessionLiveStreamState:
    """State tracking for a single active streaming session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.thinking_accumulator: list[str] = []
        self.msg_ids: dict[str, int] = {}  # chat_id -> message_id
        self.last_edit_time: float = 0.0
        self.lock = asyncio.Lock()


class LiveBridgeManager:
    """Thread-safe manager for live stream watchers and real-time event dispatching."""

    def __init__(self, min_edit_interval: float = 0.5) -> None:
        self.active_watchers: set[str] = set()
        self.min_edit_interval: float = min_edit_interval  # Max 2 edits/sec
        self._session_states: dict[str, SessionLiveStreamState] = {}
        self._lock = asyncio.Lock()
        self.load_state()

    def save_state(self) -> None:
        """Save active watchers and workspace settings to .data/telegram_state.yaml."""
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            data = {
                "active_watchers": list(self.active_watchers),
                "claude_workspace": settings.CLAUDE_WORKSPACE,
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
        except Exception as e:
            logger.warning(f"Failed to save live bridge state to {STATE_FILE}: {e}")

    def load_state(self) -> None:
        """Load active watchers and workspace settings from .data/telegram_state.yaml."""
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            watchers = data.get("active_watchers") or []
            self.active_watchers = set(str(w).strip() for w in watchers)

            ws = data.get("claude_workspace")
            if ws and os.path.exists(ws):
                settings.CLAUDE_WORKSPACE = ws

            if self.active_watchers:
                logger.info(f"LiveBridgeManager: Restored active watchers: {self.active_watchers}")
        except Exception as e:
            logger.warning(f"Failed to load live bridge state from {STATE_FILE}: {e}")

    async def toggle_live(self, chat_id: str, enable: bool | None = None) -> bool:
        """Enable or disable live stream watching for a given Chat ID.

        If enable is None, toggles current state. Returns True if now active.
        """
        async with self._lock:
            cid = str(chat_id).strip()
            res = False
            if enable is None:
                if cid in self.active_watchers:
                    self.active_watchers.remove(cid)
                    res = False
                else:
                    self.active_watchers.add(cid)
                    res = True
            else:
                if enable:
                    self.active_watchers.add(cid)
                    res = True
                else:
                    self.active_watchers.discard(cid)
                    res = False

            self.save_state()
            return res

    def is_watcher(self, chat_id: str) -> bool:
        """Check if chat_id is registered as an active live watcher."""
        return str(chat_id).strip() in self.active_watchers

    def _get_session_state(self, session_id: str) -> SessionLiveStreamState:
        if session_id not in self._session_states:
            self._session_states[session_id] = SessionLiveStreamState(session_id)
        return self._session_states[session_id]

    def get_watchers(self) -> set[str]:
        """Return the current set of active Telegram watcher Chat IDs."""
        watchers = set(self.active_watchers)
        allowed = settings.ALLOWED_TELEGRAM_USER_ID.strip()
        if allowed and (watchers or self.is_watcher(allowed)):
            watchers.add(allowed)
        return watchers

    async def dispatch_thinking_chunk(self, session_id: str | None, chunk: str) -> None:
        """Accumulate reasoning text and send throttled edits to active watchers."""
        watchers = self.get_watchers()
        if not chunk or not watchers:
            return

        sid = session_id or "default_session"
        state = self._get_session_state(sid)

        async with state.lock:
            state.thinking_accumulator.append(chunk)
            now = time.monotonic()

            # Enforce debounced edit throttling (max 2 edits / second)
            if (now - state.last_edit_time) < self.min_edit_interval:
                return

            state.last_edit_time = now
            full_thinking = "".join(state.thinking_accumulator).strip()
            if not full_thinking:
                return

            # Truncate for Telegram single message limit
            if len(full_thinking) > 3500:
                full_thinking = full_thinking[:3400] + "\n...(reasoning truncated)"

            esc_thinking = escape_markdown_v2(full_thinking, is_code_block=False)
            text = f"🧠 *Claude is thinking...*\n\n_{esc_thinking}_"

            from bot.factory import bot_factory
            tg_adapter = bot_factory.get_adapter("telegram")
            if not tg_adapter or not getattr(tg_adapter, "app", None):
                return

            app = tg_adapter.app
            for cid in list(watchers):
                try:
                    try:
                        from telegram.constants import ChatAction
                        await app.bot.send_chat_action(chat_id=cid, action=ChatAction.TYPING)
                    except Exception:
                        pass

                    msg_id = state.msg_ids.get(cid)
                    if not msg_id:
                        try:
                            sent_msg = await app.bot.send_message(
                                chat_id=cid,
                                text=text,
                                parse_mode="MarkdownV2",
                            )
                        except Exception:
                            import re
                            plain_text = re.sub(r"\\(.)", r"\1", text)
                            sent_msg = await app.bot.send_message(
                                chat_id=cid,
                                text=plain_text,
                                parse_mode=None,
                            )
                        state.msg_ids[cid] = sent_msg.message_id
                    else:
                        try:
                            await app.bot.edit_message_text(
                                chat_id=cid,
                                message_id=msg_id,
                                text=text,
                                parse_mode="MarkdownV2",
                            )
                        except Exception as e:
                            if "Message is not modified" in str(e):
                                pass
                            else:
                                import re
                                plain_text = re.sub(r"\\(.)", r"\1", text)
                                await app.bot.edit_message_text(
                                    chat_id=cid,
                                    message_id=msg_id,
                                    text=plain_text,
                                    parse_mode=None,
                                )
                except Exception as e:
                    if "Message is not modified" not in str(e):
                        logger.debug(f"Live thinking update edit error for chat {cid}: {e}")

    async def dispatch_tool_call(
        self,
        session_id: str | None,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Notify live watchers when a tool call (file edit, command execution) is intercepted."""
        watchers = self.get_watchers()
        if not watchers:
            return

        from bot.factory import bot_factory
        tg_adapter = bot_factory.get_adapter("telegram")
        if not tg_adapter or not getattr(tg_adapter, "app", None):
            return

        app = tg_adapter.app

        # Extract file path or command details
        file_path = tool_input.get("path") or tool_input.get("file_path") or tool_input.get("target_file")
        cmd = tool_input.get("command") or tool_input.get("cmd")

        details_lines = [f"🛠️ *Tool Call Intercepted:* `{escape_markdown_v2(tool_name, is_code_block=True)}`"]

        if file_path:
            esc_path = escape_markdown_v2(str(file_path), is_code_block=True)
            details_lines.append(f"📝 *Edited File:* `{esc_path}`")
        if cmd:
            esc_cmd = escape_markdown_v2(str(cmd), is_code_block=True)
            details_lines.append(f"⚙️ *Executed Command:* `{esc_cmd}`")

        msg_text = "\n".join(details_lines)

        for cid in list(watchers):
            try:
                await app.bot.send_message(
                    chat_id=cid,
                    text=msg_text,
                    parse_mode="MarkdownV2",
                )
            except Exception as e:
                logger.warning(f"Failed to dispatch tool call notification to {cid}: {e}")

    async def dispatch_diff_report(
        self,
        session_id: str | None,
        file_path: str,
        diff_content: str,
    ) -> None:
        """Send a Markdown diff report to live watchers."""
        watchers = self.get_watchers()
        if not watchers or not diff_content:
            return

        from bot.factory import bot_factory
        tg_adapter = bot_factory.get_adapter("telegram")
        if not tg_adapter or not getattr(tg_adapter, "app", None):
            return

        app = tg_adapter.app
        esc_path = escape_markdown_v2(file_path, is_code_block=True)
        esc_diff = escape_markdown_v2(diff_content, is_code_block=True)

        diff_msg = (
            f"📄 *File Update Diff Report:* `{esc_path}`\n\n"
            f"```diff\n{esc_diff}\n```"
        )

        for cid in list(self.active_watchers):
            try:
                await app.bot.send_message(
                    chat_id=cid,
                    text=diff_msg,
                    parse_mode="MarkdownV2",
                )
            except Exception as e:
                logger.warning(f"Failed to send diff report to {cid}: {e}")

    def finalize_session_stream(self, session_id: str | None) -> None:
        """Clean up tracking state when a streaming session completes."""
        sid = session_id or "default_session"
        self._session_states.pop(sid, None)


# Singleton LiveBridgeManager instance
live_bridge_manager = LiveBridgeManager()
