"""Authorization helper functions for Telegram and Discord bot platforms."""

from typing import Any

from config import settings


def is_authorized_telegram(update: Any) -> bool:
    """Verify requesting user or chat matches ALLOWED_TELEGRAM_USER_ID."""
    allowed = settings.ALLOWED_TELEGRAM_USER_ID.strip()
    if not allowed:
        return False
    user_id = str(update.effective_user.id) if getattr(update, "effective_user", None) else ""
    chat_id = str(update.effective_chat.id) if getattr(update, "effective_chat", None) else ""
    return user_id == allowed or chat_id == allowed
