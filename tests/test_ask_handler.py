"""Unit tests for /ask command handler."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.handlers.ask import handle_ask_command, is_shell_command


def test_is_shell_command_detection() -> None:
    """Test shell command conflict detection logic."""
    assert is_shell_command("pnpm dev")
    assert is_shell_command("git status")
    assert is_shell_command("pytest tests/")
    assert is_shell_command("rm -rf /")

    assert not is_shell_command("Blog.tsx dosyasındaki hataları ayıkla")
    assert not is_shell_command("How do I use FastAPI SSE streaming?")


@pytest.mark.asyncio
async def test_handle_ask_empty_prompt() -> None:
    """Test /ask command with empty prompt returns warning."""
    mock_update = MagicMock()
    mock_msg = MagicMock()
    mock_msg.reply_text = AsyncMock()
    mock_update.message = mock_msg
    mock_update.effective_user.id = 12345
    mock_update.effective_chat.id = 12345

    mock_context = MagicMock()
    mock_context.args = []

    # Mock authorization
    with pytest.MonkeyPatch.context() as m:
        m.setattr("bot.handlers.ask.is_authorized_telegram", lambda u: True)
        await handle_ask_command(mock_update, mock_context)

    mock_msg.reply_text.assert_called_once()
    text = mock_msg.reply_text.call_args.kwargs.get("text") or mock_msg.reply_text.call_args[0][0]
    assert "Please specify your prompt" in text


@pytest.mark.asyncio
async def test_handle_ask_shell_command_warning() -> None:
    """Test /ask command with shell command returns warning pointing to /run."""
    mock_update = MagicMock()
    mock_msg = MagicMock()
    mock_msg.reply_text = AsyncMock()
    mock_update.message = mock_msg
    mock_update.effective_user.id = 12345
    mock_update.effective_chat.id = 12345

    mock_context = MagicMock()
    mock_context.args = ["pnpm", "build"]

    with pytest.MonkeyPatch.context() as m:
        m.setattr("bot.handlers.ask.is_authorized_telegram", lambda u: True)
        await handle_ask_command(mock_update, mock_context)

    mock_msg.reply_text.assert_called_once()
    text = mock_msg.reply_text.call_args.kwargs.get("text") or mock_msg.reply_text.call_args[0][0]
    assert "terminal command" in text
    assert "/run" in text
