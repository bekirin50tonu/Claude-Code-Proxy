"""Unit tests for LiveBridgeManager (/live stream tracking and debounced updates)."""

import pytest

from bot.live_bridge import LiveBridgeManager


@pytest.mark.asyncio
async def test_live_bridge_toggle_watchers() -> None:
    """Test watcher registration and toggling in LiveBridgeManager."""
    manager = LiveBridgeManager()
    chat_id = "123456789"

    assert not manager.is_watcher(chat_id)

    # Enable watching
    active = await manager.toggle_live(chat_id, enable=True)
    assert active
    assert manager.is_watcher(chat_id)

    # Disable watching
    active = await manager.toggle_live(chat_id, enable=False)
    assert not active
    assert not manager.is_watcher(chat_id)

    # Toggle automatically
    active = await manager.toggle_live(chat_id)
    assert active
    assert manager.is_watcher(chat_id)


@pytest.mark.asyncio
async def test_live_bridge_dispatch_thinking_chunk() -> None:
    """Test thinking chunk accumulation and state management."""
    manager = LiveBridgeManager(min_edit_interval=0.1)
    chat_id = "987654321"

    await manager.toggle_live(chat_id, enable=True)
    state = manager._get_session_state("session_test_1")

    # Dispatch thinking chunks
    await manager.dispatch_thinking_chunk("session_test_1", "Let's analyze the problem.")
    await manager.dispatch_thinking_chunk("session_test_1", " Step 1: Check inputs.")

    assert len(state.thinking_accumulator) == 2
    assert "".join(state.thinking_accumulator) == "Let's analyze the problem. Step 1: Check inputs."

    manager.finalize_session_stream("session_test_1")
    assert "session_test_1" not in manager._session_states
