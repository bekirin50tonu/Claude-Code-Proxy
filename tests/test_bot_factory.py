"""Unit tests for BotFactory pattern, platform adapters, and English formatters."""

from unittest.mock import AsyncMock, patch

import pytest

from bot.factory import bot_factory
from bot.formatters import (
    escape_markdown_v2,
    format_circuit_breaker_alert_discord,
    format_circuit_breaker_alert_tg,
    format_status_overview_discord,
    format_status_overview_tg,
)
from core.router.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
)


def test_escape_markdown_v2() -> None:
    """Test Telegram Markdown V2 character escaping."""
    raw_text = "Error: model_id [test] (v1.0) failed! 1+1=2 & price=$10"
    escaped = escape_markdown_v2(raw_text, is_code_block=False)
    assert "\\_" in escaped
    assert "\\[" in escaped
    assert "\\]" in escaped
    assert "\\(" in escaped
    assert "\\)" in escaped
    assert "\\!" in escaped
    assert "\\+" in escaped
    assert "\\=" in escaped

    code_text = "def test(x):\n    return f'hello {x}'"
    code_escaped = escape_markdown_v2(code_text, is_code_block=True)
    assert "(" in code_escaped
    assert ")" in code_escaped


def test_format_circuit_breaker_alert_english() -> None:
    """Test generation of English proactive Circuit Breaker alerts for Telegram and Discord."""
    # Telegram alert format
    tg_alert = format_circuit_breaker_alert_tg(
        model_id="nvidia_nim/z-ai/glm-5.2",
        reason="Upstream 503 Service Unavailable",
        fallback_model="gemini-3.7-flash",
    )
    assert "Circuit Breaker Tripped" in tg_alert
    assert "nvidia_nim/z-ai/glm-5.2" in tg_alert
    assert "gemini-3.7-flash" in tg_alert
    assert "OPEN" in tg_alert
    assert "reset the circuit breaker" in tg_alert

    # Discord alert format
    ds_alert = format_circuit_breaker_alert_discord(
        model_id="nvidia_nim/z-ai/glm-5.2",
        reason="Upstream 503 Service Unavailable",
        fallback_model="gemini-3.7-flash",
    )
    assert ds_alert["title"] == "🚨 Circuit Breaker Tripped!"
    assert any(f["value"] == "`nvidia_nim/z-ai/glm-5.2`" for f in ds_alert["fields"])
    assert any(f["value"] == "`gemini-3.7-flash`" for f in ds_alert["fields"])


def test_format_status_overview_english() -> None:
    """Test English status overview formatters."""
    tg_status = format_status_overview_tg()
    assert "Claude Code Proxy Gateway Status" in tg_status
    assert "Model Mappings" in tg_status

    ds_status = format_status_overview_discord()
    assert "Claude Code Proxy Gateway Status" in ds_status
    assert "Model Mappings" in ds_status


def test_bot_factory_initialization() -> None:
    """Test that BotFactory initializes Telegram and Discord adapters."""
    assert "telegram" in bot_factory.adapters
    assert "discord" in bot_factory.adapters
    assert bot_factory.get_adapter("telegram") is not None
    assert bot_factory.get_adapter("discord") is not None


@pytest.mark.asyncio
async def test_bot_factory_broadcast() -> None:
    """Test broadcasting Circuit Breaker alerts across all factory adapters."""
    tg_adapter = bot_factory.get_adapter("telegram")
    ds_adapter = bot_factory.get_adapter("discord")

    assert tg_adapter is not None
    assert ds_adapter is not None

    with (
        patch.object(tg_adapter, "send_circuit_breaker_alert", new_callable=AsyncMock) as mock_tg_alert,
        patch.object(ds_adapter, "send_circuit_breaker_alert", new_callable=AsyncMock) as mock_ds_alert,
    ):
        await bot_factory.broadcast_circuit_breaker_alert(
            model_id="open_router/anthropic/claude-3.5-sonnet",
            reason="Test rate limit trip",
            fallback_model="gemini-3.7-flash",
        )

        mock_tg_alert.assert_called_once_with(
            "open_router/anthropic/claude-3.5-sonnet",
            "Test rate limit trip",
            "gemini-3.7-flash",
        )
        mock_ds_alert.assert_called_once_with(
            "open_router/anthropic/claude-3.5-sonnet",
            "Test rate limit trip",
            "gemini-3.7-flash",
        )


@pytest.mark.asyncio
async def test_circuit_breaker_reset_state() -> None:
    """Test resetting Circuit Breaker state."""
    registry = CircuitBreakerRegistry()
    cb = registry.get("nvidia_nim/z-ai/glm-5.2")

    with patch("core.router.circuit_breaker.circuit_breaker_registry", registry):
        cb.force_open("Test force open")
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
