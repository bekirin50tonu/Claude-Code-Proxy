"""Unit tests for Hermes Agent Inline Query Autocomplete previews."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.handlers.inline import inline_query_handler
from core.router.circuit_breaker import circuit_breaker_registry


@pytest.mark.asyncio
async def test_inline_query_run_autocomplete() -> None:
    """Test inline query autocomplete suggestions for /run command."""
    mock_update = MagicMock()
    mock_query = MagicMock()
    mock_query.query = "/run"
    mock_query.answer = AsyncMock()
    mock_update.inline_query = mock_query

    await inline_query_handler(mock_update, MagicMock())

    mock_query.answer.assert_called_once()
    results = mock_query.answer.call_args.kwargs.get("results") or mock_query.answer.call_args[1].get("results")
    assert results is not None
    assert len(results) > 0
    assert any("pnpm dev" in r.title for r in results)


@pytest.mark.asyncio
async def test_inline_query_reset_circuit_autocomplete() -> None:
    """Test inline query autocomplete suggestions for /reset_circuit command."""
    cb = circuit_breaker_registry.get("nvidia_nim/z-ai/glm-5.2")
    cb.force_open("Test timeout")

    mock_update = MagicMock()
    mock_query = MagicMock()
    mock_query.query = "/reset_circuit"
    mock_query.answer = AsyncMock()
    mock_update.inline_query = mock_query

    await inline_query_handler(mock_update, MagicMock())

    mock_query.answer.assert_called_once()
    results = mock_query.answer.call_args.kwargs.get("results") or mock_query.answer.call_args[1].get("results")
    assert results is not None
    assert any("nvidia_nim/z-ai/glm-5.2" in r.title for r in results)

    # Clean up test breaker
    cb.reset()
