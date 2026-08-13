"""
Stream Guard — timeout and stall detection for SSE streams.

Wraps an AsyncGenerator and applies two safety mechanisms:

1. Chunk timeout (default 30s): if no chunk arrives within this window,
   the stream is considered stalled.

2. Empty chunk counter (default 10): consecutive chunks with no meaningful
   content increment a counter. If the counter reaches max_empty_chunks,
   the stream is considered stalled.

On stall/timeout, an Anthropic-formatted SSE error event is yielded and
the generator closes cleanly without blocking the client.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator

from loguru import logger

_SSE_STREAM_ERROR = (
    "event: error\n"
    "data: {}\n\n"
    "event: message_stop\n"
    'data: {{"type": "message_stop"}}\n\n'
)


def _make_error_event(message: str) -> str:
    payload = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "stream_error",
                "message": message,
            },
        }
    )
    return (
        f"event: error\ndata: {payload}\n\n"
        + 'event: message_stop\ndata: {"type": "message_stop"}\n\n'
    )


class StreamGuard:
    """Wrap an Anthropic SSE string generator with timeout and stall detection.

    Parameters
    ----------
    source:
        The upstream AsyncGenerator yielding Anthropic SSE event strings.
    stream_timeout:
        Max seconds to wait between consecutive chunks. Default: 30.
    max_empty_chunks:
        Max consecutive empty/whitespace-only chunks before stall. Default: 10.
    """

    def __init__(
        self,
        source: AsyncGenerator[str, None],
        stream_timeout: float = 30.0,
        max_empty_chunks: int = 10,
    ) -> None:
        self._source = source
        self._timeout = stream_timeout
        self._max_empty = max_empty_chunks

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        empty_count = 0
        try:
            async for chunk in self._iter_with_timeout():
                if not chunk or chunk.strip() == "":
                    empty_count += 1
                    if empty_count >= self._max_empty:
                        logger.warning(
                            "StreamGuard: %d consecutive empty chunks — stall detected",
                            empty_count,
                        )
                        yield _make_error_event(
                            f"Stream stalled after {empty_count} consecutive empty chunks."
                        )
                        return
                    yield chunk
                else:
                    empty_count = 0
                    yield chunk
        except TimeoutError:
            logger.warning("StreamGuard: chunk timeout after %.1fs", self._timeout)
            yield _make_error_event(
                f"Stream timed out — no data received within {self._timeout}s."
            )
        except Exception as exc:
            logger.error("StreamGuard: unexpected error: %s", exc)
            yield _make_error_event(f"Stream error: {exc}")
        finally:
            # Ensure the underlying generator is properly closed
            with contextlib.suppress(Exception):
                await self._source.aclose()

    async def _iter_with_timeout(self) -> AsyncGenerator[str, None]:  # type: ignore[override]
        """Yield chunks from source with per-chunk timeout."""
        source_iter = self._source.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    source_iter.__anext__(),
                    timeout=self._timeout,
                )
                yield chunk
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise


# Convenience type alias for route handlers
GuardedStream = AsyncGenerator[str, None]


def guarded(
    source: AsyncGenerator[str, None],
    stream_timeout: float = 30.0,
    max_empty_chunks: int = 10,
) -> GuardedStream:
    """Shorthand factory for StreamGuard.__aiter__."""
    guard = StreamGuard(source, stream_timeout, max_empty_chunks)
    return guard.__aiter__()
