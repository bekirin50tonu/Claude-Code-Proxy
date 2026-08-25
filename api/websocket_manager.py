"""Real-time WebSocket Manager for Claude Code Proxy Dashboard."""

import asyncio
import json
from typing import Any

from fastapi import WebSocket
from loguru import logger


class DashboardWebSocketManager:
    """Manager for real-time WebSocket connections serving dashboard telemetry."""

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._background_task: asyncio.Task[None] | None = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection and send initial snapshot."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

        client_host = websocket.client.host if websocket.client else "unknown"
        logger.info(
            "🌐 [WebSocket] New dashboard client connected from {}:{} | Active clients: {}",
            client_host,
            websocket.client.port if websocket.client else 0,
            len(self.active_connections),
        )

        # Send initial snapshot payload on connect
        snapshot = await self._build_initial_snapshot()
        try:
            await websocket.send_text(json.dumps({"event": "initial_state", "data": snapshot}, ensure_ascii=False))
        except Exception as err:
            logger.warning("Failed to send initial snapshot to WS client: {}", err)

        # Start background telemetry pulse if not running
        self._ensure_pulse_task()

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister disconnected WebSocket client."""
        async with self._lock:
            self.active_connections.discard(websocket)

        logger.info(
            "🔌 [WebSocket] Client disconnected | Remaining active clients: {}",
            len(self.active_connections),
        )

    async def broadcast_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast a structured event payload to all connected dashboard WebSocket clients."""
        if not self.active_connections:
            return

        payload = {
            "event": event_type,
            "data": data,
        }
        message_str = json.dumps(payload, ensure_ascii=False)

        async with self._lock:
            clients = list(self.active_connections)

        disconnected: set[WebSocket] = set()
        for client in clients:
            try:
                await client.send_text(message_str)
            except Exception as exc:
                logger.debug("Failed to send WS message to client, removing: {}", exc)
                disconnected.add(client)

        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    self.active_connections.discard(ws)

    async def _build_initial_snapshot(self) -> dict[str, Any]:
        """Gather current snapshot state across proxy modules."""
        from config import settings, stats, model_registry
        from core.router.daily_tracker import daily_request_tracker
        from core.router.selector import model_selector

        router_status = model_selector.get_status()

        client_models = [
            ("claude_default", "1. DEFAULT (RECOMMENDED)", "Default Model (Nemotron 70B / Llama 3.3)"),
            ("claude_opus", "2. OPUS (1M CONTEXT)", "Opus (Llama 3.3 70B)"),
            ("claude_sonnet", "3. SONNET", "Sonnet (Llama 3.3 70B)"),
            ("claude_sonnet_1m", "4. SONNET 1M", "Sonnet 1M (Llama 3.3 70B)"),
            ("claude_haiku", "5. HAIKU", "Haiku (Llama 3.1 8B)"),
        ]
        client_mappings = []
        for c_model, label, desc in client_models:
            primary = model_registry.get_primary(c_model)
            fallbacks = model_registry.get_fallbacks(c_model)
            all_chain = [c for c in ([primary] + fallbacks) if c]

            resolved = "ALL_UNAVAILABLE"
            is_fallback = False
            step_name = "NONE"

            for idx, cand in enumerate(all_chain):
                if await model_selector._is_available(cand):
                    resolved = cand
                    is_fallback = idx > 0
                    step_name = "PRIMARY DIRECT" if idx == 0 else f"FALLBACK #{idx}"
                    break

            client_mappings.append({
                "client_model": c_model,
                "label": label,
                "description": desc,
                "primary": primary,
                "resolved_target": resolved,
                "is_fallback": is_fallback,
                "step_name": step_name,
                "chain": all_chain,
            })

        return {
            "stats": {
                "total_requests": stats.total_requests,
                "mocked_requests": stats.mocked_requests,
                "error_count": stats.error_count,
                "active_concurrency": stats.active_concurrency,
                "ds_bot_status": "Online" if hasattr(settings, "_ds_bot") else "Offline",
                "tg_bot_status": "Online" if hasattr(settings, "_tg_bot") else "Offline",
            },
            "router_status": {
                "summary": {
                    "total_models": len(router_status),
                    "healthy": sum(1 for v in router_status.values() if isinstance(v, dict) and v.get("circuit_breaker", {}).get("state") == "closed"),
                    "circuit_open": sum(1 for v in router_status.values() if isinstance(v, dict) and v.get("circuit_breaker", {}).get("state") == "open"),
                },
                "models": router_status,
                "client_mappings": client_mappings,
                "daily_rpd": daily_request_tracker.all_statuses(),
            },
            "recent_requests": stats.get_recent_dicts(include_payload=False)[:10],
        }

    def _ensure_pulse_task(self) -> None:
        """Ensure background task is pushing 3s telemetry pulse when clients exist."""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._telemetry_pulse_loop())

    async def _telemetry_pulse_loop(self) -> None:
        """Periodic 3.0s background pulse pushing live telemetry to active WS clients."""
        logger.debug("Starting real-time WebSocket telemetry pulse loop...")
        while True:
            try:
                await asyncio.sleep(3.0)
                if not self.active_connections:
                    continue

                snapshot = await self._build_initial_snapshot()
                await self.broadcast_event("telemetry_pulse", snapshot)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Error in WS pulse loop: {}", e)


ws_manager = DashboardWebSocketManager()
