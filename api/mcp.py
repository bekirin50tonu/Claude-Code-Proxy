"""Model Context Protocol (MCP) Server Router for Hermes Agent and External Assistants.

Provides standard MCP JSON-RPC 2.0 tools for listing & setting model mappings,
inspecting & updating system configurations, monitoring real-time provider metrics,
and controlling circuit breakers.
"""

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from api.dashboard import (
    SUBAGENTS_ENABLED,
    get_available_models,
    get_config,
    get_dev_metrics,
    get_key_statuses,
    handle_circuit_breaker_action,
    save_config,
    toggle_subagents_setting,
)

mcp_router = APIRouter()


class MCPJsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] | None = None


MCP_TOOLS_DEFINITIONS = [
    {
        "name": "get_models",
        "description": "Fetch available LLM models across providers (NVIDIA NIM, OpenRouter, Gemini, Groq, DeepSeek, Mistral, etc.) and active client alias mappings (Default, Opus, Sonnet, Sonnet 1M, Haiku).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "set_model_mapping",
        "description": "Set primary target model or fallback chain for a client alias (MODEL_OPUS, MODEL_SONNET, MODEL_SONNET_1M, MODEL_HAIKU, MODEL).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "alias_key": {
                    "type": "string",
                    "description": "The client model key to update: 'MODEL_OPUS', 'MODEL_SONNET', 'MODEL_SONNET_1M', 'MODEL_HAIKU', or 'MODEL'.",
                },
                "target_model": {
                    "type": "string",
                    "description": "The target provider model string (e.g. 'nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct' or 'open_router/google/gemini-2.5-pro').",
                },
                "fallback_order": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of fallback model IDs in fallback priority order.",
                },
            },
            "required": ["alias_key", "target_model"],
        },
    },
    {
        "name": "get_system_config",
        "description": "Retrieve current proxy system configurations, per-provider settings (RPM, TPM, RPD, Timeouts), and Subagent Emergency Switch status.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "update_system_config",
        "description": "Update system parameters or per-provider settings (e.g., PROVIDER_NVIDIA_NIM_RPM, PROVIDER_OPEN_ROUTER_TPM, subagents_enabled, thinking_modes).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "configs": {
                    "type": "object",
                    "description": "Dictionary of key-value configuration pairs to set (e.g., {'PROVIDER_NVIDIA_NIM_RPM': 38, 'SUBAGENTS_ENABLED': true}).",
                },
            },
            "required": ["configs"],
        },
    },
    {
        "name": "get_metrics",
        "description": "Get real-time proxy traffic telemetry, per-provider fundamental RPM and TPM consumption, 429 rate limit events, and circuit breaker statuses.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_throttle_metrics",
        "description": "Inspect active throttle sleeps, queue delays, max fast-fallback sleep threshold, total sleep time, and API key pool cooldown statuses.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "update_throttle_settings",
        "description": "Update throttle guard parameters dynamically in-memory and in .env (e.g. max_sleep_threshold fast fallback delay, max_queue_wait timeout budget, rpm_limit).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_sleep_threshold": {
                    "type": "number",
                    "description": "Maximum seconds allowed for throttle sleep before triggering fast fallback to secondary models (e.g. 2.0 or 3.0).",
                },
                "max_queue_wait": {
                    "type": "number",
                    "description": "Maximum queue wait budget seconds before raising queue timeout (e.g. 15.0 or 30.0).",
                },
                "rpm_limit": {
                    "type": "integer",
                    "description": "Safe sliding window RPM limit for rate-limited providers (e.g. 38).",
                },
            },
        },
    },
    {
        "name": "get_model_routing",
        "description": "Retrieve resolved client-to-target routing mappings matrix (showing which primary or fallback model currently handles Opus, Sonnet, Haiku, Default).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "control_circuit_breaker",
        "description": "Manually trip (block/close traffic) or reset (clear/enable traffic) a model's circuit breaker state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Target model ID (e.g. 'nvidia_nim/z-ai/glm-5.2').",
                },
                "action": {
                    "type": "string",
                    "enum": ["reset", "trip"],
                    "description": "'reset' to clear timeout and restore model traffic, or 'trip' to block model traffic.",
                },
            },
            "required": ["model_id", "action"],
        },
    },
]


async def execute_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute target MCP tool and return standardized MCP tool content response."""
    try:
        if tool_name == "get_models":
            models_resp = await get_available_models()
            models_data = json.loads(models_resp.body.decode("utf-8"))
            cfg_resp = await get_config()
            cfg_data = json.loads(cfg_resp.body.decode("utf-8"))
            
            res = {
                "available_models_by_provider": models_data,
                "active_client_mappings": {
                    "MODEL_OPUS": cfg_data.get("configs", {}).get("MODEL_OPUS"),
                    "MODEL_SONNET": cfg_data.get("configs", {}).get("MODEL_SONNET"),
                    "MODEL_SONNET_1M": cfg_data.get("configs", {}).get("MODEL_SONNET_1M"),
                    "MODEL_HAIKU": cfg_data.get("configs", {}).get("MODEL_HAIKU"),
                    "MODEL": cfg_data.get("configs", {}).get("MODEL"),
                },
                "fallbacks": cfg_data.get("fallbacks", {}),
            }
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}

        elif tool_name == "set_model_mapping":
            alias_key = arguments.get("alias_key", "").upper()
            target_model = arguments.get("target_model", "")
            fallback_order = arguments.get("fallback_order")

            if alias_key not in ("MODEL_OPUS", "MODEL_SONNET", "MODEL_SONNET_1M", "MODEL_HAIKU", "MODEL"):
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Invalid alias_key '{alias_key}'. Must be one of MODEL_OPUS, MODEL_SONNET, MODEL_SONNET_1M, MODEL_HAIKU, MODEL."}],
                }

            configs_to_save = {alias_key: target_model}
            if fallback_order is not None and isinstance(fallback_order, list):
                alias_map = {
                    "MODEL_OPUS": "claude_opus",
                    "MODEL_SONNET": "claude_sonnet",
                    "MODEL_SONNET_1M": "claude_sonnet_1m",
                    "MODEL_HAIKU": "claude_haiku",
                    "MODEL": "claude_default",
                }
                alias = alias_map[alias_key]
                configs_to_save[f"FALLBACK_ORDER_{alias.upper()}"] = fallback_order

            from api.dashboard import ConfigSaveRequest
            save_resp = await save_config(ConfigSaveRequest(configs=configs_to_save))
            res_data = json.loads(save_resp.body.decode("utf-8"))
            return {"content": [{"type": "text", "text": json.dumps(res_data, indent=2)}]}

        elif tool_name == "get_system_config":
            cfg_resp = await get_config()
            cfg_data = json.loads(cfg_resp.body.decode("utf-8"))
            cfg_data["subagents_enabled"] = SUBAGENTS_ENABLED
            return {"content": [{"type": "text", "text": json.dumps(cfg_data, indent=2)}]}

        elif tool_name == "update_system_config":
            configs = arguments.get("configs", {})
            if "subagents_enabled" in configs:
                from api.dashboard import SubagentsToggleRequest
                await toggle_subagents_setting(SubagentsToggleRequest(enabled=bool(configs["subagents_enabled"])))
                configs.pop("subagents_enabled")

            if configs:
                from api.dashboard import ConfigSaveRequest
                save_resp = await save_config(ConfigSaveRequest(configs=configs))
                res_data = json.loads(save_resp.body.decode("utf-8"))
            else:
                res_data = {"status": "success", "message": "System configuration updated."}
            return {"content": [{"type": "text", "text": json.dumps(res_data, indent=2)}]}

        elif tool_name == "get_metrics":
            metrics_resp = await get_dev_metrics()
            metrics_data = json.loads(metrics_resp.body.decode("utf-8"))
            return {"content": [{"type": "text", "text": json.dumps(metrics_data, indent=2)}]}

        elif tool_name == "get_throttle_metrics":
            from atomic.guards.nim_guard import nim_throttle_guard
            metrics_resp = await get_dev_metrics()
            metrics_data = json.loads(metrics_resp.body.decode("utf-8"))

            res = {
                "throttle_telemetry": metrics_data.get("throttle_telemetry", {}),
                "nim_telemetry": metrics_data.get("nim_telemetry", {}),
                "active_sleep_threshold": nim_throttle_guard.max_sleep_threshold,
                "max_queue_wait": nim_throttle_guard.max_queue_wait,
                "rpm_limit": nim_throttle_guard.rpm_limit,
            }
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}

        elif tool_name == "update_throttle_settings":
            from atomic.guards.nim_guard import nim_throttle_guard
            max_sleep = arguments.get("max_sleep_threshold")
            max_queue = arguments.get("max_queue_wait")
            rpm = arguments.get("rpm_limit")

            updated_fields = {}
            if max_sleep is not None and isinstance(max_sleep, (int, float)):
                nim_throttle_guard.max_sleep_threshold = float(max_sleep)
                updated_fields["max_sleep_threshold"] = nim_throttle_guard.max_sleep_threshold

            if max_queue is not None and isinstance(max_queue, (int, float)):
                nim_throttle_guard._max_queue_wait = float(max_queue)
                updated_fields["max_queue_wait"] = nim_throttle_guard.max_queue_wait

            if rpm is not None and isinstance(rpm, int):
                nim_throttle_guard._rpm_limit = rpm
                updated_fields["rpm_limit"] = nim_throttle_guard.rpm_limit
                from api.dashboard import ConfigSaveRequest
                await save_config(ConfigSaveRequest(configs={"PROVIDER_NVIDIA_NIM_RPM": rpm}))

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "success",
                                "message": f"Throttle settings updated successfully: {updated_fields}",
                                "current_settings": {
                                    "max_sleep_threshold": nim_throttle_guard.max_sleep_threshold,
                                    "max_queue_wait": nim_throttle_guard.max_queue_wait,
                                    "rpm_limit": nim_throttle_guard.rpm_limit,
                                },
                            },
                            indent=2,
                        ),
                    }
                ]
            }

        elif tool_name == "get_model_routing":
            from api.dashboard import get_router_status
            routing_resp = await get_router_status()
            routing_data = json.loads(routing_resp.body.decode("utf-8"))
            return {"content": [{"type": "text", "text": json.dumps(routing_data, indent=2)}]}

        elif tool_name == "control_circuit_breaker":
            model_id = arguments.get("model_id", "")
            action = arguments.get("action", "")
            from api.dashboard import CircuitBreakerActionRequest
            cb_resp = await handle_circuit_breaker_action(CircuitBreakerActionRequest(model_id=model_id, action=action))
            res_data = json.loads(cb_resp.body.decode("utf-8"))
            return {"content": [{"type": "text", "text": json.dumps(res_data, indent=2)}]}

        else:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Unknown tool: '{tool_name}'"}],
            }

    except Exception as e:
        logger.error(f"MCP Tool Execution Error ({tool_name}): {e}")
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Execution error for tool '{tool_name}': {e}"}],
        }


@mcp_router.post("/mcp")
@mcp_router.post("/api/mcp")
async def handle_mcp_request(req: Request) -> JSONResponse:
    """Standard JSON-RPC 2.0 MCP HTTP endpoint for Hermes Agent and external clients."""
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
        )

    req_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {}) or {}

    if method == "initialize":
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "claude-code-proxy-mcp",
                        "version": "1.0.0",
                        "description": "Claude Code Proxy MCP Server for Hermes Agent",
                    },
                },
            }
        )

    elif method in ("notifications/initialized", "initialized"):
        return JSONResponse(content={"jsonrpc": "2.0", "id": req_id, "result": {}})

    elif method == "ping":
        return JSONResponse(content={"jsonrpc": "2.0", "id": req_id, "result": {}})

    elif method == "tools/list":
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": MCP_TOOLS_DEFINITIONS},
            }
        )

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        tool_result = await execute_mcp_tool(tool_name, arguments)
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "result": tool_result,
            }
        )

    else:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: '{method}'"},
            },
        )


@mcp_router.get("/mcp/sse")
@mcp_router.get("/api/mcp/sse")
async def handle_mcp_sse(request: Request) -> StreamingResponse:
    """MCP SSE Transport connection endpoint."""
    async def sse_event_generator():
        yield "event: endpoint\ndata: /mcp\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
