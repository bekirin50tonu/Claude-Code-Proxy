import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
from pydantic import BaseModel

from config import settings, stats
from core.router.selector import model_selector as model_router

router = APIRouter()

# Fallback models in case API calls fail or keys are missing
FALLBACK_MODELS = {
    "nvidia_nim": [
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/nemotron-4-340b-instruct",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.3-70b-instruct",
        "z-ai/glm-5.2",
        "mistralai/mixtral-8x22b-instruct-v0.1",
    ],
    "open_router": [
        "google/gemini-2.5-flash:free",
        "google/gemini-2.5-pro",
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-r1",
        "arcee-ai/trinity-large-preview:free",
        "qwen/qwen-2.5-72b-instruct",
        "mistralai/pixtral-12b:free",
        "stepfun/step-3.5-flash:free",
    ],
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "deepseek-r1-distill-llama-70b",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ],
    "mistral": [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "codestral-latest",
        "pixtral-large-latest",
    ],
    "cerebras": [
        "llama3.3-70b",
        "llama3.1-8b",
    ],
    "fireworks": [
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "accounts/fireworks/models/deepseek-v3",
        "accounts/fireworks/models/deepseek-r1",
        "accounts/fireworks/models/qwen2p5-72b-instruct",
    ],
    "kimi": [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ],
    "lmstudio": [
        "qwen2.5-7b-instruct",
        "llama-3.2-3b-instruct",
        "phi-3-mini-4k-instruct",
        "mistral-7b-instruct",
    ],
    "ollama": [
        "llama3.3",
        "llama3.1:70b",
        "qwen2.5-coder",
        "deepseek-r1",
    ],
    "llama_cpp": [
        "local-model",
    ],
}


class ConfigSaveRequest(BaseModel):
    configs: dict[str, Any]


class SubagentsToggleRequest(BaseModel):
    enabled: bool


# Global in-memory server state for Subagents Emergency Switch
SUBAGENTS_ENABLED: bool = True



async def fetch_models_from_url(
    url: str, headers: dict[str, str] | None = None
) -> list[str]:
    """Asynchronously fetch list of models from a completions-compatible models endpoint."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                return sorted(models)
    except Exception as e:
        logger.debug(f"Failed to fetch models from {url}: {e}")
    return []


def get_env_file_keys() -> set[str]:
    """Parse .env file keys directly to check if they are configured inside the file."""
    env_path = Path(__file__).parent.parent / ".env"
    keys = set()
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    k, _ = stripped.split("=", 1)
                    keys.add(k.strip())
    return keys


def get_key_statuses() -> dict[str, str]:
    """Resolve Set / Env Locked / Not Set state badges for each setting."""
    env_keys = get_env_file_keys()
    statuses = {}
    managed_keys = [
        "NVIDIA_NIM_API_KEYS",
        "NVIDIA_NIM_API_KEY",
        "OPENROUTER_API_KEY",
        "GATEWAY_AUTH_TOKEN",
        "MISTRAL_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "CEREBRAS_API_KEY",
        "FIREWORKS_API_KEY",
        "KIMI_API_KEY",
        "LM_STUDIO_BASE_URL",
        "LLAMA_CPP_BASE_URL",
        "OLLAMA_BASE_URL",
        "NVIDIA_NIM_BASE_URL",
        "OPENROUTER_BASE_URL",
        "MISTRAL_BASE_URL",
        "GEMINI_BASE_URL",
        "GROQ_BASE_URL",
        "DEEPSEEK_BASE_URL",
        "CEREBRAS_BASE_URL",
        "FIREWORKS_BASE_URL",
        "MODEL_OPUS",
        "MODEL_SONNET",
        "MODEL_SONNET_1M",
        "MODEL_HAIKU",
        "MODEL",
        "REFRESH_TIME",
    ]

    from config import PROVIDER_DEFAULTS
    for p in PROVIDER_DEFAULTS:
        for field_name in ["rpm", "tpm", "rpd", "rate_window", "max_concurrency", "context", "max_output", "http_read_timeout", "http_write_timeout", "http_connect_timeout"]:
            managed_keys.append(f"PROVIDER_{p.upper()}_{field_name.upper()}")

    for k in managed_keys:
        val_in_env = k in env_keys
        val_in_os = k in os.environ and bool(os.environ[k].strip())

        if not val_in_env and val_in_os:
            statuses[k] = "Env Locked"
        elif val_in_env or val_in_os:
            statuses[k] = "Set"
        else:
            statuses[k] = "Not Set"
    return statuses



def save_env_values(configs: dict[str, Any]) -> None:
    """Read the current .env file, update existing lines, and append new keys."""
    env_path = Path(__file__).parent.parent / ".env"
    lines: list[str] = []
    if env_path.exists():
        with open(env_path) as f:
            lines = f.readlines()

    updated_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _ = stripped.split("=", 1)
            key = key.strip()
            if key in configs:
                val = configs[key]
                if isinstance(val, bool):
                    val_str = "true" if val else "false"
                else:
                    val_str = str(val)

                if isinstance(val, (int, float, bool)):
                    new_lines.append(f"{key}={val_str}\n")
                else:
                    cleaned = val_str.strip('"').strip("'")
                    new_lines.append(f'{key}="{cleaned}"\n')
                updated_keys.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Append any configurations that did not exist in the file
    for key, val in configs.items():
        if key not in updated_keys:
            if isinstance(val, bool):
                val_str = "true" if val else "false"
            else:
                val_str = str(val)

            if isinstance(val, (int, float, bool)):
                new_lines.append(f"{key}={val_str}\n")
            else:
                cleaned = val_str.strip('"').strip("'")
                new_lines.append(f'{key}="{cleaned}"\n')

    with open(env_path, "w") as f:
        f.writelines(new_lines)


async def check_endpoint_online(url: str) -> str:
    """Ping a local endpoint with a short timeout to see if it is online."""
    if not url:
        return "Offline"
    try:
        cleaned = url.rstrip("/")
        # If url is just host, strip path
        async with httpx.AsyncClient(timeout=0.3) as client:
            resp = await client.get(cleaned)
            if resp.status_code < 500:
                return "Online"
    except Exception:
        pass
    return "Offline"

STATIC_DIR = Path(__file__).parent.parent / "static"


@router.get("/dashboard", response_class=FileResponse)
async def get_dashboard_ui() -> FileResponse:
    """Serve the static Claude Gate dashboard HTML page."""
    html_file = STATIC_DIR / "dashboard.html"
    return FileResponse(html_file, media_type="text/html")


@router.get("/api/models")
async def get_available_models() -> JSONResponse:
    """Fetch and aggregate models across all supported providers dynamically."""
    providers_config = [
        ("open_router", f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/models", settings.OPENROUTER_API_KEY),
        ("nvidia_nim", f"{settings.NVIDIA_NIM_BASE_URL.rstrip('/')}/models", settings.NVIDIA_NIM_API_KEY),
        ("gemini", f"{settings.GEMINI_BASE_URL.rstrip('/')}/openai/models", settings.GEMINI_API_KEY),
        ("groq", f"{settings.GROQ_BASE_URL.rstrip('/')}/models", settings.GROQ_API_KEY),
        ("deepseek", f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/models", settings.DEEPSEEK_API_KEY),
        ("mistral", f"{settings.MISTRAL_BASE_URL.rstrip('/')}/models", settings.MISTRAL_API_KEY),
        ("cerebras", f"{settings.CEREBRAS_BASE_URL.rstrip('/')}/models", settings.CEREBRAS_API_KEY),
        ("fireworks", f"{settings.FIREWORKS_BASE_URL.rstrip('/')}/models", settings.FIREWORKS_API_KEY),
        ("kimi", "https://api.moonshot.cn/v1/models", settings.KIMI_API_KEY),
        ("lmstudio", f"{settings.LM_STUDIO_BASE_URL.rstrip('/')}/models", None),
        ("ollama", f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1/models", None),
        ("llama_cpp", f"{settings.LLAMA_CPP_BASE_URL.rstrip('/')}/models", None),
    ]

    tasks = []
    keys_order = []

    for name, url, api_key in providers_config:
        keys_order.append(name)
        if api_key:
            tasks.append(fetch_models_from_url(url, headers={"Authorization": f"Bearer {api_key}"}))
        elif api_key is None and url:
            # Local endpoints (LM Studio, Ollama, Llama.cpp) require no key
            tasks.append(fetch_models_from_url(url))
        else:
            tasks.append(asyncio.to_thread(list))

    results = await asyncio.gather(*tasks)

    models = {}
    for idx, name in enumerate(keys_order):
        fetched = results[idx]
        models[name] = fetched if fetched else FALLBACK_MODELS.get(name, [])

    return JSONResponse(content=models)



@router.get("/api/config")
async def get_config() -> JSONResponse:
    """Return in-memory settings configuration values, Lock Statuses, and Model Registry fallbacks."""
    from config import model_registry

    nim_key_val = settings.NVIDIA_NIM_API_KEY or getattr(settings, "NVIDIA_NIM_API_KEYS", "")
    config_data = {
        "NVIDIA_NIM_API_KEY": nim_key_val,
        "NVIDIA_NIM_API_KEYS": nim_key_val,
        "OPENROUTER_API_KEY": settings.OPENROUTER_API_KEY,
        "GATEWAY_AUTH_TOKEN": settings.GATEWAY_AUTH_TOKEN,
        "MISTRAL_API_KEY": settings.MISTRAL_API_KEY,
        "GEMINI_API_KEY": settings.GEMINI_API_KEY,
        "GROQ_API_KEY": settings.GROQ_API_KEY,
        "DEEPSEEK_API_KEY": settings.DEEPSEEK_API_KEY,
        "CEREBRAS_API_KEY": settings.CEREBRAS_API_KEY,
        "FIREWORKS_API_KEY": settings.FIREWORKS_API_KEY,
        "KIMI_API_KEY": settings.KIMI_API_KEY,
        "LM_STUDIO_BASE_URL": settings.LM_STUDIO_BASE_URL,
        "LLAMA_CPP_BASE_URL": settings.LLAMA_CPP_BASE_URL,
        "OLLAMA_BASE_URL": settings.OLLAMA_BASE_URL,
        "NVIDIA_NIM_BASE_URL": settings.NVIDIA_NIM_BASE_URL,
        "OPENROUTER_BASE_URL": settings.OPENROUTER_BASE_URL,
        "MISTRAL_BASE_URL": settings.MISTRAL_BASE_URL,
        "GEMINI_BASE_URL": settings.GEMINI_BASE_URL,
        "GROQ_BASE_URL": settings.GROQ_BASE_URL,
        "DEEPSEEK_BASE_URL": settings.DEEPSEEK_BASE_URL,
        "CEREBRAS_BASE_URL": settings.CEREBRAS_BASE_URL,
        "FIREWORKS_BASE_URL": settings.FIREWORKS_BASE_URL,
        "MODEL_OPUS": settings.MODEL_OPUS,
        "MODEL_SONNET": settings.MODEL_SONNET,
        "MODEL_SONNET_1M": getattr(settings, "MODEL_SONNET_1M", ""),
        "MODEL_HAIKU": settings.MODEL_HAIKU,
        "MODEL": settings.MODEL,

        "REFRESH_TIME": settings.REFRESH_TIME,
        "PROVIDER_RATE_LIMIT": settings.PROVIDER_RATE_LIMIT,
        "PROVIDER_RATE_WINDOW": settings.PROVIDER_RATE_WINDOW,
        "PROVIDER_MAX_CONCURRENCY": settings.PROVIDER_MAX_CONCURRENCY,
        "HTTP_READ_TIMEOUT": settings.HTTP_READ_TIMEOUT,
        "HTTP_WRITE_TIMEOUT": settings.HTTP_WRITE_TIMEOUT,
        "HTTP_CONNECT_TIMEOUT": settings.HTTP_CONNECT_TIMEOUT,
        "MESSAGING_PLATFORM": settings.MESSAGING_PLATFORM,
        "VOICE_NOTE_ENABLED": settings.VOICE_NOTE_ENABLED,
        "WHISPER_DEVICE": settings.WHISPER_DEVICE,
        "WHISPER_MODEL": settings.WHISPER_MODEL,
        "TELEGRAM_BOT_TOKEN": settings.TELEGRAM_BOT_TOKEN,
        "ALLOWED_TELEGRAM_USER_ID": settings.ALLOWED_TELEGRAM_USER_ID,
        "DISCORD_BOT_TOKEN": settings.DISCORD_BOT_TOKEN,
        "ALLOWED_DISCORD_CHANNELS": settings.ALLOWED_DISCORD_CHANNELS,
        "CLAUDE_WORKSPACE": settings.CLAUDE_WORKSPACE,
        "FAST_PREFIX_DETECTION": settings.FAST_PREFIX_DETECTION,
        "ENABLE_NETWORK_PROBE_MOCK": settings.ENABLE_NETWORK_PROBE_MOCK,
        "ENABLE_TITLE_GENERATION_SKIP": settings.ENABLE_TITLE_GENERATION_SKIP,
        "ENABLE_SUGGESTION_MODE_SKIP": settings.ENABLE_SUGGESTION_MODE_SKIP,
        "ENABLE_FILEPATH_EXTRACTION_MOCK": settings.ENABLE_FILEPATH_EXTRACTION_MOCK,
        "THINKING_MODE_OPUS": settings.THINKING_MODE_OPUS,
        "THINKING_MODE_SONNET": settings.THINKING_MODE_SONNET,
        "THINKING_MODE_HAIKU": settings.THINKING_MODE_HAIKU,
        "THINKING_MODE_DEFAULT": settings.THINKING_MODE_DEFAULT,
    }

    from config import PROVIDER_DEFAULTS
    for p in PROVIDER_DEFAULTS:
        p_cfg = settings.get_provider_config(p)
        for field_name, val in p_cfg.items():
            k = f"PROVIDER_{p.upper()}_{field_name.upper()}"
            config_data[k] = getattr(settings, k, val)

    fallbacks_data = {}
    for alias, entry in model_registry._entries.items():
        fallbacks_data[alias] = {
            "primary": entry.primary,
            "fallback_order": entry.fallback_order,
        }

    return JSONResponse(
        content={
            "configs": config_data,
            "key_statuses": get_key_statuses(),
            "fallbacks": fallbacks_data,
        }
    )


@router.post("/api/config")
async def save_config(req: ConfigSaveRequest) -> JSONResponse:
    """Save configuration parameters to .env & models.yaml and reload settings in-memory."""
    from config import model_registry

    try:
        # Prevent overwriting keys that are locked (set in OS env but not in env file)
        statuses = get_key_statuses()
        filtered_configs = {}
        fallback_updates = {}

        for key, val in req.configs.items():
            if key.startswith("FALLBACK_ORDER_"):
                alias = key.replace("FALLBACK_ORDER_", "").lower()
                if isinstance(val, str):
                    fb_list = [item.strip() for item in val.split(",") if item.strip()]
                elif isinstance(val, list):
                    fb_list = val
                else:
                    fb_list = []
                if alias not in fallback_updates:
                    fallback_updates[alias] = {}
                fallback_updates[alias]["fallback_order"] = fb_list
                continue

            if key in ("MODEL_OPUS", "MODEL_SONNET", "MODEL_SONNET_1M", "MODEL_HAIKU", "MODEL"):
                alias_map = {
                    "MODEL_OPUS": "claude_opus",
                    "MODEL_SONNET": "claude_sonnet",
                    "MODEL_SONNET_1M": "claude_sonnet_1m",
                    "MODEL_HAIKU": "claude_haiku",
                    "MODEL": "claude_default",
                }
                alias = alias_map[key]
                if alias not in fallback_updates:
                    fallback_updates[alias] = {}
                fallback_updates[alias]["primary"] = str(val).strip()

            if statuses.get(key) == "Locked":
                # Skip saving locked env variables to .env
                continue
            filtered_configs[key] = val

        if "NVIDIA_NIM_API_KEY" in filtered_configs:
            filtered_configs["NVIDIA_NIM_API_KEYS"] = filtered_configs["NVIDIA_NIM_API_KEY"]

        save_env_values(filtered_configs)
        if fallback_updates:
            model_registry.save_entries(fallback_updates)

        settings.reload()
        model_registry.reload()

        return JSONResponse(
            content={
                "status": "success",
                "message": "Configuration saved & reloaded in-memory.",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to save settings: {e}"},
        )


@router.get("/api/settings/subagents")
async def get_subagents_setting() -> JSONResponse:
    """Return current global SUBAGENTS_ENABLED status."""
    return JSONResponse(content={"subagents_enabled": SUBAGENTS_ENABLED})


@router.post("/api/settings/subagents")
async def toggle_subagents_setting(req: SubagentsToggleRequest) -> JSONResponse:
    """Update global SUBAGENTS_ENABLED emergency switch status."""
    global SUBAGENTS_ENABLED
    SUBAGENTS_ENABLED = bool(req.enabled)
    logger.info("Subagents Emergency Switch updated: SUBAGENTS_ENABLED={}", SUBAGENTS_ENABLED)
    return JSONResponse(
        content={
            "status": "success",
            "subagents_enabled": SUBAGENTS_ENABLED,
            "message": f"Subagent execution is now {'ENABLED (ON)' if SUBAGENTS_ENABLED else 'DISABLED (OFF Bypass)'}.",
        }
    )



@router.get("/api/stats")
async def get_stats() -> JSONResponse:
    """Get dynamic real-time traffic statistics of the proxy gateway."""
    tg_online = bool(settings.TELEGRAM_BOT_TOKEN)
    ds_online = bool(settings.DISCORD_BOT_TOKEN)

    # Check local endpoints pings
    lm_status = await check_endpoint_online(settings.LM_STUDIO_BASE_URL)
    llama_status = await check_endpoint_online(settings.LLAMA_CPP_BASE_URL)
    ollama_status = await check_endpoint_online(settings.OLLAMA_BASE_URL)

    router_status = model_router.get_status()

    return JSONResponse(
        content={
            "total_requests": stats.total_requests,
            "mocked_requests": stats.mocked_requests,
            "error_count": stats.error_count,
            "active_concurrency": stats.active_concurrency,
            "subagents_enabled": SUBAGENTS_ENABLED,
            "tg_bot_status": "Online" if tg_online else "Offline",
            "ds_bot_status": "Online" if ds_online else "Offline",
            "endpoints": {
                "lmstudio": lm_status,
                "llama_cpp": llama_status,
                "ollama": ollama_status,
            },
            "recent_requests": stats.get_recent_dicts(include_payload=False),
            "router_status": router_status,
        }
    )


class CircuitBreakerActionRequest(BaseModel):
    model_id: str
    action: str  # "reset" (enable/open traffic) or "trip" (block/close traffic)


@router.post("/api/circuit-breaker/action")
async def handle_circuit_breaker_action(req: CircuitBreakerActionRequest) -> JSONResponse:
    """Manually trip or reset a model's circuit breaker from Dashboard."""
    from core.router.circuit_breaker import circuit_breaker_registry

    if not req.model_id:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "model_id is required"},
        )

    cb = circuit_breaker_registry.get(req.model_id)

    action_lower = req.action.lower().strip()
    if action_lower in ("reset", "open_traffic", "enable", "clear", "open"):
        # User requested to clear timeout & reset circuit breaker to working/CLOSED state
        cb.reset()
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Circuit breaker for '{req.model_id}' reset to CLOSED (Healthy). Timeout cleared.",
                "circuit_breaker": cb.status_dict(),
            }
        )
    elif action_lower in ("trip", "block", "close_traffic", "disable", "close"):
        # User requested to block model / force circuit breaker to OPEN state (extending timeout 1m -> 5m -> 10m -> 15m -> 30m -> 60m)
        new_timeout = cb.trip_or_extend(reason="Manually blocked via Dashboard")
        mins = int(new_timeout // 60)
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Circuit breaker for '{req.model_id}' forced OPEN (Blocked) for {mins} min timeout.",
                "circuit_breaker": cb.status_dict(),
            }
        )
    else:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid action '{req.action}'. Expected 'reset' or 'trip'."},
        )


@router.get("/api/router-status")
async def get_router_status() -> JSONResponse:
    """Per-model circuit breaker, rate limit status, and client request resolution mappings."""
    from config import model_registry

    status_data = model_router.get_status()

    # Compute summary
    open_count = sum(
        1
        for v in status_data.values()
        if isinstance(v, dict) and v.get("circuit_breaker", {}).get("state") == "open"
    )
    healthy_count = len(status_data) - open_count

    # Build resolution mappings matrix matching Claude Code CLI menu (1 to 5)
    client_models = [
        ("claude_default", "1. DEFAULT (RECOMMENDED)", "Default Model (Opus 5 / Nemotron 70B)"),
        ("claude_opus", "2. OPUS (1M CONTEXT)", "Opus 5 (1M context)"),
        ("claude_sonnet", "3. SONNET", "Sonnet (Llama 3.1 70B)"),
        ("claude_sonnet_1m", "4. SONNET 5 (1M CONTEXT)", "Sonnet 5 (1M context)"),
        ("claude_haiku", "5. HAIKU", "Haiku 4.5 (Llama 3.1 8B)"),
    ]
    mappings_matrix = []
    for c_model, label, desc in client_models:
        primary = model_registry.get_primary(c_model)
        fallbacks = model_registry.get_fallbacks(c_model)
        all_chain = [c for c in ([primary] + fallbacks) if c]

        resolved = "ALL_UNAVAILABLE"
        is_fallback = False
        step_name = "NONE"

        for idx, cand in enumerate(all_chain):
            if model_router._is_available(cand):
                resolved = cand
                is_fallback = idx > 0
                step_name = "PRIMARY DIRECT" if idx == 0 else f"FALLBACK #{idx}"
                break

        mappings_matrix.append(
            {
                "client_model": c_model,
                "label": label,
                "description": desc,
                "primary": primary,
                "resolved_target": resolved,
                "is_fallback": is_fallback,
                "step_name": step_name,
                "chain": all_chain,
            }
        )


    from core.router.daily_tracker import daily_request_tracker
    return JSONResponse(
        content={
            "summary": {
                "total_models": len(status_data),
                "healthy": healthy_count,
                "circuit_open": open_count,
            },
            "client_mappings": mappings_matrix,
            "models": status_data,
            "daily_rpd": daily_request_tracker.all_statuses(),
        }
    )


@router.get("/api/doctor")
@router.get("/api/diagnostics")
async def run_doctor_checks() -> JSONResponse:
    """Run full proxy diagnostic health check."""
    reports = []
    has_errors = False

    reports.append("[Doctor] Initializing proxy gateway diagnostics...")
    reports.append("[Doctor] Checking API Keys & Providers...")

    for key_name in ["MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"]:
        val = getattr(settings, key_name, "")
        if val:
            reports.append(f"  ├─ {key_name}: Set ({val})")
        else:
            reports.append(f"  ├─ {key_name}: NOT SET (Warning)")

    if settings.OPENROUTER_API_KEY:
        reports.append("  ├─ OPENROUTER_API_KEY: Present")
    else:
        reports.append("  ├─ OPENROUTER_API_KEY: Missing (Warning)")

    if settings.NVIDIA_NIM_API_KEY:
        reports.append("  ├─ NVIDIA_NIM_API_KEY: Present")
    else:
        reports.append("  ├─ NVIDIA_NIM_API_KEY: Missing (Warning)")

    if settings.DEEPSEEK_API_KEY:
        reports.append("  ├─ DEEPSEEK_API_KEY: Present")

    reports.append("[Doctor] Checking Upstream Endpoints & Circuits...")
    status_data = model_router.get_status()
    for model_id, m_status in status_data.items():
        cb_state = m_status.get("circuit_breaker", {}).get("state")
        rl_headroom = m_status.get("rate_limit", {}).get("has_headroom")
        if cb_state == "open":
            has_errors = True
            reports.append(f"  ├─ ❌ {model_id}: CIRCUIT OPEN (Failures)")
        elif not rl_headroom:
            reports.append(f"  ├─ ⚠️ {model_id}: Rate limit headroom tight (<10%)")
        else:
            reports.append(f"  ├─ ✅ {model_id}: Healthy & Operational")

    reports.append("[Telemetry] System Model Mappings:")
    reports.append(f"  └─ OPUS:   {settings.MODEL_OPUS}")
    reports.append(f"  └─ SONNET: {settings.MODEL_SONNET}")
    reports.append(f"  └─ HAIKU:  {settings.MODEL_HAIKU}")

    return JSONResponse(content={"logs": reports, "has_errors": has_errors})


@router.get("/api/dev/payloads")
async def get_dev_payloads(
    limit: int = 20,
    page: int = 1,
    query: str = ""
) -> JSONResponse:
    """Return captured dev mode request & response payloads for inspection with server-side query filtering and RAM-saving pagination."""
    res = stats.get_paginated_payloads(limit=limit, page=page, query=query)
    return JSONResponse(content=res)


@router.get("/api/dev/payloads/{req_id}")
async def get_single_dev_payload(req_id: str) -> JSONResponse:
    """Return a single captured request payload by ID."""
    recent_dicts = stats.get_recent_dicts(include_payload=True)
    for p in recent_dicts:
        if p.get("id") == req_id:
            return JSONResponse(content=p)
    return JSONResponse(content={"error": "Not found"}, status_code=404)


@router.get("/api/dev/raw-logs")
async def get_raw_dev_logs() -> Response:
    """Return raw persisted .development/requests.jsonl log contents."""
    log_file = Path(__file__).parent.parent / ".development" / "requests.jsonl"
    if not log_file.exists():
        return Response(content="", media_type="text/plain")
    return Response(content=log_file.read_text(encoding="utf-8"), media_type="text/plain")


@router.get("/api/dev/metrics")
async def get_dev_metrics() -> JSONResponse:
    """Return real-time performance metrics, RPM per model, estimated queue wait delays, and NIM key pool telemetry."""
    import time
    now = time.time()

    recent_logs = stats.recent_requests
    logs_last_60s = [log for log in recent_logs if (now - getattr(log, "created_at", now)) <= 60.0]

    model_stats: dict[str, dict[str, Any]] = {}
    router_status = model_router.get_status()

    from config import model_registry

    all_target_models = set(router_status.keys())
    for entry in model_registry._entries.values():
        if entry.primary:
            all_target_models.add(entry.primary)
        for fb in entry.fallback_order:
            if fb:
                all_target_models.add(fb)

    for cat, m_list in FALLBACK_MODELS.items():
        for m in m_list:
            if "/" in m:
                if not m.startswith("nvidia_nim/") and not m.startswith("openrouter/") and not m.startswith("gemini/") and not m.startswith("deepseek/"):
                    prefix = "nvidia_nim" if cat == "nvidia_nim" else "openrouter"
                    all_target_models.add(f"{prefix}/{m}")
                else:
                    all_target_models.add(m)

    for model_id in sorted(all_target_models):
        model_stats[model_id] = {
            "model_id": model_id,
            "rpm_60s": 0,
            "total_requests": 0,
            "success_count": 0,
            "error_count": 0,
            "rate_limit_429_count": 0,
            "rate_limit_429_60s": 0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "success_rate": 100.0,
            "estimated_wait_s": 0.0,
            "circuit_state": "closed",
            "recovery_remaining_s": None,
            "req_remaining": None,
            "has_headroom": True,
        }

    global_429_count = 0
    global_429_60s = 0

    for log in recent_logs:
        mid = log.mapped_model
        if not mid or mid in ("unknown", "none"):
            continue
        if mid not in model_stats:
            model_stats[mid] = {
                "model_id": mid,
                "rpm_60s": 0,
                "total_requests": 0,
                "success_count": 0,
                "error_count": 0,
                "rate_limit_429_count": 0,
                "rate_limit_429_60s": 0,
                "total_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "success_rate": 100.0,
                "estimated_wait_s": 0.0,
                "circuit_state": "closed",
                "recovery_remaining_s": None,
                "req_remaining": None,
                "has_headroom": True,
            }

        st = model_stats[mid]
        st["total_requests"] += 1
        st["total_latency_ms"] += log.duration_ms
        if log.status_code < 400:
            st["success_count"] += 1
        else:
            st["error_count"] += 1
            if log.status_code == 429:
                st["rate_limit_429_count"] += 1
                global_429_count += 1

    for log in logs_last_60s:
        mid = log.mapped_model
        if mid in model_stats:
            model_stats[mid]["rpm_60s"] += 1
            if log.status_code == 429:
                model_stats[mid]["rate_limit_429_60s"] += 1
                global_429_60s += 1

    from atomic.guards.nim_guard import nim_throttle_guard
    from core.key_manager import nim_key_manager

    nim_timestamps = getattr(nim_throttle_guard, "_timestamps", [])
    nim_rpm_current = len(nim_timestamps)
    nim_rpm_limit = nim_throttle_guard.rpm_limit
    nim_wait_s = 0.0
    if nim_rpm_current >= nim_rpm_limit and nim_timestamps:
        mono_now = time.monotonic()
        oldest = nim_timestamps[0]
        nim_wait_s = max(0.0, round((oldest + nim_throttle_guard.window_seconds) - mono_now, 1))

    keys = nim_key_manager.get_configured_keys()
    passive_until_dict = getattr(nim_key_manager, "_passive_until", {})
    passive_keys = [k for k in keys if passive_until_dict.get(k, 0.0) > now]
    active_keys = [k for k in keys if passive_until_dict.get(k, 0.0) <= now]

    key_details = []
    for k in keys:
        cool = passive_until_dict.get(k, 0.0)
        rem_cool = max(0.0, round(cool - now, 1)) if cool > now else 0.0
        masked = k[:7] + "..." + k[-4:] if len(k) > 12 else k
        key_details.append({
            "key_masked": masked,
            "status": "Active" if rem_cool == 0 else f"Passive (Cooldown: {rem_cool}s)",
            "cooldown_s": rem_cool,
        })

    for mid, st in model_stats.items():
        if st["total_requests"] > 0:
            st["avg_latency_ms"] = round(st["total_latency_ms"] / st["total_requests"], 1)
            st["success_rate"] = round((st["success_count"] / st["total_requests"]) * 100, 1)

        r_info = router_status.get(mid, {})
        cb = r_info.get("circuit_breaker", {})
        rl = r_info.get("rate_limit", {})

        st["circuit_state"] = cb.get("state", "closed")
        st["recovery_remaining_s"] = cb.get("recovery_remaining_s")
        st["has_headroom"] = rl.get("has_headroom", True)
        st["req_remaining"] = rl.get("req_remaining")

        if st["circuit_state"] == "open" and st["recovery_remaining_s"]:
            st["estimated_wait_s"] = st["recovery_remaining_s"]
        elif "nvidia_nim" in mid.lower() and nim_wait_s > 0:
            st["estimated_wait_s"] = nim_wait_s

    global_rpm = len(logs_last_60s)
    total_latency_sum = sum(log.duration_ms for log in recent_logs)
    global_avg_latency = round(total_latency_sum / len(recent_logs), 1) if recent_logs else 0.0

    return JSONResponse(
        content={
            "global_metrics": {
                "global_rpm": global_rpm,
                "active_concurrency": stats.active_concurrency,
                "global_avg_latency_ms": global_avg_latency,
                "global_429_count": global_429_count,
                "global_429_60s": global_429_60s,
                "total_requests": stats.total_requests,
                "mocked_requests": stats.mocked_requests,
            },
            "nim_telemetry": {
                "current_rpm": nim_rpm_current,
                "rpm_limit": nim_rpm_limit,
                "estimated_delay_s": nim_wait_s,
                "total_keys": len(keys),
                "active_keys": len(active_keys),
                "passive_keys": len(passive_keys),
                "key_details": key_details,
            },
            "model_metrics": list(model_stats.values()),
        }
    )

