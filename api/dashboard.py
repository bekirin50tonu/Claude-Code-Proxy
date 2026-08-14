import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from config import settings, stats
from core.router.selector import model_selector as model_router

router = APIRouter()

# Fallback models in case API calls fail or keys are missing
FALLBACK_MODELS = {
    "nvidia_nim": [
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.1-70b-instruct",
        "z-ai/glm4.7",
        "mistralai/mixtral-8x22b-instruct-v0.1",
    ],
    "open_router": [
        "arcee-ai/trinity-large-preview:free",
        "google/gemini-2.5-flash:free",
        "meta-llama/llama-3-8b-instruct:free",
        "deepseek/deepseek-chat",
        "qwen/qwen-2.5-72b-instruct",
        "mistralai/pixtral-12b:free",
    ],
    "lmstudio": [
        "qwen2.5-7b-instruct",
        "llama-3.2-3b-instruct",
        "phi-3-mini-4k-instruct",
        "mistral-7b-instruct",
    ],
}


class ConfigSaveRequest(BaseModel):
    configs: dict[str, Any]


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
    ]

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
    """Serve the static Hermes Gate dashboard HTML page."""
    html_file = STATIC_DIR / "dashboard.html"
    return FileResponse(html_file, media_type="text/html")


@router.get("/api/models")
async def get_available_models() -> JSONResponse:
    """Fetch and aggregate models across openrouter and nvidia nim only (LM Studio skipped if offline)."""
    nim_key = settings.NVIDIA_NIM_API_KEY
    or_key = settings.OPENROUTER_API_KEY

    tasks = []

    if or_key:
        tasks.append(
            fetch_models_from_url(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {or_key}"},
            )
        )
    else:
        tasks.append(asyncio.to_thread(list))

    if nim_key:
        tasks.append(
            fetch_models_from_url(
                "https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {nim_key}"},
            )
        )
    else:
        tasks.append(asyncio.to_thread(list))

    results = await asyncio.gather(*tasks)

    models = {
        "open_router": results[0] if results[0] else FALLBACK_MODELS["open_router"],
        "nvidia_nim": results[1] if results[1] else FALLBACK_MODELS["nvidia_nim"],
    }

    return JSONResponse(content=models)


@router.get("/api/config")
async def get_config() -> JSONResponse:
    """Return in-memory settings configuration values, Lock Statuses, and Model Registry fallbacks."""
    from config import model_registry

    config_data = {
        "NVIDIA_NIM_API_KEY": settings.NVIDIA_NIM_API_KEY,
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


    return JSONResponse(
        content={
            "summary": {
                "total_models": len(status_data),
                "healthy": healthy_count,
                "circuit_open": open_count,
            },
            "client_mappings": mappings_matrix,
            "models": status_data,
        }
    )


@router.get("/api/doctor")
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
async def get_dev_payloads() -> JSONResponse:
    """Return captured dev mode request & response payloads for inspection."""
    recent_dicts = stats.get_recent_dicts(include_payload=True)
    return JSONResponse(
        content={
            "total_captured": len(recent_dicts),
            "payloads": recent_dicts,
        }
    )


@router.get("/api/dev/payloads/{req_id}")
async def get_single_dev_payload(req_id: str) -> JSONResponse:
    """Return a single captured request payload by ID."""
    for item in stats.recent_requests:
        if item.id == req_id:
            return JSONResponse(content=item.to_dict(include_payload=True))
    return JSONResponse(content={"error": "Payload not found"}, status_code=404)
