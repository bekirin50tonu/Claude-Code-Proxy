import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import settings, stats
from router.model_router import model_router

router = APIRouter()
logger = logging.getLogger("proxy_dashboard")

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
    """Resolve Configured / Locked / Not Configured state badges for each setting."""
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
        "MODEL_HAIKU",
        "MODEL",
    ]
    for k in managed_keys:
        val_in_env = k in env_keys
        val_in_os = k in os.environ and bool(os.environ[k].strip())

        # Determine if it's currently an active engine model
        is_active_engine = False
        active_model = settings.MODEL.split("/")[0] if "/" in settings.MODEL else ""
        if active_model and (
            active_model == "nvidia_nim"
            and k in ["NVIDIA_NIM_API_KEY", "MODEL_OPUS"]
            or active_model == "open_router"
            and k in ["OPENROUTER_API_KEY", "MODEL_SONNET"]
        ):
            is_active_engine = True

        if is_active_engine:
            statuses[k] = "Active Engine"
        elif not val_in_env and val_in_os:
            statuses[k] = "Locked"
        elif val_in_env or val_in_os:
            statuses[k] = "Configured"
        else:
            statuses[k] = "Not Configured"
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
                fallback_updates[alias] = {"fallback_order": fb_list}
                continue

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
            "recent_requests": stats.recent_requests,
            "router_status": router_status,
        }
    )


@router.get("/api/router-status")
async def get_router_status() -> JSONResponse:
    """Per-model circuit breaker and rate limit status."""
    status_data = model_router.get_status()

    # Compute summary
    open_count = sum(
        1
        for v in status_data.values()
        if isinstance(v, dict) and v.get("circuit_breaker", {}).get("state") == "open"
    )
    total = len(status_data)
    healthy = total - open_count

    return JSONResponse(
        content={
            "summary": {
                "total_models": total,
                "healthy": healthy,
                "circuit_open": open_count,
            },
            "models": status_data,
        }
    )


@router.get("/api/diagnostics")
async def get_diagnostics() -> JSONResponse:
    """Run diagnostics self-check connection targets and return logs report."""
    reports = []
    has_errors = False

    reports.append("[Telemetry] Initializing Diagnostic System Check...")
    reports.append(
        f"• NVIDIA NIM key state: {'ACTIVE' if settings.NVIDIA_NIM_API_KEY else 'ABSENT'}"
    )
    reports.append(
        f"• OpenRouter key state: {'ACTIVE' if settings.OPENROUTER_API_KEY else 'ABSENT'}"
    )

    lm_url = settings.LM_STUDIO_BASE_URL.rstrip("/")
    reports.append(f"[Telemetry] Pinging local endpoints: {lm_url}")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{lm_url}/models")
            if resp.status_code == 200:
                reports.append("✅ [Connection Success] LM Studio responsive.")
            else:
                reports.append(
                    f"❌ [Connection Alert] LM Studio returned code {resp.status_code}"
                )
                has_errors = True
    except Exception as e:
        reports.append(f"❌ [Connection Alert] Failed to connect: {e}")
        has_errors = True

    reports.append("[Telemetry] System Model Mappings:")
    reports.append(f"  └─ OPUS:   {settings.MODEL_OPUS}")
    reports.append(f"  └─ SONNET: {settings.MODEL_SONNET}")
    reports.append(f"  └─ HAIKU:  {settings.MODEL_HAIKU}")

    return JSONResponse(content={"logs": reports, "has_errors": has_errors})


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard_ui() -> HTMLResponse:
    """Render a clean grayscale, minimal, and focused Hermes Agent style UI."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hermes Gate - Claude Code Proxy Control Console</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --void-bg: #060709;
            --carbon-base: #0c0d11;
            --carbon-card: #0c0d11;
            --border-subtle: rgba(255, 255, 255, 0.07);
            --border-focus: rgba(255, 255, 255, 0.35);
            --text-main: #e5e7eb;
            --text-muted: #9ca3af;
            --text-dim: #4b5563;
            --surface-hover: rgba(255, 255, 255, 0.03);
            --bg-active: #ffffff;
            --text-active: #000000;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--void-bg);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Top Navigation Header */
        header {
            border-bottom: 1px solid var(--border-subtle);
            padding: 1.25rem 3rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--void-bg);
            z-index: 100;
        }

        .brand h1 {
            font-weight: 800;
            font-size: 1.25rem;
            letter-spacing: -0.5px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand p {
            font-size: 0.65rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 2px;
            font-weight: 600;
        }

        .header-links {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .header-link {
            color: var(--text-muted);
            text-decoration: none;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: 1px solid var(--border-subtle);
            padding: 6px 12px;
            border-radius: 6px;
            transition: all 0.2s;
        }

        .header-link:hover {
            color: #ffffff;
            border-color: var(--text-muted);
            background: var(--surface-hover);
        }

        /* Real-time Telemetry Grid */
        .live-dials-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.25rem;
            padding: 2rem 3rem 0 3rem;
            max-width: 1600px;
            width: 100%;
            margin: 0 auto;
        }

        .dial-card {
            background: var(--carbon-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
        }

        .dial-card h3 {
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-dim);
            margin-bottom: 0.5rem;
        }

        .dial-val {
            font-size: 1.5rem;
            font-weight: 800;
            color: white;
            font-family: 'JetBrains Mono', monospace;
            display: flex;
            align-items: baseline;
            gap: 6px;
        }

        .dial-val span {
            font-size: 0.8rem;
            color: var(--text-dim);
            font-weight: 400;
        }

        .dial-badge {
            position: absolute;
            top: 1.25rem;
            right: 1.25rem;
            font-size: 0.65rem;
            font-weight: 800;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
        }

        .status-text {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            font-weight: 700;
        }

        /* Main Workspace Container */
        main {
            display: flex;
            flex: 1;
            padding: 2rem 3rem;
            gap: 2rem;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        /* Left Side Controls Panel */
        .control-sidebar {
            width: 250px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .control-btn {
            background: var(--carbon-card);
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            padding: 0.85rem 1.1rem;
            border-radius: 8px;
            text-align: left;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .control-btn:hover {
            color: white;
            background: var(--surface-hover);
            border-color: var(--border-focus);
        }

        .control-btn.active {
            color: var(--text-active);
            background: var(--bg-active);
            border-color: var(--bg-active);
        }

        /* Content panel styling */
        .panel-container {
            flex: 1;
            background: var(--carbon-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 2.5rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .panel-pane {
            display: none;
        }

        .panel-pane.active {
            display: block;
        }

        .panel-header {
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 1rem;
        }

        .panel-header h2 {
            font-size: 1.2rem;
            font-weight: 700;
            color: white;
            letter-spacing: -0.5px;
        }

        .panel-header p {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* Interactive Forms and Grids */
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
            position: relative;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        .label-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        label {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }

        /* Status badges for inputs */
        .badge-status {
            font-size: 0.6rem;
            font-weight: 700;
            text-transform: uppercase;
            padding: 1px 6px;
            border-radius: 4px;
            border: 1px solid var(--border-subtle);
        }
        .badge-status.configured {
            color: #ffffff;
            border-color: #ffffff;
        }
        .badge-status.locked {
            color: var(--text-dim);
            background: rgba(255,255,255,0.02);
            border-color: var(--text-dim);
        }
        .badge-status.not-configured {
            color: var(--text-dim);
            border-style: dashed;
        }
        .badge-status.active-engine {
            color: #000000;
            background: #ffffff;
            border-color: #ffffff;
        }

        .input-wrapper {
            display: flex;
            width: 100%;
            gap: 0;
        }

        .input-wrapper input {
            flex: 1;
            min-width: 0;
            border-radius: 8px 0 0 8px;
        }

        .input-icon-btn {
            flex-shrink: 0;
            width: 60px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-subtle);
            border-left: none;
            border-radius: 0 8px 8px 0;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 0;
            transition: all 0.2s;
        }
        .input-icon-btn:hover {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.3);
        }

        input[type="text"], input[type="password"], input[type="number"], select {
            background: #050608;
            border: 1px solid var(--border-subtle);
            color: white;
            padding: 0.8rem 0.95rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-family: inherit;
            outline: none;
            transition: all 0.2s;
        }

        input:focus, select:focus {
            border-color: white;
            background: #000000;
        }

        input:disabled {
            color: var(--text-dim);
            background: rgba(255,255,255,0.01);
            cursor: not-allowed;
        }

        /* Tag Input Styling */
        .tag-input-container {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px;
            background: #050608;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 6px 10px;
            min-height: 44px;
            cursor: text;
            transition: all 0.2s;
            position: relative;
        }

        .tag-input-container:focus-within {
            border-color: white;
            background: #000000;
        }

        .tag-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border-subtle);
            color: #ffffff;
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            padding: 3px 8px;
            border-radius: 6px;
            user-select: none;
            transition: all 0.15s;
        }

        .tag-chip:hover {
            border-color: rgba(255, 255, 255, 0.4);
            background: rgba(255, 255, 255, 0.15);
        }

        .tag-chip .tag-remove {
            cursor: pointer;
            color: var(--text-muted);
            font-size: 0.8rem;
            font-weight: 700;
            line-height: 1;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            transition: all 0.15s;
        }

        .tag-chip .tag-remove:hover {
            color: #f87171;
            background: rgba(248, 113, 113, 0.2);
        }

        .tag-input-field {
            border: none !important;
            background: transparent !important;
            outline: none !important;
            color: white !important;
            padding: 4px 6px !important;
            font-size: 0.8rem !important;
            font-family: inherit !important;
            flex: 1;
            min-width: 160px;
        }

        /* Autocomplete dropdown styling */
        .autocomplete-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #08090d;
            border: 1px solid var(--border-focus);
            border-radius: 8px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 100;
            display: none;
            margin-top: 4px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        }

        .autocomplete-group {
            padding: 6px 12px;
            font-size: 0.6rem;
            font-weight: 800;
            color: var(--text-muted);
            background: rgba(255,255,255,0.03);
            letter-spacing: 1px;
            text-transform: uppercase;
            border-bottom: 1px solid var(--border-subtle);
        }

        .autocomplete-item {
            padding: 8px 12px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.15s;
            color: var(--text-main);
        }

        .autocomplete-item:hover {
            background: white;
            color: black;
            font-weight: 600;
        }

        .autocomplete-item.highlighted {
            background: white !important;
            color: black !important;
            font-weight: 600;
            padding-left: 16px;
        }

        /* Checkbox switch options */
        .setting-switch {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.95rem 1.15rem;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            background: rgba(255,255,255,0.005);
            transition: all 0.2s;
        }

        .setting-switch:hover {
            background: var(--surface-hover);
            border-color: var(--border-focus);
        }

        .setting-switch-desc {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .setting-switch-desc h4 {
            font-size: 0.85rem;
            font-weight: 600;
            color: white;
        }

        .setting-switch-desc p {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .setting-switch input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: white;
            cursor: pointer;
        }

        /* Collapsible accordion settings */
        .accordion-section {
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            margin-top: 1rem;
            overflow: hidden;
        }

        .accordion-header {
            background: rgba(255,255,255,0.01);
            padding: 0.95rem 1.25rem;
            cursor: pointer;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
            transition: background 0.2s;
        }

        .accordion-header:hover {
            background: var(--surface-hover);
        }

        .accordion-content {
            padding: 1.5rem;
            display: none;
            border-top: 1px solid var(--border-subtle);
            background: #07080a;
        }

        .accordion-content.open {
            display: block;
        }

        /* Live Request Feed Log Table */
        .feed-container {
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            overflow: hidden;
            margin-top: 1rem;
            background: #040507;
        }

        .feed-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.75rem;
            text-align: left;
        }

        .feed-table th {
            padding: 8px 12px;
            background: rgba(255,255,255,0.02);
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.65rem;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .feed-table td {
            padding: 8px 12px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-main);
            font-family: 'JetBrains Mono', monospace;
        }

        .feed-table tr:hover {
            background: rgba(255,255,255,0.01);
        }

        .feed-badge {
            font-size: 0.65rem;
            padding: 1px 4px;
            border-radius: 4px;
            font-weight: 700;
        }

        .feed-badge.status-200 { background: rgba(255,255,255,0.1); color: #ffffff; }
        .feed-badge.status-error { background: rgba(127,29,29,0.3); color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }
        .feed-badge.mock { border: 1px solid var(--border-focus); color: var(--text-muted); }

        /* Button configurations */
        .console-buttons-bar {
            margin-top: 2rem;
            display: flex;
            gap: 1rem;
            border-top: 1px solid var(--border-subtle);
            padding-top: 1.5rem;
        }

        .btn-action {
            background: white;
            color: black;
            border: none;
            padding: 0.85rem 1.75rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn-action:hover {
            background: #e2e8f0;
        }

        .btn-alt {
            background: transparent;
            border: 1px solid var(--border-subtle);
            color: white;
            padding: 0.85rem 1.5rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn-alt:hover {
            background: var(--surface-hover);
            border-color: white;
        }

        /* Diagnostics Console box */
        .terminal-box {
            background: #040507;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 1.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #ffffff;
            min-height: 250px;
            max-height: 380px;
            overflow-y: auto;
            margin-top: 1rem;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.8);
        }

        .terminal-line {
            margin-bottom: 5px;
            line-height: 1.5;
        }

        /* Gray Toaster Notification box */
        .alert-toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #0d0e12;
            border: 1px solid #ffffff;
            color: #ffffff;
            padding: 0.85rem 1.5rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
            transform: translateY(180%);
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .alert-toast.show {
            transform: translateY(0);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>HERMES GATE <span>//</span> PROXY</h1>
            <p>Monochrome Client Gateway Terminus</p>
        </div>
        <div class="header-links">
            <a href="/docs" target="_blank" class="header-link">Docs</a>
        </div>
    </header>

    <!-- Real-time Telemetry Grid -->
    <section class="live-dials-grid">
        <div class="dial-card">
            <h3>GATEWAY PORT</h3>
            <div class="dial-val">
                <span class="status-text">● ONLINE</span>
                <span>:8090</span>
            </div>
            <div class="dial-badge">STANDBY</div>
        </div>
        <div class="dial-card">
            <h3>TRAFFIC VOLUME</h3>
            <div class="dial-val" id="telemetry-requests">0 <span>reqs</span></div>
            <div class="dial-badge">ACCUMULATOR</div>
        </div>
        <div class="dial-card">
            <h3>MOCK SAVINGS</h3>
            <div class="dial-val" id="telemetry-savings">0%</div>
            <div class="dial-badge" id="telemetry-mock-badge">0 / 0 SAVES</div>
        </div>
        <div class="dial-card">
            <h3>BOT SUBSYSTEMS</h3>
            <div class="dial-val" style="font-size: 0.95rem; flex-direction: column; gap: 4px; line-height: 1.2; justify-content: center; height: 100%;">
                <div style="display: flex; justify-content: space-between; width: 100%;">
                    <span style="color: var(--text-muted); font-size: 0.75rem;">DISCORD:</span>
                    <span id="telemetry-discord" style="font-weight: 700; font-family: inherit;">...</span>
                </div>
                <div style="display: flex; justify-content: space-between; width: 100%;">
                    <span style="color: var(--text-muted); font-size: 0.75rem;">TELEGRAM:</span>
                    <span id="telemetry-telegram" style="font-weight: 700; font-family: inherit;">...</span>
                </div>
            </div>
        </div>
    </section>

    <!-- Main Workspace -->
    <main>
        <div class="control-sidebar">
            <button type="button" class="control-btn active" onclick="switchPane(event, 'pane-models')">Model Matrix</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-router')">Router & Fallbacks</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-keys')">API Credentials</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-endpoints')">Local Endpoints</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-tunnels')">Proxy Tunnels & Base URLs</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-mocks')">Mock Engines</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-limits')">Throttle & Timeouts</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-bots')">Bot Matrix</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-logs')">System Trace Feed</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-guide')">Setup Guide</button>
            <button type="button" class="control-btn" onclick="switchPane(event, 'pane-doctor')">Diagnostics</button>
        </div>

        <div class="panel-container">
            <form id="configForm" onsubmit="saveConfigs(event)">
                <!-- PANE 1: MODEL MATRIX -->
                <div id="pane-models" class="panel-pane active">
                    <div class="panel-header" style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2>Model Matrix Config</h2>
                            <p>Map Anthropic Client requests to OpenAI-compatible provider endpoints.</p>
                        </div>
                        <button type="button" class="btn-alt" onclick="refreshModelsList()" style="padding: 0.5rem 1rem; font-size: 0.75rem;">Refresh Models</button>
                    </div>

                    <!-- Provider Model Counters Bar (only cloud providers) -->
                    <div id="provider-model-counters" style="display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
                        <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-subtle); padding: 0.6rem 1rem; border-radius: 8px; font-size: 0.75rem; display: flex; align-items: center; gap: 8px;">
                            <span style="color: var(--text-muted); font-weight: 700; text-transform: uppercase; font-size: 0.65rem;">OpenRouter:</span>
                            <span id="count-open_router" style="font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #ffffff;">...</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-subtle); padding: 0.6rem 1rem; border-radius: 8px; font-size: 0.75rem; display: flex; align-items: center; gap: 8px;">
                            <span style="color: var(--text-muted); font-weight: 700; text-transform: uppercase; font-size: 0.65rem;">NVIDIA NIM:</span>
                            <span id="count-nvidia_nim" style="font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #ffffff;">...</span>
                        </div>
                    </div>
                    <div class="form-grid">
                        <div class="form-group">
                            <div class="label-row">
                                <label for="MODEL_OPUS">MODEL_OPUS (Opus primary mapping)</label>
                                <span id="badge-MODEL_OPUS" class="badge-status">...</span>
                            </div>
                            <input type="text" id="MODEL_OPUS" autocomplete="off" onfocus="handleAutocomplete('MODEL_OPUS')" onblur="blurAutocomplete('MODEL_OPUS')">
                            <div id="MODEL_OPUS-dropdown" class="autocomplete-dropdown"></div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="MODEL_SONNET">MODEL_SONNET (Sonnet primary mapping)</label>
                                <span id="badge-MODEL_SONNET" class="badge-status">...</span>
                            </div>
                            <input type="text" id="MODEL_SONNET" autocomplete="off" onfocus="handleAutocomplete('MODEL_SONNET')" onblur="blurAutocomplete('MODEL_SONNET')">
                            <div id="MODEL_SONNET-dropdown" class="autocomplete-dropdown"></div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="MODEL_HAIKU">MODEL_HAIKU (Haiku primary mapping)</label>
                                <span id="badge-MODEL_HAIKU" class="badge-status">...</span>
                            </div>
                            <input type="text" id="MODEL_HAIKU" autocomplete="off" onfocus="handleAutocomplete('MODEL_HAIKU')" onblur="blurAutocomplete('MODEL_HAIKU')">
                            <div id="MODEL_HAIKU-dropdown" class="autocomplete-dropdown"></div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="MODEL">MODEL (Global default fallback)</label>
                                <span id="badge-MODEL" class="badge-status">...</span>
                            </div>
                            <input type="text" id="MODEL" autocomplete="off" onfocus="handleAutocomplete('MODEL')" onblur="blurAutocomplete('MODEL')">
                            <div id="MODEL-dropdown" class="autocomplete-dropdown"></div>
                        </div>
                        <div class="form-group full-width" style="grid-column: span 2; margin-top: 1rem; border-top: 1px dashed var(--border-subtle); padding-top: 1rem;">
                            <h3 style="font-size: 0.85rem; font-weight: 700; color: white; margin-bottom: 0.75rem;">Per-Model Thinking Mode Controls (&lt;think&gt; Directive States)</h3>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div>
                                    <label for="THINKING_MODE_OPUS" style="font-size: 0.75rem; font-weight: 700; color: var(--text-muted);">OPUS THINKING MODE</label>
                                    <select id="THINKING_MODE_OPUS" style="width: 100%; margin-top: 4px;">
                                        <option value="open">open — Force &lt;think&gt; Reasoning</option>
                                        <option value="inherit">inherit — Native Model Behavior</option>
                                        <option value="close">close — Suppress Thinking</option>
                                    </select>
                                </div>
                                <div>
                                    <label for="THINKING_MODE_SONNET" style="font-size: 0.75rem; font-weight: 700; color: var(--text-muted);">SONNET THINKING MODE</label>
                                    <select id="THINKING_MODE_SONNET" style="width: 100%; margin-top: 4px;">
                                        <option value="open">open — Force &lt;think&gt; Reasoning</option>
                                        <option value="inherit">inherit — Native Model Behavior</option>
                                        <option value="close">close — Suppress Thinking</option>
                                    </select>
                                </div>
                                <div>
                                    <label for="THINKING_MODE_HAIKU" style="font-size: 0.75rem; font-weight: 700; color: var(--text-muted);">HAIKU THINKING MODE</label>
                                    <select id="THINKING_MODE_HAIKU" style="width: 100%; margin-top: 4px;">
                                        <option value="open">open — Force &lt;think&gt; Reasoning</option>
                                        <option value="inherit">inherit — Native Model Behavior</option>
                                        <option value="close">close — Suppress Thinking</option>
                                    </select>
                                </div>
                                <div>
                                    <label for="THINKING_MODE_DEFAULT" style="font-size: 0.75rem; font-weight: 700; color: var(--text-muted);">DEFAULT THINKING MODE</label>
                                    <select id="THINKING_MODE_DEFAULT" style="width: 100%; margin-top: 4px;">
                                        <option value="open">open — Force &lt;think&gt; Reasoning</option>
                                        <option value="inherit">inherit — Native Model Behavior</option>
                                        <option value="close">close — Suppress Thinking</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Fallback Chains Config -->
                    <div style="margin-top: 2rem; border-top: 1px solid var(--border-subtle); padding-top: 1.5rem;">
                        <h3 style="font-size: 0.95rem; font-weight: 700; color: white; margin-bottom: 0.25rem;">Fallback Order Chains</h3>
                        <p style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1.25rem;">Configure failover model priority list (inline tag chips with autocomplete) used when primary models are unavailable or rate-limited. Press Enter or comma to add a tag.</p>
                        <div class="form-grid">
                            <div class="form-group full-width">
                                <label for="FALLBACK_ORDER_CLAUDE_OPUS">Opus Fallback Order</label>
                                <div class="tag-input-container" id="tag-container-FALLBACK_ORDER_CLAUDE_OPUS" onclick="focusTagInput('FALLBACK_ORDER_CLAUDE_OPUS')">
                                    <input type="text" class="tag-input-field" id="FALLBACK_ORDER_CLAUDE_OPUS" placeholder="Type or select model..." autocomplete="off" onfocus="handleAutocomplete('FALLBACK_ORDER_CLAUDE_OPUS')" onblur="blurAutocomplete('FALLBACK_ORDER_CLAUDE_OPUS')" onkeydown="handleTagInputKeyDown(event, 'FALLBACK_ORDER_CLAUDE_OPUS')" oninput="handleAutocomplete('FALLBACK_ORDER_CLAUDE_OPUS')">
                                </div>
                                <div id="FALLBACK_ORDER_CLAUDE_OPUS-dropdown" class="autocomplete-dropdown"></div>
                            </div>
                            <div class="form-group full-width">
                                <label for="FALLBACK_ORDER_CLAUDE_SONNET">Sonnet Fallback Order</label>
                                <div class="tag-input-container" id="tag-container-FALLBACK_ORDER_CLAUDE_SONNET" onclick="focusTagInput('FALLBACK_ORDER_CLAUDE_SONNET')">
                                    <input type="text" class="tag-input-field" id="FALLBACK_ORDER_CLAUDE_SONNET" placeholder="Type or select model..." autocomplete="off" onfocus="handleAutocomplete('FALLBACK_ORDER_CLAUDE_SONNET')" onblur="blurAutocomplete('FALLBACK_ORDER_CLAUDE_SONNET')" onkeydown="handleTagInputKeyDown(event, 'FALLBACK_ORDER_CLAUDE_SONNET')" oninput="handleAutocomplete('FALLBACK_ORDER_CLAUDE_SONNET')">
                                </div>
                                <div id="FALLBACK_ORDER_CLAUDE_SONNET-dropdown" class="autocomplete-dropdown"></div>
                            </div>
                            <div class="form-group full-width">
                                <label for="FALLBACK_ORDER_CLAUDE_HAIKU">Haiku Fallback Order</label>
                                <div class="tag-input-container" id="tag-container-FALLBACK_ORDER_CLAUDE_HAIKU" onclick="focusTagInput('FALLBACK_ORDER_CLAUDE_HAIKU')">
                                    <input type="text" class="tag-input-field" id="FALLBACK_ORDER_CLAUDE_HAIKU" placeholder="Type or select model..." autocomplete="off" onfocus="handleAutocomplete('FALLBACK_ORDER_CLAUDE_HAIKU')" onblur="blurAutocomplete('FALLBACK_ORDER_CLAUDE_HAIKU')" onkeydown="handleTagInputKeyDown(event, 'FALLBACK_ORDER_CLAUDE_HAIKU')" oninput="handleAutocomplete('FALLBACK_ORDER_CLAUDE_HAIKU')">
                                </div>
                                <div id="FALLBACK_ORDER_CLAUDE_HAIKU-dropdown" class="autocomplete-dropdown"></div>
                            </div>
                            <div class="form-group full-width">
                                <label for="FALLBACK_ORDER_CLAUDE_DEFAULT">Default Fallback Order</label>
                                <div class="tag-input-container" id="tag-container-FALLBACK_ORDER_CLAUDE_DEFAULT" onclick="focusTagInput('FALLBACK_ORDER_CLAUDE_DEFAULT')">
                                    <input type="text" class="tag-input-field" id="FALLBACK_ORDER_CLAUDE_DEFAULT" placeholder="Type or select model..." autocomplete="off" onfocus="handleAutocomplete('FALLBACK_ORDER_CLAUDE_DEFAULT')" onblur="blurAutocomplete('FALLBACK_ORDER_CLAUDE_DEFAULT')" onkeydown="handleTagInputKeyDown(event, 'FALLBACK_ORDER_CLAUDE_DEFAULT')" oninput="handleAutocomplete('FALLBACK_ORDER_CLAUDE_DEFAULT')">
                                </div>
                                <div id="FALLBACK_ORDER_CLAUDE_DEFAULT-dropdown" class="autocomplete-dropdown"></div>
                            </div>
                        </div>
                    </div>
                </div><!-- /pane-models -->

                <!-- PANE: ROUTER & FALLBACKS -->
                <div id="pane-router" class="panel-pane">
                    <div class="panel-header" style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2>Resilient Multi-Model Router Status</h2>
                            <p>Real-time health status, Circuit Breakers, and Rate Limit Headroom for primary and fallback models.</p>
                        </div>
                        <button type="button" class="btn-alt" onclick="fetchRouterStatus()" style="padding: 0.5rem 1rem; font-size: 0.75rem;">Refresh Status</button>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                        <div class="dial-card" style="padding: 1rem;">
                            <h3>TOTAL REGISTERED MODELS</h3>
                            <div class="dial-val" id="router-total-models" style="font-size: 1.25rem;">-</div>
                        </div>
                        <div class="dial-card" style="padding: 1rem;">
                            <h3>HEALTHY / READY</h3>
                            <div class="dial-val" id="router-healthy-models" style="font-size: 1.25rem; color: #4ade80;">-</div>
                        </div>
                        <div class="dial-card" style="padding: 1rem;">
                            <h3>CIRCUIT BREAKER OPEN</h3>
                            <div class="dial-val" id="router-open-circuits" style="font-size: 1.25rem; color: #f87171;">-</div>
                        </div>
                    </div>

                    <div style="background: #050608; border: 1px solid var(--border-subtle); border-radius: 8px; overflow: hidden;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: left;">
                            <thead>
                                <tr style="background: rgba(255,255,255,0.03); border-bottom: 1px solid var(--border-subtle); color: var(--text-muted); font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;">
                                    <th style="padding: 0.75rem 1rem;">Model Target ID</th>
                                    <th style="padding: 0.75rem 1rem;">Circuit State</th>
                                    <th style="padding: 0.75rem 1rem;">Failures</th>
                                    <th style="padding: 0.75rem 1rem;">Quota Headroom</th>
                                    <th style="padding: 0.75rem 1rem;">Remaining Req / Tok</th>
                                </tr>
                            </thead>
                            <tbody id="router-table-body">
                                <tr><td colspan="5" style="padding: 1rem; text-align: center; color: var(--text-muted);">Loading router health matrix...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- PANE 2: API CREDENTIALS (GRID A) -->
                <div id="pane-keys" class="panel-pane">
                    <div class="panel-header">
                        <h2>API Credentials Matrix</h2>
                        <p>Configure API keys for commercial cloud models. Sensitive values are masked by default.</p>
                    </div>
                    <div class="form-grid">
                        <div class="form-group full-width">
                            <div class="label-row">
                                <label for="GATEWAY_AUTH_TOKEN">Gateway Auth Token (Enforces local client authentication. Leave empty to disable auth)</label>
                                <span id="badge-GATEWAY_AUTH_TOKEN" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="GATEWAY_AUTH_TOKEN" placeholder="Optional. If set, ANTHROPIC_AUTH_TOKEN must match this value.">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('GATEWAY_AUTH_TOKEN')">Show</button>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="NVIDIA_NIM_API_KEY">NVIDIA NIM API Key</label>
                                <span id="badge-NVIDIA_NIM_API_KEY" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="NVIDIA_NIM_API_KEY" placeholder="nvapi-...">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('NVIDIA_NIM_API_KEY')">Show</button>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="OPENROUTER_API_KEY">OpenRouter API Key</label>
                                <span id="badge-OPENROUTER_API_KEY" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="OPENROUTER_API_KEY" placeholder="sk-or-v1-...">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('OPENROUTER_API_KEY')">Show</button>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="MISTRAL_API_KEY">Mistral / Codestral API Key</label>
                                <span id="badge-MISTRAL_API_KEY" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="MISTRAL_API_KEY" placeholder="sk-...">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('MISTRAL_API_KEY')">Show</button>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="GEMINI_API_KEY">Gemini API Key (Google)</label>
                                <span id="badge-GEMINI_API_KEY" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="GEMINI_API_KEY" placeholder="AIzaSy...">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('GEMINI_API_KEY')">Show</button>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="GROQ_API_KEY">Groq API Key</label>
                                <span id="badge-GROQ_API_KEY" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="GROQ_API_KEY" placeholder="gsk_...">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('GROQ_API_KEY')">Show</button>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="DEEPSEEK_API_KEY">DeepSeek API Key</label>
                                <span id="badge-DEEPSEEK_API_KEY" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="DEEPSEEK_API_KEY" placeholder="sk-...">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('DEEPSEEK_API_KEY')">Show</button>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="CEREBRAS_API_KEY">Cerebras API Key</label>
                                <span id="badge-CEREBRAS_API_KEY" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="CEREBRAS_API_KEY" placeholder="csk-...">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('CEREBRAS_API_KEY')">Show</button>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="FIREWORKS_API_KEY">Fireworks API Key</label>
                                <span id="badge-FIREWORKS_API_KEY" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="FIREWORKS_API_KEY" placeholder="fw_...">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('FIREWORKS_API_KEY')">Show</button>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="KIMI_API_KEY">Kimi API Key</label>
                                <span id="badge-KIMI_API_KEY" class="badge-status">...</span>
                            </div>
                            <div class="input-wrapper">
                                <input type="password" id="KIMI_API_KEY" placeholder="sk-...">
                                <button type="button" class="input-icon-btn" onclick="toggleMask('KIMI_API_KEY')">Show</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- PANE 3: LOCAL ENDPOINTS (GRID B) -->
                <div id="pane-endpoints" class="panel-pane">
                    <div class="panel-header">
                        <h2>Local Model Endpoints</h2>
                        <p>Configure routing addresses for local LLM models running on your machine.</p>
                    </div>
                    <div class="form-grid">
                        <div class="form-group">
                            <div class="label-row">
                                <label for="LM_STUDIO_BASE_URL">LM Studio Base URL</label>
                                <span id="status-lmstudio" class="badge-status">Checking...</span>
                            </div>
                            <input type="text" id="LM_STUDIO_BASE_URL" placeholder="http://localhost:1234/v1">
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="LLAMA_CPP_BASE_URL">llama.cpp Base URL</label>
                                <span id="status-llama_cpp" class="badge-status">Checking...</span>
                            </div>
                            <input type="text" id="LLAMA_CPP_BASE_URL" placeholder="http://localhost:8080/v1">
                        </div>
                        <div class="form-group">
                            <div class="label-row">
                                <label for="OLLAMA_BASE_URL">Ollama Base URL</label>
                                <span id="status-ollama" class="badge-status">Checking...</span>
                            </div>
                            <input type="text" id="OLLAMA_BASE_URL" placeholder="http://localhost:11434">
                        </div>
                    </div>
                </div>

                <!-- PANE 4: MOCK ENGINES -->
                <div id="pane-mocks" class="panel-pane">
                    <div class="panel-header">
                        <h2>Mock Optimization Engines</h2>
                        <p>Toggle local processing optimizations to speed up Claude Code interactions.</p>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 1rem;">
                        <div class="setting-switch">
                            <div class="setting-switch-desc">
                                <h4>Fast Prefix Detection</h4>
                                <p>Extract command safety prefix checks locally.</p>
                            </div>
                            <input type="checkbox" id="FAST_PREFIX_DETECTION">
                        </div>
                        <div class="setting-switch">
                            <div class="setting-switch-desc">
                                <h4>Network Probe Mocking</h4>
                                <p>Immediately respond to system network and quota status diagnostics.</p>
                            </div>
                            <input type="checkbox" id="ENABLE_NETWORK_PROBE_MOCK">
                        </div>
                        <div class="setting-switch">
                            <div class="setting-switch-desc">
                                <h4>Skip Title Generation</h4>
                                <p>Prevent external requests for generation of workspace thread titles.</p>
                            </div>
                            <input type="checkbox" id="ENABLE_TITLE_GENERATION_SKIP">
                        </div>
                        <div class="setting-switch">
                            <div class="setting-switch-desc">
                                <h4>Skip Suggestion Autocomplete</h4>
                                <p>Mock auto-completions in suggestion terminal queries locally.</p>
                            </div>
                            <input type="checkbox" id="ENABLE_SUGGESTION_MODE_SKIP">
                        </div>
                        <div class="setting-switch">
                            <div class="setting-switch-desc">
                                <h4>Filepath Extraction Mocking</h4>
                                <p>Perform filepath lookup extractions with a regex parser engine.</p>
                            </div>
                            <input type="checkbox" id="ENABLE_FILEPATH_EXTRACTION_MOCK">
                        </div>
                    </div>
                </div>

                <!-- PANE 5: THROTTLE & TIMEOUTS -->
                <div id="pane-limits" class="panel-pane">
                    <div class="panel-header">
                        <h2>Throttle &amp; Timeout Settings</h2>
                        <p>Configure rate limits, concurrency caps, and HTTP socket timeouts. Select a provider preset to populate recommended values, then commit to save.</p>
                    </div>

                    <!-- Provider Preset Selector -->
                    <div style="margin-bottom: 1.75rem; background: rgba(255,255,255,0.02); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 1rem 1.25rem;">
                        <div style="font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); margin-bottom: 0.75rem;">Provider Defaults</div>
                        <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap;">
                            <select id="provider-preset-select" style="flex: 1; min-width: 200px; max-width: 320px;">
                                <option value="">-- Select provider --</option>
                                <option value="nvidia_nim">NVIDIA NIM</option>
                                <option value="openrouter">OpenRouter</option>
                                <option value="groq">Groq</option>
                                <option value="deepseek">DeepSeek</option>
                                <option value="mistral">Mistral</option>
                                <option value="cerebras">Cerebras</option>
                            </select>
                            <button type="button" class="btn-alt" onclick="applySelectedPreset()" style="padding: 0.75rem 1.5rem; font-size: 0.75rem; white-space: nowrap;">Apply Defaults</button>
                        </div>
                        <div id="preset-desc" style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.65rem; min-height: 1rem; line-height: 1.5;"></div>
                    </div>

                    <div class="form-grid">
                        <div class="form-group">
                            <label for="PROVIDER_RATE_LIMIT">Rate Limit (requests per window)</label>
                            <input type="number" id="PROVIDER_RATE_LIMIT">
                        </div>
                        <div class="form-group">
                            <label for="PROVIDER_RATE_WINDOW">Rate Window (seconds)</label>
                            <input type="number" id="PROVIDER_RATE_WINDOW">
                        </div>
                        <div class="form-group">
                            <label for="PROVIDER_MAX_CONCURRENCY">Max Concurrent Connections</label>
                            <input type="number" id="PROVIDER_MAX_CONCURRENCY">
                        </div>
                        <div class="form-group">
                            <label for="HTTP_READ_TIMEOUT">HTTP Read Timeout (seconds)</label>
                            <input type="number" id="HTTP_READ_TIMEOUT">
                        </div>
                        <div class="form-group">
                            <label for="HTTP_WRITE_TIMEOUT">HTTP Write Timeout (seconds)</label>
                            <input type="number" id="HTTP_WRITE_TIMEOUT">
                        </div>
                        <div class="form-group">
                            <label for="HTTP_CONNECT_TIMEOUT">HTTP Connect Timeout (seconds)</label>
                            <input type="number" id="HTTP_CONNECT_TIMEOUT">
                        </div>
                    </div>
                </div>

                <!-- PANE 6: BOT MATRIX -->
                <div id="pane-bots" class="panel-pane">
                    <div class="panel-header">
                        <h2>Bot Matrix Administration</h2>
                        <p>Set token mappings and workspace folders for Telegram/Discord admin endpoints.</p>
                    </div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="MESSAGING_PLATFORM">Messaging Platform</label>
                            <select id="MESSAGING_PLATFORM">
                                <option value="discord">Discord</option>
                                <option value="telegram">Telegram</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="CLAUDE_WORKSPACE">Workspace Directory</label>
                            <input type="text" id="CLAUDE_WORKSPACE">
                        </div>
                        <div class="form-group full-width">
                            <label for="TELEGRAM_BOT_TOKEN">Telegram Bot Token</label>
                            <input type="password" id="TELEGRAM_BOT_TOKEN">
                        </div>
                        <div class="form-group full-width">
                            <label for="ALLOWED_TELEGRAM_USER_ID">Telegram Chat ID / User ID</label>
                            <input type="text" id="ALLOWED_TELEGRAM_USER_ID">
                        </div>
                        <div class="form-group full-width">
                            <label for="DISCORD_BOT_TOKEN">Discord Bot Token</label>
                            <input type="password" id="DISCORD_BOT_TOKEN">
                        </div>
                        <div class="form-group full-width">
                            <label for="ALLOWED_DISCORD_CHANNELS">Discord Channel IDs (comma separated)</label>
                            <input type="text" id="ALLOWED_DISCORD_CHANNELS">
                        </div>
                    </div>
                </div>

                <!-- PANE 7: SYSTEM TRACE FEED -->
                <div id="pane-logs" class="panel-pane">
                    <div class="panel-header">
                        <h2>System Request Trace Feed</h2>
                        <p>Real-time rolling ledger tracing API calls, mapped execution models, error indicators, and response latencies.</p>
                    </div>
                    <div class="feed-container">
                        <table class="feed-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Call Path</th>
                                    <th>Client Model</th>
                                    <th>Resolved Upstream</th>
                                    <th>Latency</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody id="live-request-feed-body">
                                <tr>
                                    <td colspan="6" style="text-align: center; color: var(--text-dim); padding: 20px;">No operational traffic logged. Awaiting API events...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- PANE 8: DIAGNOSTICS -->
                <div id="pane-doctor" class="panel-pane">
                    <div class="panel-header">
                        <h2>Diagnostics Console</h2>
                        <p>Run diagnostic checks and inspect raw connection capabilities.</p>
                    </div>
                    <button type="button" class="btn-alt" onclick="runDiagnostics()">Run Diagnostics</button>
                    <div class="terminal-box" id="terminalConsole">
                        <div class="terminal-line">> Console ready. Awaiting instructions...</div>
                    </div>
                </div>

                <!-- PANE 9: SETUP GUIDE -->
                <div id="pane-guide" class="panel-pane">
                    <div class="panel-header">
                        <h2>Setup Guide & Environment Settings</h2>
                        <p>Learn how to redirect the Claude Code CLI tool through this local proxy gateway.</p>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 1.5rem; font-size: 0.85rem; line-height: 1.6;">
                        <div>
                            <h3 style="color: white; font-size: 0.95rem; margin-bottom: 0.5rem;">1. Concept</h3>
                            <p style="color: var(--text-muted);">
                                By default, Claude Code connects directly to Anthropic's cloud endpoints. We redirect its traffic locally using the <code>ANTHROPIC_BASE_URL</code> environment variable. 
                                Since Claude Code requires an authentication token to initialize, we provide a dummy token <code>freecc</code> (or your custom Gateway Token) via <code>ANTHROPIC_AUTH_TOKEN</code>.
                            </p>
                        </div>

                        <div>
                            <h3 style="color: white; font-size: 0.95rem; margin-bottom: 0.5rem;">2. Bash / Zsh Shell Integration</h3>
                            <p style="color: var(--text-muted); margin-bottom: 0.5rem;">Run this inside your terminal session to connect directly, or add it to your <code>~/.bashrc</code> or <code>~/.zshrc</code> file:</p>
                            <pre class="terminal-box" style="min-height: auto; padding: 1rem; margin-top: 0.25rem;"><code style="color: white; font-family: inherit;">export ANTHROPIC_BASE_URL="http://127.0.0.1:8090"
export ANTHROPIC_AUTH_TOKEN="<span class="guide-token-display">freecc</span>"
claude</code></pre>
                        </div>

                        <div>
                            <h3 style="color: white; font-size: 0.95rem; margin-bottom: 0.5rem;">3. Fish Shell Integration</h3>
                            <p style="color: var(--text-muted); margin-bottom: 0.5rem;">For Fish shell users, you can define a custom function (e.g., <code>fcc-claude</code>) in your fish configuration (<code>~/.config/fish/functions/fcc-claude.fish</code>):</p>
                            <pre class="terminal-box" style="min-height: auto; padding: 1rem; margin-top: 0.25rem;"><code style="color: white; font-family: inherit;">function fcc-claude
    env ANTHROPIC_BASE_URL="http://127.0.0.1:8090" ANTHROPIC_AUTH_TOKEN="<span class="guide-token-display">freecc</span>" claude $argv
end</code></pre>
                            <p style="color: var(--text-muted); margin-top: 0.5rem;">Then, simply type <code>fcc-claude</code> in your terminal to launch it with the proxy routing enabled, or use <code>claude</code> to bypass the proxy.</p>
                        </div>

                        <div style="border-top: 1px solid var(--border-subtle); padding-top: 1rem;">
                            <h3 style="color: white; font-size: 0.95rem; margin-bottom: 0.5rem;">Security Note — Gateway Auth Token</h3>
                            <p style="color: var(--text-muted);">
                                Currently, your Gateway Token state is: <strong id="guide-token-state" style="color: white;">DISABLED</strong>.
                                <br>
                                You can configure a custom token inside the <strong>API Credentials</strong> pane. If configured, you must set your <code>ANTHROPIC_AUTH_TOKEN</code> override to match that exact token. If left blank, any dummy string is accepted.
                            </p>
                        </div>
                    </div>
                </div>

                <!-- PANE: ADVANCED PROXY TUNNELS & BASE URLS -->
                <div id="pane-tunnels" class="panel-pane">
                    <div class="panel-header">
                        <h2>Advanced Proxy Tunnels & Base URLs</h2>
                        <p>Configure custom upstream base URLs and proxy endpoint destinations for commercial LLM providers.</p>
                    </div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="NVIDIA_NIM_BASE_URL">NVIDIA NIM Base URL Proxy</label>
                            <input type="text" id="NVIDIA_NIM_BASE_URL" placeholder="https://integrate.api.nvidia.com/v1">
                        </div>
                        <div class="form-group">
                            <label for="OPENROUTER_BASE_URL">OpenRouter Base URL Proxy</label>
                            <input type="text" id="OPENROUTER_BASE_URL" placeholder="https://openrouter.ai/api/v1">
                        </div>
                        <div class="form-group">
                            <label for="MISTRAL_BASE_URL">Mistral / Codestral Base URL Proxy</label>
                            <input type="text" id="MISTRAL_BASE_URL" placeholder="https://api.mistral.ai/v1">
                        </div>
                        <div class="form-group">
                            <label for="GEMINI_BASE_URL">Gemini Base URL Proxy</label>
                            <input type="text" id="GEMINI_BASE_URL" placeholder="https://generativelanguage.googleapis.com/v1beta">
                        </div>
                        <div class="form-group">
                            <label for="GROQ_BASE_URL">Groq Base URL Proxy</label>
                            <input type="text" id="GROQ_BASE_URL" placeholder="https://api.groq.com/openai/v1">
                        </div>
                        <div class="form-group">
                            <label for="DEEPSEEK_BASE_URL">DeepSeek Base URL Proxy</label>
                            <input type="text" id="DEEPSEEK_BASE_URL" placeholder="https://api.deepseek.com/v1">
                        </div>
                        <div class="form-group">
                            <label for="CEREBRAS_BASE_URL">Cerebras Base URL Proxy</label>
                            <input type="text" id="CEREBRAS_BASE_URL" placeholder="https://api.cerebras.ai/v1">
                        </div>
                        <div class="form-group">
                            <label for="FIREWORKS_BASE_URL">Fireworks Base URL Proxy</label>
                            <input type="text" id="FIREWORKS_BASE_URL" placeholder="https://api.fireworks.ai/inference/v1">
                        </div>
                    </div>
                </div>

                <div class="console-buttons-bar" id="actionButtons">
                    <button type="submit" class="btn-action">Commit Configuration</button>
                    <button type="button" class="btn-alt" onclick="revertConfigs()">Revert changes</button>
                </div>
            </form>
        </div>
    </main>

    <div class="alert-toast" id="saveToast">Configuration saved</div>

    <script>
        let configSnapshot = {};
        let modelsSnapshot = {};
        let lockStatuses = {};

        async function fetchModels() {
            try {
                const resp = await fetch('/api/models');
                modelsSnapshot = await resp.json();
                
                // Update model count badges
                for (const [provider, list] of Object.entries(modelsSnapshot)) {
                    const badge = document.getElementById('count-' + provider);
                    if (badge) {
                        badge.innerText = `${list.length} models fetched`;
                    }
                }
            } catch (e) {
                console.error("Failed to fetch autocomplete list", e);
            }
        }

        async function loadConfigs() {
            try {
                const resp = await fetch('/api/config');
                const data = await resp.json();
                configSnapshot = data.configs;
                lockStatuses = data.key_statuses;
                
                for (const [key, value] of Object.entries(configSnapshot)) {
                    const el = document.getElementById(key);
                    if (!el) continue;

                    if (el.type === 'checkbox') {
                        el.checked = !!value;
                    } else {
                        el.value = value;
                    }

                    // Apply status badge & Lock check
                    const badge = document.getElementById('badge-' + key);
                    if (badge) {
                        const status = lockStatuses[key] || "Not Configured";
                        badge.innerText = status;
                        badge.className = 'badge-status ' + status.toLowerCase().replace(/[^a-z0-9]+/g, '-');
                        
                        if (status === "Locked") {
                            el.disabled = true;
                            // Add lock indicators
                            const parent = el.closest('.form-group');
                            if (parent) parent.style.opacity = '0.7';
                        } else {
                            el.disabled = false;
                            const parent = el.closest('.form-group');
                            if (parent) parent.style.opacity = '1';
                        }
                    }
                }
                // Populate Fallback Chains
                if (data.fallbacks) {
                    for (const [alias, entry] of Object.entries(data.fallbacks)) {
                        const inputId = 'FALLBACK_ORDER_' + alias.toUpperCase();
                        if (entry.fallback_order) {
                            tagState[inputId] = [...entry.fallback_order];
                            renderTags(inputId);
                        }
                    }
                }

                // Update dynamic guide tokens
                const tokenVal = configSnapshot["GATEWAY_AUTH_TOKEN"] || "freecc";
                document.querySelectorAll(".guide-token-display").forEach(el => {
                    el.innerText = tokenVal;
                });
                const tokenState = document.getElementById("guide-token-state");
                if (tokenState) {
                    if (configSnapshot["GATEWAY_AUTH_TOKEN"]) {
                        tokenState.innerText = "ACTIVE (Value: " + configSnapshot["GATEWAY_AUTH_TOKEN"] + ")";
                        tokenState.style.color = "#ffffff";
                    } else {
                        tokenState.innerText = "DISABLED (Any dummy token will be accepted)";
                        tokenState.style.color = "var(--text-dim)";
                    }
                }
            } catch (e) {
                console.error("Failed to load configs", e);
            }
        }

        async function revertConfigs() {
            await loadConfigs();
            showToast("Changes reverted.", "success");
        }

        async function saveConfigs(event) {
            event.preventDefault();
            const payload = {};
            for (const key of Object.keys(configSnapshot)) {
                const el = document.getElementById(key);
                if (!el) continue;

                // Check lock status
                if (lockStatuses[key] === "Locked") {
                    payload[key] = configSnapshot[key];
                    continue;
                }

                if (el.type === 'checkbox') {
                    payload[key] = el.checked;
                } else if (el.type === 'number') {
                    payload[key] = parseInt(el.value) || 0;
                } else {
                    payload[key] = el.value;
                }
            }

            // Also collect Fallback Chains from tagState
            ['CLAUDE_OPUS', 'CLAUDE_SONNET', 'CLAUDE_HAIKU', 'CLAUDE_DEFAULT'].forEach(alias => {
                const inputId = 'FALLBACK_ORDER_' + alias;
                // Add any remaining text typed in input before saving
                const inputEl = document.getElementById(inputId);
                if (inputEl && inputEl.value.trim()) {
                    addTag(inputId, inputEl.value);
                }
                payload[inputId] = (tagState[inputId] || []).join(', ');
            });

            try {
                const resp = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ configs: payload })
                });
                const res = await resp.json();
                if (res.status === 'success') {
                    showToast("Configuration saved & reloaded.", "success");
                    configSnapshot = payload;
                    await loadConfigs();
                    fetchModels();
                } else {
                    showToast("Commit failed: " + res.message, "error");
                }
            } catch (e) {
                showToast("Network error saving configs.", "error");
            }
        }

        function showToast(message, type = "success") {
            const toast = document.getElementById('saveToast');
            toast.innerText = message;
            if (type === "error") {
                toast.style.background = "#1a0f0f";
                toast.style.borderColor = "#f87171";
                toast.style.color = "#f87171";
            } else {
                toast.style.background = "#0c0d11";
                toast.style.borderColor = "#ffffff";
                toast.style.color = "#ffffff";
            }
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        function switchPane(event, paneId) {
            console.log("Switching to pane:", paneId);
            document.querySelectorAll('.control-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.panel-pane').forEach(pane => pane.classList.remove('active'));

            event.currentTarget.classList.add('active');
            document.getElementById(paneId).classList.add('active');

            const actionButtons = document.getElementById('actionButtons');

            // Hide action commit bar for read-only pages
            if (paneId === 'pane-doctor' || paneId === 'pane-logs' || paneId === 'pane-guide' || paneId === 'pane-router') {
                actionButtons.style.display = 'none';
            } else {
                actionButtons.style.display = 'flex';
            }

            if (paneId === 'pane-router') {
                fetchRouterStatus();
            }
        }

        async function fetchRouterStatus() {
            try {
                const resp = await fetch('/api/router-status');
                const data = await resp.json();
                
                document.getElementById('router-total-models').innerText = data.summary.total_models || 0;
                document.getElementById('router-healthy-models').innerText = data.summary.healthy || 0;
                document.getElementById('router-open-circuits').innerText = data.summary.circuit_open || 0;

                const tbody = document.getElementById('router-table-body');
                tbody.innerHTML = '';

                const entries = Object.entries(data.models);
                if (entries.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" style="padding: 1rem; text-align: center; color: var(--text-muted);">No models routed yet. Try sending a request through the proxy.</td></tr>';
                    return;
                }

                for (const [modelId, status] of entries) {
                    const tr = document.createElement('tr');
                    tr.style.borderBottom = '1px solid var(--border-subtle)';

                    const cb = status.circuit_breaker || {};
                    const rl = status.rate_limit || {};

                    const stateColor = cb.state === 'closed' ? '#4ade80' : (cb.state === 'half_open' ? '#facc15' : '#f87171');
                    const headroomColor = rl.has_headroom ? '#4ade80' : '#f87171';

                    const reqRem = rl.req_remaining !== null && rl.req_remaining !== undefined ? rl.req_remaining : '∞';
                    const tokRem = rl.tok_remaining !== null && rl.tok_remaining !== undefined ? rl.tok_remaining : '∞';

                    tr.innerHTML = `
                        <td style="padding: 0.75rem 1rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">${modelId}</td>
                        <td style="padding: 0.75rem 1rem;"><span style="color: ${stateColor}; font-weight: 700; text-transform: uppercase; font-size: 0.7rem;">● ${cb.state || 'CLOSED'}</span></td>
                        <td style="padding: 0.75rem 1rem; color: var(--text-muted);">${cb.failure_count || 0}</td>
                        <td style="padding: 0.75rem 1rem;"><span style="color: ${headroomColor}; font-weight: 700;">${rl.has_headroom ? 'YES (≥10%)' : 'NO (LIMITED)'}</span></td>
                        <td style="padding: 0.75rem 1rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">${reqRem} req / ${tokRem} tok</td>
                    `;
                    tbody.appendChild(tr);
                }
            } catch (e) {
                console.error("Failed to fetch router status:", e);
            }
        }

        function toggleAccordion() {
            const content = document.getElementById('accordionContent');
            const arrow = document.getElementById('accordionArrow');
            const isOpen = content.classList.toggle('open');
            arrow.innerText = isOpen ? '▼' : '▶';
        }

        function toggleMask(inputId) {
            const el = document.getElementById(inputId);
            const wrapper = el.closest('.input-wrapper');
            const btn = wrapper.querySelector('.input-icon-btn');
            if (el.type === 'password') {
                el.type = 'text';
                btn.innerText = 'Hide';
            } else {
                el.type = 'password';
                btn.innerText = 'Show';
            }
        }

        async function runDiagnostics() {
            showToast("Running system diagnostics...", "success");
            const consoleEl = document.getElementById('terminalConsole');
            consoleEl.innerHTML = '<div class="terminal-line">> Initializing diagnostic check...</div>';
            try {
                const resp = await fetch('/api/diagnostics');
                const data = await resp.json();

                consoleEl.innerHTML = '';

                let i = 0;
                function printLine() {
                    if (i < data.logs.length) {
                        const div = document.createElement('div');
                        div.className = 'terminal-line';
                        div.innerText = data.logs[i];
                        consoleEl.appendChild(div);
                        consoleEl.scrollTop = consoleEl.scrollHeight;
                        i++;
                        setTimeout(printLine, 150);
                    } else {
                        const finalDiv = document.createElement('div');
                        finalDiv.className = 'terminal-line';
                        if (data.has_errors) {
                            finalDiv.style.color = '#f87171';
                            finalDiv.innerText = 'DIAGNOSTICS COMPLETE - ADVISORIES FOUND.';
                            showToast("Diagnostics completed with advisories.", "error");
                        } else {
                            finalDiv.style.color = '#ffffff';
                            finalDiv.innerText = 'ALL SYSTEMS FUNCTIONAL. READY TO PROCESS.';
                            showToast("System diagnostics completed.", "success");
                        }
                        consoleEl.appendChild(finalDiv);
                        consoleEl.scrollTop = consoleEl.scrollHeight;
                    }
                }
                printLine();
            } catch (e) {
                consoleEl.innerHTML += '<div class="terminal-line" style="color: #f87171">> System connection failed: ' + e + '</div>';
                showToast("Diagnostics check failed.", "error");
            }
        }

        // Live Telemetry statistics poller
        async function pollTelemetry() {
            try {
                const resp = await fetch('/api/stats');
                const data = await resp.json();

                // Update request counters
                document.getElementById('telemetry-requests').innerHTML = `${data.total_requests} <span>reqs</span>`;

                if (data.total_requests > 0) {
                    const ratio = Math.round((data.mocked_requests / data.total_requests) * 100);
                    document.getElementById('telemetry-savings').innerText = `${ratio}%`;
                } else {
                    document.getElementById('telemetry-savings').innerText = `0%`;
                }
                document.getElementById('telemetry-mock-badge').innerText = `${data.mocked_requests} / ${data.total_requests} SAVES`;

                // Update Bot connectivity
                const dsEl = document.getElementById('telemetry-discord');
                dsEl.innerText = data.ds_bot_status;
                dsEl.style.color = data.ds_bot_status === 'Online' ? '#ffffff' : 'var(--text-dim)';

                const tgEl = document.getElementById('telemetry-telegram');
                tgEl.innerText = data.tg_bot_status;
                tgEl.style.color = data.tg_bot_status === 'Online' ? '#ffffff' : 'var(--text-dim)';

                // Update local services online health badges
                for (const [service, status] of Object.entries(data.endpoints)) {
                    const statusBadge = document.getElementById('status-' + service);
                    if (statusBadge) {
                        statusBadge.innerText = status;
                        statusBadge.style.color = status === 'Online' ? '#ffffff' : 'var(--text-dim)';
                        statusBadge.style.borderColor = status === 'Online' ? '#ffffff' : 'var(--border-subtle)';
                    }
                }

                // Render Live Request Trace log feed
                const feedBody = document.getElementById('live-request-feed-body');
                if (data.recent_requests && data.recent_requests.length > 0) {
                    feedBody.innerHTML = '';
                    // Display in reverse order (newest first)
                    const sorted = [...data.recent_requests].reverse();
                    sorted.forEach(req => {
                        const tr = document.createElement('tr');
                        
                        const tdTime = document.createElement('td');
                        tdTime.innerText = req.timestamp;
                        tr.appendChild(tdTime);

                        const tdPath = document.createElement('td');
                        tdPath.innerText = `${req.method} ${req.path}`;
                        tr.appendChild(tdPath);

                        const tdClient = document.createElement('td');
                        tdClient.innerText = req.client_model;
                        tr.appendChild(tdClient);

                        const tdUpstream = document.createElement('td');
                        tdUpstream.innerText = req.mapped_model || req.target_model || '-';
                        tr.appendChild(tdUpstream);

                        const tdLatency = document.createElement('td');
                        const dur = req.duration_ms !== undefined ? req.duration_ms : (req.latency_ms || 0);
                        tdLatency.innerText = `${dur} ms`;
                        tr.appendChild(tdLatency);

                        const tdStatus = document.createElement('td');
                        const spanStatus = document.createElement('span');
                        spanStatus.className = 'feed-badge ' + (req.status_code === 200 ? 'status-200' : 'status-error');
                        spanStatus.innerText = req.status_code;
                        tdStatus.appendChild(spanStatus);

                        if (req.mocked) {
                            const spanMock = document.createElement('span');
                            spanMock.className = 'feed-badge mock';
                            spanMock.style.marginLeft = '6px';
                            spanMock.innerText = 'MOCK';
                            tdStatus.appendChild(spanMock);
                        }

                        if (req.fallbacks_used && req.fallbacks_used.length > 0) {
                            const spanFb = document.createElement('span');
                            spanFb.className = 'feed-badge mock';
                            spanFb.style.marginLeft = '6px';
                            spanFb.style.background = 'rgba(234, 179, 8, 0.2)';
                            spanFb.style.color = '#fef08a';
                            spanFb.innerText = `FALLBACK (${req.fallbacks_used.length})`;
                            tdStatus.appendChild(spanFb);
                        }

                        tr.appendChild(tdStatus);
                        feedBody.appendChild(tr);
                    });
                } else {
                    feedBody.innerHTML = `<tr>
                        <td colspan="6" style="text-align: center; color: var(--text-dim); padding: 20px;">No operational traffic logged. Awaiting API events...</td>
                    </tr>`;
                }

            } catch (e) {
                console.error("Telemetry link lost", e);
            }
        }

        let activeSuggestionIndex = -1;
        let currentVisibleItems = [];

        async function refreshModelsList() {
            showToast("🔄 Refreshing available models list...", "success");
            await fetchModels();
            showToast("✅ Models list refreshed successfully.", "success");
        }

        // Search Autocomplete Suggestion Logic
        function handleAutocomplete(inputId) {
            const inputEl = document.getElementById(inputId);
            const dropdownEl = document.getElementById(inputId + '-dropdown');
            
            dropdownEl.innerHTML = '';
            let count = 0;
            currentVisibleItems = [];
            activeSuggestionIndex = -1;

            for (const [provider, list] of Object.entries(modelsSnapshot)) {
                if (list.length === 0) continue;

                const textVal = inputEl.value.toLowerCase();
                const providerPrefix = provider + '/';
                const filtered = list.filter(m => {
                    const full = providerPrefix + m;
                    return !textVal || full.toLowerCase().includes(textVal);
                });

                if (filtered.length > 0) {
                    const header = document.createElement('div');
                    header.className = 'autocomplete-group';
                    header.innerText = provider.replace('_', ' ');
                    dropdownEl.appendChild(header);

                    filtered.forEach(m => {
                        count++;
                        const fullStr = providerPrefix + m;
                        const item = document.createElement('div');
                        item.className = 'autocomplete-item';
                        item.innerText = fullStr;
                        item.onmousedown = () => {
                            selectSuggestion(inputId, fullStr);
                        };
                        dropdownEl.appendChild(item);
                        currentVisibleItems.push(item);
                    });
                }
            }

            if (count > 0) {
                dropdownEl.style.display = 'block';
            } else {
                dropdownEl.style.display = 'none';
            }
        }

        // Tag State Management
        const tagState = {
            FALLBACK_ORDER_CLAUDE_OPUS: [],
            FALLBACK_ORDER_CLAUDE_SONNET: [],
            FALLBACK_ORDER_CLAUDE_HAIKU: [],
            FALLBACK_ORDER_CLAUDE_DEFAULT: [],
        };

        function renderTags(inputId) {
            const container = document.getElementById('tag-container-' + inputId);
            if (!container) return;

            // Clear existing chips
            container.querySelectorAll('.tag-chip').forEach(chip => chip.remove());

            const tags = tagState[inputId] || [];
            const inputEl = document.getElementById(inputId);

            tags.forEach((tag, idx) => {
                const chip = document.createElement('div');
                chip.className = 'tag-chip';
                
                const spanText = document.createElement('span');
                spanText.innerText = tag;
                chip.appendChild(spanText);

                const removeBtn = document.createElement('span');
                removeBtn.className = 'tag-remove';
                removeBtn.innerHTML = '&times;';
                removeBtn.onclick = (e) => {
                    e.stopPropagation();
                    removeTag(inputId, idx);
                };
                chip.appendChild(removeBtn);

                container.insertBefore(chip, inputEl);
            });
        }

        function addTag(inputId, val) {
            const trimmed = val.trim().replace(/,/g, '');
            if (!trimmed) return;

            if (!tagState[inputId]) tagState[inputId] = [];
            if (!tagState[inputId].includes(trimmed)) {
                tagState[inputId].push(trimmed);
                renderTags(inputId);
            }
            const inputEl = document.getElementById(inputId);
            if (inputEl) inputEl.value = '';
        }

        function removeTag(inputId, index) {
            if (tagState[inputId]) {
                tagState[inputId].splice(index, 1);
                renderTags(inputId);
            }
        }

        function focusTagInput(inputId) {
            const inputEl = document.getElementById(inputId);
            if (inputEl) inputEl.focus();
        }

        function closeAllAutocompleteDropdowns() {
            document.querySelectorAll('.autocomplete-dropdown').forEach(dropdown => {
                dropdown.style.display = 'none';
            });
            activeSuggestionIndex = -1;
            currentVisibleItems = [];
        }

        function handleTagInputKeyDown(event, inputId) {
            const inputEl = document.getElementById(inputId);
            const dropdownEl = document.getElementById(inputId + '-dropdown');

            if (event.key === 'Escape') {
                event.preventDefault();
                closeAllAutocompleteDropdowns();
                return;
            }

            // If dropdown active with selection
            if (dropdownEl && dropdownEl.style.display !== 'none' && currentVisibleItems.length > 0) {
                if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                    handleInputKeyDown(event, inputId);
                    return;
                }
                if (event.key === 'Enter') {
                    if (activeSuggestionIndex >= 0 && activeSuggestionIndex < currentVisibleItems.length) {
                        event.preventDefault();
                        const selectedText = currentVisibleItems[activeSuggestionIndex].innerText;
                        selectSuggestion(inputId, selectedText);
                        return;
                    }
                }
            }

            if (event.key === 'Enter' || event.key === ',') {
                event.preventDefault();
                addTag(inputId, inputEl.value);
                if (dropdownEl) dropdownEl.style.display = 'none';
            } else if (event.key === 'Backspace' && !inputEl.value) {
                if (tagState[inputId] && tagState[inputId].length > 0) {
                    removeTag(inputId, tagState[inputId].length - 1);
                }
            }
        }

        function selectSuggestion(inputId, value) {
            if (inputId.startsWith('FALLBACK_ORDER_')) {
                addTag(inputId, value);
            } else {
                const inputEl = document.getElementById(inputId);
                inputEl.value = value;
            }
            closeAllAutocompleteDropdowns();
        }

        // Close dropdowns when clicking outside
        document.addEventListener('click', (event) => {
            const isClickInsideGroup = event.target.closest('.form-group');
            if (!isClickInsideGroup) {
                closeAllAutocompleteDropdowns();
            }
        });

        // Wire event listeners for live filtering
        document.querySelectorAll('input[type="text"]').forEach(input => {
            if (['MODEL_OPUS', 'MODEL_SONNET', 'MODEL_HAIKU', 'MODEL'].includes(input.id)) {
                input.addEventListener('input', () => handleAutocomplete(input.id));
                input.addEventListener('keydown', (e) => handleInputKeyDown(e, input.id));
            }
        });

        // Provider default preset configs
        const PROVIDER_PRESETS = {
            nvidia_nim: {
                label: 'NVIDIA NIM — high-throughput, optimised for long generation timeouts. Rate: 60 req/min, Concurrency: 10, Read: 300s.',
                PROVIDER_RATE_LIMIT: 60,
                PROVIDER_RATE_WINDOW: 60,
                PROVIDER_MAX_CONCURRENCY: 10,
                HTTP_READ_TIMEOUT: 300,
                HTTP_WRITE_TIMEOUT: 30,
                HTTP_CONNECT_TIMEOUT: 5,
            },
            openrouter: {
                label: 'OpenRouter — balanced defaults for free-tier cloud routing. Rate: 40 req/min, Concurrency: 5, Read: 180s.',
                PROVIDER_RATE_LIMIT: 40,
                PROVIDER_RATE_WINDOW: 60,
                PROVIDER_MAX_CONCURRENCY: 5,
                HTTP_READ_TIMEOUT: 180,
                HTTP_WRITE_TIMEOUT: 20,
                HTTP_CONNECT_TIMEOUT: 5,
            },
            groq: {
                label: 'Groq — ultra-fast inference, conservative rate limits. Rate: 30 req/min, Concurrency: 4, Read: 60s.',
                PROVIDER_RATE_LIMIT: 30,
                PROVIDER_RATE_WINDOW: 60,
                PROVIDER_MAX_CONCURRENCY: 4,
                HTTP_READ_TIMEOUT: 60,
                HTTP_WRITE_TIMEOUT: 10,
                HTTP_CONNECT_TIMEOUT: 5,
            },
            deepseek: {
                label: 'DeepSeek — moderate throughput with generous timeouts. Rate: 50 req/min, Concurrency: 6, Read: 180s.',
                PROVIDER_RATE_LIMIT: 50,
                PROVIDER_RATE_WINDOW: 60,
                PROVIDER_MAX_CONCURRENCY: 6,
                HTTP_READ_TIMEOUT: 180,
                HTTP_WRITE_TIMEOUT: 20,
                HTTP_CONNECT_TIMEOUT: 5,
            },
            mistral: {
                label: 'Mistral — standard cloud inference defaults. Rate: 40 req/min, Concurrency: 5, Read: 120s.',
                PROVIDER_RATE_LIMIT: 40,
                PROVIDER_RATE_WINDOW: 60,
                PROVIDER_MAX_CONCURRENCY: 5,
                HTTP_READ_TIMEOUT: 120,
                HTTP_WRITE_TIMEOUT: 15,
                HTTP_CONNECT_TIMEOUT: 5,
            },
            cerebras: {
                label: 'Cerebras — wafer-scale fast inference. Rate: 60 req/min, Concurrency: 8, Read: 60s.',
                PROVIDER_RATE_LIMIT: 60,
                PROVIDER_RATE_WINDOW: 60,
                PROVIDER_MAX_CONCURRENCY: 8,
                HTTP_READ_TIMEOUT: 60,
                HTTP_WRITE_TIMEOUT: 10,
                HTTP_CONNECT_TIMEOUT: 5,
            },
        };

        function applySelectedPreset() {
            const sel = document.getElementById('provider-preset-select');
            const providerKey = sel ? sel.value : '';
            if (!providerKey) {
                showToast('Select a provider from the dropdown first.', 'error');
                return;
            }
            applyProviderPreset(providerKey);
        }

        function applyProviderPreset(providerKey) {
            const preset = PROVIDER_PRESETS[providerKey];
            if (!preset) return;
            const fields = ['PROVIDER_RATE_LIMIT', 'PROVIDER_RATE_WINDOW', 'PROVIDER_MAX_CONCURRENCY',
                            'HTTP_READ_TIMEOUT', 'HTTP_WRITE_TIMEOUT', 'HTTP_CONNECT_TIMEOUT'];
            fields.forEach(field => {
                const el = document.getElementById(field);
                if (el) el.value = preset[field];
            });
            const descEl = document.getElementById('preset-desc');
            if (descEl) descEl.innerText = preset.label;
            showToast('Preset applied. Click Commit to save.', 'success');
        }

        // Auto-populate description when dropdown changes
        document.getElementById('provider-preset-select').addEventListener('change', function() {
            const preset = PROVIDER_PRESETS[this.value];
            const descEl = document.getElementById('preset-desc');
            if (descEl) descEl.innerText = preset ? preset.label : '';
        });

        // Loop execution hooks
        fetchModels();
        loadConfigs();
        pollTelemetry();
        setInterval(pollTelemetry, 3000);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)
