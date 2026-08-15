import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

from models.telemetry import RequestLogEntry

# Configure Loguru logger default format and stdout sink
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Load env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

DATA_DIR = Path(__file__).parent.parent / ".data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_MODELS_YAML_PATH = DATA_DIR / "models.yaml"


def _ensure_models_yaml_exists() -> None:
    """Ensure .data/models.yaml exists, generating or migrating from config/models.yaml if missing."""
    if _MODELS_YAML_PATH.exists():
        return

    legacy_path = Path(__file__).parent / "models.yaml"
    if legacy_path.exists():
        try:
            _MODELS_YAML_PATH.write_text(legacy_path.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("Migrated config/models.yaml -> .data/models.yaml")
            return
        except Exception as e:
            logger.warning(f"Failed to migrate legacy models.yaml: {e}")

    default_yaml_content = """claude_default:
  display_name: 1. Default (Recommended - Opus 5 / Nemotron 70B)
  description: "Claude Code CLI Default Selection — Highest capacity 1M context model"
  primary: open_router/poolside/laguna-xs-2.1:free
  fallback_order: []
  metadata:
    context: 1000000
    max_output: 32768
    rpm_limit: 15
    tpm_limit: 200000
    tags:
    - default
    - opus-5
    - 1m-context
    - agentic
    - coding
claude_opus:
  display_name: 2. Opus (1M context - Nemotron 70B / Llama 3.3)
  description: "Opus 5 with 1M context — Best for everyday, complex tasks"
  primary: nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b
  fallback_order: []
  metadata:
    context: 1000000
    max_output: 32768
    rpm_limit: 15
    tpm_limit: 200000
    tags:
    - opus-5
    - 1m-context
    - reasoning
    - agentic
    - coding
claude_sonnet:
  display_name: 3. Sonnet (Llama 3.1 70B)
  description: "Sonnet — Efficient for routine tasks"
  primary: nvidia_nim/z-ai/glm-5.2
  fallback_order: []
  metadata:
    context: 1000000
    max_output: 16384
    rpm_limit: 15
    tpm_limit: 200000
    tags:
    - sonnet
    - coding
    - tool-calling
    - agentic
claude_sonnet_1m:
  display_name: 4. Sonnet 5 (1M context - Llama 3.3 70B)
  description: Sonnet 5 for long sessions with 1M context
  primary: nvidia_nim/z-ai/glm-5.2
  fallback_order:
  - nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b
  - nvidia_nim/poolside/laguna-xs-2.1
  - gemini/models/gemini-3.1-flash-lite
  metadata:
    context: 1000000
    max_output: 32768
    rpm_limit: 15
    tpm_limit: 200000
    tags:
    - sonnet-5
    - 1m-context
    - coding
    - tool-calling
claude_haiku:
  display_name: 5. Haiku (Fastest 8B / Flash)
  description: "Haiku 4.5 — Fastest for quick answers"
  primary: nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b
  fallback_order: []
  metadata:
    context: 200000
    max_output: 8192
    rpm_limit: 20
    tpm_limit: 100000
    tags:
    - haiku-4.5
    - fast
    - coding
"""
    _MODELS_YAML_PATH.write_text(default_yaml_content, encoding="utf-8")
    logger.info("Generated default .data/models.yaml")


@dataclass
class ModelMetadata:
    context: int = 128000
    max_output: int = 16384
    rpm_limit: int = 15
    tpm_limit: int = 200000
    tags: list[str] = field(default_factory=list)


@dataclass
class ModelEntry:
    primary: str
    fallback_order: list[str]
    metadata: ModelMetadata


class ModelRegistry:
    """Load models.yaml and resolve Claude alias → upstream model chain."""

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}
        self._load()

    def _load(self) -> None:
        _ensure_models_yaml_exists()
        if not _MODELS_YAML_PATH.exists():
            return
        raw: dict[str, Any] = (
            yaml.safe_load(_MODELS_YAML_PATH.read_text(encoding="utf-8")) or {}
        )
        for alias, data in raw.items():
            meta_raw = data.get("metadata", {})
            self._entries[alias] = ModelEntry(
                primary=data.get("primary", ""),
                fallback_order=data.get("fallback_order", []),
                metadata=ModelMetadata(
                    context=meta_raw.get("context", 128000),
                    max_output=meta_raw.get("max_output", 16384),
                    rpm_limit=meta_raw.get("rpm_limit", 15),
                    tpm_limit=meta_raw.get("tpm_limit", 200000),
                    tags=meta_raw.get("tags", []),
                ),
            )

    def _resolve_alias(self, client_model: str) -> str:
        """Map a Claude client model string to a registry alias.

        Ordered matching corresponding to Claude Code CLI selection menu (1 to 5):
        1. Default (recommended) -> claude_default
        2. Opus (1M context)     -> claude_opus
        3. Sonnet                -> claude_sonnet
        4. Sonnet 5 (1M context) -> claude_sonnet_1m
        5. Haiku                 -> claude_haiku
        """
        lower = client_model.lower()
        if "default" in lower:
            return "claude_default"
        if "opus" in lower:
            return "claude_opus"
        if any(k in lower for k in ["3-7-sonnet", "3.7-sonnet", "sonnet-5", "sonnet-1m"]) or ("1m" in lower and "sonnet" in lower):
            return "claude_sonnet_1m"
        if "sonnet" in lower:
            return "claude_sonnet"
        if "haiku" in lower:
            return "claude_haiku"
        return "claude_default"

    def get_primary(self, client_model: str) -> str:
        # If client_model is a full provider model string (e.g. open_router/model or nvidia_nim/model)
        if "/" in client_model and not client_model.startswith("claude"):
            return client_model

        alias = self._resolve_alias(client_model)

        # 1. Prioritize active Settings (from .env / dashboard edits)
        settings_obj = globals().get("settings")
        if settings_obj:
            if alias == "claude_opus" and getattr(settings_obj, "MODEL_OPUS", None):
                return settings_obj.MODEL_OPUS
            if alias == "claude_sonnet" and getattr(settings_obj, "MODEL_SONNET", None):
                return settings_obj.MODEL_SONNET
            if alias == "claude_sonnet_1m" and getattr(settings_obj, "MODEL_SONNET_1M", None):
                return settings_obj.MODEL_SONNET_1M
            if alias == "claude_haiku" and getattr(settings_obj, "MODEL_HAIKU", None):
                return settings_obj.MODEL_HAIKU
            if alias == "claude_default" and getattr(settings_obj, "MODEL", None):
                return settings_obj.MODEL

        # 2. Fall back to models.yaml entry
        entry = self._entries.get(alias)
        if entry and entry.primary:
            return entry.primary
        return "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct"

    def get_fallbacks(self, client_model: str) -> list[str]:
        if "/" in client_model and not client_model.startswith("claude"):
            return []
        alias = self._resolve_alias(client_model)
        entry = self._entries.get(alias)
        return list(entry.fallback_order) if entry and entry.fallback_order is not None else []


    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Return metadata for a concrete upstream model_id or a client alias."""
        # Try by alias first
        alias = self._resolve_alias(model_id)
        entry = self._entries.get(alias)
        if entry:
            return entry.metadata
        # Try matching by primary/fallback model_id
        for e in self._entries.values():
            if e.primary == model_id or model_id in e.fallback_order:
                return e.metadata
        return ModelMetadata()

    def all_model_ids(self) -> list[str]:
        """All unique upstream model IDs known to the registry."""
        ids: list[str] = []
        for e in self._entries.values():
            if e.primary and e.primary not in ids:
                ids.append(e.primary)
            for fb in e.fallback_order:
                if fb not in ids:
                    ids.append(fb)
        return ids

    def save_entries(self, updates: dict[str, dict[str, Any]]) -> None:
        """Update models.yaml with new primary or fallback_order definitions."""
        _ensure_models_yaml_exists()
        if not _MODELS_YAML_PATH.exists():
            return
        raw: dict[str, Any] = (
            yaml.safe_load(_MODELS_YAML_PATH.read_text(encoding="utf-8")) or {}
        )
        for alias, data in updates.items():
            if alias in raw:
                if "primary" in data:
                    raw[alias]["primary"] = data["primary"]
                if "fallback_order" in data:
                    raw[alias]["fallback_order"] = data["fallback_order"]

        with open(_MODELS_YAML_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        self.reload()

    def reload(self) -> None:
        self._entries.clear()
        self._load()


def get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "t", "y", "yes")


def get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


PROVIDER_DEFAULTS: dict[str, dict[str, int]] = {
    "nvidia_nim": {
        "rpm": 38,
        "tpm": 200000,
        "rpd": 1000,
        "rate_window": 60,
        "max_concurrency": 5,
        "context": 1000000,
        "max_output": 32768,
        "http_read_timeout": 120,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "open_router": {
        "rpm": 60,
        "tpm": 300000,
        "rpd": 10000,
        "rate_window": 60,
        "max_concurrency": 10,
        "context": 200000,
        "max_output": 16384,
        "http_read_timeout": 120,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "gemini": {
        "rpm": 30,
        "tpm": 1000000,
        "rpd": 1500,
        "rate_window": 60,
        "max_concurrency": 5,
        "context": 1000000,
        "max_output": 8192,
        "http_read_timeout": 120,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "groq": {
        "rpm": 30,
        "tpm": 100000,
        "rpd": 1440,
        "rate_window": 60,
        "max_concurrency": 5,
        "context": 128000,
        "max_output": 8192,
        "http_read_timeout": 60,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "deepseek": {
        "rpm": 60,
        "tpm": 200000,
        "rpd": 5000,
        "rate_window": 60,
        "max_concurrency": 5,
        "context": 64000,
        "max_output": 8192,
        "http_read_timeout": 120,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "mistral": {
        "rpm": 30,
        "tpm": 200000,
        "rpd": 2000,
        "rate_window": 60,
        "max_concurrency": 5,
        "context": 128000,
        "max_output": 16384,
        "http_read_timeout": 120,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "cerebras": {
        "rpm": 30,
        "tpm": 100000,
        "rpd": 1440,
        "rate_window": 60,
        "max_concurrency": 5,
        "context": 128000,
        "max_output": 8192,
        "http_read_timeout": 60,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "fireworks": {
        "rpm": 60,
        "tpm": 200000,
        "rpd": 5000,
        "rate_window": 60,
        "max_concurrency": 5,
        "context": 128000,
        "max_output": 16384,
        "http_read_timeout": 120,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "kimi": {
        "rpm": 30,
        "tpm": 200000,
        "rpd": 1000,
        "rate_window": 60,
        "max_concurrency": 5,
        "context": 128000,
        "max_output": 8192,
        "http_read_timeout": 120,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "lmstudio": {
        "rpm": 120,
        "tpm": 1000000,
        "rpd": 100000,
        "rate_window": 60,
        "max_concurrency": 10,
        "context": 128000,
        "max_output": 16384,
        "http_read_timeout": 300,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "ollama": {
        "rpm": 120,
        "tpm": 1000000,
        "rpd": 100000,
        "rate_window": 60,
        "max_concurrency": 10,
        "context": 128000,
        "max_output": 16384,
        "http_read_timeout": 300,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
    "llama_cpp": {
        "rpm": 120,
        "tpm": 1000000,
        "rpd": 100000,
        "rate_window": 60,
        "max_concurrency": 10,
        "context": 128000,
        "max_output": 16384,
        "http_read_timeout": 300,
        "http_write_timeout": 10,
        "http_connect_timeout": 2,
    },
}


class Settings:
    # Upstream API keys and endpoints
    NVIDIA_NIM_API_KEY: str = os.getenv("NVIDIA_NIM_API_KEY", "")

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    GATEWAY_AUTH_TOKEN: str = os.getenv("GATEWAY_AUTH_TOKEN", "")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
    FIREWORKS_API_KEY: str = os.getenv("FIREWORKS_API_KEY", "")
    KIMI_API_KEY: str = os.getenv("KIMI_API_KEY", "")

    # Base URLs / Proxies
    NVIDIA_NIM_BASE_URL: str = os.getenv(
        "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    MISTRAL_BASE_URL: str = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    GEMINI_BASE_URL: str = os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
    )
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    DEEPSEEK_BASE_URL: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
    )
    CEREBRAS_BASE_URL: str = os.getenv(
        "CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"
    )
    FIREWORKS_BASE_URL: str = os.getenv(
        "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
    )
    LM_STUDIO_BASE_URL: str = os.getenv(
        "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
    )
    LLAMA_CPP_BASE_URL: str = os.getenv(
        "LLAMA_CPP_BASE_URL", "http://localhost:8080/v1"
    )
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Model Mappings (format: provider_type/model/name)
    MODEL_OPUS: str = os.getenv("MODEL_OPUS", "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct")
    MODEL_SONNET: str = os.getenv(
        "MODEL_SONNET", "nvidia_nim/meta/llama-3.1-70b-instruct"
    )
    MODEL_SONNET_1M: str = os.getenv(
        "MODEL_SONNET_1M", "open_router/meta-llama/llama-3.3-70b-instruct"
    )
    MODEL_HAIKU: str = os.getenv(
        "MODEL_HAIKU", "nvidia_nim/meta/llama-3.1-8b-instruct"
    )
    MODEL: str = os.getenv("MODEL", "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct")


    # Provider rate limits and performance controls
    REFRESH_TIME: int = get_int("REFRESH_TIME", 4)
    PROVIDER_RATE_LIMIT: int = get_int("PROVIDER_RATE_LIMIT", 40)
    PROVIDER_RATE_WINDOW: int = get_int("PROVIDER_RATE_WINDOW", 60)
    PROVIDER_MAX_CONCURRENCY: int = get_int("PROVIDER_MAX_CONCURRENCY", 5)

    # Dedicated NVIDIA NIM Proactive Rate Limiter & Guard settings
    NVIDIA_NIM_SAFE_RPM: int = get_int("NVIDIA_NIM_SAFE_RPM", 38)
    NVIDIA_NIM_WINDOW_SECONDS: int = get_int("NVIDIA_NIM_WINDOW_SECONDS", 60)
    NVIDIA_NIM_MAX_QUEUE_WAIT: int = get_int("NVIDIA_NIM_MAX_QUEUE_WAIT", 30)

    # HTTP client timeouts
    HTTP_READ_TIMEOUT: int = get_int("HTTP_READ_TIMEOUT", 120)
    HTTP_WRITE_TIMEOUT: int = get_int("HTTP_WRITE_TIMEOUT", 10)
    HTTP_CONNECT_TIMEOUT: int = get_int("HTTP_CONNECT_TIMEOUT", 2)

    # Messaging integration (Telegram / Discord)
    MESSAGING_PLATFORM: str = os.getenv("MESSAGING_PLATFORM", "discord")
    MESSAGING_RATE_LIMIT: int = get_int("MESSAGING_RATE_LIMIT", 1)
    MESSAGING_RATE_WINDOW: int = get_int("MESSAGING_RATE_WINDOW", 1)

    # Voice Note Transcription options
    VOICE_NOTE_ENABLED: bool = get_bool("VOICE_NOTE_ENABLED", False)
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "nvidia_nim")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Telegram Specific config
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    ALLOWED_TELEGRAM_USER_ID: str = os.getenv("ALLOWED_TELEGRAM_USER_ID", "")

    # Discord Specific config
    DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
    ALLOWED_DISCORD_CHANNELS: str = os.getenv("ALLOWED_DISCORD_CHANNELS", "")

    # Agent / Mock configurations
    CLAUDE_WORKSPACE: str = os.getenv("CLAUDE_WORKSPACE", "./agent_workspace")
    ALLOWED_DIR: str = os.getenv("ALLOWED_DIR", "")
    FAST_PREFIX_DETECTION: bool = get_bool("FAST_PREFIX_DETECTION", True)
    ENABLE_NETWORK_PROBE_MOCK: bool = get_bool("ENABLE_NETWORK_PROBE_MOCK", True)
    ENABLE_TITLE_GENERATION_SKIP: bool = get_bool("ENABLE_TITLE_GENERATION_SKIP", True)
    ENABLE_SUGGESTION_MODE_SKIP: bool = get_bool("ENABLE_SUGGESTION_MODE_SKIP", True)
    ENABLE_FILEPATH_EXTRACTION_MOCK: bool = get_bool(
        "ENABLE_FILEPATH_EXTRACTION_MOCK", True
    )

    # Thinking directive modes per model: "inherit" (native), "open" (force think tags), "close" (suppress think tags)
    THINKING_MODE_OPUS: str = os.getenv("THINKING_MODE_OPUS", "inherit")
    THINKING_MODE_SONNET: str = os.getenv("THINKING_MODE_SONNET", "inherit")
    THINKING_MODE_HAIKU: str = os.getenv("THINKING_MODE_HAIKU", "inherit")
    THINKING_MODE_DEFAULT: str = os.getenv("THINKING_MODE_DEFAULT", "inherit")

    def get_thinking_mode(self, client_model: str) -> str:
        lower = client_model.lower()
        if "opus" in lower:
            return self.THINKING_MODE_OPUS
        if "sonnet" in lower:
            return self.THINKING_MODE_SONNET
        if "haiku" in lower:
            return self.THINKING_MODE_HAIKU
        return self.THINKING_MODE_DEFAULT

    def get_provider_config(self, provider_name: str) -> dict[str, int]:
        p = provider_name.lower().strip()
        defaults = PROVIDER_DEFAULTS.get(p, PROVIDER_DEFAULTS["nvidia_nim"])
        res = {}
        for field_name, def_val in defaults.items():
            env_key = f"PROVIDER_{p.upper()}_{field_name.upper()}"
            val_attr = getattr(self, env_key, None)
            if val_attr is not None:
                try:
                    res[field_name] = int(val_attr)
                except (ValueError, TypeError):
                    res[field_name] = def_val
            else:
                res[field_name] = get_int(env_key, def_val)
        return res

    def reload(self) -> None:
        """Reload configurations from the .env file and update settings in-memory."""
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            load_dotenv(override=True)

        self.NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY", "")
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
        self.GATEWAY_AUTH_TOKEN = os.getenv("GATEWAY_AUTH_TOKEN", "")
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
        self.CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
        self.FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
        self.KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")

        self.NVIDIA_NIM_BASE_URL = os.getenv(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.OPENROUTER_BASE_URL = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.MISTRAL_BASE_URL = os.getenv(
            "MISTRAL_BASE_URL", "https://api.mistral.ai/v1"
        )
        self.GEMINI_BASE_URL = os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )
        self.GROQ_BASE_URL = os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )
        self.DEEPSEEK_BASE_URL = os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        )
        self.CEREBRAS_BASE_URL = os.getenv(
            "CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"
        )
        self.FIREWORKS_BASE_URL = os.getenv(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
        self.LM_STUDIO_BASE_URL = os.getenv(
            "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
        )
        self.LLAMA_CPP_BASE_URL = os.getenv(
            "LLAMA_CPP_BASE_URL", "http://localhost:8080/v1"
        )
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.MODEL_OPUS = os.getenv("MODEL_OPUS", "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct")
        self.MODEL_SONNET = os.getenv(
            "MODEL_SONNET", "nvidia_nim/meta/llama-3.1-70b-instruct"
        )
        self.MODEL_SONNET_1M = os.getenv(
            "MODEL_SONNET_1M", "open_router/meta-llama/llama-3.3-70b-instruct"
        )
        self.MODEL_HAIKU = os.getenv(
            "MODEL_HAIKU", "nvidia_nim/meta/llama-3.1-8b-instruct"
        )
        self.MODEL = os.getenv("MODEL", "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct")

        self.REFRESH_TIME = get_int("REFRESH_TIME", 4)
        self.PROVIDER_RATE_LIMIT = get_int("PROVIDER_RATE_LIMIT", 40)
        self.PROVIDER_RATE_WINDOW = get_int("PROVIDER_RATE_WINDOW", 60)
        self.PROVIDER_MAX_CONCURRENCY = get_int("PROVIDER_MAX_CONCURRENCY", 5)

        self.NVIDIA_NIM_SAFE_RPM = get_int("NVIDIA_NIM_SAFE_RPM", 38)
        self.NVIDIA_NIM_WINDOW_SECONDS = get_int("NVIDIA_NIM_WINDOW_SECONDS", 60)
        self.NVIDIA_NIM_MAX_QUEUE_WAIT = get_int("NVIDIA_NIM_MAX_QUEUE_WAIT", 30)

        self.HTTP_READ_TIMEOUT = get_int("HTTP_READ_TIMEOUT", 120)
        self.HTTP_WRITE_TIMEOUT = get_int("HTTP_WRITE_TIMEOUT", 10)
        self.HTTP_CONNECT_TIMEOUT = get_int("HTTP_CONNECT_TIMEOUT", 2)

        self.MESSAGING_PLATFORM = os.getenv("MESSAGING_PLATFORM", "discord")
        self.MESSAGING_RATE_LIMIT = get_int("MESSAGING_RATE_LIMIT", 1)
        self.MESSAGING_RATE_WINDOW = get_int("MESSAGING_RATE_WINDOW", 1)

        self.VOICE_NOTE_ENABLED = get_bool("VOICE_NOTE_ENABLED", False)
        self.WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "nvidia_nim")
        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3")
        self.HF_TOKEN = os.getenv("HF_TOKEN", "")

        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.ALLOWED_TELEGRAM_USER_ID = os.getenv("ALLOWED_TELEGRAM_USER_ID", "")

        self.DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
        self.ALLOWED_DISCORD_CHANNELS = os.getenv("ALLOWED_DISCORD_CHANNELS", "")

        self.CLAUDE_WORKSPACE = os.getenv("CLAUDE_WORKSPACE", "./agent_workspace")
        self.ALLOWED_DIR = os.getenv("ALLOWED_DIR", "")
        self.FAST_PREFIX_DETECTION = get_bool("FAST_PREFIX_DETECTION", True)
        self.ENABLE_NETWORK_PROBE_MOCK = get_bool("ENABLE_NETWORK_PROBE_MOCK", True)
        self.ENABLE_TITLE_GENERATION_SKIP = get_bool(
            "ENABLE_TITLE_GENERATION_SKIP", True
        )
        self.ENABLE_SUGGESTION_MODE_SKIP = get_bool("ENABLE_SUGGESTION_MODE_SKIP", True)
        self.ENABLE_FILEPATH_EXTRACTION_MOCK = get_bool(
            "ENABLE_FILEPATH_EXTRACTION_MOCK", True
        )
        self.THINKING_MODE_OPUS = os.getenv("THINKING_MODE_OPUS", "inherit")
        self.THINKING_MODE_SONNET = os.getenv("THINKING_MODE_SONNET", "inherit")
        self.THINKING_MODE_HAIKU = os.getenv("THINKING_MODE_HAIKU", "inherit")
        self.THINKING_MODE_DEFAULT = os.getenv("THINKING_MODE_DEFAULT", "inherit")

        for p, d in PROVIDER_DEFAULTS.items():
            for field_name, def_val in d.items():
                attr_name = f"PROVIDER_{p.upper()}_{field_name.upper()}"
                setattr(self, attr_name, get_int(attr_name, def_val))


for _p, _d in PROVIDER_DEFAULTS.items():
    for _field_name, _def_val in _d.items():
        _attr_name = f"PROVIDER_{_p.upper()}_{_field_name.upper()}"
        setattr(Settings, _attr_name, get_int(_attr_name, _def_val))





_LOG_FILE_PATH = Path(__file__).parent.parent / ".development" / "requests.jsonl"


class ProxyStats:
    def __init__(self) -> None:
        self.total_requests: int = 0
        self.mocked_requests: int = 0
        self.error_count: int = 0
        self.active_concurrency: int = 0
        self.recent_requests: list[RequestLogEntry] = []
        self._request_counter: int = 0
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load persisted transaction records from disk on startup."""
        if not _LOG_FILE_PATH.exists():
            return
        try:
            with open(_LOG_FILE_PATH, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                        entry = RequestLogEntry.from_dict(data)
                        self.recent_requests.insert(0, entry)
                        self.total_requests += 1
                        if entry.mocked:
                            self.mocked_requests += 1
                        if entry.is_error:
                            self.error_count += 1
                        self._request_counter += 1
                    except Exception as parse_err:
                        logger.debug("Failed to parse log line: %s", parse_err)
            # Keep up to 100 recent entries in memory
            if len(self.recent_requests) > 100:
                self.recent_requests = self.recent_requests[:100]
        except Exception as e:
            logger.warning("Failed to load request logs from disk: %s", e)

    def record_log(
        self,
        method: str,
        path: str,
        client_model: str,
        mapped_model: str,
        status_code: int,
        start_time: float,
        mocked: bool = False,
        fallbacks_used: list[str] | None = None,
        request_body: dict[str, Any] | None = None,
        response_body: dict[str, Any] | str | None = None,
        headers: dict[str, str] | None = None,
        error_details: dict[str, Any] | None = None,
        attempt_history: list[dict[str, Any]] | None = None,
    ) -> None:
        self._request_counter += 1
        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Calculate token counts
        input_tokens = 0
        output_tokens = 0

        if request_body and isinstance(request_body, dict):
            from atomic.guards.token_budget import TokenBudgetGuard
            guard = TokenBudgetGuard(client_model)
            input_tokens = guard.count_prompt_tokens(
                request_body.get("messages", []),
                request_body.get("system"),
                request_body.get("tools"),
            )

        if response_body:
            if isinstance(response_body, dict):
                usage = response_body.get("usage", {})
                if isinstance(usage, dict) and usage.get("output_tokens"):
                    output_tokens = int(usage["output_tokens"])
                elif isinstance(usage, dict) and usage.get("completion_tokens"):
                    output_tokens = int(usage["completion_tokens"])
                else:
                    output_tokens = len(json.dumps(response_body)) // 4
            elif isinstance(response_body, str):
                output_tokens = len(response_body) // 4

        entry = RequestLogEntry(
            id=f"req_{self._request_counter}_{int(time.time())}",
            timestamp=time.strftime("%H:%M:%S"),
            method=method,
            path=path,
            client_model=client_model,
            mapped_model=mapped_model,
            status_code=status_code,
            duration_ms=duration_ms,
            mocked=mocked,
            fallbacks_used=fallbacks_used or [],
            request_body=request_body,
            response_body=response_body,
            headers=headers or {},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error_details=error_details,
            attempt_history=attempt_history or [],
        )
        self.recent_requests.insert(0, entry)
        # Keep last 100 in memory
        if len(self.recent_requests) > 100:
            self.recent_requests.pop()

        # Append to persistent disk log file
        try:
            _LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(include_payload=True)) + "\n")
        except Exception as write_err:
            logger.warning("Failed to persist request log to disk: %s", write_err)

    def get_recent_dicts(self, include_payload: bool = True) -> list[dict[str, Any]]:
        return [entry.to_dict(include_payload=include_payload) for entry in self.recent_requests]

    def get_paginated_payloads(
        self,
        limit: int = 20,
        page: int = 1,
        query: str = "",
    ) -> dict[str, Any]:
        """Return server-side filtered and paginated payloads to conserve RAM and bandwidth."""
        import math
        filtered_entries = []
        q = (query or "").strip().lower()

        for req in self.recent_requests:
            if not q:
                filtered_entries.append(req)
            else:
                c_model = (req.client_model or "").lower()
                m_model = (req.mapped_model or "").lower()
                path = (req.path or "").lower()
                method = (req.method or "").lower()
                status = str(req.status_code)
                req_b = json.dumps(req.request_body).lower() if isinstance(req.request_body, (dict, list)) else str(req.request_body or "").lower()
                resp_b = json.dumps(req.response_body).lower() if isinstance(req.response_body, (dict, list)) else str(req.response_body or "").lower()

                if (q in c_model or q in m_model or q in path or q in method or q in status or q in req_b or q in resp_b):
                    filtered_entries.append(req)

        total = len(filtered_entries)
        limit = max(1, limit)
        total_pages = max(1, math.ceil(total / limit))
        page = min(total_pages, max(1, page))

        start_idx = (page - 1) * limit
        end_idx = min(total, start_idx + limit)
        paged_items = filtered_entries[start_idx:end_idx]

        return {
            "total": total,
            "total_captured": self._request_counter,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "payloads": [item.to_dict(include_payload=True) for item in paged_items],
        }



settings = Settings()
stats = ProxyStats()
model_registry = ModelRegistry()
