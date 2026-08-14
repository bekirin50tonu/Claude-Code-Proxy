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

_MODELS_YAML_PATH = Path(__file__).parent / "models.yaml"


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
        """Map a Claude client model string to a registry alias."""
        lower = client_model.lower()
        if "opus" in lower:
            return "claude_opus"
        if "sonnet" in lower:
            return "claude_sonnet"
        if "haiku" in lower:
            return "claude_haiku"
        return "claude_default"

    def get_primary(self, client_model: str) -> str:
        alias = self._resolve_alias(client_model)
        entry = self._entries.get(alias)
        if entry and entry.primary:
            return entry.primary
        # Fallback to .env values
        settings_obj = globals().get("settings")
        if settings_obj:
            lower = client_model.lower()
            if "opus" in lower:
                return settings_obj.MODEL_OPUS
            if "sonnet" in lower:
                return settings_obj.MODEL_SONNET
            if "haiku" in lower:
                return settings_obj.MODEL_HAIKU
            return settings_obj.MODEL
        return "nvidia_nim/z-ai/glm4.7"

    def get_fallbacks(self, client_model: str) -> list[str]:
        alias = self._resolve_alias(client_model)
        entry = self._entries.get(alias)
        return entry.fallback_order if entry else []

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
    MODEL_HAIKU: str = os.getenv(
        "MODEL_HAIKU", "nvidia_nim/meta/llama-3.1-8b-instruct"
    )
    MODEL: str = os.getenv("MODEL", "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct")

    # Provider rate limits and performance controls
    PROVIDER_RATE_LIMIT: int = get_int("PROVIDER_RATE_LIMIT", 40)
    PROVIDER_RATE_WINDOW: int = get_int("PROVIDER_RATE_WINDOW", 60)
    PROVIDER_MAX_CONCURRENCY: int = get_int("PROVIDER_MAX_CONCURRENCY", 5)

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
        self.MODEL_HAIKU = os.getenv(
            "MODEL_HAIKU", "nvidia_nim/meta/llama-3.1-8b-instruct"
        )
        self.MODEL = os.getenv("MODEL", "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct")

        self.PROVIDER_RATE_LIMIT = get_int("PROVIDER_RATE_LIMIT", 40)
        self.PROVIDER_RATE_WINDOW = get_int("PROVIDER_RATE_WINDOW", 60)
        self.PROVIDER_MAX_CONCURRENCY = get_int("PROVIDER_MAX_CONCURRENCY", 5)

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





class ProxyStats:
    def __init__(self) -> None:
        self.total_requests: int = 0
        self.mocked_requests: int = 0
        self.error_count: int = 0
        self.active_concurrency: int = 0
        self.recent_requests: list[RequestLogEntry] = []
        self._request_counter: int = 0

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
    ) -> None:
        self._request_counter += 1
        duration_ms = round((time.time() - start_time) * 1000, 2)
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
        )
        self.recent_requests.insert(0, entry)
        # Keep last 50 requests
        if len(self.recent_requests) > 50:
            self.recent_requests.pop()

    def get_recent_dicts(self, include_payload: bool = True) -> list[dict[str, Any]]:
        return [entry.to_dict(include_payload=include_payload) for entry in self.recent_requests]


settings = Settings()
stats = ProxyStats()
model_registry = ModelRegistry()
