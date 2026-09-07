"""Provider Registry Module for Claude Code Proxy.

Provides structured, configuration-driven provider definitions to eliminate
hardcoded provider logic and enable clean registry lookup.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderSpec:
    """Specification of an upstream LLM Provider."""

    name: str
    base_url_attr: str
    api_key_attr: str = ""
    url_path_suffix: str = ""
    default_base_url: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "nvidia_nim": ProviderSpec(
        name="nvidia_nim",
        base_url_attr="NVIDIA_NIM_BASE_URL",
        api_key_attr="NVIDIA_NIM_API_KEYS",
        default_base_url="https://integrate.api.nvidia.com/v1",
    ),
    "open_router": ProviderSpec(
        name="open_router",
        base_url_attr="OPENROUTER_BASE_URL",
        api_key_attr="OPENROUTER_API_KEY",
        default_base_url="https://openrouter.ai/api/v1",
        extra_headers={
            "HTTP-Referer": "https://github.com/bekirin50tonu/Claude-Code-Proxy",
            "X-Title": "Claude Code Proxy",
        },
    ),
    "tokenrouter": ProviderSpec(
        name="tokenrouter",
        base_url_attr="TOKENROUTER_BASE_URL",
        api_key_attr="TOKENROUTER_API_KEY",
        default_base_url="https://api.tokenrouter.com/v1",
    ),
    "groq": ProviderSpec(
        name="groq",
        base_url_attr="GROQ_BASE_URL",
        api_key_attr="GROQ_API_KEY",
        default_base_url="https://api.groq.com/openai/v1",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        base_url_attr="DEEPSEEK_BASE_URL",
        api_key_attr="DEEPSEEK_API_KEY",
        default_base_url="https://api.deepseek.com/v1",
    ),
    "mistral": ProviderSpec(
        name="mistral",
        base_url_attr="MISTRAL_BASE_URL",
        api_key_attr="MISTRAL_API_KEY",
        default_base_url="https://api.mistral.ai/v1",
    ),
    "cerebras": ProviderSpec(
        name="cerebras",
        base_url_attr="CEREBRAS_BASE_URL",
        api_key_attr="CEREBRAS_API_KEY",
        default_base_url="https://api.cerebras.ai/v1",
    ),
    "fireworks": ProviderSpec(
        name="fireworks",
        base_url_attr="FIREWORKS_BASE_URL",
        api_key_attr="FIREWORKS_API_KEY",
        default_base_url="https://api.fireworks.ai/inference/v1",
    ),
    "kimi": ProviderSpec(
        name="kimi",
        base_url_attr="KIMI_BASE_URL",
        api_key_attr="KIMI_API_KEY",
        default_base_url="https://api.moonshot.cn/v1",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        base_url_attr="GEMINI_BASE_URL",
        api_key_attr="GEMINI_API_KEY",
        url_path_suffix="/openai",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
    ),
    "lmstudio": ProviderSpec(
        name="lmstudio",
        base_url_attr="LM_STUDIO_BASE_URL",
        api_key_attr="",
        default_base_url="http://localhost:1234/v1",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        base_url_attr="OLLAMA_BASE_URL",
        api_key_attr="",
        default_base_url="http://localhost:11434",
    ),
    "llama_cpp": ProviderSpec(
        name="llama_cpp",
        base_url_attr="LLAMA_CPP_BASE_URL",
        api_key_attr="",
        default_base_url="http://localhost:8080/v1",
    ),
}


class UnknownProviderError(ValueError):
    """Raised when an unrecognized provider name is specified and strict mode is active."""

    def __init__(self, provider_name: str, available_providers: list[str]) -> None:
        self.provider_name = provider_name
        self.available_providers = available_providers
        super().__init__(
            f"Unknown provider '{provider_name}'. Available providers: {', '.join(sorted(available_providers))}"
        )
