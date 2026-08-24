"""Model selection and candidate model chain resolution for Core Gateway."""

from typing import Any

from config import model_registry, settings


def pick_model_with_fallbacks(client_model: str, body: dict[str, Any]) -> tuple[list[str], str | list[dict[str, Any]] | None]:
    """Resolve candidate model chain for client_model and apply thinking mode prompt directive if active.

    Returns:
        tuple of (candidate_models, system_prompt)
    """
    primary = model_registry.get_primary(client_model)
    fallbacks = model_registry.get_fallbacks(client_model)
    candidates = [c for c in ([primary] + fallbacks) if c]

    system = body.get("system")
    model_thinking_mode = settings.get_thinking_mode(client_model)
    if model_thinking_mode == "open":
        thinking_prompt = "\n\nImportant: Analyze the task and write your step-by-step reasoning inside <think>...</think> tags before providing your answer or executing tools."
        if system:
            if isinstance(system, str):
                system = system + thinking_prompt
            elif isinstance(system, list):
                system = list(system) + [{"type": "text", "text": thinking_prompt}]
        else:
            system = thinking_prompt

    return candidates, system
