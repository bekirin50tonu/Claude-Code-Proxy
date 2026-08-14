"""OpenAI / NIM / OpenRouter Request, Response, and Chunk Pydantic Models."""

from typing import Any

from pydantic import BaseModel, Field


class OpenAIChatFunction(BaseModel):
    """OpenAI Function descriptor."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class OpenAIChatTool(BaseModel):
    """OpenAI Tool descriptor."""

    type: str = "function"
    function: OpenAIChatFunction


class OpenAIChatMessage(BaseModel):
    """OpenAI Chat Message object."""

    role: str
    content: Any | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible Chat Completion API request schema."""

    model: str
    messages: list[OpenAIChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False
    tools: list[OpenAIChatTool] | None = None
    tool_choice: Any | None = None


class OpenAIStreamChoiceDelta(BaseModel):
    """Delta object within streaming choices."""

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAIStreamChoice(BaseModel):
    """Choice object within streaming chunks."""

    index: int = 0
    delta: OpenAIStreamChoiceDelta
    finish_reason: str | None = None


class OpenAIStreamChunk(BaseModel):
    """OpenAI-compatible streaming chunk chunk schema."""

    id: str | None = None
    object: str = "chat.completion.chunk"
    created: int | None = None
    model: str | None = None
    choices: list[OpenAIStreamChoice] = Field(default_factory=list)
