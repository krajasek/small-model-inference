"""Pydantic schemas for API request/response models."""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from ..backends.base import GenerationConfig


# Shared schemas
class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# Completion endpoint schemas
class CompletionRequest(BaseModel):
    """Request for text completion."""

    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1)
    do_sample: bool = True
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stream: bool = False
    # Tracing fields (optional)
    user: str | None = Field(None, description="User ID for tracing")

    def to_generation_config(self) -> GenerationConfig:
        """Convert to GenerationConfig."""
        return GenerationConfig(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
        )


class CompletionChoice(BaseModel):
    """A completion choice."""

    index: int
    text: str
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage


# Chat completion endpoint schemas (OpenAI-compatible)
class ChatMessage(BaseModel):
    """A chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    messages: list[ChatMessage]
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1)
    do_sample: bool = True
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stream: bool = False
    # Tracing fields (optional)
    user: str | None = Field(None, description="User ID for tracing")

    def to_generation_config(self) -> GenerationConfig:
        """Convert to GenerationConfig."""
        return GenerationConfig(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
        )

    def format_prompt(self) -> str:
        """Format messages into a prompt string for LLaMA/Mistral style models."""
        parts = []
        for msg in self.messages:
            if msg.role == "system":
                parts.append(f"[INST] <<SYS>>\n{msg.content}\n<</SYS>>\n\n")
            elif msg.role == "user":
                if parts and not parts[-1].endswith("[/INST]"):
                    parts.append(f"{msg.content} [/INST]")
                else:
                    parts.append(f"[INST] {msg.content} [/INST]")
            elif msg.role == "assistant":
                parts.append(f" {msg.content} ")
        return "".join(parts)


class ChatCompletionChoice(BaseModel):
    """A chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# Streaming response schemas
class ChatCompletionChunkDelta(BaseModel):
    """Delta for streaming chat completion."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """Choice for streaming chat completion."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for chat completion."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# Model info schemas
class ModelInfo(BaseModel):
    """Information about a loaded model."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    """Response for model list endpoint."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


# Health check schemas
class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    device: str | None = None
