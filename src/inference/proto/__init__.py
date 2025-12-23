"""Protocol buffer definitions for WebSocket inference API."""

from .inference import (
    ChatCompletionRequest,
    ChatMessage,
    ChunkChoice,
    ChunkDelta,
    ClientMessage,
    CompletionRequest,
    ErrorResponse,
    GenerationParams,
    ServerMessage,
    StreamChunk,
    StreamComplete,
    Usage,
)

__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "ChunkChoice",
    "ChunkDelta",
    "ClientMessage",
    "CompletionRequest",
    "ErrorResponse",
    "GenerationParams",
    "ServerMessage",
    "StreamChunk",
    "StreamComplete",
    "Usage",
]
