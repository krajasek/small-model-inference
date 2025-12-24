"""WebSocket endpoint for streaming inference with protobuf wire format."""

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import betterproto
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..backends.base import GenerationConfig, InferenceBackend
from ..proto import (
    ChatMessage,
    ChunkChoice,
    ChunkDelta,
    ClientMessage,
    ErrorResponse,
    GenerationParams,
    ServerMessage,
    StreamChunk,
    StreamComplete,
    Usage,
)

logger = logging.getLogger(__name__)

ws_router = APIRouter()


# =============================================================================
# Caching for WebSocket
# =============================================================================


@dataclass
class CacheEntry:
    """Entry in the formatted prompt cache."""

    formatted_prompt: str
    created_at: float = field(default_factory=time.time)


class FormattedPromptCache:
    """LRU cache for formatted chat prompts.

    Caches the result of formatting chat messages into prompt strings.
    Particularly useful when system prompts are repeated across requests.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 1800) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, messages: list[ChatMessage]) -> str:
        """Generate cache key from messages."""
        # Create a deterministic string representation of messages
        msg_str = "|".join(f"{m.role}:{m.content}" for m in messages)
        return hashlib.md5(msg_str.encode()).hexdigest()

    def get(self, messages: list[ChatMessage]) -> str | None:
        """Get cached formatted prompt if available."""
        key = self._make_key(messages)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.created_at > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"Formatted prompt cache hit for key {key[:8]}...")
            return entry.formatted_prompt

    def put(self, messages: list[ChatMessage], formatted_prompt: str) -> None:
        """Store formatted prompt in cache."""
        key = self._make_key(messages)

        with self._lock:
            # Remove oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(formatted_prompt=formatted_prompt)
            logger.debug(f"Cached formatted prompt for key {key[:8]}...")

    def clear(self) -> None:
        """Clear all cached prompts."""
        with self._lock:
            self._cache.clear()
            logger.info("Formatted prompt cache cleared")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class SystemPromptCache:
    """Cache for formatted system prompts.

    System prompts often remain constant across many chat requests.
    This cache stores the formatted version of system prompts for reuse.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, content: str) -> str:
        """Generate cache key from system prompt content."""
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, content: str) -> str | None:
        """Get cached formatted system prompt if available."""
        key = self._make_key(content)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.created_at > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            self._cache.move_to_end(key)
            self._hits += 1
            return entry.formatted_prompt

    def put(self, content: str, formatted: str) -> None:
        """Store formatted system prompt in cache."""
        key = self._make_key(content)

        with self._lock:
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(formatted_prompt=formatted)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class WebSocketCacheManager:
    """Manages all WebSocket-layer caches."""

    def __init__(
        self,
        formatted_prompt_cache_size: int = 500,
        system_prompt_cache_size: int = 100,
    ) -> None:
        self.formatted_prompt_cache = FormattedPromptCache(
            max_size=formatted_prompt_cache_size
        )
        self.system_prompt_cache = SystemPromptCache(
            max_size=system_prompt_cache_size
        )
        logger.info(
            f"WebSocketCacheManager initialized: "
            f"formatted_prompt_cache_size={formatted_prompt_cache_size}, "
            f"system_prompt_cache_size={system_prompt_cache_size}"
        )

    def clear_all(self) -> None:
        """Clear all caches."""
        self.formatted_prompt_cache.clear()
        self.system_prompt_cache._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "formatted_prompt_cache": self.formatted_prompt_cache.stats(),
            "system_prompt_cache": self.system_prompt_cache.stats(),
        }


# Global cache manager instance
_cache_manager: WebSocketCacheManager | None = None


def get_cache_manager() -> WebSocketCacheManager:
    """Get or create the global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = WebSocketCacheManager()
    return _cache_manager


def get_websocket_cache_stats() -> dict[str, Any]:
    """Get WebSocket cache statistics."""
    return get_cache_manager().stats()


def clear_websocket_caches() -> None:
    """Clear all WebSocket caches."""
    get_cache_manager().clear_all()


def _params_to_generation_config(params: GenerationParams | None) -> GenerationConfig:
    """Convert protobuf GenerationParams to GenerationConfig."""
    if params is None:
        return GenerationConfig()
    return GenerationConfig(
        max_new_tokens=params.max_tokens or 256,
        temperature=params.temperature or 0.7,
        top_p=params.top_p or 0.9,
        top_k=params.top_k or 50,
        do_sample=params.do_sample if params.do_sample else True,
        repetition_penalty=params.repetition_penalty or 1.1,
    )


def _format_system_prompt(content: str) -> str:
    """Format a system prompt message."""
    return f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"


def _format_chat_prompt_uncached(messages: list[ChatMessage]) -> str:
    """Format chat messages into a prompt string (no caching)."""
    cache_manager = get_cache_manager()
    parts: list[str] = []

    for msg in messages:
        if msg.role == "system":
            # Check system prompt cache
            cached_sys = cache_manager.system_prompt_cache.get(msg.content)
            if cached_sys is not None:
                parts.append(cached_sys)
            else:
                formatted_sys = _format_system_prompt(msg.content)
                cache_manager.system_prompt_cache.put(msg.content, formatted_sys)
                parts.append(formatted_sys)
        elif msg.role == "user":
            if parts and not parts[-1].endswith("[/INST]"):
                parts.append(f"{msg.content} [/INST]")
            else:
                parts.append(f"[INST] {msg.content} [/INST]")
        elif msg.role == "assistant":
            parts.append(f" {msg.content} ")

    return "".join(parts)


def _format_chat_prompt(messages: list[ChatMessage]) -> str:
    """Format chat messages into a prompt string with caching.

    Uses two levels of caching:
    1. Full formatted prompt cache - for identical message sequences
    2. System prompt cache - for reusing formatted system prompts
    """
    cache_manager = get_cache_manager()

    # Check full prompt cache first
    cached = cache_manager.formatted_prompt_cache.get(messages)
    if cached is not None:
        return cached

    # Format with system prompt caching
    formatted = _format_chat_prompt_uncached(messages)

    # Cache the full formatted prompt
    cache_manager.formatted_prompt_cache.put(messages, formatted)

    return formatted


async def _iter_stream_tokens(
    engine: InferenceBackend,
    prompt: str,
    config: GenerationConfig,
    user_id: str | None,
) -> AsyncIterator[tuple[str, bool]]:
    """Iterate over tokens from generate_stream, yielding control periodically."""
    for token_data in engine.generate_stream(prompt, config, user_id=user_id):
        yield token_data
        # Allow event loop to process other tasks
        await asyncio.sleep(0)


async def _stream_completion(
    request_id: str,
    prompt: str,
    config: GenerationConfig,
    engine: InferenceBackend,
    user_id: str | None = None,
) -> AsyncIterator[ServerMessage]:
    """Generate streaming completion and yield ServerMessage chunks."""
    response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    token_count = 0
    async for token, is_finished in _iter_stream_tokens(engine, prompt, config, user_id):
        if is_finished:
            # Send final chunk with finish_reason
            yield ServerMessage(
                request_id=request_id,
                chunk=StreamChunk(
                    id=response_id,
                    created=created,
                    model=engine.model_name,
                    choice=ChunkChoice(
                        index=0,
                        delta=ChunkDelta(),
                        finish_reason="stop",
                    ),
                ),
            )
            # Send completion message with usage stats
            yield ServerMessage(
                request_id=request_id,
                complete=StreamComplete(
                    id=response_id,
                    created=created,
                    model=engine.model_name,
                    usage=Usage(
                        prompt_tokens=0,
                        completion_tokens=token_count,
                        total_tokens=token_count,
                    ),
                ),
            )
            break

        token_count += 1
        yield ServerMessage(
            request_id=request_id,
            chunk=StreamChunk(
                id=response_id,
                created=created,
                model=engine.model_name,
                choice=ChunkChoice(
                    index=0,
                    delta=ChunkDelta(content=token),
                ),
            ),
        )


async def _stream_chat_completion(
    request_id: str,
    messages: list[ChatMessage],
    config: GenerationConfig,
    engine: InferenceBackend,
    user_id: str | None = None,
) -> AsyncIterator[ServerMessage]:
    """Generate streaming chat completion and yield ServerMessage chunks."""
    prompt = _format_chat_prompt(messages)
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Send initial chunk with role
    yield ServerMessage(
        request_id=request_id,
        chunk=StreamChunk(
            id=response_id,
            created=created,
            model=engine.model_name,
            choice=ChunkChoice(
                index=0,
                delta=ChunkDelta(role="assistant"),
            ),
        ),
    )

    token_count = 0
    async for token, is_finished in _iter_stream_tokens(engine, prompt, config, user_id):
        if is_finished:
            # Send final chunk with finish_reason
            yield ServerMessage(
                request_id=request_id,
                chunk=StreamChunk(
                    id=response_id,
                    created=created,
                    model=engine.model_name,
                    choice=ChunkChoice(
                        index=0,
                        delta=ChunkDelta(),
                        finish_reason="stop",
                    ),
                ),
            )
            # Send completion message with usage stats
            yield ServerMessage(
                request_id=request_id,
                complete=StreamComplete(
                    id=response_id,
                    created=created,
                    model=engine.model_name,
                    usage=Usage(
                        prompt_tokens=0,
                        completion_tokens=token_count,
                        total_tokens=token_count,
                    ),
                ),
            )
            break

        token_count += 1
        yield ServerMessage(
            request_id=request_id,
            chunk=StreamChunk(
                id=response_id,
                created=created,
                model=engine.model_name,
                choice=ChunkChoice(
                    index=0,
                    delta=ChunkDelta(content=token),
                ),
            ),
        )


async def _handle_client_message(
    client_msg: ClientMessage,
    engine: InferenceBackend,
    websocket: WebSocket,
) -> None:
    """Handle a single client message and stream responses."""
    request_id = client_msg.request_id

    # Use betterproto.which_one_of to check which oneof field is set
    payload_type, _ = betterproto.which_one_of(client_msg, "payload")

    try:
        if payload_type == "completion":
            req = client_msg.completion
            if not req.prompt:
                raise ValueError("completion.prompt is required")
            config = _params_to_generation_config(req.params)
            user_id = req.params.user if req.params else None

            async for server_msg in _stream_completion(
                request_id, req.prompt, config, engine, user_id
            ):
                await websocket.send_bytes(bytes(server_msg))

        elif payload_type == "chat_completion":
            req = client_msg.chat_completion
            if not req.messages:
                raise ValueError("chat_completion.messages is required")
            config = _params_to_generation_config(req.params)
            user_id = req.params.user if req.params else None

            async for server_msg in _stream_chat_completion(
                request_id, req.messages, config, engine, user_id
            ):
                await websocket.send_bytes(bytes(server_msg))

        else:
            # Unknown or empty payload
            error_msg = ServerMessage(
                request_id=request_id,
                error=ErrorResponse(
                    code=400,
                    message="Invalid request. Use completion or chat_completion payload.",
                    type="validation_error",
                ),
            )
            await websocket.send_bytes(bytes(error_msg))

    except Exception as e:
        logger.exception("Error processing WebSocket message")
        error_msg = ServerMessage(
            request_id=request_id,
            error=ErrorResponse(
                code=500,
                message=str(e),
                type="internal_error",
            ),
        )
        await websocket.send_bytes(bytes(error_msg))


@ws_router.websocket("/v1/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming inference.

    Protocol:
    1. Client sends binary protobuf ClientMessage
    2. Server responds with binary protobuf ServerMessage stream
    3. Multiple requests can be sent sequentially (connection reuse)

    Message format:
    - ClientMessage: Contains request_id and either completion or chat_completion payload
    - ServerMessage: Contains request_id and either chunk, complete, or error payload
    """
    await websocket.accept()
    engine: InferenceBackend = websocket.app.state.engine

    logger.info("WebSocket client connected")

    try:
        while True:
            # Receive binary protobuf message
            data = await websocket.receive_bytes()

            try:
                # Parse ClientMessage
                client_msg = ClientMessage().parse(data)
                await _handle_client_message(client_msg, engine, websocket)

            except Exception as e:
                logger.exception("Error parsing client message")
                error_msg = ServerMessage(
                    request_id="",
                    error=ErrorResponse(
                        code=400,
                        message=f"Failed to parse message: {e}",
                        type="parse_error",
                    ),
                )
                await websocket.send_bytes(bytes(error_msg))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
