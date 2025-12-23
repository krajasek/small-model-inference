"""WebSocket endpoint for streaming inference with protobuf wire format."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator

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


def _format_chat_prompt(messages: list[ChatMessage]) -> str:
    """Format chat messages into a prompt string (mirrors REST API logic)."""
    parts: list[str] = []
    for msg in messages:
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
