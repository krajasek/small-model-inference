"""API route definitions."""

import json
import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..backends.base import InferenceBackend
from .dependencies import get_engine
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    Usage,
)

# Type alias for dependency injection
EngineDep = Annotated[InferenceBackend, Depends(get_engine)]

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return HealthResponse(status="unhealthy", model_loaded=False)
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=engine.device,
    )


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(engine: EngineDep) -> ModelListResponse:
    """List available models."""
    model_info = ModelInfo(id=engine.model_name)
    return ModelListResponse(data=[model_info])


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    engine: EngineDep,
) -> CompletionResponse | StreamingResponse:
    """Create a text completion."""
    if request.stream:
        return _stream_completion(request, engine)

    try:
        config = request.to_generation_config()
        generated_text, usage_stats = engine.generate(
            request.prompt,
            config,
            user_id=request.user,
        )

        return CompletionResponse(
            model=engine.model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=generated_text,
                    finish_reason="stop",
                )
            ],
            usage=Usage(**usage_stats),
        )
    except Exception as e:
        logger.exception("Error during completion")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _stream_completion(
    request: CompletionRequest,
    engine: InferenceBackend,
) -> StreamingResponse:
    """Stream completion response."""

    async def generate():
        config = request.to_generation_config()
        response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        for token, is_finished in engine.generate_stream(
            request.prompt,
            config,
            user_id=request.user,
        ):
            if is_finished:
                yield "data: [DONE]\n\n"
                break

            chunk = {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": engine.model_name,
                "choices": [{"index": 0, "text": token, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    engine: EngineDep,
) -> ChatCompletionResponse | StreamingResponse:
    """Create a chat completion (OpenAI-compatible)."""
    if request.stream:
        return _stream_chat_completion(request, engine)

    try:
        config = request.to_generation_config()
        prompt = request.format_prompt()
        generated_text, usage_stats = engine.generate(
            prompt,
            config,
            user_id=request.user,
        )

        # Extract just the assistant's response (after the prompt)
        response_text = generated_text[len(prompt) :].strip()

        return ChatCompletionResponse(
            model=engine.model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(**usage_stats),
        )
    except Exception as e:
        logger.exception("Error during chat completion")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _stream_chat_completion(
    request: ChatCompletionRequest,
    engine: InferenceBackend,
) -> StreamingResponse:
    """Stream chat completion response."""

    async def generate():
        config = request.to_generation_config()
        prompt = request.format_prompt()
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Send initial chunk with role
        initial_chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=engine.model_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"

        # Stream content
        for token, is_finished in engine.generate_stream(
            prompt,
            config,
            user_id=request.user,
        ):
            if is_finished:
                # Send final chunk
                final_chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=engine.model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                break

            chunk = ChatCompletionChunk(
                id=response_id,
                created=created,
                model=engine.model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=token),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/v1/cache/stats")
async def get_cache_stats(engine: EngineDep) -> dict:
    """Get cache statistics for monitoring."""
    from .websocket import get_websocket_cache_stats

    return {
        "caches": engine.get_cache_stats(),
        "websocket_caches": get_websocket_cache_stats(),
        "model_info": engine.get_model_info(),
    }


@router.post("/v1/cache/clear")
async def clear_caches(engine: EngineDep) -> dict:
    """Clear all inference caches."""
    from .websocket import clear_websocket_caches

    engine.clear_caches()
    clear_websocket_caches()
    return {"status": "ok", "message": "All caches cleared"}


@router.get("/v1/tracing/status")
async def get_tracing_status(request: Request) -> dict:
    """Get tracing status and configuration."""
    tracer = getattr(request.app.state, "tracer", None)
    if tracer is None:
        return {
            "enabled": False,
            "provider": None,
        }
    return {
        "enabled": tracer.enabled,
        "provider": "langfuse",
        "model_name": tracer._model_name,
    }
