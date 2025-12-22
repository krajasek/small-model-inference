"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router
from .backends import create_backend
from .config import Settings
from .observability.tracing import InferenceTracer, init_tracer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle - load model on startup, cleanup on shutdown."""
    settings: Settings = app.state.settings

    # Initialize tracer if tracing is enabled
    tracer: InferenceTracer | None = None
    if settings.enable_tracing:
        tracer = init_tracer(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            enabled=True,
            debug=settings.langfuse_debug,
        )
        logger.info("Langfuse tracing initialized")

    # Create backend (handles model loading internally)
    logger.info(f"Creating {settings.backend} backend...")
    backend = create_backend(settings, tracer)
    app.state.engine = backend
    app.state.tracer = tracer

    logger.info(f"Backend ready: {backend.model_name} on {backend.device}")

    # Log optimization status
    if settings.backend == "pytorch":
        logger.info(
            f"Optimizations: kv_cache={settings.use_kv_cache}, "
            f"response_cache={settings.enable_response_cache}, "
            f"prompt_cache={settings.enable_prompt_cache}, "
            f"tokenizer_cache={settings.enable_tokenizer_cache}, "
            f"speculative={settings.enable_speculative_decoding}, "
            f"tracing={tracer is not None and tracer.enabled}"
        )
    else:
        logger.info(
            f"llama-cpp config: n_ctx={settings.llama_cpp_n_ctx}, "
            f"n_threads={settings.num_threads}, "
            f"n_gpu_layers={settings.llama_cpp_n_gpu_layers}, "
            f"tracing={tracer is not None and tracer.enabled}"
        )

    yield

    # Cleanup on shutdown
    logger.info("Shutting down, cleaning up resources...")

    # Flush and shutdown tracer
    if tracer:
        tracer.shutdown()
        logger.info("Langfuse tracer shut down")

    # Shutdown backend
    backend.shutdown()

    del app.state.engine


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = FastAPI(
        title="Small Model Inference",
        description="CPU-friendly inference layer for serving small language models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store settings in app state
    app.state.settings = settings

    # Include API routes
    app.include_router(router)

    return app
