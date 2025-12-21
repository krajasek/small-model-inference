"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router
from .config import Settings
from .engine.cpu_optimizer import CPUOptimizer
from .engine.inference import InferenceEngine, load_draft_model
from .engine.batching import ContinuousBatcher
from .models.loader import ModelLoader, resolve_device

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle - load model on startup, cleanup on shutdown."""
    settings: Settings = app.state.settings

    # Resolve device
    device = resolve_device(settings.device)
    logger.info(f"Using device: {device}")

    # Setup CPU optimizations if on CPU
    if device == "cpu":
        optimizer = CPUOptimizer(settings)
        optimizer.setup_environment()

    # Load model
    logger.info("Loading model...")
    loader = ModelLoader(settings)
    model, tokenizer = loader.load()

    # Apply CPU optimizations to model
    if device == "cpu":
        model = optimizer.optimize_model(model)

    # Load draft model for speculative decoding if enabled
    draft_model = None
    if settings.enable_speculative_decoding and settings.draft_model_path:
        draft_model = load_draft_model(
            settings.draft_model_path,
            device,
            settings,
        )
        if draft_model and device == "cpu":
            draft_model = optimizer.optimize_model(draft_model)

    # Create inference engine
    engine = InferenceEngine(
        model, tokenizer, device, settings, draft_model=draft_model
    )
    app.state.engine = engine

    # Start continuous batcher if enabled
    batcher = None
    if settings.enable_batching:
        batcher = ContinuousBatcher(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_batch_size=settings.max_batch_size,
            max_wait_time_ms=settings.batch_wait_time_ms,
        )
        await batcher.start()
        app.state.batcher = batcher
        logger.info("Continuous batching enabled")

    logger.info("Model loaded and ready for inference")

    # Log optimization status
    logger.info(
        f"Optimizations: kv_cache={settings.use_kv_cache}, "
        f"response_cache={settings.enable_response_cache}, "
        f"prompt_cache={settings.enable_prompt_cache}, "
        f"tokenizer_cache={settings.enable_tokenizer_cache}, "
        f"batching={settings.enable_batching}, "
        f"speculative={settings.enable_speculative_decoding and draft_model is not None}"
    )

    yield

    # Cleanup on shutdown
    logger.info("Shutting down, cleaning up resources...")

    # Stop batcher
    if batcher:
        await batcher.stop()

    # Clear caches
    engine.clear_caches()

    del app.state.engine
    if draft_model:
        del draft_model
    del model
    del tokenizer


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
