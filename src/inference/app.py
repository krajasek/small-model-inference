"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router
from .config import Settings
from .engine.cpu_optimizer import CPUOptimizer
from .engine.inference import InferenceEngine
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

    # Create inference engine
    engine = InferenceEngine(model, tokenizer, device, settings)
    app.state.engine = engine

    logger.info("Model loaded and ready for inference")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down, cleaning up resources...")
    del app.state.engine
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
