"""FastAPI dependencies for dependency injection."""

from fastapi import Request

from ..engine.inference import InferenceEngine


def get_engine(request: Request) -> InferenceEngine:
    """Get the inference engine from application state."""
    return request.app.state.engine
