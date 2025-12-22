"""FastAPI dependencies for dependency injection."""

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from ..backends.base import InferenceBackend


def get_engine(request: Request) -> "InferenceBackend":
    """Get the inference backend from application state."""
    return request.app.state.engine
