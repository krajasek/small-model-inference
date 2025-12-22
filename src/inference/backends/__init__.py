"""Backend module for inference engines."""

from .base import GenerationConfig, InferenceBackend

__all__ = ["InferenceBackend", "GenerationConfig", "create_backend"]


def create_backend(
    settings: "Settings",  # noqa: F821
    tracer: "InferenceTracer | None" = None,  # noqa: F821
) -> InferenceBackend:
    """Create an inference backend based on settings.

    Args:
        settings: Application settings with backend configuration
        tracer: Optional tracer for observability

    Returns:
        Configured inference backend

    Raises:
        ValueError: If backend type is unknown
        ImportError: If backend dependencies are not installed
    """
    from ..config import Settings
    from ..observability.tracing import InferenceTracer

    # Type narrowing for proper typing
    _settings: Settings = settings
    _tracer: InferenceTracer | None = tracer

    if _settings.backend == "pytorch":
        from .pytorch import PyTorchBackend

        return PyTorchBackend(_settings, _tracer)
    elif _settings.backend == "llama-cpp":
        from .llamacpp import LlamaCppBackend

        return LlamaCppBackend(_settings, _tracer)
    else:
        raise ValueError(f"Unknown backend: {_settings.backend}")
