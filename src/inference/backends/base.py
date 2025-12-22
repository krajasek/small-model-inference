"""Base protocol for inference backends."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


@runtime_checkable
class InferenceBackend(Protocol):
    """Abstract interface for inference backends.

    All inference backends must implement this protocol to be compatible
    with the inference server. This allows swapping between different
    inference engines (PyTorch, llama-cpp, etc.) transparently.
    """

    @property
    def model_name(self) -> str:
        """Get the name of the loaded model."""
        ...

    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        ...

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Generate text completion synchronously.

        Args:
            prompt: Input text to complete
            config: Generation configuration
            user_id: Optional user ID for tracing
            session_id: Optional session ID for tracing

        Returns:
            Tuple of (generated_text, usage_stats)
            usage_stats contains: prompt_tokens, completion_tokens, total_tokens
        """
        ...

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[tuple[str, bool]]:
        """Generate text completion with streaming.

        Args:
            prompt: Input text to complete
            config: Generation configuration
            user_id: Optional user ID for tracing
            session_id: Optional session ID for tracing

        Yields:
            Tuples of (token_text, is_finished)
        """
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model metadata (name, device, config, etc.)
        """
        ...

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache hit rates, sizes, etc.
        """
        ...

    def clear_caches(self) -> None:
        """Clear all caches."""
        ...

    def shutdown(self) -> None:
        """Shutdown the backend and release resources."""
        ...
