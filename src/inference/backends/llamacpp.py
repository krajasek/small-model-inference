"""llama-cpp-python backend for fast CPU inference."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..config import Settings
from ..engine.cache import ResponseCache
from ..observability.tracing import GenerationTrace, InferenceTracer
from .base import GenerationConfig

logger = logging.getLogger(__name__)


class LlamaCppBackend:
    """llama-cpp-python backend for fast CPU inference.

    This backend uses llama-cpp-python for optimized CPU inference.
    Supports GGUF model format (single .gguf file).
    """

    def __init__(
        self,
        settings: Settings,
        tracer: InferenceTracer | None = None,
    ) -> None:
        """Initialize the llama-cpp backend.

        Args:
            settings: Application settings
            tracer: Optional tracer for observability

        Raises:
            ImportError: If llama-cpp-python is not installed
            FileNotFoundError: If model file doesn't exist
        """
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            ) from e

        self._settings = settings
        self._tracer = tracer
        self._model_path = Path(settings.model_path)

        # Validate model path
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self._model_path}")

        # Determine model file path
        if self._model_path.is_file() and self._model_path.suffix == ".gguf":
            model_file = str(self._model_path)
            self._model_name = settings.model_name or self._model_path.stem
        elif self._model_path.is_dir():
            # Look for .gguf file in directory
            gguf_files = list(self._model_path.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(
                    f"No .gguf files found in {self._model_path}. "
                    "llama-cpp backend requires GGUF format models."
                )
            model_file = str(gguf_files[0])
            self._model_name = settings.model_name or gguf_files[0].stem
            if len(gguf_files) > 1:
                logger.warning(
                    f"Multiple .gguf files found, using: {gguf_files[0].name}"
                )
        else:
            raise FileNotFoundError(
                f"Invalid model path: {self._model_path}. "
                "Expected .gguf file or directory containing .gguf files."
            )

        logger.info(f"Loading llama-cpp model from {model_file}")

        # Initialize llama-cpp model
        self._model = Llama(
            model_path=model_file,
            n_ctx=settings.llama_cpp_n_ctx,
            n_threads=settings.num_threads,
            n_gpu_layers=settings.llama_cpp_n_gpu_layers,
            n_batch=settings.llama_cpp_n_batch,
            verbose=False,
        )

        # Determine device string for reporting
        self._device = "cuda" if settings.llama_cpp_n_gpu_layers > 0 else "cpu"

        # Initialize response cache if enabled
        self._response_cache: ResponseCache | None = None
        if settings.enable_response_cache:
            self._response_cache = ResponseCache(
                max_size=settings.response_cache_size,
                ttl_seconds=settings.response_cache_ttl,
            )

        # Set model info on tracer
        if self._tracer:
            self._tracer.set_model_info(self._model_name, self.get_model_info())

        logger.info(
            f"LlamaCppBackend initialized: model={self._model_name}, "
            f"n_ctx={settings.llama_cpp_n_ctx}, "
            f"n_threads={settings.num_threads}, "
            f"n_gpu_layers={settings.llama_cpp_n_gpu_layers}, "
            f"response_cache={settings.enable_response_cache}"
        )

    @property
    def model_name(self) -> str:
        """Get the name of the loaded model."""
        return self._model_name

    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        return self._device

    def _create_trace(
        self,
        name: str = "generation",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> GenerationTrace | None:
        """Create a trace context if tracing is enabled."""
        if self._tracer and self._tracer.enabled:
            return self._tracer.trace_generation(
                name=name,
                user_id=user_id,
                session_id=session_id,
            )
        return None

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Generate text completion synchronously with response caching.

        Returns:
            Tuple of (generated_text, usage_stats)
        """
        trace = self._create_trace("completion", user_id, session_id)

        if trace:
            trace.__enter__()
            trace.set_input(prompt=prompt)
            trace.set_parameters(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
            )

        try:
            # Check response cache first
            if self._response_cache:
                cached = self._response_cache.get(
                    prompt=prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                )
                if cached:
                    logger.debug("Returning cached response")
                    if trace:
                        trace.record_cache_hit("response")
                        trace.set_output(text=cached[0], usage=cached[1])
                        trace.__exit__(None, None, None)
                    return cached

            # Generate with llama-cpp
            output = self._model(
                prompt,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 0.0,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repetition_penalty,
                echo=False,  # Don't include prompt in output
            )

            # Extract text and usage from output
            generated_text = output["choices"][0]["text"]
            usage = {
                "prompt_tokens": output["usage"]["prompt_tokens"],
                "completion_tokens": output["usage"]["completion_tokens"],
                "total_tokens": output["usage"]["total_tokens"],
            }

            # Prepend prompt to match PyTorch backend behavior
            full_text = prompt + generated_text

            # Cache the response
            if self._response_cache:
                self._response_cache.put(
                    prompt=prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    response=full_text,
                    usage=usage,
                )

            if trace:
                trace.set_output(text=full_text, usage=usage)
                trace.__exit__(None, None, None)
                # Flush immediately to ensure trace is sent
                if self._tracer:
                    self._tracer.flush()

            return full_text, usage

        except Exception as e:
            if trace:
                trace.__exit__(type(e), e, e.__traceback__)
                if self._tracer:
                    self._tracer.flush()
            raise

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[tuple[str, bool]]:
        """Generate text completion with streaming.

        Yields:
            Tuples of (token_text, is_finished)
        """
        trace = self._create_trace("stream_completion", user_id, session_id)

        if trace:
            trace.__enter__()
            trace.set_input(prompt=prompt)
            trace.set_parameters(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                streaming=True,
            )

        try:
            generated_text_parts: list[str] = []
            first_token = True
            prompt_tokens = 0
            completion_tokens = 0

            # Stream with llama-cpp
            for output in self._model(
                prompt,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 0.0,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repetition_penalty,
                echo=False,
                stream=True,
            ):
                if first_token:
                    if trace:
                        trace.record_first_token()
                    first_token = False

                token = output["choices"][0]["text"]
                generated_text_parts.append(token)

                # Track token counts from final chunk
                if "usage" in output:
                    prompt_tokens = output["usage"].get("prompt_tokens", 0)
                    completion_tokens = output["usage"].get("completion_tokens", 0)

                yield token, False

            # Record final metrics
            if trace:
                full_text = prompt + "".join(generated_text_parts)
                comp_tokens = completion_tokens or len(generated_text_parts)
                trace.set_output(
                    text=full_text,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": comp_tokens,
                        "total_tokens": prompt_tokens + comp_tokens,
                    },
                )
                trace.__exit__(None, None, None)
                # Flush immediately to ensure trace is sent
                if self._tracer:
                    self._tracer.flush()

            yield "", True

        except Exception as e:
            if trace:
                trace.__exit__(type(e), e, e.__traceback__)
                if self._tracer:
                    self._tracer.flush()
            raise

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self._model_name,
            "backend": "llama-cpp",
            "device": self._device,
            "n_ctx": self._settings.llama_cpp_n_ctx,
            "n_threads": self._settings.num_threads,
            "n_gpu_layers": self._settings.llama_cpp_n_gpu_layers,
            "n_batch": self._settings.llama_cpp_n_batch,
            "max_sequence_length": self._settings.llama_cpp_n_ctx,
        }

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats: dict[str, Any] = {
            "backend": "llama-cpp",
            "note": "llama-cpp manages KV cache internally",
        }

        if self._response_cache:
            stats["response_cache"] = self._response_cache.stats()

        return stats

    def clear_caches(self) -> None:
        """Clear response cache."""
        if self._response_cache:
            self._response_cache.clear()
            logger.info("Response cache cleared")
        else:
            logger.debug("No caches to clear (llama-cpp manages KV cache internally)")

    def shutdown(self) -> None:
        """Shutdown the backend and release resources."""
        logger.info("Shutting down llama-cpp backend")
        # llama-cpp-python handles cleanup via __del__
        del self._model
