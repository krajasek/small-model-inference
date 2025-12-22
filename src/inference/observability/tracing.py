"""Langfuse-based tracing for inference observability."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langfuse import Langfuse

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer: "InferenceTracer | None" = None


@dataclass
class GenerationMetrics:
    """Metrics captured during generation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token_ms: float | None = None
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    cache_hit: bool = False
    cache_type: str | None = None  # "response", "prompt", "tokenizer"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceTracer:
    """Langfuse-based tracer for inference observability.

    Captures key metrics:
    - Generation latency (total and time-to-first-token)
    - Token counts (prompt, completion, total)
    - Tokens per second throughput
    - Cache hit/miss rates
    - Generation parameters (temperature, top_p, etc.)
    - Model information
    - Errors and exceptions
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        enabled: bool = True,
        debug: bool = False,
        flush_at: int = 15,
        flush_interval: float = 10.0,
    ) -> None:
        """Initialize the Langfuse tracer.

        Args:
            public_key: Langfuse public key (or set LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (or set LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL (or set LANGFUSE_HOST env var)
            enabled: Whether tracing is enabled
            debug: Enable debug logging
            flush_at: Number of events before flushing
            flush_interval: Seconds between flushes
        """
        self.enabled = enabled
        self._client: Langfuse | None = None
        self._model_name: str = "local-model"
        self._model_info: dict[str, Any] = {}

        if not enabled:
            logger.info("Langfuse tracing disabled")
            return

        try:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                debug=debug,
                flush_at=flush_at,
                flush_interval=flush_interval,
            )
            # Check if client is properly initialized (has keys)
            if not self._client.auth_check():
                logger.warning("Langfuse authentication failed. Tracing disabled.")
                self.enabled = False
                self._client = None
            else:
                logger.info(f"Langfuse tracing initialized (host: {host or 'default'})")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}. Tracing disabled.")
            self.enabled = False

    def set_model_info(self, model_name: str, model_info: dict[str, Any]) -> None:
        """Set model information for traces."""
        self._model_name = model_name
        self._model_info = model_info

    def trace_generation(
        self,
        name: str = "generation",
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "GenerationTrace":
        """Create a new generation trace context.

        Usage:
            with tracer.trace_generation(name="completion") as trace:
                trace.set_input(prompt="Hello")
                trace.set_parameters(temperature=0.7, max_tokens=100)
                # ... do generation ...
                trace.set_output(text="World", usage={"prompt_tokens": 1, ...})
        """
        return GenerationTrace(
            tracer=self,
            name=name,
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
            metadata=metadata or {},
        )

    def flush(self) -> None:
        """Flush pending events to Langfuse."""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse events: {e}")

    def shutdown(self) -> None:
        """Shutdown the tracer and flush remaining events."""
        if self._client:
            try:
                self._client.flush()
                self._client.shutdown()
            except Exception as e:
                logger.warning(f"Error during Langfuse shutdown: {e}")


class GenerationTrace:
    """Context manager for tracing a generation request."""

    def __init__(
        self,
        tracer: InferenceTracer,
        name: str,
        user_id: str | None,
        session_id: str | None,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._user_id = user_id
        self._session_id = session_id
        self._tags = tags
        self._metadata = metadata
        self._generation: Any = None
        self._start_time: float = 0.0
        self._first_token_time: float | None = None
        self._input_data: Any = None
        self._parameters: dict[str, Any] = {}
        self._metrics = GenerationMetrics()

    def __enter__(self) -> "GenerationTrace":
        """Start the trace."""
        self._start_time = time.perf_counter()

        if self._tracer.enabled and self._tracer._client:
            try:
                # Create a generation span
                self._generation = self._tracer._client.start_generation(
                    name=self._name,
                    model=self._tracer._model_name,
                    metadata={
                        **self._metadata,
                        "model_info": self._tracer._model_info,
                        "user_id": self._user_id,
                        "session_id": self._session_id,
                        "tags": self._tags,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to start generation trace: {e}")

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End the trace and record metrics."""
        end_time = time.perf_counter()
        self._metrics.total_latency_ms = (end_time - self._start_time) * 1000

        # Calculate tokens per second
        if self._metrics.completion_tokens > 0 and self._metrics.total_latency_ms > 0:
            self._metrics.tokens_per_second = self._metrics.completion_tokens / (
                self._metrics.total_latency_ms / 1000
            )

        # Record error if present
        if exc_val is not None:
            self._metrics.error = str(exc_val)

        # End the generation with final metrics
        if self._generation:
            try:
                level = "ERROR" if self._metrics.error else "DEFAULT"

                self._generation.end(
                    input=self._input_data,
                    output=self._metrics.metadata.get("output"),
                    usage_details={
                        "input": self._metrics.prompt_tokens,
                        "output": self._metrics.completion_tokens,
                        "total": self._metrics.total_tokens,
                    },
                    level=level,
                    status_message=self._metrics.error,
                    model_parameters=self._parameters,
                    metadata={
                        "latency_ms": self._metrics.total_latency_ms,
                        "time_to_first_token_ms": self._metrics.time_to_first_token_ms,
                        "tokens_per_second": self._metrics.tokens_per_second,
                        "cache_hit": self._metrics.cache_hit,
                        "cache_type": self._metrics.cache_type,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to end generation: {e}")

    def set_input(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """Set the input prompt or messages."""
        if prompt is not None:
            self._input_data = prompt
        elif messages is not None:
            self._input_data = messages

    def set_parameters(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        do_sample: bool | None = None,
        repetition_penalty: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Set generation parameters."""
        if temperature is not None:
            self._parameters["temperature"] = temperature
        if top_p is not None:
            self._parameters["top_p"] = top_p
        if top_k is not None:
            self._parameters["top_k"] = top_k
        if max_tokens is not None:
            self._parameters["max_tokens"] = max_tokens
        if do_sample is not None:
            self._parameters["do_sample"] = do_sample
        if repetition_penalty is not None:
            self._parameters["repetition_penalty"] = repetition_penalty
        self._parameters.update(kwargs)

    def set_output(
        self,
        text: str,
        usage: dict[str, int],
    ) -> None:
        """Set the generation output and usage."""
        self._metrics.prompt_tokens = usage.get("prompt_tokens", 0)
        self._metrics.completion_tokens = usage.get("completion_tokens", 0)
        self._metrics.total_tokens = usage.get("total_tokens", 0)
        self._metrics.metadata["output"] = text

    def record_first_token(self) -> None:
        """Record when the first token was generated (for streaming)."""
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()
            self._metrics.time_to_first_token_ms = (
                self._first_token_time - self._start_time
            ) * 1000

    def record_cache_hit(self, cache_type: str) -> None:
        """Record that this generation was served from cache."""
        self._metrics.cache_hit = True
        self._metrics.cache_type = cache_type

    def add_metadata(self, **kwargs: Any) -> None:
        """Add additional metadata to the trace."""
        self._metrics.metadata.update(kwargs)


def init_tracer(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    enabled: bool = True,
    debug: bool = False,
    flush_at: int = 15,
    flush_interval: float = 10.0,
) -> InferenceTracer:
    """Initialize the global tracer instance.

    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse host URL
        enabled: Whether tracing is enabled
        debug: Enable debug logging
        flush_at: Number of events before flushing (lower for quicker visibility)
        flush_interval: Seconds between flushes
    """
    global _tracer
    _tracer = InferenceTracer(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        enabled=enabled,
        debug=debug,
        flush_at=flush_at,
        flush_interval=flush_interval,
    )
    return _tracer


def get_tracer() -> InferenceTracer | None:
    """Get the global tracer instance."""
    return _tracer


@contextmanager
def trace_generation(
    name: str = "generation",
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[GenerationTrace | None]:
    """Convenience context manager for tracing generation.

    Returns None if tracing is not initialized.
    """
    tracer = get_tracer()
    if tracer is None or not tracer.enabled:
        yield None
        return

    with tracer.trace_generation(
        name=name,
        user_id=user_id,
        session_id=session_id,
        tags=tags,
        metadata=metadata,
    ) as trace:
        yield trace
