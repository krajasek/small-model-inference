"""llama-cpp-python backend for fast CPU inference."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ..config import Settings
from ..engine.cache import ResponseCache
from ..observability.tracing import GenerationTrace, InferenceTracer
from .base import GenerationConfig

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaCppDraftModelWrapper:
    """Wrapper to use a Llama model as a draft model for speculative decoding.

    Implements the LlamaDraftModel interface expected by llama-cpp-python.
    """

    def __init__(self, draft_llama: Llama, num_pred_tokens: int = 4) -> None:
        """Initialize the draft model wrapper.

        Args:
            draft_llama: The Llama instance to use as draft model
            num_pred_tokens: Number of tokens to predict per speculation step
        """
        self._draft = draft_llama
        self._num_pred_tokens = num_pred_tokens

    def __call__(
        self,
        input_ids: npt.NDArray[np.intc],
        /,
        **kwargs: Any,
    ) -> npt.NDArray[np.intc]:
        """Generate draft tokens from the input.

        Args:
            input_ids: Input token IDs

        Returns:
            Predicted token IDs
        """
        # Reset the draft model's KV cache and eval the input
        self._draft.reset()
        self._draft.eval(input_ids.tolist())

        # Generate draft tokens
        draft_tokens = []
        for _ in range(self._num_pred_tokens):
            # Sample next token
            token = self._draft.sample(
                temp=0.0,  # Greedy for draft
            )
            if token == self._draft.token_eos():
                break
            draft_tokens.append(token)
            self._draft.eval([token])

        return np.array(draft_tokens, dtype=np.intc)


@dataclass
class StateEntry:
    """Entry in the llama-cpp state cache."""

    prompt_hash: str
    state_data: bytes
    num_tokens: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class LlamaCppStateCache:
    """Persistent state cache for llama-cpp models.

    Caches model state (including KV cache) after processing prompts,
    enabling faster continuation for repeated prefixes.
    """

    def __init__(
        self,
        cache_dir: Path,
        max_memory_entries: int = 10,
        max_disk_size_gb: float = 5.0,
        ttl_days: int = 7,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        self.max_disk_size_bytes = int(max_disk_size_gb * 1024 * 1024 * 1024)
        self.ttl_seconds = ttl_days * 24 * 60 * 60

        self._memory_cache: OrderedDict[str, StateEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._disk_size_bytes = 0
        self._hits = 0
        self._misses = 0

        # Initialize cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()

    def _make_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]

    def _get_path(self, key: str) -> Path:
        """Get file path for cache entry."""
        return self.cache_dir / f"{key}.state"

    def _load_metadata(self) -> None:
        """Load disk cache metadata."""
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                    self._disk_size_bytes = data.get("total_size", 0)
            except Exception as e:
                logger.warning(f"Failed to load state cache metadata: {e}")

    def _save_metadata(self) -> None:
        """Save disk cache metadata."""
        metadata_file = self.cache_dir / "metadata.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump({"total_size": self._disk_size_bytes}, f)
        except Exception as e:
            logger.warning(f"Failed to save state cache metadata: {e}")

    def get(self, prompt: str) -> StateEntry | None:
        """Get cached state for prompt."""
        key = self._make_key(prompt)

        with self._lock:
            # Check memory first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                entry.last_accessed = time.time()
                self._memory_cache.move_to_end(key)
                self._hits += 1
                return entry

            # Check disk
            path = self._get_path(key)
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        state_data = f.read()

                    # Check TTL based on file modification time
                    mtime = path.stat().st_mtime
                    if time.time() - mtime > self.ttl_seconds:
                        path.unlink(missing_ok=True)
                        self._misses += 1
                        return None

                    entry = StateEntry(
                        prompt_hash=key,
                        state_data=state_data,
                        num_tokens=0,  # Unknown from disk
                        created_at=mtime,
                        last_accessed=time.time(),
                    )

                    # Promote to memory if space available
                    if len(self._memory_cache) < self.max_memory_entries:
                        self._memory_cache[key] = entry

                    self._hits += 1
                    return entry
                except Exception as e:
                    logger.warning(f"Failed to load state from disk: {e}")

            self._misses += 1
            return None

    def put(self, prompt: str, state_data: bytes, num_tokens: int = 0) -> None:
        """Store state for prompt."""
        key = self._make_key(prompt)
        size = len(state_data)

        with self._lock:
            # Evict from memory if at capacity
            while len(self._memory_cache) >= self.max_memory_entries:
                evicted_key, _ = self._memory_cache.popitem(last=False)
                logger.debug(f"Evicted state from memory: {evicted_key[:8]}...")

            entry = StateEntry(
                prompt_hash=key,
                state_data=state_data,
                num_tokens=num_tokens,
            )
            self._memory_cache[key] = entry

        # Write to disk asynchronously (simple sync for now)
        try:
            path = self._get_path(key)
            with open(path, "wb") as f:
                f.write(state_data)
            self._disk_size_bytes += size
            self._save_metadata()
            logger.debug(f"Saved state to disk: {key[:8]}... ({size / 1024:.1f} KB)")
        except Exception as e:
            logger.warning(f"Failed to save state to disk: {e}")

    def clear(self) -> None:
        """Clear all cached states."""
        with self._lock:
            self._memory_cache.clear()
            self._hits = 0
            self._misses = 0

        # Clear disk cache
        try:
            for path in self.cache_dir.glob("*.state"):
                path.unlink(missing_ok=True)
            self._disk_size_bytes = 0
            self._save_metadata()
        except Exception as e:
            logger.warning(f"Failed to clear disk state cache: {e}")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "memory_entries": len(self._memory_cache),
                "max_memory_entries": self.max_memory_entries,
                "disk_size_mb": self._disk_size_bytes / 1024 / 1024,
                "max_disk_size_gb": self.max_disk_size_bytes / 1024 / 1024 / 1024,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


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
                "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
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
                logger.warning(f"Multiple .gguf files found, using: {gguf_files[0].name}")
        else:
            raise FileNotFoundError(
                f"Invalid model path: {self._model_path}. "
                "Expected .gguf file or directory containing .gguf files."
            )

        logger.info(f"Loading llama-cpp model from {model_file}")

        # Load draft model for speculative decoding if configured
        # Must be done before main model initialization
        self._draft_model_wrapper: LlamaCppDraftModelWrapper | None = None
        self._num_speculative_tokens = settings.num_speculative_tokens
        self._speculative_decoding_enabled = False

        if settings.enable_speculative_decoding and settings.draft_model_path:
            draft_llama = self._load_draft_model(settings.draft_model_path, Llama)
            if draft_llama is not None:
                self._draft_model_wrapper = LlamaCppDraftModelWrapper(
                    draft_llama=draft_llama,
                    num_pred_tokens=settings.num_speculative_tokens,
                )
                self._speculative_decoding_enabled = True
                logger.info(
                    f"Speculative decoding enabled with {settings.num_speculative_tokens} "
                    "draft tokens per step"
                )

        # Initialize llama-cpp model with optional draft model
        self._model = Llama(
            model_path=model_file,
            n_ctx=settings.llama_cpp_n_ctx,
            n_threads=settings.num_threads,
            n_gpu_layers=settings.llama_cpp_n_gpu_layers,
            n_batch=settings.llama_cpp_n_batch,
            verbose=False,
            draft_model=self._draft_model_wrapper,
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

        # Initialize persistent state cache if enabled
        self._state_cache: LlamaCppStateCache | None = None
        self._supports_state_persistence = self._check_state_support()
        if settings.enable_persistent_cache and self._supports_state_persistence:
            cache_dir = Path(settings.persistent_cache_dir) / "llama-cpp-state"
            self._state_cache = LlamaCppStateCache(
                cache_dir=cache_dir,
                max_memory_entries=min(10, settings.persistent_cache_memory_size // 10),
                max_disk_size_gb=settings.persistent_cache_disk_size_gb / 2,  # Share with PyTorch
                ttl_days=settings.persistent_cache_ttl_days,
            )

        # Set model info on tracer
        if self._tracer:
            self._tracer.set_model_info(self._model_name, self.get_model_info())

        logger.info(
            f"LlamaCppBackend initialized: model={self._model_name}, "
            f"n_ctx={settings.llama_cpp_n_ctx}, "
            f"n_threads={settings.num_threads}, "
            f"n_gpu_layers={settings.llama_cpp_n_gpu_layers}, "
            f"response_cache={settings.enable_response_cache}, "
            f"speculative_decoding={self._speculative_decoding_enabled}, "
            f"state_cache={self._state_cache is not None}"
        )

    @property
    def model_name(self) -> str:
        """Get the name of the loaded model."""
        return self._model_name

    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        return self._device

    def _check_state_support(self) -> bool:
        """Check if the model supports state save/load operations."""
        return hasattr(self._model, "save_state") and hasattr(self._model, "load_state")

    def _save_state_for_prompt(self, prompt: str) -> None:
        """Save model state after processing a prompt."""
        if not self._state_cache or not self._supports_state_persistence:
            return

        try:
            state = self._model.save_state()
            if state:
                # Get token count from tokenization
                tokens = self._model.tokenize(prompt.encode())
                self._state_cache.put(prompt, state, num_tokens=len(tokens))
        except Exception as e:
            logger.warning(f"Failed to save model state: {e}")

    def _restore_state_for_prompt(self, prompt: str) -> bool:
        """Restore model state if prompt was previously processed.

        Returns True if state was restored, False otherwise.
        """
        if not self._state_cache or not self._supports_state_persistence:
            return False

        try:
            entry = self._state_cache.get(prompt)
            if entry:
                self._model.load_state(entry.state_data)
                logger.debug(f"Restored state for prompt (hash={entry.prompt_hash[:8]}...)")
                return True
        except Exception as e:
            logger.warning(f"Failed to restore model state: {e}")

        return False

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

    def _load_draft_model(self, draft_model_path: str, llama_cls: type[Llama]) -> Llama | None:
        """Load a draft model for speculative decoding.

        Args:
            draft_model_path: Path to the draft model GGUF file
            llama_cls: The Llama class to use for loading

        Returns:
            Loaded draft model or None if loading fails
        """
        try:
            draft_path = Path(draft_model_path)

            if not draft_path.exists():
                logger.warning(f"Draft model path does not exist: {draft_path}")
                return None

            # Determine the GGUF file path
            if draft_path.is_file() and draft_path.suffix == ".gguf":
                draft_file = str(draft_path)
            elif draft_path.is_dir():
                gguf_files = list(draft_path.glob("*.gguf"))
                if not gguf_files:
                    logger.warning(f"No .gguf files found in {draft_path}")
                    return None
                draft_file = str(gguf_files[0])
                if len(gguf_files) > 1:
                    logger.warning(f"Multiple .gguf files found, using: {gguf_files[0].name}")
            else:
                logger.warning(f"Invalid draft model path: {draft_path}")
                return None

            logger.info(f"Loading draft model from {draft_file}")

            # Load draft model with same settings as main model
            draft_model = llama_cls(
                model_path=draft_file,
                n_ctx=self._settings.llama_cpp_n_ctx,
                n_threads=self._settings.num_threads,
                n_gpu_layers=self._settings.llama_cpp_n_gpu_layers,
                n_batch=self._settings.llama_cpp_n_batch,
                verbose=False,
            )

            logger.info("Draft model loaded successfully for speculative decoding")
            return draft_model

        except Exception as e:
            logger.warning(f"Failed to load draft model: {e}")
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

            # Build generation kwargs
            gen_kwargs: dict[str, Any] = {
                "max_tokens": config.max_new_tokens,
                "temperature": config.temperature if config.do_sample else 0.0,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repetition_penalty,
                "echo": False,  # Don't include prompt in output
            }

            # Generate with llama-cpp
            # Note: Draft model for speculative decoding is set at initialization
            output = self._model(prompt, **gen_kwargs)

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

            # Build generation kwargs
            gen_kwargs: dict[str, Any] = {
                "max_tokens": config.max_new_tokens,
                "temperature": config.temperature if config.do_sample else 0.0,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repetition_penalty,
                "echo": False,
                "stream": True,
            }

            # Stream with llama-cpp
            # Note: Draft model for speculative decoding is set at initialization
            for output in self._model(prompt, **gen_kwargs):
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
            "speculative_decoding": self._speculative_decoding_enabled,
            "state_cache_enabled": self._state_cache is not None,
            "state_persistence_supported": self._supports_state_persistence,
        }

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats: dict[str, Any] = {
            "backend": "llama-cpp",
            "note": "llama-cpp manages KV cache internally",
        }

        if self._response_cache:
            stats["response_cache"] = self._response_cache.stats()

        if self._state_cache:
            stats["state_cache"] = self._state_cache.stats()

        return stats

    def clear_caches(self) -> None:
        """Clear response and state caches."""
        cleared = False

        if self._response_cache:
            self._response_cache.clear()
            logger.info("Response cache cleared")
            cleared = True

        if self._state_cache:
            self._state_cache.clear()
            logger.info("State cache cleared")
            cleared = True

        if not cleared:
            logger.debug("No caches to clear (llama-cpp manages KV cache internally)")

    def shutdown(self) -> None:
        """Shutdown the backend and release resources."""
        logger.info("Shutting down llama-cpp backend")

        # Clean up state cache
        if self._state_cache:
            logger.debug("State cache will persist to disk for next startup")

        # llama-cpp-python handles cleanup via __del__
        if self._draft_model_wrapper:
            del self._draft_model_wrapper
        del self._model
