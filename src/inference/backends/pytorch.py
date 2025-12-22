"""PyTorch/HuggingFace backend for inference."""

import logging
from collections.abc import Iterator
from threading import Thread
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    TextIteratorStreamer,
)

from ..config import Settings
from ..engine.cache import CacheManager
from ..engine.cpu_optimizer import CPUOptimizer
from ..models.loader import ModelLoader
from ..observability.tracing import GenerationTrace, InferenceTracer
from .base import GenerationConfig

logger = logging.getLogger(__name__)


class PyTorchBackend:
    """PyTorch/HuggingFace backend for inference.

    This backend uses HuggingFace transformers for model loading and inference.
    Supports safetensors/bin model formats in HuggingFace directory layout.
    """

    def __init__(
        self,
        settings: Settings,
        tracer: InferenceTracer | None = None,
    ) -> None:
        """Initialize the PyTorch backend.

        Args:
            settings: Application settings
            tracer: Optional tracer for observability
        """
        self._settings = settings
        self._tracer = tracer

        # Load model and tokenizer
        loader = ModelLoader(settings)
        self._model, self._tokenizer = loader.load()
        self._device = loader.device
        self._model_name = settings.model_name or "local-model"

        # Apply CPU optimizations if on CPU
        cpu_optimizer: CPUOptimizer | None = None
        if self._device == "cpu":
            cpu_optimizer = CPUOptimizer(settings)
            cpu_optimizer.setup_environment()
            self._model = cpu_optimizer.optimize_model(self._model)

        # Load draft model for speculative decoding if configured
        self._draft_model: PreTrainedModel | None = None
        if settings.enable_speculative_decoding and settings.draft_model_path:
            self._draft_model = self._load_draft_model(settings.draft_model_path)
            # Apply CPU optimizations to draft model too
            if self._draft_model and cpu_optimizer:
                self._draft_model = cpu_optimizer.optimize_model(self._draft_model)

        # Initialize cache manager
        self._cache_manager = CacheManager(
            response_cache_size=settings.response_cache_size,
            response_cache_ttl=settings.response_cache_ttl,
            prompt_cache_size=settings.prompt_cache_size,
            prompt_cache_ttl=settings.prompt_cache_ttl,
            tokenizer_cache_size=settings.tokenizer_cache_size,
            enable_response_cache=settings.enable_response_cache,
            enable_prompt_cache=settings.enable_prompt_cache,
            enable_tokenizer_cache=settings.enable_tokenizer_cache,
        )

        # Configure static KV cache if enabled
        if settings.static_kv_cache:
            self._setup_static_cache()

        # Set model info on tracer
        if self._tracer:
            self._tracer.set_model_info(self._model_name, self.get_model_info())

        logger.info(
            f"PyTorchBackend initialized: device={self._device}, "
            f"caching: response={settings.enable_response_cache}, "
            f"prompt={settings.enable_prompt_cache}, "
            f"tokenizer={settings.enable_tokenizer_cache}, "
            f"tracing={tracer is not None and tracer.enabled}"
        )

    def _load_draft_model(self, draft_model_path: str) -> PreTrainedModel | None:
        """Load a draft model for speculative decoding."""
        try:
            logger.info(f"Loading draft model from {draft_model_path}")
            draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_path,
                torch_dtype=torch.float32 if self._device == "cpu" else torch.float16,
                low_cpu_mem_usage=self._settings.low_cpu_mem_usage,
            )
            draft_model = draft_model.to(self._device)
            draft_model.eval()
            logger.info("Draft model loaded successfully")
            return draft_model
        except Exception as e:
            logger.warning(f"Failed to load draft model: {e}")
            return None

    def _setup_static_cache(self) -> None:
        """Configure static KV cache for faster generation."""
        try:
            if hasattr(self._model, "generation_config"):
                self._model.generation_config.cache_implementation = "static"
                logger.info("Static KV cache enabled")
        except Exception as e:
            logger.warning(f"Failed to enable static KV cache: {e}")

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
        """Generate text completion synchronously with caching.

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
            if self._cache_manager.response_cache:
                cached = self._cache_manager.response_cache.get(
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

            # Tokenize with caching
            if self._cache_manager.tokenizer_cache:
                inputs = self._cache_manager.tokenizer_cache.tokenize(
                    prompt,
                    self._tokenizer,
                    max_length=self._settings.max_sequence_length,
                    truncation=True,
                )
            else:
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self._settings.max_sequence_length,
                )

            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            prompt_tokens = inputs["input_ids"].shape[1]

            # Build generation kwargs
            generation_kwargs = self._build_generation_kwargs(inputs, config)

            # Generate with appropriate method
            if self._settings.enable_speculative_decoding and self._draft_model:
                outputs = self._generate_with_speculation(inputs, generation_kwargs)
            else:
                outputs = self._generate_standard(generation_kwargs)

            # Decode output
            generated_ids = outputs[0]
            completion_tokens = len(generated_ids) - prompt_tokens
            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            # Cache the response
            if self._cache_manager.response_cache:
                self._cache_manager.response_cache.put(
                    prompt=prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    response=generated_text,
                    usage=usage,
                )

            if trace:
                trace.set_output(text=generated_text, usage=usage)
                trace.__exit__(None, None, None)

            return generated_text, usage

        except Exception as e:
            if trace:
                trace.__exit__(type(e), e, e.__traceback__)
            raise

    def _build_generation_kwargs(
        self,
        inputs: dict[str, torch.Tensor],
        config: GenerationConfig,
    ) -> dict[str, Any]:
        """Build kwargs for model.generate()."""
        kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature if config.do_sample else 1.0,
            "top_p": config.top_p if config.do_sample else 1.0,
            "top_k": config.top_k if config.do_sample else 0,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "use_cache": self._settings.use_kv_cache,
        }
        return kwargs

    def _generate_standard(self, generation_kwargs: dict[str, Any]) -> torch.Tensor:
        """Standard generation without speculation."""
        with torch.inference_mode():
            return self._model.generate(**generation_kwargs)

    def _generate_with_speculation(
        self,
        inputs: dict[str, torch.Tensor],
        generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Generation with speculative decoding using draft model."""
        logger.debug("Using speculative decoding")

        gen_kwargs = {
            k: v
            for k, v in generation_kwargs.items()
            if k not in ("input_ids", "attention_mask")
        }

        with torch.inference_mode():
            return self._model.generate(
                **inputs,
                **gen_kwargs,
                assistant_model=self._draft_model,
            )

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
            # Tokenize with caching
            if self._cache_manager.tokenizer_cache:
                inputs = self._cache_manager.tokenizer_cache.tokenize(
                    prompt,
                    self._tokenizer,
                    max_length=self._settings.max_sequence_length,
                    truncation=True,
                )
            else:
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self._settings.max_sequence_length,
                )

            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            prompt_tokens = inputs["input_ids"].shape[1]

            # Create streamer for token-by-token streaming
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # Generation kwargs
            generation_kwargs: dict[str, Any] = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature if config.do_sample else 1.0,
                "top_p": config.top_p if config.do_sample else 1.0,
                "top_k": config.top_k if config.do_sample else 0,
                "do_sample": config.do_sample,
                "repetition_penalty": config.repetition_penalty,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "use_cache": self._settings.use_kv_cache,
            }

            # Run generation in a separate thread
            thread = Thread(target=self._generate_in_thread, args=(generation_kwargs,))
            thread.start()

            # Track generated tokens for metrics
            generated_text_parts: list[str] = []
            first_token = True

            # Yield tokens as they're generated
            for text in streamer:
                if first_token and trace:
                    trace.record_first_token()
                    first_token = False
                generated_text_parts.append(text)
                yield text, False

            thread.join()

            # Record final metrics
            if trace:
                full_text = "".join(generated_text_parts)
                completion_tokens = len(self._tokenizer.encode(full_text))
                trace.set_output(
                    text=full_text,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                )
                trace.__exit__(None, None, None)

            yield "", True

        except Exception as e:
            if trace:
                trace.__exit__(type(e), e, e.__traceback__)
            raise

    def _generate_in_thread(self, generation_kwargs: dict[str, Any]) -> None:
        """Run generation in a separate thread for streaming."""
        with torch.inference_mode():
            self._model.generate(**generation_kwargs)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        model_config = self._model.config
        return {
            "model_name": self._model_name,
            "backend": "pytorch",
            "device": self._device,
            "quantization": self._settings.quantization,
            "max_sequence_length": self._settings.max_sequence_length,
            "vocab_size": getattr(model_config, "vocab_size", None),
            "hidden_size": getattr(model_config, "hidden_size", None),
            "num_layers": getattr(model_config, "num_hidden_layers", None),
            "kv_cache_enabled": self._settings.use_kv_cache,
            "static_kv_cache": self._settings.static_kv_cache,
            "speculative_decoding": (
                self._settings.enable_speculative_decoding and self._draft_model is not None
            ),
        }

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache_manager.stats()

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._cache_manager.clear_all()
        logger.info("All caches cleared")

    def shutdown(self) -> None:
        """Shutdown the backend and release resources."""
        logger.info("Shutting down PyTorch backend")
        self.clear_caches()
        # Clear model references to help with memory cleanup
        del self._model
        if self._draft_model:
            del self._draft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
