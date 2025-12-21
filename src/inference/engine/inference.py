"""Core inference engine for text generation with optimizations."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Thread
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)

from ..config import Settings
from .cache import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


class InferenceEngine:
    """Core engine for running model inference with optimizations."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str,
        settings: Settings,
        draft_model: PreTrainedModel | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.settings = settings
        self.model_name = settings.model_name or "local-model"
        self.draft_model = draft_model

        # Initialize cache manager
        self.cache_manager = CacheManager(
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

        logger.info(
            f"InferenceEngine initialized with caching: "
            f"response={settings.enable_response_cache}, "
            f"prompt={settings.enable_prompt_cache}, "
            f"tokenizer={settings.enable_tokenizer_cache}"
        )

    def _setup_static_cache(self) -> None:
        """Configure static KV cache for faster generation."""
        try:
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.cache_implementation = "static"
                logger.info("Static KV cache enabled")
        except Exception as e:
            logger.warning(f"Failed to enable static KV cache: {e}")

    def generate(
        self, prompt: str, config: GenerationConfig
    ) -> tuple[str, dict[str, int]]:
        """Generate text completion synchronously with caching.

        Returns:
            Tuple of (generated_text, usage_stats)
        """
        # Check response cache first
        if self.cache_manager.response_cache:
            cached = self.cache_manager.response_cache.get(
                prompt=prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )
            if cached:
                logger.debug("Returning cached response")
                return cached

        # Tokenize with caching
        if self.cache_manager.tokenizer_cache:
            inputs = self.cache_manager.tokenizer_cache.tokenize(
                prompt,
                self.tokenizer,
                max_length=self.settings.max_sequence_length,
                truncation=True,
            )
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings.max_sequence_length,
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        # Build generation kwargs
        generation_kwargs = self._build_generation_kwargs(inputs, config)

        # Generate with appropriate method
        if self.settings.enable_speculative_decoding and self.draft_model:
            outputs = self._generate_with_speculation(inputs, generation_kwargs)
        else:
            outputs = self._generate_standard(generation_kwargs)

        # Decode output
        generated_ids = outputs[0]
        completion_tokens = len(generated_ids) - prompt_tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        # Cache the response
        if self.cache_manager.response_cache:
            self.cache_manager.response_cache.put(
                prompt=prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                response=generated_text,
                usage=usage,
            )

        return generated_text, usage

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
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.settings.use_kv_cache,
        }
        return kwargs

    def _generate_standard(
        self, generation_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Standard generation without speculation."""
        with torch.inference_mode():
            return self.model.generate(**generation_kwargs)

    def _generate_with_speculation(
        self,
        inputs: dict[str, torch.Tensor],
        generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Generation with speculative decoding using draft model."""
        logger.debug("Using speculative decoding")

        # Remove inputs from kwargs since we pass assistant_model separately
        gen_kwargs = {
            k: v for k, v in generation_kwargs.items()
            if k not in ("input_ids", "attention_mask")
        }

        with torch.inference_mode():
            return self.model.generate(
                **inputs,
                **gen_kwargs,
                assistant_model=self.draft_model,
            )

    def generate_with_prefix_cache(
        self,
        prefix: str,
        continuation: str,
        config: GenerationConfig,
    ) -> tuple[str, dict[str, int]]:
        """Generate with cached prefix KV states.

        Useful for chat with system prompts or few-shot examples.
        """
        if not self.cache_manager.prompt_cache:
            # Fallback to regular generation
            return self.generate(prefix + continuation, config)

        # Get or compute KV cache for prefix
        prefix_ids, prefix_mask, past_key_values = (
            self.cache_manager.prompt_cache.get_or_compute_kv(
                prefix,
                self.model,
                self.tokenizer,
                self.device,
            )
        )

        # Tokenize continuation
        continuation_inputs = self.tokenizer(
            continuation,
            return_tensors="pt",
            truncation=True,
        )
        continuation_ids = continuation_inputs["input_ids"].to(self.device)

        # Combine for generation
        input_ids = torch.cat([prefix_ids, continuation_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        prompt_tokens = input_ids.shape[1]

        # Generate with cached KV states
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p if config.do_sample else 1.0,
                top_k=config.top_k if config.do_sample else 0,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_ids = outputs[0]
        completion_tokens = len(generated_ids) - prompt_tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        return generated_text, usage

    def generate_stream(
        self, prompt: str, config: GenerationConfig
    ) -> Iterator[tuple[str, bool]]:
        """Generate text completion with streaming.

        Yields:
            Tuples of (token_text, is_finished)
        """
        # Tokenize with caching
        if self.cache_manager.tokenizer_cache:
            inputs = self.cache_manager.tokenizer_cache.tokenize(
                prompt,
                self.tokenizer,
                max_length=self.settings.max_sequence_length,
                truncation=True,
            )
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings.max_sequence_length,
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
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
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.settings.use_kv_cache,
        }

        # Run generation in a separate thread
        thread = Thread(target=self._generate_in_thread, args=(generation_kwargs,))
        thread.start()

        # Yield tokens as they're generated
        for text in streamer:
            yield text, False

        thread.join()
        yield "", True

    def _generate_in_thread(self, generation_kwargs: dict[str, Any]) -> None:
        """Run generation in a separate thread for streaming."""
        with torch.inference_mode():
            self.model.generate(**generation_kwargs)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        model_config = self.model.config
        return {
            "model_name": self.model_name,
            "device": self.device,
            "quantization": self.settings.quantization,
            "max_sequence_length": self.settings.max_sequence_length,
            "vocab_size": getattr(model_config, "vocab_size", None),
            "hidden_size": getattr(model_config, "hidden_size", None),
            "num_layers": getattr(model_config, "num_hidden_layers", None),
            "kv_cache_enabled": self.settings.use_kv_cache,
            "static_kv_cache": self.settings.static_kv_cache,
            "speculative_decoding": (
                self.settings.enable_speculative_decoding and self.draft_model is not None
            ),
        }

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache_manager.stats()

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.cache_manager.clear_all()
        logger.info("All caches cleared")


def load_draft_model(
    draft_model_path: str,
    device: str,
    settings: Settings,
) -> PreTrainedModel | None:
    """Load a draft model for speculative decoding."""
    if not draft_model_path:
        return None

    try:
        logger.info(f"Loading draft model from {draft_model_path}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            low_cpu_mem_usage=settings.low_cpu_mem_usage,
        )
        draft_model = draft_model.to(device)
        draft_model.eval()
        logger.info("Draft model loaded successfully")
        return draft_model
    except Exception as e:
        logger.warning(f"Failed to load draft model: {e}")
        return None
