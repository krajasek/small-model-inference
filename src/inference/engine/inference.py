"""Core inference engine for text generation."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Thread
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, TextIteratorStreamer

from ..config import Settings

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
    """Core engine for running model inference."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str,
        settings: Settings,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.settings = settings
        self.model_name = settings.model_name or "local-model"

    def generate(self, prompt: str, config: GenerationConfig) -> tuple[str, dict[str, int]]:
        """Generate text completion synchronously.

        Returns:
            Tuple of (generated_text, usage_stats)
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_sequence_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p if config.do_sample else 1.0,
                top_k=config.top_k if config.do_sample else 0,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_ids = outputs[0]
        completion_tokens = len(generated_ids) - prompt_tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        return generated_text, usage

    def generate_stream(self, prompt: str, config: GenerationConfig) -> Iterator[tuple[str, bool]]:
        """Generate text completion with streaming.

        Yields:
            Tuples of (token_text, is_finished)
        """
        # Tokenize input
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
        config = self.model.config
        return {
            "model_name": self.model_name,
            "device": self.device,
            "quantization": self.settings.quantization,
            "max_sequence_length": self.settings.max_sequence_length,
            "vocab_size": getattr(config, "vocab_size", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_layers": getattr(config, "num_hidden_layers", None),
        }
