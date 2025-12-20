"""Tests for the inference engine module."""

from unittest.mock import MagicMock, patch

import torch

from inference.config import Settings
from inference.engine.inference import GenerationConfig, InferenceEngine


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = GenerationConfig()

        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.do_sample is True
        assert config.repetition_penalty == 1.1

    def test_custom_values(self) -> None:
        """Test that custom values can be set."""
        config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.5,
            top_p=0.95,
            top_k=100,
            do_sample=False,
            repetition_penalty=1.2,
        )

        assert config.max_new_tokens == 128
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 100
        assert config.do_sample is False
        assert config.repetition_penalty == 1.2


class TestInferenceEngine:
    """Test InferenceEngine class."""

    def test_init(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_settings: Settings,
    ) -> None:
        """Test InferenceEngine initialization."""
        engine = InferenceEngine(mock_model, mock_tokenizer, "cpu", mock_settings)

        assert engine.model == mock_model
        assert engine.tokenizer == mock_tokenizer
        assert engine.device == "cpu"
        assert engine.settings == mock_settings
        assert engine.model_name == "local-model"  # Default when model_name is None

    def test_init_with_custom_model_name(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_settings: Settings,
    ) -> None:
        """Test InferenceEngine with custom model name."""
        mock_settings.model_name = "my-custom-model"
        engine = InferenceEngine(mock_model, mock_tokenizer, "cpu", mock_settings)

        assert engine.model_name == "my-custom-model"

    def test_generate(
        self,
        mock_engine: InferenceEngine,
        generation_config: GenerationConfig,
    ) -> None:
        """Test synchronous generation."""
        prompt = "Test prompt"

        generated_text, usage = mock_engine.generate(prompt, generation_config)

        # Should return decoded text
        assert generated_text == "Test generated text response"
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_generate_calls_model(
        self,
        mock_engine: InferenceEngine,
        generation_config: GenerationConfig,
    ) -> None:
        """Test that generate calls the model correctly."""
        mock_engine.generate("Test", generation_config)

        mock_engine.model.generate.assert_called_once()
        call_kwargs = mock_engine.model.generate.call_args.kwargs

        assert "max_new_tokens" in call_kwargs
        assert call_kwargs["max_new_tokens"] == generation_config.max_new_tokens
        assert call_kwargs["do_sample"] == generation_config.do_sample

    def test_generate_with_sampling_disabled(
        self,
        mock_engine: InferenceEngine,
    ) -> None:
        """Test generation with sampling disabled (greedy)."""
        config = GenerationConfig(do_sample=False)

        mock_engine.generate("Test", config)

        call_kwargs = mock_engine.model.generate.call_args.kwargs
        assert call_kwargs["do_sample"] is False
        # When do_sample is False, temperature should be 1.0
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["top_p"] == 1.0
        assert call_kwargs["top_k"] == 0

    def test_generate_stream(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_settings: Settings,
        generation_config: GenerationConfig,
    ) -> None:
        """Test streaming generation."""
        # Create a mock streamer that yields tokens
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = lambda self: iter(["Hello", " ", "World"])

        streamer_patch = "inference.engine.inference.TextIteratorStreamer"
        thread_patch = "inference.engine.inference.Thread"
        with patch(streamer_patch, return_value=mock_streamer), patch(thread_patch) as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            engine = InferenceEngine(mock_model, mock_tokenizer, "cpu", mock_settings)
            tokens = list(engine.generate_stream("Test", generation_config))

        # Should yield tokens plus final (empty, True) tuple
        assert len(tokens) > 0
        # Last token should be the finished signal
        assert tokens[-1] == ("", True)

    def test_generate_stream_starts_thread(
        self,
        mock_engine: InferenceEngine,
        generation_config: GenerationConfig,
    ) -> None:
        """Test that streaming generation starts a thread."""
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = lambda self: iter([])

        streamer_patch = "inference.engine.inference.TextIteratorStreamer"
        thread_patch = "inference.engine.inference.Thread"
        with patch(streamer_patch, return_value=mock_streamer), patch(thread_patch) as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            # Consume the generator
            list(mock_engine.generate_stream("Test", generation_config))

            mock_thread_instance.start.assert_called_once()
            mock_thread_instance.join.assert_called_once()

    def test_get_model_info(
        self,
        mock_engine: InferenceEngine,
    ) -> None:
        """Test getting model information."""
        info = mock_engine.get_model_info()

        assert info["model_name"] == "local-model"
        assert info["device"] == "cpu"
        assert info["quantization"] == "none"
        assert info["max_sequence_length"] == 512
        assert "vocab_size" in info
        assert "hidden_size" in info
        assert "num_layers" in info

    def test_generate_in_thread(
        self,
        mock_engine: InferenceEngine,
    ) -> None:
        """Test the _generate_in_thread helper method."""
        generation_kwargs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "max_new_tokens": 10,
        }

        mock_engine._generate_in_thread(generation_kwargs)

        mock_engine.model.generate.assert_called_once_with(**generation_kwargs)
