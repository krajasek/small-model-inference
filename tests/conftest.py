"""Pytest fixtures for the inference test suite."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from inference.api.schemas import ChatMessage
from inference.config import Settings
from inference.engine.inference import GenerationConfig, InferenceEngine


@pytest.fixture
def mock_settings() -> Settings:
    """Create test settings."""
    return Settings(
        model_path="/tmp/test-model",
        device="cpu",
        num_threads=2,
        quantization="none",
        enable_torch_compile=False,
        max_sequence_length=512,
        max_new_tokens=64,
    )


@pytest.fixture
def mock_model_config() -> MagicMock:
    """Create a mock model config."""
    config = MagicMock()
    config.hidden_size = 256
    config.num_hidden_layers = 4
    config.vocab_size = 1000
    config.intermediate_size = 512
    return config


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"

    # Mock tokenization
    def mock_call(text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        # Return simple mock tensors
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer.side_effect = mock_call
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones((1, 5)),
    }

    # Mock decode
    tokenizer.decode.return_value = "Test generated text response"

    # Mock apply_chat_template
    tokenizer.apply_chat_template.return_value = "<|user|>\nHello\n</s>\n<|assistant|>"

    return tokenizer


@pytest.fixture
def mock_model(mock_model_config: MagicMock) -> MagicMock:
    """Create a mock language model."""
    model = MagicMock()
    model.config = mock_model_config
    model.device = torch.device("cpu")

    # Mock generate method
    def mock_generate(**kwargs: Any) -> torch.Tensor:
        # Return tensor with input + generated tokens
        input_ids = kwargs.get("input_ids", torch.tensor([[1, 2, 3]]))
        # Simulate generating 10 new tokens
        new_tokens = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        return torch.cat([input_ids, new_tokens[:, :10]], dim=1)

    model.generate.side_effect = mock_generate
    model.eval.return_value = model
    model.to.return_value = model

    return model


@pytest.fixture
def mock_engine(
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
    mock_settings: Settings,
) -> InferenceEngine:
    """Create a mock inference engine."""
    return InferenceEngine(
        model=mock_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        settings=mock_settings,
    )


@pytest.fixture
def generation_config() -> GenerationConfig:
    """Create a default generation config."""
    return GenerationConfig(
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.1,
    )


@pytest.fixture
def sample_chat_messages() -> list[ChatMessage]:
    """Create sample chat messages for testing."""
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello, how are you?"),
    ]


@pytest.fixture
def mock_app_with_engine(mock_engine: InferenceEngine) -> Iterator[TestClient]:
    """Create a test client with a mock engine."""
    from inference.app import create_app

    # Create app without lifespan to avoid loading real model
    with patch("inference.app.lifespan"):
        app = create_app(Settings(model_path="/tmp/test"))
        app.state.engine = mock_engine

        with TestClient(app) as client:
            yield client


@pytest.fixture
def temp_model_dir(tmp_path: Any) -> Any:
    """Create a temporary directory simulating a model folder."""
    import json

    model_dir = tmp_path / "test-model"
    model_dir.mkdir()

    # Create config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "vocab_size": 1000,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "model_type": "llama",
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    return model_dir
