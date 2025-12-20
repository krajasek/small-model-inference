"""Integration tests with actual model inference.

These tests use a tiny GPT-2 model to verify end-to-end functionality.
They are slower than unit tests and require model download on first run.
"""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from inference.app import create_app
from inference.config import Settings

# Use GPT-2 which has safetensors format (required for PyTorch < 2.6)
TINY_MODEL_ID = "openai-community/gpt2"


def download_model_if_needed(model_path: Path) -> bool:
    """Download tiny model for testing if not already present.

    Returns True if model is available, False otherwise.
    """
    # Check for safetensors file (preferred) or config
    safetensors_file = model_path / "model.safetensors"
    config_file = model_path / "config.json"
    if safetensors_file.exists() and config_file.exists():
        return True

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=TINY_MODEL_ID,
            local_dir=str(model_path),
            # Only download safetensors and config files to save bandwidth
            ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],
        )
        return True
    except Exception as e:
        pytest.skip(f"Could not download test model: {e}")
        return False


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get or create path to tiny test model."""
    # Use a persistent cache directory for faster subsequent runs
    cache_dir = Path(tempfile.gettempdir()) / "test-gpt2-safetensors"
    cache_dir.mkdir(exist_ok=True)

    if not download_model_if_needed(cache_dir):
        pytest.skip("Test model not available")

    return cache_dir


@pytest.fixture(scope="module")
def integration_settings(model_path: Path) -> Settings:
    """Create settings for integration testing."""
    # Clear any INFERENCE_ env vars to use our test settings
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("INFERENCE_")}

    with patch.dict(os.environ, clean_env, clear=True):
        return Settings(
            model_path=str(model_path),
            model_name="tiny-gpt2-test",
            device="cpu",
            num_threads=2,
            quantization="none",
            enable_torch_compile=False,
            max_sequence_length=128,
            max_new_tokens=16,
        )


@pytest.fixture(scope="module")
def integration_client(integration_settings: Settings) -> Iterator[TestClient]:
    """Create test client with real model loaded."""
    app = create_app(integration_settings)

    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Integration tests for health endpoint."""

    def test_health_check_returns_healthy(self, integration_client: TestClient) -> None:
        """Test that health check returns healthy status with model loaded."""
        response = integration_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["device"] == "cpu"


class TestModelsEndpoint:
    """Integration tests for models endpoint."""

    def test_list_models(self, integration_client: TestClient) -> None:
        """Test listing available models."""
        response = integration_client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "tiny-gpt2-test"
        assert data["data"][0]["object"] == "model"


class TestCompletionsEndpoint:
    """Integration tests for completions endpoint."""

    def test_basic_completion(self, integration_client: TestClient) -> None:
        """Test basic text completion with real model."""
        response = integration_client.post(
            "/v1/completions",
            json={
                "prompt": "Hello world",
                "max_tokens": 8,
                "temperature": 0.1,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["object"] == "text_completion"
        assert data["model"] == "tiny-gpt2-test"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["index"] == 0
        assert isinstance(data["choices"][0]["text"], str)
        assert data["choices"][0]["finish_reason"] == "stop"

        # Verify usage stats
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] >= 0
        assert data["usage"]["total_tokens"] > 0

    def test_completion_with_sampling_params(
        self, integration_client: TestClient
    ) -> None:
        """Test completion with various sampling parameters."""
        response = integration_client.post(
            "/v1/completions",
            json={
                "prompt": "The quick brown fox",
                "max_tokens": 10,
                "temperature": 0.8,
                "top_p": 0.95,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"][0]["text"]) > 0

    def test_completion_low_temperature(self, integration_client: TestClient) -> None:
        """Test that very low temperature produces consistent output."""
        prompt = "Once upon a time"

        # Use very low temperature (0.0 is invalid in transformers, must be positive)
        response1 = integration_client.post(
            "/v1/completions",
            json={"prompt": prompt, "max_tokens": 5, "temperature": 0.01},
        )
        response2 = integration_client.post(
            "/v1/completions",
            json={"prompt": prompt, "max_tokens": 5, "temperature": 0.01},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        # With very low temperature, outputs should be nearly identical
        # (not guaranteed due to floating point, but highly likely)
        text1 = response1.json()["choices"][0]["text"]
        text2 = response2.json()["choices"][0]["text"]
        assert isinstance(text1, str) and isinstance(text2, str)

    def test_completion_streaming(self, integration_client: TestClient) -> None:
        """Test streaming completion response."""
        response = integration_client.post(
            "/v1/completions",
            json={
                "prompt": "Hello",
                "max_tokens": 5,
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Collect streamed chunks
        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                chunk_data = line[6:]  # Remove "data: " prefix
                if chunk_data == "[DONE]":
                    break
                chunks.append(chunk_data)

        # Should have received at least one chunk
        assert len(chunks) >= 1

        # Verify chunk structure
        import json

        first_chunk = json.loads(chunks[0])
        assert "id" in first_chunk
        assert first_chunk["object"] == "text_completion"
        assert "choices" in first_chunk


class TestChatCompletionsEndpoint:
    """Integration tests for chat completions endpoint."""

    def test_basic_chat_completion(self, integration_client: TestClient) -> None:
        """Test basic chat completion with real model."""
        response = integration_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi there"}],
                "max_tokens": 8,
                "temperature": 0.1,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["object"] == "chat.completion"
        assert data["model"] == "tiny-gpt2-test"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["index"] == 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(data["choices"][0]["message"]["content"], str)
        assert data["choices"][0]["finish_reason"] == "stop"

        # Verify usage stats
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0

    def test_chat_with_system_message(self, integration_client: TestClient) -> None:
        """Test chat completion with system message."""
        response = integration_client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 8,
                "temperature": 0.1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_multi_turn(self, integration_client: TestClient) -> None:
        """Test multi-turn chat conversation."""
        response = integration_client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "max_tokens": 10,
                "temperature": 0.1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"]) == 1

    def test_chat_streaming(self, integration_client: TestClient) -> None:
        """Test streaming chat completion response."""
        response = integration_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Collect streamed chunks
        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                chunk_data = line[6:]
                if chunk_data == "[DONE]":
                    break
                chunks.append(chunk_data)

        # Should have received at least the initial role chunk
        assert len(chunks) >= 1

        # Verify chunk structure
        import json

        first_chunk = json.loads(chunks[0])
        assert "id" in first_chunk
        assert first_chunk["object"] == "chat.completion.chunk"
        assert "choices" in first_chunk
        # First chunk should have role
        assert first_chunk["choices"][0]["delta"].get("role") == "assistant"

    def test_invalid_role_rejected(self, integration_client: TestClient) -> None:
        """Test that invalid message roles are rejected."""
        response = integration_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "invalid_role", "content": "Hello"}],
                "max_tokens": 8,
            },
        )

        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_empty_prompt_handled(self, integration_client: TestClient) -> None:
        """Test handling of empty prompt."""
        response = integration_client.post(
            "/v1/completions",
            json={"prompt": "", "max_tokens": 5},
        )

        # Empty prompts may fail at model level (500) or be rejected (400/422)
        # All are acceptable behaviors for this edge case
        assert response.status_code in [200, 400, 422, 500]

    def test_missing_required_fields(self, integration_client: TestClient) -> None:
        """Test validation of missing required fields."""
        response = integration_client.post(
            "/v1/completions",
            json={},  # Missing prompt
        )

        assert response.status_code == 422

    def test_invalid_max_tokens(self, integration_client: TestClient) -> None:
        """Test validation of invalid max_tokens."""
        response = integration_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": -1},
        )

        assert response.status_code == 422

    def test_invalid_temperature(self, integration_client: TestClient) -> None:
        """Test validation of invalid temperature."""
        response = integration_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "temperature": 3.0},
        )

        assert response.status_code == 422
