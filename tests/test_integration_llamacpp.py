"""Integration tests for llama-cpp backend with actual GGUF model inference.

These tests use a TinyLlama GGUF model to verify llama-cpp backend functionality.
They are slower than unit tests and require:
1. llama-cpp-python to be installed (uv sync --extra llama-cpp)
2. GGUF model download on first run (~600MB)

Run with: uv run pytest tests/test_integration_llamacpp.py -v
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

# Skip all tests if llama-cpp-python is not installed
try:
    import llama_cpp  # noqa: F401

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LLAMA_CPP_AVAILABLE,
    reason="llama-cpp-python not installed. Install with: uv sync --extra llama-cpp",
)

# Use TinyLlama GGUF - small but functional model
GGUF_MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
GGUF_MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def download_gguf_model_if_needed(model_path: Path) -> bool:
    """Download GGUF model for testing if not already present.

    Returns True if model is available, False otherwise.
    """
    gguf_file = model_path / GGUF_MODEL_FILE
    if gguf_file.exists():
        return True

    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=GGUF_MODEL_REPO,
            filename=GGUF_MODEL_FILE,
            local_dir=str(model_path),
        )
        return True
    except Exception as e:
        pytest.skip(f"Could not download GGUF test model: {e}")
        return False


@pytest.fixture(scope="module")
def gguf_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get or create path to GGUF test model."""
    # Use a persistent cache directory for faster subsequent runs
    cache_dir = Path(tempfile.gettempdir()) / "test-tinyllama-gguf"
    cache_dir.mkdir(exist_ok=True)

    if not download_gguf_model_if_needed(cache_dir):
        pytest.skip("GGUF test model not available")

    return cache_dir / GGUF_MODEL_FILE


@pytest.fixture(scope="module")
def llamacpp_settings(gguf_model_path: Path) -> Settings:
    """Create settings for llama-cpp integration testing."""
    # Clear any INFERENCE_ env vars to use our test settings
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("INFERENCE_")}

    with patch.dict(os.environ, clean_env, clear=True):
        return Settings(
            _env_file=None,  # Don't read .env file
            backend="llama-cpp",
            model_path=str(gguf_model_path),
            model_name="tinyllama-gguf-test",
            num_threads=2,
            llama_cpp_n_ctx=512,  # Smaller context for faster tests
            llama_cpp_n_batch=128,
        )


@pytest.fixture(scope="module")
def llamacpp_client(llamacpp_settings: Settings) -> Iterator[TestClient]:
    """Create test client with llama-cpp backend loaded."""
    app = create_app(llamacpp_settings)

    with TestClient(app) as client:
        yield client


class TestLlamaCppHealthEndpoint:
    """Integration tests for health endpoint with llama-cpp backend."""

    def test_health_check_returns_healthy(self, llamacpp_client: TestClient) -> None:
        """Test that health check returns healthy status with model loaded."""
        response = llamacpp_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["device"] == "cpu"


class TestLlamaCppModelsEndpoint:
    """Integration tests for models endpoint with llama-cpp backend."""

    def test_list_models(self, llamacpp_client: TestClient) -> None:
        """Test listing available models."""
        response = llamacpp_client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "tinyllama-gguf-test"
        assert data["data"][0]["object"] == "model"


class TestLlamaCppCompletionsEndpoint:
    """Integration tests for completions endpoint with llama-cpp backend."""

    def test_basic_completion(self, llamacpp_client: TestClient) -> None:
        """Test basic text completion with llama-cpp backend."""
        response = llamacpp_client.post(
            "/v1/completions",
            json={
                "prompt": "Hello, my name is",
                "max_tokens": 8,
                "temperature": 0.1,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["object"] == "text_completion"
        assert data["model"] == "tinyllama-gguf-test"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["index"] == 0
        assert isinstance(data["choices"][0]["text"], str)
        assert data["choices"][0]["finish_reason"] == "stop"

        # Verify usage stats
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] >= 0
        assert data["usage"]["total_tokens"] > 0

    def test_completion_streaming(self, llamacpp_client: TestClient) -> None:
        """Test streaming completion response with llama-cpp backend."""
        response = llamacpp_client.post(
            "/v1/completions",
            json={
                "prompt": "Once upon a time",
                "max_tokens": 10,
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


class TestLlamaCppChatCompletionsEndpoint:
    """Integration tests for chat completions endpoint with llama-cpp backend."""

    def test_basic_chat_completion(self, llamacpp_client: TestClient) -> None:
        """Test basic chat completion with llama-cpp backend."""
        response = llamacpp_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 8,
                "temperature": 0.1,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["object"] == "chat.completion"
        assert data["model"] == "tinyllama-gguf-test"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["index"] == 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(data["choices"][0]["message"]["content"], str)
        assert data["choices"][0]["finish_reason"] == "stop"

        # Verify usage stats
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0

    def test_chat_streaming(self, llamacpp_client: TestClient) -> None:
        """Test streaming chat completion with llama-cpp backend."""
        response = llamacpp_client.post(
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


class TestLlamaCppModelInfo:
    """Test model info endpoint with llama-cpp backend."""

    def test_cache_stats_shows_backend(self, llamacpp_client: TestClient) -> None:
        """Test that cache stats endpoint shows llama-cpp backend info."""
        response = llamacpp_client.get("/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()

        assert "model_info" in data
        assert data["model_info"]["backend"] == "llama-cpp"
        assert data["model_info"]["model_name"] == "tinyllama-gguf-test"
