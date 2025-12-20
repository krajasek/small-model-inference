"""Tests for the API routes module."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference.api.routes import router
from inference.engine.inference import InferenceEngine


@pytest.fixture
def app_with_engine(mock_engine: InferenceEngine) -> FastAPI:
    """Create a FastAPI app with a mock engine."""
    app = FastAPI()
    app.state.engine = mock_engine
    app.include_router(router)
    return app


@pytest.fixture
def client(app_with_engine: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app_with_engine)


class TestHealthCheck:
    """Test health check endpoint."""

    def test_healthy_with_engine(self, client: TestClient) -> None:
        """Test health check returns healthy when engine is loaded."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["device"] == "cpu"

    def test_unhealthy_without_engine(self) -> None:
        """Test health check returns unhealthy when engine is not loaded."""
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestListModels:
    """Test list models endpoint."""

    def test_list_models(self, client: TestClient) -> None:
        """Test listing models returns the loaded model."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "local-model"


class TestCompletions:
    """Test completions endpoint."""

    def test_create_completion(self, client: TestClient) -> None:
        """Test creating a text completion."""
        response = client.post(
            "/v1/completions",
            json={"prompt": "Hello, world!", "max_tokens": 50},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "local-model"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert "usage" in data

    def test_completion_with_all_params(self, client: TestClient) -> None:
        """Test completion with all parameters specified."""
        response = client.post(
            "/v1/completions",
            json={
                "prompt": "Test",
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.95,
                "top_k": 40,
                "do_sample": True,
                "repetition_penalty": 1.2,
            },
        )

        assert response.status_code == 200

    def test_completion_streaming(self, client: TestClient, mock_engine: InferenceEngine) -> None:
        """Test streaming completion."""
        # Mock the generate_stream method
        mock_engine.generate_stream = MagicMock(
            return_value=iter([("Hello", False), (" world", False), ("", True)])
        )

        response = client.post(
            "/v1/completions",
            json={"prompt": "Test", "max_tokens": 50, "stream": True},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        lines = response.text.strip().split("\n\n")
        events = [line for line in lines if line.startswith("data: ")]

        # Should have content events plus [DONE]
        assert len(events) >= 2
        assert events[-1] == "data: [DONE]"

    def test_completion_error_handling(
        self, client: TestClient, mock_engine: InferenceEngine
    ) -> None:
        """Test error handling in completion endpoint."""
        mock_engine.generate = MagicMock(side_effect=Exception("Generation failed"))

        response = client.post(
            "/v1/completions",
            json={"prompt": "Test"},
        )

        assert response.status_code == 500
        assert "Generation failed" in response.json()["detail"]


class TestChatCompletions:
    """Test chat completions endpoint."""

    def test_create_chat_completion(self, client: TestClient) -> None:
        """Test creating a chat completion."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 50,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "local-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completion_with_system_message(self, client: TestClient) -> None:
        """Test chat completion with system message."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "max_tokens": 50,
            },
        )

        assert response.status_code == 200

    def test_chat_completion_streaming(
        self, client: TestClient, mock_engine: InferenceEngine
    ) -> None:
        """Test streaming chat completion."""
        mock_engine.generate_stream = MagicMock(
            return_value=iter([("Hello", False), (" there!", False), ("", True)])
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        lines = response.text.strip().split("\n\n")
        events = [line for line in lines if line.startswith("data: ")]

        # Should have initial role chunk, content chunks, final chunk, and [DONE]
        assert len(events) >= 4
        assert events[-1] == "data: [DONE]"

        # Check first chunk has role
        first_data = json.loads(events[0].replace("data: ", ""))
        assert first_data["choices"][0]["delta"].get("role") == "assistant"

    def test_chat_completion_error_handling(
        self, client: TestClient, mock_engine: InferenceEngine
    ) -> None:
        """Test error handling in chat completion endpoint."""
        mock_engine.generate = MagicMock(side_effect=Exception("Chat generation failed"))

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Test"}]},
        )

        assert response.status_code == 500
        assert "Chat generation failed" in response.json()["detail"]

    def test_invalid_message_role(self, client: TestClient) -> None:
        """Test that invalid message roles are rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "invalid", "content": "Test"}],
            },
        )

        assert response.status_code == 422  # Validation error
