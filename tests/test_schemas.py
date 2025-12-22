"""Tests for the API schemas module."""

import time

import pytest
from pydantic import ValidationError

from inference.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    Usage,
)
from inference.backends.base import GenerationConfig


class TestUsage:
    """Test Usage schema."""

    def test_create_usage(self) -> None:
        """Test creating a Usage instance."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestCompletionRequest:
    """Test CompletionRequest schema."""

    def test_minimal_request(self) -> None:
        """Test creating a minimal completion request."""
        request = CompletionRequest(prompt="Hello")

        assert request.prompt == "Hello"
        assert request.max_tokens == 256
        assert request.temperature == 0.7
        assert request.stream is False

    def test_full_request(self) -> None:
        """Test creating a full completion request."""
        request = CompletionRequest(
            prompt="Hello",
            max_tokens=100,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            do_sample=False,
            repetition_penalty=1.2,
            stream=True,
        )

        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is True

    def test_max_tokens_validation(self) -> None:
        """Test that max_tokens is validated."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="Test", max_tokens=0)

        with pytest.raises(ValidationError):
            CompletionRequest(prompt="Test", max_tokens=5000)

    def test_temperature_validation(self) -> None:
        """Test that temperature is validated."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="Test", temperature=-0.1)

        with pytest.raises(ValidationError):
            CompletionRequest(prompt="Test", temperature=2.5)

    def test_to_generation_config(self) -> None:
        """Test conversion to GenerationConfig."""
        request = CompletionRequest(
            prompt="Test",
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            do_sample=True,
            repetition_penalty=1.15,
        )

        config = request.to_generation_config()

        assert isinstance(config, GenerationConfig)
        assert config.max_new_tokens == 100
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.do_sample is True
        assert config.repetition_penalty == 1.15


class TestCompletionResponse:
    """Test CompletionResponse schema."""

    def test_create_response(self) -> None:
        """Test creating a completion response."""
        response = CompletionResponse(
            model="test-model",
            choices=[CompletionChoice(index=0, text="Generated text")],
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )

        assert response.object == "text_completion"
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].text == "Generated text"

    def test_auto_generated_fields(self) -> None:
        """Test that id and created are auto-generated."""
        response = CompletionResponse(
            model="test-model",
            choices=[CompletionChoice(index=0, text="Test")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

        assert response.id.startswith("cmpl-")
        assert response.created <= int(time.time())


class TestChatMessage:
    """Test ChatMessage schema."""

    def test_create_message(self) -> None:
        """Test creating chat messages with different roles."""
        system = ChatMessage(role="system", content="You are helpful.")
        user = ChatMessage(role="user", content="Hello")
        assistant = ChatMessage(role="assistant", content="Hi there!")

        assert system.role == "system"
        assert user.role == "user"
        assert assistant.role == "assistant"

    def test_invalid_role(self) -> None:
        """Test that invalid roles are rejected."""
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Test")  # type: ignore[arg-type]


class TestChatCompletionRequest:
    """Test ChatCompletionRequest schema."""

    def test_minimal_request(self, sample_chat_messages: list[ChatMessage]) -> None:
        """Test creating a minimal chat request."""
        request = ChatCompletionRequest(messages=sample_chat_messages)

        assert len(request.messages) == 2
        assert request.max_tokens == 256
        assert request.stream is False

    def test_to_generation_config(self, sample_chat_messages: list[ChatMessage]) -> None:
        """Test conversion to GenerationConfig."""
        request = ChatCompletionRequest(
            messages=sample_chat_messages,
            max_tokens=100,
            temperature=0.5,
        )

        config = request.to_generation_config()

        assert config.max_new_tokens == 100
        assert config.temperature == 0.5

    def test_format_prompt_system_and_user(self) -> None:
        """Test prompt formatting with system and user messages."""
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="Be helpful."),
                ChatMessage(role="user", content="Hello"),
            ]
        )

        prompt = request.format_prompt()

        assert "<<SYS>>" in prompt
        assert "Be helpful." in prompt
        assert "[INST]" in prompt
        assert "[/INST]" in prompt

    def test_format_prompt_user_only(self) -> None:
        """Test prompt formatting with user message only."""
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hello")])

        prompt = request.format_prompt()

        assert "[INST] Hello [/INST]" in prompt

    def test_format_prompt_with_assistant(self) -> None:
        """Test prompt formatting with assistant message."""
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi!"),
                ChatMessage(role="user", content="How are you?"),
            ]
        )

        prompt = request.format_prompt()

        assert "Hi!" in prompt
        assert "How are you?" in prompt


class TestChatCompletionResponse:
    """Test ChatCompletionResponse schema."""

    def test_create_response(self) -> None:
        """Test creating a chat completion response."""
        response = ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello!"),
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )

        assert response.object == "chat.completion"
        assert response.id.startswith("chatcmpl-")


class TestChatCompletionChunk:
    """Test streaming chunk schemas."""

    def test_chunk_delta(self) -> None:
        """Test ChatCompletionChunkDelta."""
        delta = ChatCompletionChunkDelta(role="assistant", content="Hello")

        assert delta.role == "assistant"
        assert delta.content == "Hello"

    def test_chunk_delta_empty(self) -> None:
        """Test empty delta for final chunk."""
        delta = ChatCompletionChunkDelta()

        assert delta.role is None
        assert delta.content is None

    def test_chunk_choice(self) -> None:
        """Test ChatCompletionChunkChoice."""
        choice = ChatCompletionChunkChoice(
            index=0,
            delta=ChatCompletionChunkDelta(content="Token"),
            finish_reason=None,
        )

        assert choice.index == 0
        assert choice.delta.content == "Token"
        assert choice.finish_reason is None

    def test_chunk(self) -> None:
        """Test ChatCompletionChunk."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content="Hi"),
                )
            ],
        )

        assert chunk.object == "chat.completion.chunk"
        assert chunk.model == "test-model"


class TestModelInfo:
    """Test ModelInfo schema."""

    def test_create_model_info(self) -> None:
        """Test creating model info."""
        info = ModelInfo(id="my-model")

        assert info.id == "my-model"
        assert info.object == "model"
        assert info.owned_by == "local"


class TestModelListResponse:
    """Test ModelListResponse schema."""

    def test_create_model_list(self) -> None:
        """Test creating model list response."""
        response = ModelListResponse(data=[ModelInfo(id="model-1"), ModelInfo(id="model-2")])

        assert response.object == "list"
        assert len(response.data) == 2


class TestHealthResponse:
    """Test HealthResponse schema."""

    def test_healthy_response(self) -> None:
        """Test healthy status response."""
        response = HealthResponse(status="healthy", model_loaded=True, device="cpu")

        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.device == "cpu"

    def test_unhealthy_response(self) -> None:
        """Test unhealthy status response."""
        response = HealthResponse(status="unhealthy", model_loaded=False)

        assert response.status == "unhealthy"
        assert response.model_loaded is False
        assert response.device is None
