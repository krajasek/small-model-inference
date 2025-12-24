"""Tests for the WebSocket streaming endpoint with protobuf."""

from unittest.mock import MagicMock

import betterproto
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference.api.websocket import ws_router
from inference.engine.inference import InferenceEngine
from inference.proto import (
    ChatCompletionRequest,
    ChatMessage,
    ClientMessage,
    CompletionRequest,
    GenerationParams,
    ServerMessage,
)


def get_payload_type(msg: ServerMessage) -> str:
    """Get the type of payload in a ServerMessage using betterproto's which_one_of."""
    field_name, _ = betterproto.which_one_of(msg, "payload")
    return field_name


def has_complete(msg: ServerMessage) -> bool:
    """Check if ServerMessage has complete payload."""
    return get_payload_type(msg) == "complete"


def has_chunk(msg: ServerMessage) -> bool:
    """Check if ServerMessage has chunk payload."""
    return get_payload_type(msg) == "chunk"


def has_error(msg: ServerMessage) -> bool:
    """Check if ServerMessage has error payload."""
    return get_payload_type(msg) == "error"


@pytest.fixture
def ws_app(mock_engine: InferenceEngine) -> FastAPI:
    """Create a FastAPI app with WebSocket router and mock engine."""
    app = FastAPI()
    app.state.engine = mock_engine
    app.include_router(ws_router)
    return app


@pytest.fixture
def ws_client(ws_app: FastAPI) -> TestClient:
    """Create a test client for WebSocket testing."""
    return TestClient(ws_app)


class TestProtoSerialization:
    """Test protobuf message serialization."""

    def test_completion_request_roundtrip(self) -> None:
        """Test CompletionRequest serialization roundtrip."""
        msg = ClientMessage(
            request_id="test-123",
            completion=CompletionRequest(
                prompt="Hello, world!",
                params=GenerationParams(max_tokens=100, temperature=0.8),
            ),
        )

        # Serialize and deserialize
        data = bytes(msg)
        parsed = ClientMessage().parse(data)

        assert parsed.request_id == "test-123"
        assert parsed.completion.prompt == "Hello, world!"
        assert parsed.completion.params.max_tokens == 100
        assert parsed.completion.params.temperature == pytest.approx(0.8, rel=0.01)

    def test_chat_request_with_messages(self) -> None:
        """Test ChatCompletionRequest with multiple messages."""
        msg = ClientMessage(
            request_id="chat-456",
            chat_completion=ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content="You are helpful."),
                    ChatMessage(role="user", content="Hello!"),
                ],
                params=GenerationParams(max_tokens=50),
            ),
        )

        data = bytes(msg)
        parsed = ClientMessage().parse(data)

        assert parsed.request_id == "chat-456"
        assert len(parsed.chat_completion.messages) == 2
        assert parsed.chat_completion.messages[0].role == "system"
        assert parsed.chat_completion.messages[1].content == "Hello!"

    def test_server_message_chunk(self) -> None:
        """Test ServerMessage with chunk payload."""
        from inference.proto import ChunkChoice, ChunkDelta, StreamChunk

        msg = ServerMessage(
            request_id="req-1",
            chunk=StreamChunk(
                id="cmpl-abc",
                created=1234567890,
                model="test-model",
                choice=ChunkChoice(
                    index=0,
                    delta=ChunkDelta(content="Hello"),
                ),
            ),
        )

        data = bytes(msg)
        parsed = ServerMessage().parse(data)

        assert parsed.request_id == "req-1"
        assert has_chunk(parsed)
        assert parsed.chunk.id == "cmpl-abc"
        assert parsed.chunk.choice.delta.content == "Hello"

    def test_server_message_error(self) -> None:
        """Test ServerMessage with error payload."""
        from inference.proto import ErrorResponse

        msg = ServerMessage(
            request_id="req-err",
            error=ErrorResponse(
                code=400,
                message="Invalid request",
                type="validation_error",
            ),
        )

        data = bytes(msg)
        parsed = ServerMessage().parse(data)

        assert parsed.request_id == "req-err"
        assert has_error(parsed)
        assert parsed.error.code == 400
        assert parsed.error.message == "Invalid request"
        assert parsed.error.type == "validation_error"


class TestWebSocketCompletion:
    """Test WebSocket completion streaming."""

    def test_completion_streaming(
        self, ws_client: TestClient, mock_engine: InferenceEngine
    ) -> None:
        """Test streaming completion via WebSocket."""
        # Mock the generate_stream method
        mock_engine.generate_stream = MagicMock(
            return_value=iter([("Hello", False), (" world", False), ("", True)])
        )

        with ws_client.websocket_connect("/v1/stream") as websocket:
            # Send completion request
            request = ClientMessage(
                request_id="req-1",
                completion=CompletionRequest(
                    prompt="Test prompt",
                    params=GenerationParams(max_tokens=50),
                ),
            )
            websocket.send_bytes(bytes(request))

            # Receive streaming responses
            messages: list[ServerMessage] = []
            while True:
                data = websocket.receive_bytes()
                msg = ServerMessage().parse(data)
                messages.append(msg)
                if has_complete(msg):
                    break

            # Verify response structure
            assert all(m.request_id == "req-1" for m in messages)

            # Check for content chunks
            content_chunks = [
                m for m in messages if has_chunk(m) and m.chunk.choice.delta.content
            ]
            assert len(content_chunks) >= 1

            # Check for completion message
            complete_msgs = [m for m in messages if has_complete(m)]
            assert len(complete_msgs) == 1

    def test_completion_with_params(
        self, ws_client: TestClient, mock_engine: InferenceEngine
    ) -> None:
        """Test completion with custom generation parameters."""
        mock_engine.generate_stream = MagicMock(
            return_value=iter([("Test", False), ("", True)])
        )

        with ws_client.websocket_connect("/v1/stream") as websocket:
            request = ClientMessage(
                request_id="req-params",
                completion=CompletionRequest(
                    prompt="Test",
                    params=GenerationParams(
                        max_tokens=100,
                        temperature=0.5,
                        top_p=0.95,
                        top_k=40,
                        do_sample=True,
                        repetition_penalty=1.2,
                    ),
                ),
            )
            websocket.send_bytes(bytes(request))

            # Receive responses until complete
            while True:
                data = websocket.receive_bytes()
                msg = ServerMessage().parse(data)
                if has_complete(msg):
                    break

            # Verify generate_stream was called (params are converted internally)
            mock_engine.generate_stream.assert_called_once()


class TestWebSocketChatCompletion:
    """Test WebSocket chat completion streaming."""

    def test_chat_completion_streaming(
        self, ws_client: TestClient, mock_engine: InferenceEngine
    ) -> None:
        """Test streaming chat completion via WebSocket."""
        mock_engine.generate_stream = MagicMock(
            return_value=iter([("Hi", False), (" there!", False), ("", True)])
        )

        with ws_client.websocket_connect("/v1/stream") as websocket:
            request = ClientMessage(
                request_id="chat-1",
                chat_completion=ChatCompletionRequest(
                    messages=[
                        ChatMessage(role="user", content="Hello!"),
                    ],
                    params=GenerationParams(max_tokens=50),
                ),
            )
            websocket.send_bytes(bytes(request))

            # Receive streaming responses
            messages: list[ServerMessage] = []
            while True:
                data = websocket.receive_bytes()
                msg = ServerMessage().parse(data)
                messages.append(msg)
                if has_complete(msg):
                    break

            # Verify all have correct request_id
            assert all(m.request_id == "chat-1" for m in messages)

            # First chunk should have role
            first_chunk = messages[0]
            assert has_chunk(first_chunk)
            assert first_chunk.chunk.choice.delta.role == "assistant"

            # Should have completion message
            complete_msgs = [m for m in messages if has_complete(m)]
            assert len(complete_msgs) == 1

    def test_chat_with_system_message(
        self, ws_client: TestClient, mock_engine: InferenceEngine
    ) -> None:
        """Test chat completion with system message."""
        mock_engine.generate_stream = MagicMock(
            return_value=iter([("Response", False), ("", True)])
        )

        with ws_client.websocket_connect("/v1/stream") as websocket:
            request = ClientMessage(
                request_id="chat-sys",
                chat_completion=ChatCompletionRequest(
                    messages=[
                        ChatMessage(role="system", content="You are helpful."),
                        ChatMessage(role="user", content="Hello!"),
                    ],
                    params=GenerationParams(max_tokens=50),
                ),
            )
            websocket.send_bytes(bytes(request))

            # Receive responses until complete
            while True:
                data = websocket.receive_bytes()
                msg = ServerMessage().parse(data)
                if has_complete(msg):
                    break


class TestWebSocketErrorHandling:
    """Test WebSocket error handling."""

    def test_empty_payload(self, ws_client: TestClient) -> None:
        """Test handling of empty payload."""
        with ws_client.websocket_connect("/v1/stream") as websocket:
            # Send request with no payload
            request = ClientMessage(request_id="empty-req")
            websocket.send_bytes(bytes(request))

            # Should receive error response
            data = websocket.receive_bytes()
            msg = ServerMessage().parse(data)

            assert msg.request_id == "empty-req"
            assert has_error(msg)
            assert msg.error.code == 400
            assert "Invalid request" in msg.error.message

    def test_generation_error(
        self, ws_client: TestClient, mock_engine: InferenceEngine
    ) -> None:
        """Test handling of generation errors."""
        mock_engine.generate_stream = MagicMock(
            side_effect=Exception("Generation failed")
        )

        with ws_client.websocket_connect("/v1/stream") as websocket:
            request = ClientMessage(
                request_id="err-req",
                completion=CompletionRequest(
                    prompt="Test",
                    params=GenerationParams(max_tokens=50),
                ),
            )
            websocket.send_bytes(bytes(request))

            # Should receive error response
            data = websocket.receive_bytes()
            msg = ServerMessage().parse(data)

            assert msg.request_id == "err-req"
            assert has_error(msg)
            assert msg.error.code == 500
            assert "Generation failed" in msg.error.message

    def test_invalid_protobuf(self, ws_client: TestClient) -> None:
        """Test handling of invalid/empty protobuf data."""
        with ws_client.websocket_connect("/v1/stream") as websocket:
            # Send invalid binary data (parsed as empty ClientMessage)
            websocket.send_bytes(b"invalid protobuf data")

            # Should receive error response (validation or parse error)
            data = websocket.receive_bytes()
            msg = ServerMessage().parse(data)

            assert has_error(msg)
            assert msg.error.code == 400
            # Invalid data may be parsed as empty payload, triggering validation error
            assert msg.error.type in ("parse_error", "validation_error")


class TestWebSocketMultipleRequests:
    """Test multiple sequential requests over same WebSocket connection."""

    def test_sequential_requests(
        self, ws_client: TestClient, mock_engine: InferenceEngine
    ) -> None:
        """Test sending multiple requests over same connection."""
        mock_engine.generate_stream = MagicMock(
            return_value=iter([("Response", False), ("", True)])
        )

        with ws_client.websocket_connect("/v1/stream") as websocket:
            # Send first request
            request1 = ClientMessage(
                request_id="req-1",
                completion=CompletionRequest(
                    prompt="First",
                    params=GenerationParams(max_tokens=10),
                ),
            )
            websocket.send_bytes(bytes(request1))

            # Receive first response
            while True:
                data = websocket.receive_bytes()
                msg = ServerMessage().parse(data)
                if has_complete(msg):
                    assert msg.request_id == "req-1"
                    break

            # Reset mock for second request
            mock_engine.generate_stream = MagicMock(
                return_value=iter([("Second", False), ("", True)])
            )

            # Send second request
            request2 = ClientMessage(
                request_id="req-2",
                completion=CompletionRequest(
                    prompt="Second",
                    params=GenerationParams(max_tokens=10),
                ),
            )
            websocket.send_bytes(bytes(request2))

            # Receive second response
            while True:
                data = websocket.receive_bytes()
                msg = ServerMessage().parse(data)
                if has_complete(msg):
                    assert msg.request_id == "req-2"
                    break


class TestWebSocketCaching:
    """Test WebSocket caching functionality."""

    def test_formatted_prompt_cache(self) -> None:
        """Test that formatted prompts are cached."""
        from inference.api.websocket import (
            FormattedPromptCache,
            _format_chat_prompt,
            get_cache_manager,
        )
        from inference.proto import ChatMessage

        # Clear cache and reset counters
        cache_manager = get_cache_manager()
        cache_manager.clear_all()
        # Reset hit/miss counters
        cache_manager.formatted_prompt_cache._hits = 0
        cache_manager.formatted_prompt_cache._misses = 0
        cache_manager.system_prompt_cache._hits = 0
        cache_manager.system_prompt_cache._misses = 0

        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello!"),
        ]

        # First call should miss cache
        prompt1 = _format_chat_prompt(messages)
        stats1 = cache_manager.formatted_prompt_cache.stats()
        assert stats1["misses"] == 1
        assert stats1["hits"] == 0

        # Second call with same messages should hit cache
        prompt2 = _format_chat_prompt(messages)
        stats2 = cache_manager.formatted_prompt_cache.stats()
        assert stats2["hits"] == 1
        assert prompt1 == prompt2

    def test_system_prompt_cache(self) -> None:
        """Test that system prompts are cached separately."""
        from inference.api.websocket import (
            _format_chat_prompt,
            get_cache_manager,
        )
        from inference.proto import ChatMessage

        # Clear cache and reset counters
        cache_manager = get_cache_manager()
        cache_manager.clear_all()
        cache_manager.formatted_prompt_cache._hits = 0
        cache_manager.formatted_prompt_cache._misses = 0
        cache_manager.system_prompt_cache._hits = 0
        cache_manager.system_prompt_cache._misses = 0

        system_content = "You are a helpful assistant."

        # First request with system prompt
        messages1 = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content="Hello!"),
        ]
        _format_chat_prompt(messages1)

        # Check system prompt was cached
        sys_stats1 = cache_manager.system_prompt_cache.stats()
        assert sys_stats1["size"] == 1

        # Second request with same system but different user message
        # (will miss formatted prompt cache but hit system prompt cache)
        messages2 = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content="Different message!"),
        ]
        _format_chat_prompt(messages2)

        sys_stats2 = cache_manager.system_prompt_cache.stats()
        assert sys_stats2["hits"] >= 1  # System prompt was reused

    def test_cache_stats_and_clear(self) -> None:
        """Test cache statistics and clearing."""
        from inference.api.websocket import (
            clear_websocket_caches,
            get_cache_manager,
            get_websocket_cache_stats,
        )

        cache_manager = get_cache_manager()

        # Get stats
        stats = get_websocket_cache_stats()
        assert "formatted_prompt_cache" in stats
        assert "system_prompt_cache" in stats
        assert "size" in stats["formatted_prompt_cache"]
        assert "hit_rate" in stats["formatted_prompt_cache"]

        # Clear caches
        clear_websocket_caches()
        stats_after = get_websocket_cache_stats()
        assert stats_after["formatted_prompt_cache"]["size"] == 0
        assert stats_after["system_prompt_cache"]["size"] == 0

    def test_formatted_prompt_cache_lru_eviction(self) -> None:
        """Test that LRU eviction works."""
        from inference.api.websocket import FormattedPromptCache
        from inference.proto import ChatMessage

        cache = FormattedPromptCache(max_size=3)

        # Add 4 items (exceeds max_size of 3)
        for i in range(4):
            messages = [ChatMessage(role="user", content=f"Message {i}")]
            cache.put(messages, f"formatted_{i}")

        # Cache should have 3 items (oldest was evicted)
        assert cache.stats()["size"] == 3

        # First message should be evicted
        messages_0 = [ChatMessage(role="user", content="Message 0")]
        assert cache.get(messages_0) is None

        # Last 3 messages should still be cached
        for i in range(1, 4):
            messages = [ChatMessage(role="user", content=f"Message {i}")]
            assert cache.get(messages) == f"formatted_{i}"
