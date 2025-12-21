"""Tests for the observability/tracing module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from inference.observability.tracing import (
    GenerationMetrics,
    GenerationTrace,
    InferenceTracer,
    get_tracer,
    init_tracer,
    trace_generation,
)


class TestGenerationMetrics:
    """Tests for GenerationMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default metric values."""
        metrics = GenerationMetrics()
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.total_tokens == 0
        assert metrics.time_to_first_token_ms is None
        assert metrics.total_latency_ms == 0.0
        assert metrics.tokens_per_second == 0.0
        assert metrics.cache_hit is False
        assert metrics.cache_type is None
        assert metrics.error is None
        assert metrics.metadata == {}

    def test_custom_values(self) -> None:
        """Test custom metric values."""
        metrics = GenerationMetrics(
            prompt_tokens=10,
            completion_tokens=50,
            total_tokens=60,
            time_to_first_token_ms=100.0,
            total_latency_ms=500.0,
            tokens_per_second=100.0,
            cache_hit=True,
            cache_type="response",
            error="test error",
            metadata={"key": "value"},
        )
        assert metrics.prompt_tokens == 10
        assert metrics.completion_tokens == 50
        assert metrics.cache_hit is True
        assert metrics.cache_type == "response"


class TestInferenceTracer:
    """Tests for InferenceTracer class."""

    def test_init_disabled(self) -> None:
        """Test tracer initialization when disabled."""
        tracer = InferenceTracer(enabled=False)
        assert tracer.enabled is False
        assert tracer._client is None

    def test_init_no_keys(self) -> None:
        """Test tracer initialization without API keys."""
        # Without keys, Langfuse should be disabled
        tracer = InferenceTracer(
            public_key=None,
            secret_key=None,
            enabled=True,
        )
        # Langfuse auth will fail without keys
        assert tracer.enabled is False

    @patch("inference.observability.tracing.Langfuse")
    def test_init_with_keys(self, mock_langfuse: MagicMock) -> None:
        """Test tracer initialization with API keys."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse.return_value = mock_client

        tracer = InferenceTracer(
            public_key="test-key",
            secret_key="test-secret",
            host="https://test.langfuse.com",
            enabled=True,
        )

        assert tracer.enabled is True
        assert tracer._client is not None
        mock_langfuse.assert_called_once()

    @patch("inference.observability.tracing.Langfuse")
    def test_init_auth_failure(self, mock_langfuse: MagicMock) -> None:
        """Test tracer initialization when auth fails."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = False
        mock_langfuse.return_value = mock_client

        tracer = InferenceTracer(
            public_key="test-key",
            secret_key="test-secret",
            enabled=True,
        )

        assert tracer.enabled is False
        assert tracer._client is None

    @patch("inference.observability.tracing.Langfuse")
    def test_init_exception(self, mock_langfuse: MagicMock) -> None:
        """Test tracer initialization when exception occurs."""
        mock_langfuse.side_effect = Exception("Connection failed")

        tracer = InferenceTracer(
            public_key="test-key",
            secret_key="test-secret",
            enabled=True,
        )

        assert tracer.enabled is False

    def test_set_model_info(self) -> None:
        """Test setting model info."""
        tracer = InferenceTracer(enabled=False)
        tracer.set_model_info("test-model", {"size": "1B"})

        assert tracer._model_name == "test-model"
        assert tracer._model_info == {"size": "1B"}

    def test_trace_generation_returns_context(self) -> None:
        """Test trace_generation returns GenerationTrace context."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(
            name="test",
            user_id="user-123",
            session_id="session-456",
        )

        assert isinstance(trace, GenerationTrace)
        assert trace._name == "test"
        assert trace._user_id == "user-123"
        assert trace._session_id == "session-456"

    @patch("inference.observability.tracing.Langfuse")
    def test_flush(self, mock_langfuse: MagicMock) -> None:
        """Test flush method."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse.return_value = mock_client

        tracer = InferenceTracer(
            public_key="test-key",
            secret_key="test-secret",
            enabled=True,
        )
        tracer.flush()

        mock_client.flush.assert_called_once()

    def test_flush_disabled(self) -> None:
        """Test flush when disabled."""
        tracer = InferenceTracer(enabled=False)
        # Should not raise
        tracer.flush()

    @patch("inference.observability.tracing.Langfuse")
    def test_shutdown(self, mock_langfuse: MagicMock) -> None:
        """Test shutdown method."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse.return_value = mock_client

        tracer = InferenceTracer(
            public_key="test-key",
            secret_key="test-secret",
            enabled=True,
        )
        tracer.shutdown()

        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()


class TestGenerationTrace:
    """Tests for GenerationTrace context manager."""

    def test_enter_exit_disabled(self) -> None:
        """Test context manager when tracing is disabled."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        with trace:
            trace.set_input(prompt="Hello")
            trace.set_parameters(temperature=0.7)
            trace.set_output(
                text="World", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            )

        # Should complete without errors
        assert trace._metrics.prompt_tokens == 1
        assert trace._metrics.completion_tokens == 1

    @patch("inference.observability.tracing.Langfuse")
    def test_enter_creates_generation(self, mock_langfuse: MagicMock) -> None:
        """Test that entering creates a generation span."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_generation = MagicMock()
        mock_client.start_generation.return_value = mock_generation
        mock_langfuse.return_value = mock_client

        tracer = InferenceTracer(
            public_key="test-key",
            secret_key="test-secret",
            enabled=True,
        )
        trace = tracer.trace_generation(name="test")

        with trace:
            pass

        mock_client.start_generation.assert_called_once()
        mock_generation.end.assert_called_once()

    def test_set_input_prompt(self) -> None:
        """Test setting input prompt."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        trace.set_input(prompt="Hello, world!")
        assert trace._input_data == "Hello, world!"

    def test_set_input_messages(self) -> None:
        """Test setting input messages."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        messages = [{"role": "user", "content": "Hello"}]
        trace.set_input(messages=messages)
        assert trace._input_data == messages

    def test_set_parameters(self) -> None:
        """Test setting parameters."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        trace.set_parameters(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=100,
            do_sample=True,
            repetition_penalty=1.1,
            custom_param="value",
        )

        assert trace._parameters["temperature"] == 0.7
        assert trace._parameters["top_p"] == 0.9
        assert trace._parameters["top_k"] == 50
        assert trace._parameters["max_tokens"] == 100
        assert trace._parameters["do_sample"] is True
        assert trace._parameters["repetition_penalty"] == 1.1
        assert trace._parameters["custom_param"] == "value"

    def test_set_output(self) -> None:
        """Test setting output."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        trace.set_output(
            text="Generated text",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )

        assert trace._metrics.prompt_tokens == 10
        assert trace._metrics.completion_tokens == 20
        assert trace._metrics.total_tokens == 30
        assert trace._metrics.metadata["output"] == "Generated text"

    def test_record_first_token(self) -> None:
        """Test recording first token time."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        with trace:
            time.sleep(0.01)  # Small delay
            trace.record_first_token()

        assert trace._metrics.time_to_first_token_ms is not None
        assert trace._metrics.time_to_first_token_ms > 0

    def test_record_first_token_only_once(self) -> None:
        """Test that first token time is only recorded once."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        with trace:
            trace.record_first_token()
            first_time = trace._metrics.time_to_first_token_ms
            time.sleep(0.01)
            trace.record_first_token()  # Should not update

        assert trace._metrics.time_to_first_token_ms == first_time

    def test_record_cache_hit(self) -> None:
        """Test recording cache hit."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        trace.record_cache_hit("response")

        assert trace._metrics.cache_hit is True
        assert trace._metrics.cache_type == "response"

    def test_add_metadata(self) -> None:
        """Test adding metadata."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        trace.add_metadata(key1="value1", key2="value2")

        assert trace._metrics.metadata["key1"] == "value1"
        assert trace._metrics.metadata["key2"] == "value2"

    def test_latency_calculation(self) -> None:
        """Test latency is calculated on exit."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        with trace:
            time.sleep(0.01)  # Small delay

        assert trace._metrics.total_latency_ms > 0

    def test_tokens_per_second_calculation(self) -> None:
        """Test tokens per second is calculated."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        with trace:
            time.sleep(0.01)
            trace.set_output(
                text="test",
                usage={"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
            )

        assert trace._metrics.tokens_per_second > 0

    def test_error_recording(self) -> None:
        """Test error is recorded on exception."""
        tracer = InferenceTracer(enabled=False)
        trace = tracer.trace_generation(name="test")

        with pytest.raises(ValueError):
            with trace:
                raise ValueError("Test error")

        assert trace._metrics.error == "Test error"


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_init_tracer(self) -> None:
        """Test init_tracer creates and stores tracer."""
        tracer = init_tracer(enabled=False)

        assert tracer is not None
        assert get_tracer() is tracer

    def test_get_tracer_returns_none_initially(self) -> None:
        """Test get_tracer before initialization."""
        # Reset global state
        import inference.observability.tracing as tracing_module

        tracing_module._tracer = None

        assert get_tracer() is None

    def test_trace_generation_context_manager_disabled(self) -> None:
        """Test trace_generation context manager when disabled."""
        init_tracer(enabled=False)

        with trace_generation(name="test") as trace:
            # trace should be None since tracing is disabled
            if trace:
                trace.set_input(prompt="Hello")

    def test_trace_generation_context_manager_no_tracer(self) -> None:
        """Test trace_generation when no tracer is initialized."""
        import inference.observability.tracing as tracing_module

        tracing_module._tracer = None

        with trace_generation(name="test") as trace:
            assert trace is None
