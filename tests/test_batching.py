"""Tests for the continuous batching module."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch

from inference.engine.batching import BatchRequest, BatchResult, ContinuousBatcher
from inference.engine.inference import GenerationConfig


class TestBatchRequest:
    """Tests for BatchRequest dataclass."""

    def test_create_request(self) -> None:
        """Test creating a batch request."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()

        request = BatchRequest(
            prompt="Hello",
            config=GenerationConfig(),
            future=future,
        )

        assert request.prompt == "Hello"
        assert request.config.max_new_tokens == 256
        assert request.created_at > 0
        loop.close()


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a batch result."""
        result = BatchResult(
            text="Generated text",
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )

        assert result.text == "Generated text"
        assert result.usage["total_tokens"] == 15


class TestContinuousBatcher:
    """Tests for ContinuousBatcher."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock()

        def mock_generate(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            # Return extended sequences
            return torch.cat(
                [kwargs["input_ids"], torch.ones(batch_size, 10, dtype=torch.long)],
                dim=1,
            )

        model.generate.side_effect = mock_generate
        return model

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2

        def mock_call(texts, **kwargs):
            if isinstance(texts, list):
                batch_size = len(texts)
            else:
                batch_size = 1
            return {
                "input_ids": torch.ones(batch_size, 5, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
            }

        tokenizer.side_effect = mock_call
        tokenizer.return_value = mock_call(["test"])
        tokenizer.decode.return_value = "Generated response"

        return tokenizer

    def test_init(self, mock_model: MagicMock, mock_tokenizer: MagicMock) -> None:
        """Test batcher initialization."""
        batcher = ContinuousBatcher(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            max_batch_size=8,
            max_wait_time_ms=50,
        )

        assert batcher.max_batch_size == 8
        assert batcher.max_wait_time_ms == 50
        assert not batcher._running

    @pytest.mark.asyncio
    async def test_start_and_stop(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test starting and stopping the batcher."""
        batcher = ContinuousBatcher(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        await batcher.start()
        assert batcher._running

        await batcher.stop()
        assert not batcher._running

    @pytest.mark.asyncio
    async def test_submit_request(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test submitting a request."""
        batcher = ContinuousBatcher(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            max_batch_size=1,
            max_wait_time_ms=10,
        )

        await batcher.start()

        try:
            result = await asyncio.wait_for(
                batcher.submit("Hello", GenerationConfig(max_new_tokens=10)),
                timeout=5.0,
            )

            text, usage = result
            assert isinstance(text, str)
            assert "prompt_tokens" in usage
            assert "completion_tokens" in usage
        finally:
            await batcher.stop()

    def test_stats(self, mock_model: MagicMock, mock_tokenizer: MagicMock) -> None:
        """Test getting batcher statistics."""
        batcher = ContinuousBatcher(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        stats = batcher.stats()

        assert "total_batches" in stats
        assert "total_requests" in stats
        assert "average_batch_size" in stats
        assert "pending_requests" in stats
        assert "running" in stats

    def test_generate_batch_sync(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test synchronous batch generation."""
        batcher = ContinuousBatcher(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        prompts = ["Hello", "World"]
        config = GenerationConfig(max_new_tokens=10)

        results = batcher._generate_batch_sync(prompts, config)

        assert len(results) == 2
        assert all(isinstance(r, BatchResult) for r in results)
        assert all(isinstance(r.text, str) for r in results)
        assert all("total_tokens" in r.usage for r in results)
