"""Tests for the caching module."""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from inference.engine.cache import (
    CacheManager,
    PromptCache,
    ResponseCache,
    TokenizerCache,
)


class TestResponseCache:
    """Tests for ResponseCache."""

    def test_init(self) -> None:
        """Test cache initialization."""
        cache = ResponseCache(max_size=100, ttl_seconds=3600)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 3600

    def test_put_and_get(self) -> None:
        """Test storing and retrieving cached responses."""
        cache = ResponseCache(max_size=100, temperature_threshold=0.1)

        # Cache a response with low temperature (cacheable)
        cache.put(
            prompt="Hello",
            max_new_tokens=50,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            response="Hello world!",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

        # Retrieve it
        result = cache.get(
            prompt="Hello",
            max_new_tokens=50,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
        )

        assert result is not None
        response, usage = result
        assert response == "Hello world!"
        assert usage["total_tokens"] == 3

    def test_high_temperature_not_cached(self) -> None:
        """Test that high temperature responses are not cached."""
        cache = ResponseCache(temperature_threshold=0.1)

        # Try to cache with high temperature
        cache.put(
            prompt="Hello",
            max_new_tokens=50,
            temperature=0.8,  # Above threshold
            top_p=1.0,
            top_k=50,
            response="Response",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

        # Should not be cached
        result = cache.get(
            prompt="Hello",
            max_new_tokens=50,
            temperature=0.8,
            top_p=1.0,
            top_k=50,
        )
        assert result is None

    def test_cache_miss(self) -> None:
        """Test cache miss for non-existent entry."""
        cache = ResponseCache()

        result = cache.get(
            prompt="Not cached",
            max_new_tokens=50,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
        )
        assert result is None

    def test_ttl_expiration(self) -> None:
        """Test that entries expire after TTL."""
        cache = ResponseCache(ttl_seconds=1)

        cache.put(
            prompt="Hello",
            max_new_tokens=50,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            response="Response",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

        # Should be cached initially
        assert cache.get("Hello", 50, 0.0, 1.0, 50) is not None

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should be expired now
        assert cache.get("Hello", 50, 0.0, 1.0, 50) is None

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = ResponseCache(max_size=2)

        # Add first entry
        cache.put("prompt1", 50, 0.0, 1.0, 50, "response1", {"total_tokens": 1})
        # Add second entry
        cache.put("prompt2", 50, 0.0, 1.0, 50, "response2", {"total_tokens": 2})
        # Add third entry (should evict first)
        cache.put("prompt3", 50, 0.0, 1.0, 50, "response3", {"total_tokens": 3})

        # First should be evicted
        assert cache.get("prompt1", 50, 0.0, 1.0, 50) is None
        # Second and third should exist
        assert cache.get("prompt2", 50, 0.0, 1.0, 50) is not None
        assert cache.get("prompt3", 50, 0.0, 1.0, 50) is not None

    def test_stats(self) -> None:
        """Test cache statistics."""
        cache = ResponseCache(max_size=100)

        # Add an entry
        cache.put("prompt", 50, 0.0, 1.0, 50, "response", {"total_tokens": 1})

        # Get it (hit)
        cache.get("prompt", 50, 0.0, 1.0, 50)
        # Miss
        cache.get("other", 50, 0.0, 1.0, 50)

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self) -> None:
        """Test clearing the cache."""
        cache = ResponseCache()
        cache.put("prompt", 50, 0.0, 1.0, 50, "response", {"total_tokens": 1})

        cache.clear()

        assert cache.get("prompt", 50, 0.0, 1.0, 50) is None
        assert cache.stats()["size"] == 0


class TestPromptCache:
    """Tests for PromptCache."""

    def test_init(self) -> None:
        """Test cache initialization."""
        cache = PromptCache(max_size=50, ttl_seconds=1800)
        assert cache.max_size == 50
        assert cache.ttl_seconds == 1800

    def test_put_and_get(self) -> None:
        """Test storing and retrieving cached prompts."""
        cache = PromptCache()

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        cache.put("system prompt", input_ids, attention_mask)

        entry = cache.get("system prompt")
        assert entry is not None
        assert torch.equal(entry.input_ids, input_ids)
        assert torch.equal(entry.attention_mask, attention_mask)

    def test_cache_miss(self) -> None:
        """Test cache miss."""
        cache = PromptCache()
        assert cache.get("not cached") is None

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        cache = PromptCache(ttl_seconds=1)

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        cache.put("prompt", input_ids, attention_mask)
        assert cache.get("prompt") is not None

        time.sleep(1.1)
        assert cache.get("prompt") is None

    def test_stats(self) -> None:
        """Test cache statistics."""
        cache = PromptCache()

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        cache.put("prompt", input_ids, attention_mask)
        cache.get("prompt")  # Hit
        cache.get("other")  # Miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestTokenizerCache:
    """Tests for TokenizerCache."""

    def test_init(self) -> None:
        """Test cache initialization."""
        cache = TokenizerCache(max_size=1000)
        assert cache.max_size == 1000

    def test_tokenize_caches_result(self) -> None:
        """Test that tokenization results are cached."""
        cache = TokenizerCache()

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # First call
        result1 = cache.tokenize("Hello world", mock_tokenizer)
        assert mock_tokenizer.call_count == 1

        # Second call (should use cache)
        result2 = cache.tokenize("Hello world", mock_tokenizer)
        assert mock_tokenizer.call_count == 1  # Not called again

        # Results should be equal tensors
        assert torch.equal(result1["input_ids"], result2["input_ids"])

    def test_different_texts_not_shared(self) -> None:
        """Test that different texts have separate cache entries."""
        cache = TokenizerCache()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        cache.tokenize("text1", mock_tokenizer)
        cache.tokenize("text2", mock_tokenizer)

        assert mock_tokenizer.call_count == 2

    def test_stats(self) -> None:
        """Test cache statistics."""
        cache = TokenizerCache()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        cache.tokenize("text", mock_tokenizer)  # Miss
        cache.tokenize("text", mock_tokenizer)  # Hit

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_clear(self) -> None:
        """Test clearing the cache."""
        cache = TokenizerCache()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        cache.tokenize("text", mock_tokenizer)
        cache.clear()

        # After clear, should be a miss again
        cache.tokenize("text", mock_tokenizer)
        assert mock_tokenizer.call_count == 2


class TestCacheManager:
    """Tests for CacheManager."""

    def test_init_all_enabled(self) -> None:
        """Test initialization with all caches enabled."""
        manager = CacheManager(
            enable_response_cache=True,
            enable_prompt_cache=True,
            enable_tokenizer_cache=True,
        )

        assert manager.response_cache is not None
        assert manager.prompt_cache is not None
        assert manager.tokenizer_cache is not None

    def test_init_all_disabled(self) -> None:
        """Test initialization with all caches disabled."""
        manager = CacheManager(
            enable_response_cache=False,
            enable_prompt_cache=False,
            enable_tokenizer_cache=False,
        )

        assert manager.response_cache is None
        assert manager.prompt_cache is None
        assert manager.tokenizer_cache is None

    def test_clear_all(self) -> None:
        """Test clearing all caches."""
        manager = CacheManager()

        # Add some data to caches
        if manager.response_cache:
            manager.response_cache.put(
                "prompt", 50, 0.0, 1.0, 50, "response", {"total_tokens": 1}
            )

        manager.clear_all()

        # Verify caches are empty
        if manager.response_cache:
            assert manager.response_cache.stats()["size"] == 0

    def test_stats(self) -> None:
        """Test getting stats from all caches."""
        manager = CacheManager()

        stats = manager.stats()

        assert "response_cache" in stats
        assert "prompt_cache" in stats
        assert "tokenizer_cache" in stats

    def test_custom_sizes(self) -> None:
        """Test custom cache sizes."""
        manager = CacheManager(
            response_cache_size=500,
            prompt_cache_size=25,
            tokenizer_cache_size=2000,
        )

        assert manager.response_cache.max_size == 500
        assert manager.prompt_cache.max_size == 25
        assert manager.tokenizer_cache.max_size == 2000
