"""Caching utilities for inference optimization."""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the response cache."""

    response: str
    usage: dict[str, int]
    created_at: float = field(default_factory=time.time)


@dataclass
class PromptCacheEntry:
    """Entry in the prompt/KV cache."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: DynamicCache | None
    created_at: float = field(default_factory=time.time)


class ResponseCache:
    """LRU cache for complete inference responses.

    Caches responses for deterministic requests (low temperature)
    to avoid redundant computation for identical prompts.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        temperature_threshold: float = 0.1,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.temperature_threshold = temperature_threshold
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str | None:
        """Generate cache key for a request. Returns None if not cacheable."""
        # Only cache deterministic/near-deterministic requests
        if temperature > self.temperature_threshold:
            return None

        key_data = f"{prompt}|{max_new_tokens}|{temperature:.4f}|{top_p:.4f}|{top_k}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> tuple[str, dict[str, int]] | None:
        """Get cached response if available."""
        key = self._make_key(prompt, max_new_tokens, temperature, top_p, top_k)
        if key is None:
            return None

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.created_at > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"Response cache hit for key {key[:16]}...")
            return entry.response, entry.usage

    def put(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        response: str,
        usage: dict[str, int],
    ) -> None:
        """Store response in cache."""
        key = self._make_key(prompt, max_new_tokens, temperature, top_p, top_k)
        if key is None:
            return

        with self._lock:
            # Remove oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(response=response, usage=usage)
            logger.debug(f"Cached response for key {key[:16]}...")

    def clear(self) -> None:
        """Clear all cached responses."""
        with self._lock:
            self._cache.clear()
            logger.info("Response cache cleared")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class PromptCache:
    """Cache for tokenized prompts and their KV states.

    Useful for caching system prompts or common prefixes
    to avoid recomputation in chat scenarios.
    """

    def __init__(
        self,
        max_size: int = 50,
        ttl_seconds: int = 1800,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, PromptCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, prefix: str) -> str:
        """Generate cache key for a prefix."""
        return hashlib.md5(prefix.encode()).hexdigest()

    def get(self, prefix: str) -> PromptCacheEntry | None:
        """Get cached prompt entry if available."""
        key = self._make_key(prefix)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.created_at > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"Prompt cache hit for prefix: {prefix[:50]}...")
            return entry

    def put(
        self,
        prefix: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: DynamicCache | None = None,
    ) -> None:
        """Store tokenized prompt and optional KV states in cache."""
        key = self._make_key(prefix)

        with self._lock:
            # Remove oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                oldest_entry = self._cache.pop(oldest_key)
                # Clean up tensors
                del oldest_entry.input_ids
                del oldest_entry.attention_mask
                if oldest_entry.past_key_values is not None:
                    del oldest_entry.past_key_values

            # Clone tensors to avoid reference issues
            self._cache[key] = PromptCacheEntry(
                input_ids=input_ids.clone(),
                attention_mask=attention_mask.clone(),
                past_key_values=past_key_values,
            )
            logger.debug(f"Cached prompt: {prefix[:50]}...")

    def get_or_compute_kv(
        self,
        prefix: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, DynamicCache | None]:
        """Get cached KV states or compute them for a prefix."""
        entry = self.get(prefix)

        if entry is not None and entry.past_key_values is not None:
            return (
                entry.input_ids.to(device),
                entry.attention_mask.to(device),
                entry.past_key_values,
            )

        # Compute tokens and KV states
        inputs = tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass to get KV cache
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

        # Cache the results
        self.put(prefix, input_ids, attention_mask, past_key_values)

        return input_ids, attention_mask, past_key_values

    def clear(self) -> None:
        """Clear all cached prompts."""
        with self._lock:
            self._cache.clear()
            logger.info("Prompt cache cleared")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class TokenizerCache:
    """Cache for tokenized strings using LRU eviction.

    Avoids repeated tokenization of the same prompts.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

        # Create cached tokenize function
        @lru_cache(maxsize=max_size)
        def _cached_tokenize(
            text: str,
            tokenizer_id: int,
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            """Tokenize text and return as tuples (hashable for caching)."""
            return ((), ())  # Placeholder, actual implementation in tokenize()

        self._lru_cache = _cached_tokenize

    def tokenize(
        self,
        text: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
        truncation: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize text with caching."""
        # Create cache key from text and tokenizer identity
        cache_key = (text, id(tokenizer), max_length, truncation)

        with self._lock:
            # Check if we have a simple dict-based cache
            if not hasattr(self, "_dict_cache"):
                self._dict_cache: OrderedDict[
                    tuple[str, int, int | None, bool], dict[str, torch.Tensor]
                ] = OrderedDict()

            if cache_key in self._dict_cache:
                self._hits += 1
                # Move to end and return cloned tensors
                self._dict_cache.move_to_end(cache_key)
                cached = self._dict_cache[cache_key]
                return {k: v.clone() for k, v in cached.items()}

            self._misses += 1

        # Tokenize
        kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "truncation": truncation,
        }
        if max_length is not None:
            kwargs["max_length"] = max_length

        result = tokenizer(text, **kwargs)

        with self._lock:
            # Evict oldest if at capacity
            while len(self._dict_cache) >= self.max_size:
                self._dict_cache.popitem(last=False)

            # Cache the result
            self._dict_cache[cache_key] = {k: v.clone() for k, v in result.items()}

        return result

    def clear(self) -> None:
        """Clear the tokenizer cache."""
        with self._lock:
            if hasattr(self, "_dict_cache"):
                self._dict_cache.clear()
            self._lru_cache.cache_clear()
            logger.info("Tokenizer cache cleared")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            size = len(self._dict_cache) if hasattr(self, "_dict_cache") else 0
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": size,
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class CacheManager:
    """Unified manager for all inference caches."""

    def __init__(
        self,
        response_cache_size: int = 1000,
        response_cache_ttl: int = 3600,
        prompt_cache_size: int = 50,
        prompt_cache_ttl: int = 1800,
        tokenizer_cache_size: int = 1000,
        enable_response_cache: bool = True,
        enable_prompt_cache: bool = True,
        enable_tokenizer_cache: bool = True,
    ) -> None:
        self.response_cache = (
            ResponseCache(max_size=response_cache_size, ttl_seconds=response_cache_ttl)
            if enable_response_cache
            else None
        )
        self.prompt_cache = (
            PromptCache(max_size=prompt_cache_size, ttl_seconds=prompt_cache_ttl)
            if enable_prompt_cache
            else None
        )
        self.tokenizer_cache = (
            TokenizerCache(max_size=tokenizer_cache_size) if enable_tokenizer_cache else None
        )

        logger.info(
            f"CacheManager initialized: response={enable_response_cache}, "
            f"prompt={enable_prompt_cache}, tokenizer={enable_tokenizer_cache}"
        )

    def clear_all(self) -> None:
        """Clear all caches."""
        if self.response_cache:
            self.response_cache.clear()
        if self.prompt_cache:
            self.prompt_cache.clear()
        if self.tokenizer_cache:
            self.tokenizer_cache.clear()

    def stats(self) -> dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "response_cache": (self.response_cache.stats() if self.response_cache else None),
            "prompt_cache": self.prompt_cache.stats() if self.prompt_cache else None,
            "tokenizer_cache": (self.tokenizer_cache.stats() if self.tokenizer_cache else None),
        }
