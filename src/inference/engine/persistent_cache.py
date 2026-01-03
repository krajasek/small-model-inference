"""Persistent KV cache implementation inspired by LMCache.

This module provides tiered storage (Memory -> Disk) for KV cache states,
enabling cache persistence across server restarts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)

# Magic bytes for cache file format
CACHE_MAGIC = b"LMKV"
CACHE_VERSION = 1

# Compression type constants
COMPRESSION_NONE = 0
COMPRESSION_ZSTD = 1
COMPRESSION_LZ4 = 2


@dataclass
class CacheKey:
    """Hash-based cache key for KV cache entries."""

    prompt_hash: str
    model_name: str
    chunk_index: int = 0

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        model_name: str,
        chunk_index: int = 0,
    ) -> CacheKey:
        """Create cache key from prompt text."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:32]
        return cls(
            prompt_hash=prompt_hash,
            model_name=model_name,
            chunk_index=chunk_index,
        )

    def to_path(self, base_dir: Path) -> Path:
        """Convert cache key to file path."""
        model_hash = hashlib.md5(self.model_name.encode()).hexdigest()[:16]
        return base_dir / model_hash / self.prompt_hash / f"chunk_{self.chunk_index}.bin"

    def __hash__(self) -> int:
        return hash((self.prompt_hash, self.model_name, self.chunk_index))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CacheKey):
            return False
        return (
            self.prompt_hash == other.prompt_hash
            and self.model_name == other.model_name
            and self.chunk_index == other.chunk_index
        )


@dataclass
class CacheEntryMetadata:
    """Metadata for a cached KV entry."""

    key: CacheKey
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    num_tokens: int = 0
    num_layers: int = 0
    is_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_hash": self.key.prompt_hash,
            "model_name": self.key.model_name,
            "chunk_index": self.key.chunk_index,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "size_bytes": self.size_bytes,
            "num_tokens": self.num_tokens,
            "num_layers": self.num_layers,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntryMetadata:
        """Create from dictionary."""
        key = CacheKey(
            prompt_hash=data["prompt_hash"],
            model_name=data["model_name"],
            chunk_index=data.get("chunk_index", 0),
        )
        return cls(
            key=key,
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            size_bytes=data.get("size_bytes", 0),
            num_tokens=data.get("num_tokens", 0),
            num_layers=data.get("num_layers", 0),
            is_complete=data.get("is_complete", True),
        )


@dataclass
class CachedKVEntry:
    """A cached KV entry with data and metadata."""

    metadata: CacheEntryMetadata
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    key_cache: list[torch.Tensor]  # List of key tensors per layer
    value_cache: list[torch.Tensor]  # List of value tensors per layer


class CacheSerializer:
    """Serializes and deserializes KV cache entries."""

    def __init__(
        self,
        compression: Literal["none", "zstd", "lz4"] = "zstd",
    ) -> None:
        self.compression = compression
        self._compressor = self._get_compressor()
        self._decompressor = self._get_decompressor()

    def _get_compressor(self) -> Any:
        """Get compression function based on settings."""
        if self.compression == "zstd":
            try:
                import zstandard as zstd

                return zstd.ZstdCompressor(level=3).compress
            except ImportError:
                logger.warning("zstd not available, falling back to no compression")
                return None
        elif self.compression == "lz4":
            try:
                import lz4.frame

                return lz4.frame.compress
            except ImportError:
                logger.warning("lz4 not available, falling back to no compression")
                return None
        return None

    def _get_decompressor(self) -> Any:
        """Get decompression function based on settings."""
        if self.compression == "zstd":
            try:
                import zstandard as zstd

                return zstd.ZstdDecompressor().decompress
            except ImportError:
                return None
        elif self.compression == "lz4":
            try:
                import lz4.frame

                return lz4.frame.decompress
            except ImportError:
                return None
        return None

    def _get_compression_byte(self) -> int:
        """Get compression type byte for header."""
        if self.compression == "zstd" and self._compressor:
            return COMPRESSION_ZSTD
        elif self.compression == "lz4" and self._compressor:
            return COMPRESSION_LZ4
        return COMPRESSION_NONE

    def serialize(self, entry: CachedKVEntry) -> bytes:
        """Serialize a cached KV entry to bytes.

        Format:
        - Header (64 bytes): magic, version, compression, metadata
        - Tensor data: input_ids, attention_mask, key/value caches
        """
        # Prepare tensor data using torch serialization to buffer
        import io

        buffer = io.BytesIO()

        # Save all tensors
        torch.save(
            {
                "input_ids": entry.input_ids,
                "attention_mask": entry.attention_mask,
                "key_cache": entry.key_cache,
                "value_cache": entry.value_cache,
            },
            buffer,
        )

        tensor_data = buffer.getvalue()

        # Compress if available
        compression_byte = self._get_compression_byte()
        if self._compressor and compression_byte != COMPRESSION_NONE:
            tensor_data = self._compressor(tensor_data)

        # Build header (64 bytes)
        # Magic (4) + Version (2) + Compression (1) + dtype (1) +
        # num_layers (4) + num_tokens (4) + reserved (48)
        num_layers = len(entry.key_cache)
        num_tokens = entry.metadata.num_tokens

        header = struct.pack(
            "<4sHBBII48s",
            CACHE_MAGIC,
            CACHE_VERSION,
            compression_byte,
            0,  # dtype placeholder
            num_layers,
            num_tokens,
            b"\x00" * 48,  # reserved
        )

        return header + tensor_data

    def deserialize(self, data: bytes) -> CachedKVEntry | None:
        """Deserialize bytes to a cached KV entry."""
        if len(data) < 64:
            logger.error("Cache data too short")
            return None

        # Parse header
        header = data[:64]
        magic, version, compression, dtype, num_layers, num_tokens, _ = struct.unpack(
            "<4sHBBII48s", header
        )

        if magic != CACHE_MAGIC:
            logger.error(f"Invalid cache magic: {magic}")
            return None

        if version != CACHE_VERSION:
            logger.warning(f"Cache version mismatch: {version} != {CACHE_VERSION}")

        # Decompress tensor data
        tensor_data = data[64:]

        if compression == COMPRESSION_ZSTD:
            if not self._decompressor:
                try:
                    import zstandard as zstd

                    self._decompressor = zstd.ZstdDecompressor().decompress
                except ImportError:
                    logger.error("zstd required but not available")
                    return None
            tensor_data = self._decompressor(tensor_data)
        elif compression == COMPRESSION_LZ4:
            if not self._decompressor:
                try:
                    import lz4.frame

                    self._decompressor = lz4.frame.decompress
                except ImportError:
                    logger.error("lz4 required but not available")
                    return None
            tensor_data = self._decompressor(tensor_data)

        # Load tensors
        import io

        buffer = io.BytesIO(tensor_data)
        tensors = torch.load(buffer, weights_only=True)

        # Create metadata (will be updated by caller)
        metadata = CacheEntryMetadata(
            key=CacheKey(prompt_hash="", model_name="", chunk_index=0),
            num_tokens=num_tokens,
            num_layers=num_layers,
            size_bytes=len(data),
        )

        return CachedKVEntry(
            metadata=metadata,
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            key_cache=tensors["key_cache"],
            value_cache=tensors["value_cache"],
        )


class BaseCacheTier(ABC):
    """Abstract base class for cache tiers."""

    @abstractmethod
    def get(self, key: CacheKey) -> CachedKVEntry | None:
        """Get entry by key."""
        ...

    @abstractmethod
    def put(self, key: CacheKey, entry: CachedKVEntry) -> None:
        """Store entry."""
        ...

    @abstractmethod
    def delete(self, key: CacheKey) -> bool:
        """Delete entry by key."""
        ...

    @abstractmethod
    def exists(self, key: CacheKey) -> bool:
        """Check if key exists."""
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Get tier statistics."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        ...

    @abstractmethod
    def evict_lru(self) -> CacheKey | None:
        """Evict least recently used entry, return evicted key."""
        ...


class MemoryCacheTier(BaseCacheTier):
    """In-memory LRU cache tier."""

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[CacheKey, CachedKVEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: CacheKey) -> CachedKVEntry | None:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry = self._cache[key]
            entry.metadata.last_accessed = time.time()
            self._hits += 1
            return entry

    def put(self, key: CacheKey, entry: CachedKVEntry) -> None:
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self.evict_lru()

            self._cache[key] = entry
            self._cache.move_to_end(key)

    def delete(self, key: CacheKey) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: CacheKey) -> bool:
        with self._lock:
            return key in self._cache

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "tier": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def evict_lru(self) -> CacheKey | None:
        with self._lock:
            if not self._cache:
                return None
            key, entry = self._cache.popitem(last=False)
            # Clean up tensors
            del entry.input_ids
            del entry.attention_mask
            del entry.key_cache
            del entry.value_cache
            return key


class DiskCacheTier(BaseCacheTier):
    """Disk-based persistent cache tier with async I/O."""

    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 10.0,
        serializer: CacheSerializer | None = None,
        async_writes: bool = True,
        ttl_days: int = 7,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.serializer = serializer or CacheSerializer()
        self.async_writes = async_writes
        self.ttl_seconds = ttl_days * 24 * 60 * 60

        self._lock = threading.RLock()
        self._metadata: dict[CacheKey, CacheEntryMetadata] = {}
        self._current_size_bytes = 0
        self._hits = 0
        self._misses = 0

        # Thread pool for async I/O
        self._executor = ThreadPoolExecutor(max_workers=2) if async_writes else None

        # Initialize cache directory and load metadata
        self._init_cache_dir()

    def _init_cache_dir(self) -> None:
        """Initialize cache directory and load existing metadata."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        meta = CacheEntryMetadata.from_dict(entry_data)
                        # Check if file still exists and not expired
                        file_path = meta.key.to_path(self.cache_dir)
                        if file_path.exists():
                            age = time.time() - meta.created_at
                            if age < self.ttl_seconds:
                                self._metadata[meta.key] = meta
                                self._current_size_bytes += meta.size_bytes
                            else:
                                # Expired, remove file
                                file_path.unlink(missing_ok=True)
                logger.info(
                    f"Loaded {len(self._metadata)} entries from disk cache "
                    f"({self._current_size_bytes / 1024 / 1024:.1f} MB)"
                )
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")

    def _save_metadata(self) -> None:
        """Save metadata index to disk."""
        metadata_file = self.cache_dir / "metadata.json"
        try:
            with self._lock:
                data = {
                    "version": CACHE_VERSION,
                    "entries": [meta.to_dict() for meta in self._metadata.values()],
                }
            with open(metadata_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def get(self, key: CacheKey) -> CachedKVEntry | None:
        with self._lock:
            if key not in self._metadata:
                self._misses += 1
                return None

            meta = self._metadata[key]

            # Check TTL
            if time.time() - meta.created_at > self.ttl_seconds:
                self.delete(key)
                self._misses += 1
                return None

        # Read from disk (outside lock to avoid blocking)
        file_path = key.to_path(self.cache_dir)
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            entry = self.serializer.deserialize(data)
            if entry:
                entry.metadata = meta
                entry.metadata.last_accessed = time.time()
                with self._lock:
                    self._hits += 1
                return entry
        except Exception as e:
            logger.error(f"Failed to read cache file {file_path}: {e}")
            self.delete(key)

        with self._lock:
            self._misses += 1
        return None

    def put(self, key: CacheKey, entry: CachedKVEntry) -> None:
        """Store entry to disk."""
        # Serialize data
        try:
            data = self.serializer.serialize(entry)
        except Exception as e:
            logger.error(f"Failed to serialize cache entry: {e}")
            return

        file_path = key.to_path(self.cache_dir)
        size_bytes = len(data)

        # Evict if needed
        with self._lock:
            while self._current_size_bytes + size_bytes > self.max_size_bytes and self._metadata:
                self.evict_lru()

        def write_to_disk() -> None:
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(data)

                # Update metadata
                with self._lock:
                    entry.metadata.size_bytes = size_bytes
                    entry.metadata.key = key
                    self._metadata[key] = entry.metadata
                    self._current_size_bytes += size_bytes

                # Periodically save metadata
                if len(self._metadata) % 10 == 0:
                    self._save_metadata()

            except Exception as e:
                logger.error(f"Failed to write cache file {file_path}: {e}")

        if self._executor and self.async_writes:
            self._executor.submit(write_to_disk)
        else:
            write_to_disk()

    def delete(self, key: CacheKey) -> bool:
        with self._lock:
            if key not in self._metadata:
                return False

            meta = self._metadata.pop(key)
            self._current_size_bytes -= meta.size_bytes

        file_path = key.to_path(self.cache_dir)
        try:
            file_path.unlink(missing_ok=True)
            # Clean up empty directories
            parent = file_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache file {file_path}: {e}")
            return False

    def exists(self, key: CacheKey) -> bool:
        with self._lock:
            return key in self._metadata

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "tier": "disk",
                "entries": len(self._metadata),
                "size_bytes": self._current_size_bytes,
                "size_mb": self._current_size_bytes / 1024 / 1024,
                "max_size_gb": self.max_size_bytes / 1024 / 1024 / 1024,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def clear(self) -> None:
        with self._lock:
            self._metadata.clear()
            self._current_size_bytes = 0
            self._hits = 0
            self._misses = 0

        # Remove all cache files
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to clear cache directory: {e}")

    def evict_lru(self) -> CacheKey | None:
        with self._lock:
            if not self._metadata:
                return None

            # Find LRU entry
            lru_key = min(
                self._metadata.keys(),
                key=lambda k: self._metadata[k].last_accessed,
            )

        self.delete(lru_key)
        return lru_key

    def flush(self) -> None:
        """Flush pending writes and save metadata."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = ThreadPoolExecutor(max_workers=2)
        self._save_metadata()

    def shutdown(self) -> None:
        """Shutdown the disk tier and save state."""
        self.flush()
        if self._executor:
            self._executor.shutdown(wait=False)


class PersistentCacheManager:
    """Manages tiered KV cache with memory and disk tiers."""

    def __init__(
        self,
        cache_dir: Path,
        memory_size: int = 100,
        disk_size_gb: float = 10.0,
        compression: Literal["none", "zstd", "lz4"] = "zstd",
        async_writes: bool = True,
        ttl_days: int = 7,
        warm_on_startup: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)

        # Initialize serializer
        self.serializer = CacheSerializer(compression=compression)

        # Initialize tiers
        self.memory_tier = MemoryCacheTier(max_size=memory_size)
        self.disk_tier = DiskCacheTier(
            cache_dir=self.cache_dir,
            max_size_gb=disk_size_gb,
            serializer=self.serializer,
            async_writes=async_writes,
            ttl_days=ttl_days,
        )

        self._lock = threading.RLock()

        # Warm cache from disk
        if warm_on_startup:
            self._warm_cache()

        logger.info(
            f"PersistentCacheManager initialized: "
            f"memory_size={memory_size}, disk_size_gb={disk_size_gb}, "
            f"compression={compression}"
        )

    def _warm_cache(self, max_entries: int = 20) -> None:
        """Pre-load frequently accessed entries from disk to memory."""
        with self._lock:
            # Get most recently accessed entries from disk
            entries = sorted(
                self.disk_tier._metadata.items(),
                key=lambda x: x[1].last_accessed,
                reverse=True,
            )[:max_entries]

        loaded = 0
        for key, _ in entries:
            entry = self.disk_tier.get(key)
            if entry:
                self.memory_tier.put(key, entry)
                loaded += 1

        if loaded > 0:
            logger.info(f"Warmed cache with {loaded} entries from disk")

    def get(
        self,
        prompt: str,
        model_name: str,
    ) -> CachedKVEntry | None:
        """Get cached KV entry for prompt."""
        key = CacheKey.from_prompt(prompt, model_name)

        # Try memory first
        entry = self.memory_tier.get(key)
        if entry:
            return entry

        # Try disk
        entry = self.disk_tier.get(key)
        if entry:
            # Promote to memory
            self.memory_tier.put(key, entry)
            return entry

        return None

    def put(
        self,
        prompt: str,
        model_name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        key_cache: list[torch.Tensor],
        value_cache: list[torch.Tensor],
    ) -> None:
        """Store KV cache entry."""
        cache_key = CacheKey.from_prompt(prompt, model_name)

        # Calculate sequence length
        seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]

        metadata = CacheEntryMetadata(
            key=cache_key,
            num_tokens=seq_len,
            num_layers=len(key_cache),
        )

        entry = CachedKVEntry(
            metadata=metadata,
            input_ids=input_ids.clone().cpu(),
            attention_mask=attention_mask.clone().cpu(),
            key_cache=[k.clone().cpu() for k in key_cache],
            value_cache=[v.clone().cpu() for v in value_cache],
        )

        # Store in both tiers
        self.memory_tier.put(cache_key, entry)
        self.disk_tier.put(cache_key, entry)

    def exists(self, prompt: str, model_name: str) -> bool:
        """Check if prompt is cached."""
        key = CacheKey.from_prompt(prompt, model_name)
        return self.memory_tier.exists(key) or self.disk_tier.exists(key)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics for all tiers."""
        return {
            "memory": self.memory_tier.stats(),
            "disk": self.disk_tier.stats(),
        }

    def clear(self) -> None:
        """Clear all cache tiers."""
        self.memory_tier.clear()
        self.disk_tier.clear()
        logger.info("Persistent cache cleared")

    def flush(self) -> None:
        """Flush pending disk writes."""
        self.disk_tier.flush()

    def shutdown(self) -> None:
        """Gracefully shutdown the cache manager."""
        logger.info("Shutting down persistent cache manager")
        self.disk_tier.shutdown()
