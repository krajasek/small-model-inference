"""Tests for the configuration module."""

import os
from unittest.mock import patch

from inference.config import Settings, get_settings


class TestSettings:
    """Test Settings class."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Clear INFERENCE_ prefixed env vars to test actual defaults
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("INFERENCE_")}
        with patch.dict(os.environ, clean_env, clear=True):
            # Create Settings with _env_file=None to skip .env file reading
            settings = Settings(_env_file=None)

        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.model_path == "/models"
        assert settings.backend == "pytorch"
        assert settings.device == "auto"
        assert settings.num_threads == 4
        assert settings.enable_torch_compile is False
        assert settings.quantization == "none"
        assert settings.use_sdpa is True
        assert settings.low_cpu_mem_usage is True
        assert settings.max_sequence_length == 2048
        assert settings.max_new_tokens == 256
        assert settings.max_parameters == 10_000_000_000
        # llama-cpp settings
        assert settings.llama_cpp_n_ctx == 2048
        assert settings.llama_cpp_n_gpu_layers == 0
        assert settings.llama_cpp_n_batch == 512
        # Langfuse settings
        assert settings.langfuse_flush_at == 1
        assert settings.langfuse_flush_interval == 1.0

    def test_model_name_optional(self) -> None:
        """Test that model_name is optional."""
        settings = Settings()
        assert settings.model_name is None

        settings_with_name = Settings(model_name="my-model")
        assert settings_with_name.model_name == "my-model"

    def test_max_memory_gb_optional(self) -> None:
        """Test that max_memory_gb is optional."""
        settings = Settings()
        assert settings.max_memory_gb is None

        settings_with_mem = Settings(max_memory_gb=8.0)
        assert settings_with_mem.max_memory_gb == 8.0

    def test_env_prefix(self) -> None:
        """Test that environment variables with INFERENCE_ prefix are loaded."""
        env_vars = {
            "INFERENCE_PORT": "9000",
            "INFERENCE_MODEL_PATH": "/custom/path",
            "INFERENCE_DEVICE": "cuda",
            "INFERENCE_NUM_THREADS": "8",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

        assert settings.port == 9000
        assert settings.model_path == "/custom/path"
        assert settings.device == "cuda"
        assert settings.num_threads == 8

    def test_device_literal_values(self) -> None:
        """Test that device accepts valid literal values."""
        for device in ["cpu", "cuda", "mps", "auto"]:
            settings = Settings(device=device)  # type: ignore[arg-type]
            assert settings.device == device

    def test_quantization_literal_values(self) -> None:
        """Test that quantization accepts valid literal values."""
        for quant in ["none", "int8"]:
            settings = Settings(quantization=quant)  # type: ignore[arg-type]
            assert settings.quantization == quant

    def test_boolean_settings(self) -> None:
        """Test boolean settings can be toggled."""
        settings = Settings(
            enable_torch_compile=True,
            use_sdpa=False,
            low_cpu_mem_usage=False,
        )

        assert settings.enable_torch_compile is True
        assert settings.use_sdpa is False
        assert settings.low_cpu_mem_usage is False


class TestGetSettings:
    """Test get_settings function."""

    def test_returns_settings_instance(self) -> None:
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_returns_new_instance_each_call(self) -> None:
        """Test that get_settings creates a new instance each call."""
        settings1 = get_settings()
        settings2 = get_settings()
        # They are equal but not the same object
        assert settings1.model_path == settings2.model_path


class TestSpeculativeDecodingSettings:
    """Test speculative decoding configuration settings."""

    def test_default_values(self) -> None:
        """Test speculative decoding defaults."""
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("INFERENCE_")}
        with patch.dict(os.environ, clean_env, clear=True):
            settings = Settings(_env_file=None)

        assert settings.enable_speculative_decoding is False
        assert settings.draft_model_path is None
        assert settings.num_speculative_tokens == 4

    def test_enable_speculative_decoding(self) -> None:
        """Test enabling speculative decoding."""
        settings = Settings(enable_speculative_decoding=True)
        assert settings.enable_speculative_decoding is True

    def test_draft_model_path(self) -> None:
        """Test setting draft model path."""
        settings = Settings(draft_model_path="/path/to/draft.gguf")
        assert settings.draft_model_path == "/path/to/draft.gguf"

    def test_num_speculative_tokens(self) -> None:
        """Test setting number of speculative tokens."""
        settings = Settings(num_speculative_tokens=8)
        assert settings.num_speculative_tokens == 8

    def test_env_vars(self) -> None:
        """Test speculative decoding env vars."""
        env_vars = {
            "INFERENCE_ENABLE_SPECULATIVE_DECODING": "true",
            "INFERENCE_DRAFT_MODEL_PATH": "/models/draft.gguf",
            "INFERENCE_NUM_SPECULATIVE_TOKENS": "6",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

        assert settings.enable_speculative_decoding is True
        assert settings.draft_model_path == "/models/draft.gguf"
        assert settings.num_speculative_tokens == 6


class TestPersistentCacheSettings:
    """Test persistent cache configuration settings."""

    def test_default_values(self) -> None:
        """Test persistent cache defaults."""
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("INFERENCE_")}
        with patch.dict(os.environ, clean_env, clear=True):
            settings = Settings(_env_file=None)

        assert settings.enable_persistent_cache is False
        assert settings.persistent_cache_dir == ".cache/kv"
        assert settings.persistent_cache_memory_size == 100
        assert settings.persistent_cache_disk_size_gb == 10.0
        assert settings.persistent_cache_chunk_size == 256
        assert settings.persistent_cache_compression == "zstd"
        assert settings.persistent_cache_async_writes is True
        assert settings.persistent_cache_warm_on_startup is True
        assert settings.persistent_cache_ttl_days == 7

    def test_enable_persistent_cache(self) -> None:
        """Test enabling persistent cache."""
        settings = Settings(enable_persistent_cache=True)
        assert settings.enable_persistent_cache is True

    def test_cache_directory(self) -> None:
        """Test setting cache directory."""
        settings = Settings(persistent_cache_dir="/tmp/my_cache")
        assert settings.persistent_cache_dir == "/tmp/my_cache"

    def test_memory_size(self) -> None:
        """Test setting memory tier size."""
        settings = Settings(persistent_cache_memory_size=200)
        assert settings.persistent_cache_memory_size == 200

    def test_disk_size(self) -> None:
        """Test setting disk cache size."""
        settings = Settings(persistent_cache_disk_size_gb=20.0)
        assert settings.persistent_cache_disk_size_gb == 20.0

    def test_chunk_size(self) -> None:
        """Test setting chunk size."""
        settings = Settings(persistent_cache_chunk_size=512)
        assert settings.persistent_cache_chunk_size == 512

    def test_compression_options(self) -> None:
        """Test compression algorithm options."""
        for compression in ["none", "zstd", "lz4"]:
            settings = Settings(persistent_cache_compression=compression)  # type: ignore[arg-type]
            assert settings.persistent_cache_compression == compression

    def test_async_writes(self) -> None:
        """Test async writes setting."""
        settings = Settings(persistent_cache_async_writes=False)
        assert settings.persistent_cache_async_writes is False

    def test_warm_on_startup(self) -> None:
        """Test warm on startup setting."""
        settings = Settings(persistent_cache_warm_on_startup=False)
        assert settings.persistent_cache_warm_on_startup is False

    def test_ttl_days(self) -> None:
        """Test TTL in days setting."""
        settings = Settings(persistent_cache_ttl_days=30)
        assert settings.persistent_cache_ttl_days == 30

    def test_env_vars(self) -> None:
        """Test persistent cache env vars."""
        env_vars = {
            "INFERENCE_ENABLE_PERSISTENT_CACHE": "true",
            "INFERENCE_PERSISTENT_CACHE_DIR": "/data/cache",
            "INFERENCE_PERSISTENT_CACHE_MEMORY_SIZE": "50",
            "INFERENCE_PERSISTENT_CACHE_DISK_SIZE_GB": "5.0",
            "INFERENCE_PERSISTENT_CACHE_COMPRESSION": "lz4",
            "INFERENCE_PERSISTENT_CACHE_TTL_DAYS": "14",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

        assert settings.enable_persistent_cache is True
        assert settings.persistent_cache_dir == "/data/cache"
        assert settings.persistent_cache_memory_size == 50
        assert settings.persistent_cache_disk_size_gb == 5.0
        assert settings.persistent_cache_compression == "lz4"
        assert settings.persistent_cache_ttl_days == 14


class TestKVCacheSettings:
    """Test KV cache configuration settings."""

    def test_default_values(self) -> None:
        """Test KV cache defaults."""
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("INFERENCE_")}
        with patch.dict(os.environ, clean_env, clear=True):
            settings = Settings(_env_file=None)

        assert settings.use_kv_cache is True
        assert settings.static_kv_cache is False

    def test_use_kv_cache(self) -> None:
        """Test enabling/disabling KV cache."""
        settings_enabled = Settings(use_kv_cache=True)
        assert settings_enabled.use_kv_cache is True

        settings_disabled = Settings(use_kv_cache=False)
        assert settings_disabled.use_kv_cache is False

    def test_static_kv_cache(self) -> None:
        """Test static KV cache setting."""
        settings = Settings(static_kv_cache=True)
        assert settings.static_kv_cache is True


class TestWebSocketSettings:
    """Test WebSocket configuration settings."""

    def test_default_values(self) -> None:
        """Test WebSocket defaults."""
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("INFERENCE_")}
        with patch.dict(os.environ, clean_env, clear=True):
            settings = Settings(_env_file=None)

        assert settings.websocket_enabled is True
        assert settings.websocket_max_connections == 100
        assert settings.websocket_ping_interval is None
        assert settings.websocket_ping_timeout is None

    def test_websocket_enabled(self) -> None:
        """Test enabling/disabling WebSocket."""
        settings = Settings(websocket_enabled=False)
        assert settings.websocket_enabled is False

    def test_websocket_max_connections(self) -> None:
        """Test max connections setting."""
        settings = Settings(websocket_max_connections=50)
        assert settings.websocket_max_connections == 50

    def test_websocket_ping_settings(self) -> None:
        """Test ping interval and timeout settings."""
        settings = Settings(
            websocket_ping_interval=30.0,
            websocket_ping_timeout=10.0,
        )
        assert settings.websocket_ping_interval == 30.0
        assert settings.websocket_ping_timeout == 10.0


class TestBatchingSettings:
    """Test batching configuration settings."""

    def test_default_values(self) -> None:
        """Test batching defaults."""
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("INFERENCE_")}
        with patch.dict(os.environ, clean_env, clear=True):
            settings = Settings(_env_file=None)

        assert settings.enable_batching is False
        assert settings.max_batch_size == 8
        assert settings.batch_wait_time_ms == 50

    def test_enable_batching(self) -> None:
        """Test enabling batching."""
        settings = Settings(enable_batching=True)
        assert settings.enable_batching is True

    def test_max_batch_size(self) -> None:
        """Test max batch size setting."""
        settings = Settings(max_batch_size=16)
        assert settings.max_batch_size == 16

    def test_batch_wait_time(self) -> None:
        """Test batch wait time setting."""
        settings = Settings(batch_wait_time_ms=100)
        assert settings.batch_wait_time_ms == 100
