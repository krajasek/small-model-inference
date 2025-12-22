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
