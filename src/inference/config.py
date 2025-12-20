"""Configuration management using Pydantic Settings."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Model configuration
    model_path: str = "/models"
    model_name: str | None = None  # Optional override for model name in responses

    # Device configuration
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    num_threads: int = 4  # CPU thread count

    # Optimization settings
    enable_torch_compile: bool = False  # Disabled by default for compatibility
    quantization: Literal["none", "int8"] = "none"
    use_sdpa: bool = True  # Use scaled dot product attention

    # Memory settings
    low_cpu_mem_usage: bool = True
    max_memory_gb: float | None = None  # Optional memory limit

    # Generation defaults
    max_sequence_length: int = 2048
    max_new_tokens: int = 256

    # Model size limit (10B parameters)
    max_parameters: int = 10_000_000_000


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
