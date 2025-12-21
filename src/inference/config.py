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
    quantization: Literal["none", "int8", "int4"] = "none"
    use_sdpa: bool = True  # Use scaled dot product attention

    # Memory settings
    low_cpu_mem_usage: bool = True
    max_memory_gb: float | None = None  # Optional memory limit

    # Generation defaults
    max_sequence_length: int = 2048
    max_new_tokens: int = 256

    # Model size limit (10B parameters)
    max_parameters: int = 10_000_000_000

    # KV Cache settings
    use_kv_cache: bool = True  # Enable KV caching during generation
    static_kv_cache: bool = False  # Use static cache allocation (faster but fixed size)

    # Response caching
    enable_response_cache: bool = True
    response_cache_size: int = 1000  # Max cached responses
    response_cache_ttl: int = 3600  # Cache TTL in seconds

    # Prompt/prefix caching
    enable_prompt_cache: bool = True
    prompt_cache_size: int = 50  # Max cached prompts
    prompt_cache_ttl: int = 1800  # Cache TTL in seconds

    # Tokenizer caching
    enable_tokenizer_cache: bool = True
    tokenizer_cache_size: int = 1000  # Max cached tokenizations

    # Batching settings
    enable_batching: bool = False  # Disabled by default (adds latency for single requests)
    max_batch_size: int = 8
    batch_wait_time_ms: int = 50  # Max time to wait for batch to fill

    # Speculative decoding
    enable_speculative_decoding: bool = False
    draft_model_path: str | None = None  # Path to smaller draft model
    num_speculative_tokens: int = 4  # Tokens to speculate per step

    # Observability settings (Langfuse)
    enable_tracing: bool = False  # Enable Langfuse tracing
    langfuse_public_key: str | None = None  # Or set LANGFUSE_PUBLIC_KEY env var
    langfuse_secret_key: str | None = None  # Or set LANGFUSE_SECRET_KEY env var
    langfuse_host: str | None = None  # Custom host (default: cloud.langfuse.com)
    langfuse_debug: bool = False  # Enable Langfuse debug logging
    langfuse_flush_at: int = 15  # Number of events before flushing
    langfuse_flush_interval: float = 10.0  # Seconds between flushes


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
