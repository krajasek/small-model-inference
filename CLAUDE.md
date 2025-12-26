# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CPU-friendly inference layer for serving small language models (up to 10B parameters) via REST API.

**Python is the primary language for this project.**

## Development Setup

This project uses `uv` for dependency management (Python 3.11+).

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --extra dev

# Run the inference server
INFERENCE_MODEL_PATH=/path/to/model uv run python main.py

# Run with specific settings
INFERENCE_DEVICE=cpu INFERENCE_PORT=8000 uv run python main.py
```

## Code Quality

- **Linting & Formatting**: Use Ruff for both linting and formatting
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- **Type Checking**: Use mypy
  ```bash
  uv run mypy src/
  ```
- **Pre-commit Hooks**: Enforce code hygiene via pre-commit hooks

## Git Workflow

Use `gh` CLI for git management (PRs, issues, etc.).

## Architecture

```
src/inference/
├── app.py              # FastAPI application factory with lifespan
├── config.py           # Pydantic Settings configuration
├── api/
│   ├── routes.py       # REST API endpoints (/v1/completions, /v1/chat/completions)
│   ├── websocket.py    # WebSocket endpoint (/v1/stream) with protobuf
│   ├── schemas.py      # OpenAI-compatible request/response schemas
│   └── dependencies.py # FastAPI dependency injection
├── proto/              # Protocol buffer definitions
│   ├── inference.proto # Protobuf schema for WebSocket API
│   └── inference/      # Generated betterproto dataclasses
├── backends/           # Pluggable inference backends
│   ├── base.py         # InferenceBackend protocol and GenerationConfig
│   ├── pytorch.py      # PyTorch/HuggingFace backend
│   └── llamacpp.py     # llama-cpp-python backend (GGUF models)
├── models/
│   └── loader.py       # HuggingFace model loading from local paths
├── engine/
│   ├── inference.py    # Legacy engine (used by PyTorch backend)
│   ├── cache.py        # Response, prompt, and tokenizer caching
│   ├── batching.py     # Continuous batching for throughput
│   └── cpu_optimizer.py# CPU-specific optimizations (threads, quantization)
├── observability/
│   └── tracing.py      # Langfuse integration for metrics and tracing
└── utils/
    └── logging.py      # Logging configuration
```

## Backend Selection

The server supports two backends, selectable via `INFERENCE_BACKEND`:

- **pytorch** (default): Uses HuggingFace Transformers. Supports HuggingFace model format.
- **llama-cpp**: Uses llama-cpp-python. Supports GGUF model format with optimized CPU streaming.

```bash
# PyTorch backend (default)
INFERENCE_BACKEND=pytorch INFERENCE_MODEL_PATH=/path/to/hf/model uv run python main.py

# llama-cpp backend (for GGUF models)
uv sync --extra llama-cpp  # Install optional dependency first
INFERENCE_BACKEND=llama-cpp INFERENCE_MODEL_PATH=/path/to/model.gguf uv run python main.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion (sync + streaming) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat (sync + streaming) |
| `/v1/stream` | WebSocket | Streaming inference with protobuf wire format |
| `/v1/models` | GET | List loaded model info |
| `/v1/cache/stats` | GET | Cache statistics and hit rates |
| `/v1/cache/clear` | POST | Clear all caches |
| `/v1/tracing/status` | GET | Tracing status and configuration |
| `/health` | GET | Health check |

## Environment Variables

### Core Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_BACKEND` | `pytorch` | Backend: pytorch or llama-cpp |
| `INFERENCE_MODEL_PATH` | `/models` | Path to model weights folder or GGUF file |
| `INFERENCE_DEVICE` | `auto` | Device: cpu, cuda, mps, auto |
| `INFERENCE_PORT` | `8000` | Server port |
| `INFERENCE_NUM_THREADS` | `4` | CPU thread count |
| `INFERENCE_QUANTIZATION` | `none` | Quantization: none, int8, int4 (PyTorch) |
| `INFERENCE_ENABLE_TORCH_COMPILE` | `false` | Enable torch.compile optimization |

### llama-cpp Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_LLAMA_CPP_N_CTX` | `2048` | Context size |
| `INFERENCE_LLAMA_CPP_N_GPU_LAYERS` | `0` | GPU layers (0 = CPU only) |
| `INFERENCE_LLAMA_CPP_N_BATCH` | `512` | Batch size for prompt processing |

### Caching Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ENABLE_RESPONSE_CACHE` | `true` | Cache responses for identical requests |
| `INFERENCE_ENABLE_PROMPT_CACHE` | `true` | Cache prompt KV states |
| `INFERENCE_ENABLE_TOKENIZER_CACHE` | `true` | Cache tokenization results |
| `INFERENCE_USE_KV_CACHE` | `true` | Enable KV caching in generation |

### Advanced Optimizations
| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ENABLE_BATCHING` | `false` | Enable continuous batching |
| `INFERENCE_MAX_BATCH_SIZE` | `8` | Max batch size |
| `INFERENCE_ENABLE_SPECULATIVE_DECODING` | `false` | Use draft model |
| `INFERENCE_DRAFT_MODEL_PATH` | | Path to draft model |

### WebSocket Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_WEBSOCKET_ENABLED` | `true` | Enable WebSocket endpoint |
| `INFERENCE_WEBSOCKET_MAX_CONNECTIONS` | `100` | Max concurrent WebSocket connections |
| `INFERENCE_WEBSOCKET_PING_INTERVAL` | `None` | Ping interval in seconds (None = disabled) |
| `INFERENCE_WEBSOCKET_PING_TIMEOUT` | `None` | Ping timeout in seconds (None = disabled) |

### Observability (Langfuse)
| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ENABLE_TRACING` | `false` | Enable Langfuse tracing |
| `INFERENCE_LANGFUSE_PUBLIC_KEY` | | Langfuse public key |
| `INFERENCE_LANGFUSE_SECRET_KEY` | | Langfuse secret key |
| `INFERENCE_LANGFUSE_HOST` | | Custom Langfuse host |

## Key Dependencies

- PyTorch 2.2.2 (pinned for older Intel CPU support)
- transformers, accelerate for model loading
- FastAPI, uvicorn for REST API and WebSocket
- betterproto for protobuf serialization (WebSocket API)
- langfuse for observability and tracing
- llama-cpp-python (optional) for GGUF model support

## Docker

Build and run with Docker:

```bash
# Build the image
docker build -t small-model-inference:latest .

# Run with a mounted model
docker run -p 8000:8000 -v /path/to/model:/models:ro small-model-inference:latest

# Or use docker-compose
MODEL_PATH=/path/to/model docker compose up -d
```

## Documentation

- [API Usage Guide](docs/API_USAGE.md) - Code examples for all endpoints
- [Docker Guide](docs/DOCKER.md) - Container deployment instructions
