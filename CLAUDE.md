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
│   ├── routes.py       # API endpoints (/v1/completions, /v1/chat/completions)
│   ├── schemas.py      # OpenAI-compatible request/response schemas
│   └── dependencies.py # FastAPI dependency injection
├── models/
│   └── loader.py       # HuggingFace model loading from local paths
├── engine/
│   ├── inference.py    # Core generation logic with streaming
│   └── cpu_optimizer.py# CPU-specific optimizations (threads, quantization)
└── utils/
    └── logging.py      # Logging configuration
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion (sync + streaming) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat (sync + streaming) |
| `/v1/models` | GET | List loaded model info |
| `/health` | GET | Health check |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_MODEL_PATH` | `/models` | Path to model weights folder |
| `INFERENCE_DEVICE` | `auto` | Device: cpu, cuda, mps, auto |
| `INFERENCE_PORT` | `8000` | Server port |
| `INFERENCE_NUM_THREADS` | `4` | CPU thread count |
| `INFERENCE_QUANTIZATION` | `none` | Quantization: none, int8 |
| `INFERENCE_ENABLE_TORCH_COMPILE` | `false` | Enable torch.compile optimization |

## Key Dependencies

- PyTorch 2.2.2 (pinned for older Intel CPU support)
- transformers, accelerate for model loading
- FastAPI, uvicorn for REST API

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
