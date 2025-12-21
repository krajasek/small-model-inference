# Small Model Inference

A CPU-friendly inference server for serving small language models (up to 10B parameters) via OpenAI-compatible REST API.

## Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI's `/v1/completions` and `/v1/chat/completions` endpoints
- **Streaming Support** - Real-time token streaming for both completion types
- **CPU Optimized** - Designed for efficient CPU inference with threading and quantization
- **Multiple Caching Layers** - Response, prompt/KV, and tokenizer caching for faster responses
- **Quantization** - int8 and int4 quantization support for reduced memory and faster inference
- **Continuous Batching** - Optional request batching for high-throughput scenarios
- **Speculative Decoding** - Use draft models to accelerate generation
- **Docker Ready** - Multi-stage Dockerfile with CPU-only PyTorch for minimal image size

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Docker Deployment](#docker-deployment)
- [Examples](#examples)

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run the server
INFERENCE_MODEL_PATH=/path/to/model uv run python main.py

# Test it
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

---

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- A HuggingFace-compatible model (safetensors format recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/krajasek/small-model-inference.git
cd small-model-inference

# Install dependencies
uv sync

# Install dev dependencies (for testing/linting)
uv sync --extra dev

# Download a model (example: GPT-2)
huggingface-cli download openai-community/gpt2 --local-dir ./models/gpt2

# Run the server
INFERENCE_MODEL_PATH=./models/gpt2 uv run python main.py
```

The server starts on `http://localhost:8000` by default.

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion (sync + streaming) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat (sync + streaming) |
| `/v1/models` | GET | List loaded model info |
| `/v1/cache/stats` | GET | Cache statistics and hit rates |
| `/v1/cache/clear` | POST | Clear all caches |
| `/health` | GET | Health check |

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Input text (completions endpoint) |
| `messages` | array | required | Chat messages (chat endpoint) |
| `max_tokens` | int | 256 | Maximum tokens to generate (1-4096) |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.9 | Nucleus sampling threshold (0.0-1.0) |
| `top_k` | int | 50 | Top-k sampling |
| `do_sample` | bool | true | Enable sampling (false = greedy) |
| `repetition_penalty` | float | 1.1 | Repetition penalty (1.0-2.0) |
| `stream` | bool | false | Enable streaming response |

### Response Format

**Completion Response:**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "local-model",
  "choices": [{"index": 0, "text": "Generated text...", "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35}
}
```

**Chat Completion Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "local-model",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Response..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 15, "completion_tokens": 30, "total_tokens": 45}
}
```

---

## Configuration

All configuration is done via environment variables with the `INFERENCE_` prefix.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_MODEL_PATH` | `/models` | Path to model weights folder |
| `INFERENCE_MODEL_NAME` | (auto) | Override model name in responses |
| `INFERENCE_DEVICE` | `auto` | Device: cpu, cuda, mps, auto |
| `INFERENCE_PORT` | `8000` | Server port |
| `INFERENCE_HOST` | `0.0.0.0` | Server host |
| `INFERENCE_NUM_THREADS` | `4` | CPU thread count |
| `INFERENCE_MAX_SEQUENCE_LENGTH` | `2048` | Maximum input sequence length |
| `INFERENCE_MAX_NEW_TOKENS` | `256` | Default max tokens to generate |

### Optimization Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_QUANTIZATION` | `none` | Quantization: `none`, `int8`, `int4` |
| `INFERENCE_ENABLE_TORCH_COMPILE` | `false` | Enable torch.compile optimization |
| `INFERENCE_LOW_CPU_MEM_USAGE` | `true` | Reduce memory during loading |

### Caching Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ENABLE_RESPONSE_CACHE` | `true` | Cache responses for identical requests |
| `INFERENCE_RESPONSE_CACHE_SIZE` | `1000` | Max cached responses |
| `INFERENCE_RESPONSE_CACHE_TTL` | `3600` | Response cache TTL (seconds) |
| `INFERENCE_ENABLE_PROMPT_CACHE` | `true` | Cache prompt KV states |
| `INFERENCE_PROMPT_CACHE_SIZE` | `50` | Max cached prompts |
| `INFERENCE_ENABLE_TOKENIZER_CACHE` | `true` | Cache tokenization results |
| `INFERENCE_USE_KV_CACHE` | `true` | Enable KV caching during generation |
| `INFERENCE_STATIC_KV_CACHE` | `false` | Use static cache allocation |

### Batching Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ENABLE_BATCHING` | `false` | Enable continuous batching |
| `INFERENCE_MAX_BATCH_SIZE` | `8` | Max requests per batch |
| `INFERENCE_BATCH_WAIT_TIME_MS` | `50` | Max wait time for batch to fill |

### Speculative Decoding

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ENABLE_SPECULATIVE_DECODING` | `false` | Enable speculative decoding |
| `INFERENCE_DRAFT_MODEL_PATH` | | Path to smaller draft model |
| `INFERENCE_NUM_SPECULATIVE_TOKENS` | `4` | Tokens to speculate per step |

---

## Performance Optimization

### Caching

The server includes multiple caching layers enabled by default:

1. **Response Cache** - Caches complete responses for deterministic requests (temperature ≤ 0.1)
2. **Prompt Cache** - Caches tokenized prompts and KV states for repeated prefixes
3. **Tokenizer Cache** - Caches tokenization results to avoid repeated processing

Monitor cache performance:
```bash
curl http://localhost:8000/v1/cache/stats
```

Clear caches if needed:
```bash
curl -X POST http://localhost:8000/v1/cache/clear
```

### Quantization

Use int8 quantization for ~2x speedup with minimal quality loss:
```bash
INFERENCE_QUANTIZATION=int8 uv run python main.py
```

### Recommended Settings by Model Size

| Model Size | Memory | Threads | Quantization |
|------------|--------|---------|--------------|
| < 1B params | 4-6 GB | 4 | none |
| 1-3B params | 8-12 GB | 4-8 | int8 |
| 3-7B params | 16-24 GB | 8 | int8 |
| 7-10B params | 24-32 GB | 8+ | int8 |

---

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Set your model path
export MODEL_PATH=/path/to/your/model

# Start the server
docker compose up -d

# Check health
curl http://localhost:8000/health
```

### Build and Run Directly

```bash
# Build the image (uses CPU-only PyTorch for minimal size)
docker build -t small-model-inference:latest .

# Run with mounted model
docker run -d \
  --name inference-server \
  -p 8000:8000 \
  -v /path/to/model:/models:ro \
  -e INFERENCE_NUM_THREADS=4 \
  -e INFERENCE_QUANTIZATION=int8 \
  small-model-inference:latest
```

### Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models` | Host path to model (docker-compose) |
| `INFERENCE_PORT` | `8000` | Host port to expose |
| `MEMORY_LIMIT` | `8G` | Container memory limit |
| `MEMORY_RESERVATION` | `4G` | Container memory reservation |

### Using .env File

Create a `.env` file:
```env
MODEL_PATH=/home/user/models/my-model
INFERENCE_PORT=8080
INFERENCE_NUM_THREADS=8
INFERENCE_QUANTIZATION=int8
MEMORY_LIMIT=16G
```

Then run:
```bash
docker compose up -d
```

### Resource Management

Set memory limits based on model size:
```bash
docker run -d \
  --memory=16g \
  -e INFERENCE_NUM_THREADS=$(nproc) \
  ...
```

### Troubleshooting

**Container exits immediately:**
```bash
docker logs inference-server
```

**Out of memory:**
```bash
# Enable quantization and increase memory limit
docker run -e INFERENCE_QUANTIZATION=int8 --memory=16g ...
```

**Slow startup:**
Model loading takes 30-120 seconds. The health check has a 120-second start period.

---

## Examples

### Python with httpx

```python
import httpx

# Text completion
response = httpx.post(
    "http://localhost:8000/v1/completions",
    json={"prompt": "The capital of France is", "max_tokens": 50},
    timeout=60.0,
)
print(response.json()["choices"][0]["text"])

# Chat completion
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
        ],
        "max_tokens": 150,
    },
    timeout=60.0,
)
print(response.json()["choices"][0]["message"]["content"])
```

### Python with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

# Standard request
response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Count to 5."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### curl

```bash
# Text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 50}'

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi!"}], "max_tokens": 50}'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi!"}], "stream": true}'
```

### Streaming with Python

```python
import httpx
import json

with httpx.stream(
    "POST",
    "http://localhost:8000/v1/chat/completions",
    json={"messages": [{"role": "user", "content": "Tell me a joke."}], "stream": True},
    timeout=60.0,
) as response:
    for line in response.iter_lines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunk = json.loads(line[6:])
            content = chunk["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)
```

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Linting
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/
```

## License

MIT
