# Small Model Inference

A CPU-friendly inference server for serving small language models (up to 10B parameters) via OpenAI-compatible REST API.

## Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI's `/v1/completions` and `/v1/chat/completions` endpoints
- **WebSocket API** - Low-latency streaming with protobuf wire format for high-throughput applications
- **Dual Backend Support** - Choose between PyTorch/HuggingFace or llama-cpp-python for inference
- **Streaming Support** - Real-time token streaming via HTTP SSE or WebSocket
- **CPU Optimized** - Designed for efficient CPU inference with threading and quantization
- **Multiple Caching Layers** - Response, prompt/KV, and tokenizer caching for faster responses
- **Persistent KV Cache** - LMCache-inspired disk persistence for KV cache states across restarts
- **Quantization** - int8 and int4 quantization support for reduced memory and faster inference
- **Continuous Batching** - Optional request batching for high-throughput scenarios
- **Speculative Decoding** - Use draft models to accelerate generation
- **Observability** - Langfuse integration for tracing, metrics, and monitoring
- **Docker Ready** - Multi-stage Dockerfile with CPU-only PyTorch for minimal image size

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Backend Selection](#backend-selection)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Persistent KV Cache](#persistent-kv-cache)
- [Observability](#observability)
- [Docker Deployment](#docker-deployment)
- [Examples](#examples)
- [Speculative Decoding](#speculative-decoding)
- [CLI Client](#cli-client)

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

# Download a model (example: TinyLlama-1.1B-Chat)
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./models/tinyllama-1.1b-chat

# Run the server
INFERENCE_MODEL_PATH=./models/tinyllama-1.1b-chat uv run python main.py
```

The server starts on `http://localhost:8000` by default.

---

## Backend Selection

The server supports two inference backends:

### PyTorch Backend (Default)

Uses HuggingFace Transformers for inference. Best for:
- HuggingFace model format (safetensors/bin files)
- Advanced features (speculative decoding, prompt caching)
- GPU acceleration with CUDA/MPS

```bash
# PyTorch backend (default)
INFERENCE_BACKEND=pytorch \
INFERENCE_MODEL_PATH=./models/tinyllama-1.1b-chat \
uv run python main.py
```

### llama-cpp Backend

Uses llama-cpp-python for optimized CPU inference. Best for:
- GGUF quantized models
- Faster CPU inference with better streaming
- Lower memory usage

```bash
# Install llama-cpp-python dependency
uv sync --extra llama-cpp

# Download a GGUF model
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --local-dir ./models

# Run with llama-cpp backend
INFERENCE_BACKEND=llama-cpp \
INFERENCE_MODEL_PATH=./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
uv run python main.py
```

### Backend Comparison

| Feature | PyTorch | llama-cpp |
|---------|---------|-----------|
| Model format | HuggingFace (safetensors) | GGUF |
| CPU streaming | Basic | Optimized |
| GPU support | CUDA, MPS | Optional GPU layers |
| Memory usage | Higher | Lower (quantized) |
| Prompt caching | Yes | Internal |
| Speculative decoding | Yes | Yes |
| Response caching | Yes | Yes |

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion (sync + streaming) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat (sync + streaming) |
| `/v1/stream` | WebSocket | Streaming inference with protobuf wire format |
| `/v1/models` | GET | List loaded model info |
| `/v1/cache/stats` | GET | Cache statistics and hit rates |
| `/v1/cache/clear` | POST | Clear all caches |
| `/v1/cache/persistent/flush` | POST | Force flush persistent cache to disk |
| `/v1/cache/persistent/warm` | POST | Trigger persistent cache warming |
| `/v1/tracing/status` | GET | Tracing status and configuration |
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
| `user` | string | null | User ID for tracing (optional) |

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
| `INFERENCE_BACKEND` | `pytorch` | Backend: `pytorch` or `llama-cpp` |
| `INFERENCE_MODEL_PATH` | `/models` | Path to model weights folder or GGUF file |
| `INFERENCE_MODEL_NAME` | (auto) | Override model name in responses |
| `INFERENCE_DEVICE` | `auto` | Device: cpu, cuda, mps, auto |
| `INFERENCE_PORT` | `8000` | Server port |
| `INFERENCE_HOST` | `0.0.0.0` | Server host |
| `INFERENCE_NUM_THREADS` | `4` | CPU thread count |
| `INFERENCE_MAX_SEQUENCE_LENGTH` | `2048` | Maximum input sequence length |
| `INFERENCE_MAX_NEW_TOKENS` | `256` | Default max tokens to generate |

### llama-cpp Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_LLAMA_CPP_N_CTX` | `2048` | Context size |
| `INFERENCE_LLAMA_CPP_N_GPU_LAYERS` | `0` | GPU layers (0 = CPU only) |
| `INFERENCE_LLAMA_CPP_N_BATCH` | `512` | Batch size for prompt processing |

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

### Persistent KV Cache Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ENABLE_PERSISTENT_CACHE` | `false` | Enable persistent KV cache to disk |
| `INFERENCE_PERSISTENT_CACHE_DIR` | `.cache/kv` | Directory for disk cache |
| `INFERENCE_PERSISTENT_CACHE_MEMORY_SIZE` | `100` | Max entries in memory tier |
| `INFERENCE_PERSISTENT_CACHE_DISK_SIZE_GB` | `10.0` | Max disk cache size (GB) |
| `INFERENCE_PERSISTENT_CACHE_CHUNK_SIZE` | `256` | Tokens per cache chunk |
| `INFERENCE_PERSISTENT_CACHE_COMPRESSION` | `zstd` | Compression: `none`, `zstd`, `lz4` |
| `INFERENCE_PERSISTENT_CACHE_ASYNC_WRITES` | `true` | Use async disk writes |
| `INFERENCE_PERSISTENT_CACHE_WARM_ON_STARTUP` | `true` | Warm cache on server start |
| `INFERENCE_PERSISTENT_CACHE_TTL_DAYS` | `7` | Disk entry TTL in days |

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

### Observability (Langfuse)

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_ENABLE_TRACING` | `false` | Enable Langfuse tracing |
| `INFERENCE_LANGFUSE_PUBLIC_KEY` | | Langfuse public key |
| `INFERENCE_LANGFUSE_SECRET_KEY` | | Langfuse secret key |
| `INFERENCE_LANGFUSE_HOST` | | Custom Langfuse host URL |
| `INFERENCE_LANGFUSE_DEBUG` | `false` | Enable debug logging |

Or use standard Langfuse environment variables:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`

### WebSocket Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_WEBSOCKET_ENABLED` | `true` | Enable WebSocket endpoint |
| `INFERENCE_WEBSOCKET_MAX_CONNECTIONS` | `100` | Max concurrent WebSocket connections |
| `INFERENCE_WEBSOCKET_PING_INTERVAL` | `None` | Ping interval in seconds (None = disabled) |
| `INFERENCE_WEBSOCKET_PING_TIMEOUT` | `None` | Ping timeout in seconds (None = disabled) |

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

## Persistent KV Cache

The persistent KV cache provides LMCache-inspired disk persistence for KV cache states, enabling faster inference for repeated prompts even across server restarts. This is especially useful for:

- **System prompts** - Cache the KV states for common system prompts
- **Few-shot examples** - Reuse cached states for repeated example prefixes
- **Chat continuations** - Speed up multi-turn conversations with cached context
- **Server restarts** - Preserve cache across deployments and restarts

### Architecture

The persistent cache uses a two-tier storage architecture:

```
┌─────────────────────────────────────────┐
│         PersistentCacheManager          │
│  - Coordinates tiered cache operations  │
│  - Handles cache warming on startup     │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│ MemoryCacheTier│    │ DiskCacheTier  │
│ (LRU, fast)   │    │ (persistent)   │
└───────────────┘    └────────────────┘
```

- **Memory Tier**: Fast LRU cache for frequently accessed entries
- **Disk Tier**: Compressed persistent storage with async I/O

### Backend Support

| Backend | Cache Type | Description |
|---------|------------|-------------|
| PyTorch | DynamicCache | Full KV cache serialization with compression |
| llama-cpp | Model State | State-based caching using save/load_state |

### Enabling Persistent Cache

```bash
# Basic setup with defaults
INFERENCE_ENABLE_PERSISTENT_CACHE=true \
INFERENCE_MODEL_PATH=./models/tinyllama-1.1b-chat \
uv run python main.py
```

### Production Configuration

```bash
# Production setup with tuned settings
INFERENCE_ENABLE_PERSISTENT_CACHE=true \
INFERENCE_PERSISTENT_CACHE_DIR=/data/kv-cache \
INFERENCE_PERSISTENT_CACHE_MEMORY_SIZE=200 \
INFERENCE_PERSISTENT_CACHE_DISK_SIZE_GB=50.0 \
INFERENCE_PERSISTENT_CACHE_COMPRESSION=zstd \
INFERENCE_PERSISTENT_CACHE_TTL_DAYS=14 \
INFERENCE_MODEL_PATH=./models/mistral-7b-instruct \
uv run python main.py
```

### Docker with Persistent Cache

Mount a volume for cache persistence across container restarts:

```bash
docker run -d \
  --name inference-server \
  -p 8000:8000 \
  -v /path/to/model:/models:ro \
  -v /path/to/cache:/app/.cache/kv \
  -e INFERENCE_ENABLE_PERSISTENT_CACHE=true \
  -e INFERENCE_PERSISTENT_CACHE_DIR=/app/.cache/kv \
  small-model-inference:latest
```

### Monitoring Cache Performance

```bash
# Get cache statistics including persistent cache
curl http://localhost:8000/v1/cache/stats
```

Example response:
```json
{
  "caches": {
    "prompt_cache": {
      "size": 25,
      "max_size": 50,
      "hits": 150,
      "misses": 30,
      "hit_rate": 0.833,
      "persistent_hits": 45,
      "persistent_backend": {
        "memory_entries": 25,
        "disk_entries": 120,
        "disk_size_mb": 245.5,
        "memory_hit_rate": 0.75,
        "disk_hit_rate": 0.92
      }
    }
  }
}
```

### Cache Management Endpoints

```bash
# Force flush pending writes to disk
curl -X POST http://localhost:8000/v1/cache/persistent/flush

# Manually trigger cache warming (loads disk entries to memory)
curl -X POST http://localhost:8000/v1/cache/persistent/warm

# Clear all caches including persistent
curl -X POST http://localhost:8000/v1/cache/clear
```

### Python Usage Example

```python
import httpx

# First request - computes and caches KV states
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "What is Python?"},
        ],
        "max_tokens": 100,
    },
    timeout=60.0,
)
print("First request:", response.json()["usage"])

# Second request with same system prompt - uses cached KV states
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "How do I read a file?"},
        ],
        "max_tokens": 100,
    },
    timeout=60.0,
)
print("Second request (cached):", response.json()["usage"])

# Check cache stats
stats = httpx.get("http://localhost:8000/v1/cache/stats").json()
print("Persistent cache hits:", stats["caches"]["prompt_cache"]["persistent_hits"])
```

### Compression Options

| Algorithm | Compression Ratio | Speed | Use Case |
|-----------|-------------------|-------|----------|
| `none` | 1.0x | Fastest | Development, SSDs |
| `zstd` | ~3-4x | Fast | Production (default) |
| `lz4` | ~2x | Faster | High throughput, lower latency |

```bash
# Use LZ4 for lower latency
INFERENCE_PERSISTENT_CACHE_COMPRESSION=lz4 uv run python main.py

# Disable compression for fastest I/O
INFERENCE_PERSISTENT_CACHE_COMPRESSION=none uv run python main.py
```

### Cache Warming

On server startup, the cache automatically warms by loading recently used entries from disk into memory. This provides immediate cache hits for common prompts.

```bash
# Disable automatic warming (manual control)
INFERENCE_PERSISTENT_CACHE_WARM_ON_STARTUP=false uv run python main.py
```

Then trigger warming manually when ready:
```bash
curl -X POST http://localhost:8000/v1/cache/persistent/warm
```

### Storage Format

The cache uses a binary format with the following header:

```
Header (64 bytes):
├── Magic: "LMKV" (4 bytes)
├── Version (2 bytes)
├── Compression (1 byte): 0=none, 1=zstd, 2=lz4
├── Tensor dtype (1 byte)
├── Num layers (4 bytes)
├── Num heads (4 bytes)
├── Head dim (4 bytes)
├── Seq length (4 bytes)
├── Checksum xxhash64 (8 bytes)
└── Reserved (32 bytes)

Body: Layer-wise K/V tensors (compressed)
```

### Testing the Persistent Cache

Follow these steps to verify the persistent KV cache is working:

**Step 1: Start the server with persistent cache enabled**

```bash
INFERENCE_ENABLE_PERSISTENT_CACHE=true \
INFERENCE_PERSISTENT_CACHE_DIR=.cache/kv \
INFERENCE_MODEL_PATH=/path/to/your/model \
uv run python main.py
```

**Step 2: Make requests with repeated prefixes**

```bash
# First request - computes and caches KV states
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "What is Python?"}
    ],
    "max_tokens": 100
  }'

# Second request with same system prompt - should use cached KV states
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "How do I read a file?"}
    ],
    "max_tokens": 100
  }'
```

**Step 3: Check cache statistics**

```bash
curl http://localhost:8000/v1/cache/stats | python -m json.tool
```

Look for these fields:
- `persistent_hits` - Cache hits from persistent storage
- `persistent_backend.memory_entries` - Entries in memory tier
- `persistent_backend.disk_entries` - Entries on disk
- `persistent_backend.disk_size_mb` - Disk usage

**Step 4: Test persistence across restarts**

```bash
# Stop the server (Ctrl+C), then restart it
INFERENCE_ENABLE_PERSISTENT_CACHE=true \
INFERENCE_PERSISTENT_CACHE_DIR=.cache/kv \
INFERENCE_MODEL_PATH=/path/to/your/model \
uv run python main.py

# Make a request with the same system prompt - should load from disk
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "What is recursion?"}
    ],
    "max_tokens": 100
  }'

# Verify disk cache was used
curl http://localhost:8000/v1/cache/stats
```

**Step 5: Inspect disk cache files**

```bash
# View cached files
ls -la .cache/kv/

# Check disk usage
du -sh .cache/kv/
```

### Recommended Settings

| Scenario | Memory Size | Disk Size | Compression | TTL |
|----------|-------------|-----------|-------------|-----|
| Development | 50 | 5 GB | none | 1 day |
| Production (small) | 100 | 20 GB | zstd | 7 days |
| Production (large) | 200 | 100 GB | zstd | 14 days |
| High throughput | 500 | 50 GB | lz4 | 7 days |

---

## Observability

The server integrates with [Langfuse](https://langfuse.com) for observability and tracing. When enabled, it captures:

- **Generation latency** - Total time and time-to-first-token for streaming
- **Token usage** - Prompt tokens, completion tokens, and tokens per second
- **Model parameters** - Temperature, top_p, top_k, and other generation settings
- **Cache metrics** - Cache hits/misses for response, prompt, and tokenizer caches
- **Errors** - Exception details and error traces

### Enabling Tracing

```bash
INFERENCE_ENABLE_TRACING=true \
INFERENCE_LANGFUSE_PUBLIC_KEY=pk-lf-... \
INFERENCE_LANGFUSE_SECRET_KEY=sk-lf-... \
uv run python main.py
```

Or use standard Langfuse environment variables:
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-... \
LANGFUSE_SECRET_KEY=sk-lf-... \
INFERENCE_ENABLE_TRACING=true \
uv run python main.py
```

### User Tracking

Pass a `user` field in requests to track usage by user:

```python
response = httpx.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "Hello",
        "max_tokens": 50,
        "user": "user-123",  # Tracked in Langfuse
    },
)
```

### Check Tracing Status

```bash
curl http://localhost:8000/v1/tracing/status
# {"enabled": true, "provider": "langfuse", "model_name": "local-model"}
```

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

### WebSocket with Protobuf

The WebSocket API provides low-latency streaming with binary protobuf wire format. It's ideal for high-throughput applications and native clients.

```python
import asyncio
import websockets
import betterproto
from inference.proto import (
    ClientMessage,
    CompletionRequest,
    GenerationParams,
    ServerMessage,
)

async def stream_completion():
    uri = "ws://localhost:8000/v1/stream"

    async with websockets.connect(uri) as websocket:
        # Create request
        request = ClientMessage(
            request_id="req-001",
            completion=CompletionRequest(
                prompt="The capital of France is",
                params=GenerationParams(
                    max_tokens=50,
                    temperature=0.7,
                ),
            ),
        )

        # Send binary protobuf
        await websocket.send(bytes(request))

        # Receive streaming responses
        while True:
            data = await websocket.recv()
            msg = ServerMessage().parse(data)

            payload_type, _ = betterproto.which_one_of(msg, "payload")

            if payload_type == "chunk":
                content = msg.chunk.choice.delta.content
                if content:
                    print(content, end="", flush=True)
            elif payload_type == "complete":
                print(f"\n[Done - {msg.complete.usage.total_tokens} tokens]")
                break
            elif payload_type == "error":
                print(f"Error: {msg.error.message}")
                break

asyncio.run(stream_completion())
```

### WebSocket Chat Completion

```python
import asyncio
import websockets
import betterproto
from inference.proto import (
    ChatCompletionRequest,
    ChatMessage,
    ClientMessage,
    GenerationParams,
    ServerMessage,
)

async def stream_chat():
    uri = "ws://localhost:8000/v1/stream"

    async with websockets.connect(uri) as websocket:
        request = ClientMessage(
            request_id="chat-001",
            chat_completion=ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content="You are a helpful assistant."),
                    ChatMessage(role="user", content="What is Python?"),
                ],
                params=GenerationParams(max_tokens=150, temperature=0.7),
            ),
        )

        await websocket.send(bytes(request))

        while True:
            data = await websocket.recv()
            msg = ServerMessage().parse(data)

            payload_type, _ = betterproto.which_one_of(msg, "payload")

            if payload_type == "chunk":
                if msg.chunk.choice.delta.role:
                    print(f"[{msg.chunk.choice.delta.role}]: ", end="")
                content = msg.chunk.choice.delta.content
                if content:
                    print(content, end="", flush=True)
            elif payload_type == "complete":
                print("\n[Done]")
                break
            elif payload_type == "error":
                print(f"Error: {msg.error.message}")
                break

asyncio.run(stream_chat())
```

### Speculative Decoding

Speculative decoding uses a smaller "draft" model to predict multiple tokens ahead, which are then verified by the main model. This can significantly speed up generation for compatible model pairs.

Both backends support speculative decoding:
- **PyTorch**: Uses HuggingFace model format
- **llama-cpp**: Uses GGUF model format (recommended for CPU inference)

#### Setup with llama-cpp (GGUF Models)

```bash
# Install llama-cpp-python
uv sync --extra llama-cpp

# Download Mistral 7B Instruct GGUF as the main model
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --local-dir ./models

# Download TinyLlama 1.1B Chat GGUF as the draft model
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --local-dir ./models
```

#### Running the Server (llama-cpp)

```bash
# Start with speculative decoding enabled (llama-cpp backend)
INFERENCE_BACKEND=llama-cpp \
INFERENCE_MODEL_PATH=./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
INFERENCE_ENABLE_SPECULATIVE_DECODING=true \
INFERENCE_DRAFT_MODEL_PATH=./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
INFERENCE_NUM_THREADS=8 \
INFERENCE_LLAMA_CPP_N_CTX=4096 \
uv run python main.py
```

#### Setup with PyTorch (HuggingFace Models)

```bash
# Download Mistral 7B Instruct as the main model
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
  --local-dir ./models/mistral-7b-instruct

# Download TinyLlama 1.1B Chat as the draft model
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir ./models/tinyllama-1.1b-chat
```

#### Running the Server (PyTorch)

```bash
# Start with speculative decoding enabled (PyTorch backend)
INFERENCE_BACKEND=pytorch \
INFERENCE_MODEL_PATH=./models/mistral-7b-instruct \
INFERENCE_ENABLE_SPECULATIVE_DECODING=true \
INFERENCE_DRAFT_MODEL_PATH=./models/tinyllama-1.1b-chat \
INFERENCE_NUM_SPECULATIVE_TOKENS=4 \
INFERENCE_DEVICE=cpu \
INFERENCE_NUM_THREADS=8 \
uv run python main.py
```

#### REST API Example

```bash
# Chat completion with speculative decoding
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

#### WebSocket API Example (Low-Latency Streaming)

The WebSocket API provides the lowest latency for streaming responses with speculative decoding. It uses a binary protobuf wire format for efficient communication.

```python
import asyncio
import betterproto
import websockets
from inference.proto import (
    ChatCompletionRequest,
    ChatMessage,
    ClientMessage,
    GenerationParams,
    ServerMessage,
)


async def stream_with_speculative_decoding():
    """Stream chat completion using WebSocket with speculative decoding."""
    uri = "ws://localhost:8000/v1/stream"

    async with websockets.connect(uri) as websocket:
        # Create chat completion request
        request = ClientMessage(
            request_id="spec-decode-001",
            chat_completion=ChatCompletionRequest(
                messages=[
                    ChatMessage(
                        role="system",
                        content="You are a helpful coding assistant.",
                    ),
                    ChatMessage(
                        role="user",
                        content="Write a Python function to calculate fibonacci numbers.",
                    ),
                ],
                params=GenerationParams(
                    max_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                ),
            ),
        )

        # Send binary protobuf request
        await websocket.send(bytes(request))

        # Stream responses
        full_response = []
        while True:
            data = await websocket.recv()
            msg = ServerMessage().parse(data)

            payload_type, _ = betterproto.which_one_of(msg, "payload")

            if payload_type == "chunk":
                content = msg.chunk.choice.delta.content
                if content:
                    print(content, end="", flush=True)
                    full_response.append(content)
            elif payload_type == "complete":
                usage = msg.complete.usage
                print(f"\n\n[Completed: {usage.total_tokens} tokens]")
                break
            elif payload_type == "error":
                print(f"\nError: {msg.error.message}")
                break

        return "".join(full_response)


# Run the example
asyncio.run(stream_with_speculative_decoding())
```

#### WebSocket Text Completion Example

```python
import asyncio
import betterproto
import websockets
from inference.proto import (
    ClientMessage,
    CompletionRequest,
    GenerationParams,
    ServerMessage,
)


async def stream_completion():
    """Stream text completion using WebSocket with speculative decoding."""
    uri = "ws://localhost:8000/v1/stream"

    async with websockets.connect(uri) as websocket:
        request = ClientMessage(
            request_id="completion-001",
            completion=CompletionRequest(
                prompt="The key benefits of speculative decoding are:",
                params=GenerationParams(
                    max_tokens=150,
                    temperature=0.7,
                ),
            ),
        )

        await websocket.send(bytes(request))

        while True:
            data = await websocket.recv()
            msg = ServerMessage().parse(data)

            payload_type, _ = betterproto.which_one_of(msg, "payload")

            if payload_type == "chunk":
                content = msg.chunk.choice.delta.content
                if content:
                    print(content, end="", flush=True)
            elif payload_type == "complete":
                print(f"\n[Done - {msg.complete.usage.total_tokens} tokens]")
                break
            elif payload_type == "error":
                print(f"Error: {msg.error.message}")
                break


asyncio.run(stream_completion())
```

#### Python OpenAI Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

# Speculative decoding works transparently with the API
response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about programming."},
    ],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

#### Configuration Tips

| Setting | Recommendation |
|---------|----------------|
| `NUM_SPECULATIVE_TOKENS` | Start with 4, increase to 6-8 for longer generations |
| Draft model size | ~10-20% of main model size works well |
| Memory | Ensure enough RAM for both models (main + draft) |
| Backend | Use llama-cpp for best CPU performance with GGUF models |

---

## CLI Client

An interactive CLI client is included for conversational chat using the low-latency WebSocket API.

### Installation

The CLI is installed automatically with the package:

```bash
uv sync
```

### Usage

```bash
# Basic usage (connects to localhost:8000)
uv run inference-cli

# Or run as module
uv run python -m inference.cli

# With options
uv run inference-cli \
    --host 192.168.1.100 \
    --port 8080 \
    --system "You are a helpful coding assistant." \
    --max-tokens 512 \
    --temperature 0.8
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | | `localhost` | Server host |
| `--port` | | `8000` | Server port |
| `--system` | `-s` | | System prompt for the conversation |
| `--max-tokens` | `-m` | `256` | Maximum tokens to generate |
| `--temperature` | `-t` | `0.7` | Sampling temperature |
| `--top-p` | | `0.9` | Top-p sampling parameter |
| `--top-k` | | `50` | Top-k sampling parameter |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history |
| `/stats` | Show session statistics |
| `/quit` | Exit the chat |

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
