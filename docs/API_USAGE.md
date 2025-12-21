# API Usage Guide

This guide provides code examples for using the inference API endpoints.

## Setup

Start the server:
```bash
INFERENCE_MODEL_PATH=/path/to/model INFERENCE_DEVICE=cpu uv run python main.py
```

The server runs on `http://localhost:8000` by default.

### With Optimizations

Enable caching and other optimizations for better performance:
```bash
INFERENCE_MODEL_PATH=/path/to/model \
INFERENCE_DEVICE=cpu \
INFERENCE_ENABLE_RESPONSE_CACHE=true \
INFERENCE_ENABLE_PROMPT_CACHE=true \
INFERENCE_QUANTIZATION=int8 \
uv run python main.py
```

## Health Check

```python
import httpx

response = httpx.get("http://localhost:8000/health")
print(response.json())
# {"status": "healthy", "model_loaded": true, "device": "cpu"}
```

## List Models

```python
import httpx

response = httpx.get("http://localhost:8000/v1/models")
print(response.json())
# {"object": "list", "data": [{"id": "local-model", "object": "model", ...}]}
```

---

## Text Completions

### Standard Mode

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.7,
    },
    timeout=60.0,
)

result = response.json()
print(result["choices"][0]["text"])
```

### Streaming Mode

```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "Write a haiku about programming:",
        "max_tokens": 100,
        "temperature": 0.8,
        "stream": True,
    },
    timeout=60.0,
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[6:]  # Remove "data: " prefix
            if data == "[DONE]":
                break
            import json
            chunk = json.loads(data)
            print(chunk["choices"][0]["text"], end="", flush=True)
    print()  # Newline at end
```

---

## Chat Completions (OpenAI-Compatible)

### Standard Mode

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
        ],
        "max_tokens": 150,
        "temperature": 0.7,
    },
    timeout=60.0,
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Streaming Mode

```python
import httpx
import json

with httpx.stream(
    "POST",
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "Explain recursion in 3 sentences."},
        ],
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": True,
    },
    timeout=60.0,
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            content = chunk["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)
    print()
```

---

## Using with OpenAI Python Client

The chat completions endpoint is compatible with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # API key not required for local server
)

# Standard mode
response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
)
print(response.choices[0].message.content)

# Streaming mode
stream = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "Count from 1 to 5."}
    ],
    max_tokens=100,
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

---

## Using with curl

### Text Completion

```bash
# Standard
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'

# Streaming
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50, "stream": true}'
```

### Chat Completion

```bash
# Standard
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi!"}],
    "max_tokens": 50
  }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi!"}],
    "max_tokens": 50,
    "stream": true
  }'
```

---

## Request Parameters

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

---

## Response Format

### Completion Response
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "text": "Generated text here...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  }
}
```

### Chat Completion Response
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response text here..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 30,
    "total_tokens": 45
  }
}
```

---

## Cache Management

The server includes multiple caching layers for improved performance. Use these endpoints to monitor and manage caches.

### Get Cache Statistics

```python
import httpx

response = httpx.get("http://localhost:8000/v1/cache/stats")
print(response.json())
```

Response:
```json
{
  "caches": {
    "response_cache": {
      "size": 42,
      "max_size": 1000,
      "hits": 156,
      "misses": 89,
      "hit_rate": 0.636
    },
    "prompt_cache": {
      "size": 5,
      "max_size": 50,
      "hits": 230,
      "misses": 12,
      "hit_rate": 0.95
    },
    "tokenizer_cache": {
      "size": 128,
      "max_size": 1000,
      "hits": 445,
      "misses": 128,
      "hit_rate": 0.776
    }
  },
  "model_info": {
    "model_name": "local-model",
    "device": "cpu",
    "quantization": "int8",
    "kv_cache_enabled": true,
    "static_kv_cache": false,
    "speculative_decoding": false
  }
}
```

### Clear All Caches

```python
import httpx

response = httpx.post("http://localhost:8000/v1/cache/clear")
print(response.json())
# {"status": "ok", "message": "All caches cleared"}
```

### Using curl

```bash
# Get cache stats
curl http://localhost:8000/v1/cache/stats

# Clear caches
curl -X POST http://localhost:8000/v1/cache/clear
```

---

## Performance Optimization

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **Caching** | | |
| `INFERENCE_ENABLE_RESPONSE_CACHE` | `true` | Cache complete responses for identical requests |
| `INFERENCE_RESPONSE_CACHE_SIZE` | `1000` | Max cached responses |
| `INFERENCE_RESPONSE_CACHE_TTL` | `3600` | Response cache TTL (seconds) |
| `INFERENCE_ENABLE_PROMPT_CACHE` | `true` | Cache tokenized prompts and KV states |
| `INFERENCE_PROMPT_CACHE_SIZE` | `50` | Max cached prompts |
| `INFERENCE_ENABLE_TOKENIZER_CACHE` | `true` | Cache tokenization results |
| **KV Cache** | | |
| `INFERENCE_USE_KV_CACHE` | `true` | Enable KV caching during generation |
| `INFERENCE_STATIC_KV_CACHE` | `false` | Use static cache allocation |
| **Quantization** | | |
| `INFERENCE_QUANTIZATION` | `none` | Quantization: `none`, `int8`, `int4` |
| **Batching** | | |
| `INFERENCE_ENABLE_BATCHING` | `false` | Enable continuous batching |
| `INFERENCE_MAX_BATCH_SIZE` | `8` | Max requests per batch |
| `INFERENCE_BATCH_WAIT_TIME_MS` | `50` | Max wait time for batch |
| **Speculative Decoding** | | |
| `INFERENCE_ENABLE_SPECULATIVE_DECODING` | `false` | Enable speculative decoding |
| `INFERENCE_DRAFT_MODEL_PATH` | | Path to draft model |
| `INFERENCE_NUM_SPECULATIVE_TOKENS` | `4` | Tokens to speculate per step |

### Optimization Tips

1. **Response Caching**: Automatically caches responses for deterministic requests (temperature ≤ 0.1). Great for repeated queries.

2. **Prompt Caching**: Caches system prompts and their KV states. Speeds up chat applications with consistent system prompts.

3. **Quantization**: Use `int8` for ~2x speedup with minimal quality loss. Use `int4` for maximum speed (requires bitsandbytes).

4. **Batching**: Enable for high-throughput scenarios. Adds latency for single requests but improves overall throughput.

5. **Speculative Decoding**: Use a smaller draft model to speed up generation. Best with models that have matching tokenizers.
