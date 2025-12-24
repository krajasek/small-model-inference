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

## WebSocket API (Protobuf)

The server supports WebSocket connections for streaming inference with protobuf wire format. This provides lower latency than HTTP SSE and efficient binary serialization.

### Endpoint

- **URL**: `ws://localhost:8000/v1/stream`
- **Protocol**: Binary protobuf messages
- **Mode**: Streaming only (use REST API for non-streaming)

### Proto Schema

The protobuf schema is defined in `src/inference/proto/inference.proto`. Key message types:

```protobuf
// Client sends this
message ClientMessage {
    string request_id = 1;  // For correlating responses
    oneof payload {
        CompletionRequest completion = 2;
        ChatCompletionRequest chat_completion = 3;
    }
}

// Server responds with stream of these
message ServerMessage {
    string request_id = 1;  // Echoed from request
    oneof payload {
        StreamChunk chunk = 2;      // Token chunks
        StreamComplete complete = 3; // Final message
        ErrorResponse error = 4;     // Error response
    }
}
```

### Python Example

```python
import asyncio
import websockets
from inference.proto import (
    ClientMessage,
    CompletionRequest,
    GenerationParams,
    ServerMessage,
)
import betterproto

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

            # Check which payload type using betterproto
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

        print()

asyncio.run(stream_completion())
```

### Chat Completion Example

```python
import asyncio
import websockets
from inference.proto import (
    ChatCompletionRequest,
    ChatMessage,
    ClientMessage,
    GenerationParams,
    ServerMessage,
)
import betterproto

async def stream_chat():
    uri = "ws://localhost:8000/v1/stream"

    async with websockets.connect(uri) as websocket:
        # Create chat request
        request = ClientMessage(
            request_id="chat-001",
            chat_completion=ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content="You are a helpful assistant."),
                    ChatMessage(role="user", content="What is Python?"),
                ],
                params=GenerationParams(
                    max_tokens=150,
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
                # First chunk has role, subsequent have content
                if msg.chunk.choice.delta.role:
                    print(f"[{msg.chunk.choice.delta.role}]: ", end="")
                content = msg.chunk.choice.delta.content
                if content:
                    print(content, end="", flush=True)
            elif payload_type == "complete":
                print(f"\n[Done]")
                break
            elif payload_type == "error":
                print(f"Error: {msg.error.message}")
                break

asyncio.run(stream_chat())
```

### Multiple Requests (Connection Reuse)

WebSocket connections can be reused for multiple sequential requests:

```python
import asyncio
import websockets
from inference.proto import (
    ClientMessage,
    CompletionRequest,
    GenerationParams,
    ServerMessage,
)
import betterproto

async def multi_request():
    uri = "ws://localhost:8000/v1/stream"

    prompts = [
        "The sky is",
        "Water is",
        "Fire is",
    ]

    async with websockets.connect(uri) as websocket:
        for i, prompt in enumerate(prompts):
            request = ClientMessage(
                request_id=f"req-{i}",
                completion=CompletionRequest(
                    prompt=prompt,
                    params=GenerationParams(max_tokens=20),
                ),
            )

            await websocket.send(bytes(request))

            print(f"Prompt: {prompt}")
            print("Response: ", end="")

            while True:
                data = await websocket.recv()
                msg = ServerMessage().parse(data)
                payload_type, _ = betterproto.which_one_of(msg, "payload")

                if payload_type == "chunk" and msg.chunk.choice.delta.content:
                    print(msg.chunk.choice.delta.content, end="", flush=True)
                elif payload_type == "complete":
                    print("\n")
                    break
                elif payload_type == "error":
                    print(f"Error: {msg.error.message}\n")
                    break

asyncio.run(multi_request())
```

### Response Message Types

| Message Type | Description |
|--------------|-------------|
| `StreamChunk` | Token chunk with `delta.content` (text) or `delta.role` (first chat chunk) |
| `StreamComplete` | Final message with `usage` stats (prompt_tokens, completion_tokens, total_tokens) |
| `ErrorResponse` | Error with `code` (400/500), `message`, and `type` |

### WebSocket vs REST SSE

| Feature | WebSocket (Protobuf) | REST (SSE) |
|---------|---------------------|------------|
| Wire format | Binary protobuf | JSON text |
| Connection | Persistent, bidirectional | New per request |
| Multiple requests | Reuse connection | New connection each |
| Latency | Lower (binary, no HTTP overhead) | Higher |
| Browser support | Requires protobuf library | Native EventSource |
| Best for | High-throughput apps, native clients | Web apps, simple integrations |

### WebSocket Caching

The WebSocket endpoint includes its own caching layers for improved performance:

| Cache Type | Description | Benefit |
|------------|-------------|---------|
| Formatted Prompt Cache | Caches the result of formatting chat messages into prompt strings | Avoids repeated string formatting for identical message sequences |
| System Prompt Cache | Caches formatted system prompts separately | Reuses formatted system prompts across different user messages |

These caches are particularly useful for chat applications where:
- The same system prompt is used across many requests
- Users send similar or identical message sequences

WebSocket cache statistics are included in the `/v1/cache/stats` endpoint response under `websocket_caches`.

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
