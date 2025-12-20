# Docker Usage Guide

This guide covers running the inference server in a Docker container.

## Prerequisites

- Docker 20.10+ installed
- Docker Compose v2 (optional, for easier management)
- A compatible model downloaded locally (HuggingFace format with safetensors)

## Quick Start

### Using Docker Compose (Recommended)

1. Set the path to your model:

```bash
export MODEL_PATH=/path/to/your/model
```

2. Start the server:

```bash
docker compose up -d
```

3. Check the server is running:

```bash
curl http://localhost:8000/health
```

### Using Docker Directly

1. Build the image:

```bash
docker build -t small-model-inference:latest .
```

2. Run the container with a mounted model:

```bash
docker run -d \
  --name inference-server \
  -p 8000:8000 \
  -v /path/to/your/model:/models:ro \
  -e INFERENCE_MODEL_PATH=/models \
  -e INFERENCE_DEVICE=cpu \
  -e INFERENCE_NUM_THREADS=4 \
  small-model-inference:latest
```

## Configuration

### Environment Variables

All configuration is done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_MODEL_PATH` | `/models` | Path to model inside container |
| `INFERENCE_MODEL_NAME` | (auto) | Override model name in responses |
| `INFERENCE_DEVICE` | `cpu` | Device: cpu, cuda, mps, auto |
| `INFERENCE_NUM_THREADS` | `4` | CPU thread count |
| `INFERENCE_QUANTIZATION` | `none` | Quantization: none, int8 |
| `INFERENCE_ENABLE_TORCH_COMPILE` | `false` | Enable torch.compile |
| `INFERENCE_LOW_CPU_MEM_USAGE` | `true` | Reduce memory during loading |
| `INFERENCE_MAX_MEMORY_GB` | (none) | Optional memory limit |
| `INFERENCE_MAX_SEQUENCE_LENGTH` | `2048` | Maximum input sequence length |
| `INFERENCE_MAX_NEW_TOKENS` | `256` | Default max tokens to generate |

### Docker Compose Environment Variables

When using docker-compose, you can also set:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models` | Host path to model directory |
| `INFERENCE_PORT` | `8000` | Host port to expose |
| `MEMORY_LIMIT` | `8G` | Container memory limit |
| `MEMORY_RESERVATION` | `4G` | Container memory reservation |

## Examples

### Basic Usage with GPT-2

```bash
# Download a model first
huggingface-cli download openai-community/gpt2 --local-dir ./models/gpt2

# Run with docker-compose
MODEL_PATH=./models/gpt2 docker compose up -d

# Test it
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

### Running with Custom Settings

```bash
docker run -d \
  --name inference-server \
  -p 8000:8000 \
  -v /home/user/models/llama-3b:/models:ro \
  -e INFERENCE_MODEL_PATH=/models \
  -e INFERENCE_NUM_THREADS=8 \
  -e INFERENCE_QUANTIZATION=int8 \
  -e INFERENCE_MAX_SEQUENCE_LENGTH=4096 \
  --memory=16g \
  small-model-inference:latest
```

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

## Volume Mounts

The model directory must be mounted into the container. The mount should be read-only (`:ro`) for security:

```bash
-v /host/path/to/model:/models:ro
```

The model directory should contain:
- `config.json` - Model configuration
- `model.safetensors` or `pytorch_model.bin` - Model weights
- `tokenizer.json` or `tokenizer_config.json` - Tokenizer files

## Resource Management

### Memory

Adjust memory limits based on your model size:

| Model Size | Recommended Memory |
|------------|-------------------|
| < 1B params | 4-6 GB |
| 1-3B params | 8-12 GB |
| 3-7B params | 16-24 GB |
| 7-10B params | 24-32 GB |

### CPU Threads

Set `INFERENCE_NUM_THREADS` based on available CPU cores:

```bash
-e INFERENCE_NUM_THREADS=$(nproc)
```

## Health Checks

The container includes a health check that queries the `/health` endpoint. You can verify the server status:

```bash
docker inspect --format='{{.State.Health.Status}}' inference-server
```

## Logs

View container logs:

```bash
docker logs inference-server

# Follow logs
docker logs -f inference-server
```

## Stopping the Server

```bash
# Using docker-compose
docker compose down

# Using docker directly
docker stop inference-server
docker rm inference-server
```

## Troubleshooting

### Container exits immediately

Check logs for errors:

```bash
docker logs inference-server
```

Common issues:
- Model path not mounted correctly
- Model files missing or corrupted
- Insufficient memory

### Out of memory errors

Increase memory limit or enable quantization:

```bash
-e INFERENCE_QUANTIZATION=int8
--memory=16g
```

### Slow startup

Model loading can take 30-120 seconds depending on size. The health check has a 120-second start period to accommodate this.

### Permission denied

Ensure the model directory is readable:

```bash
chmod -R 755 /path/to/model
```
