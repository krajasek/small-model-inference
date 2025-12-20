# syntax=docker/dockerfile:1

# Multi-stage build for minimal image size
# Stage 1: Build dependencies with CPU-only PyTorch
FROM python:3.11-slim AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install CPU-only PyTorch and dependencies
# --index-url: Use PyTorch CPU-only wheels (saves ~1.5GB vs CUDA version)
# --no-install-project: Skip installing the project itself
# --no-dev: Exclude dev dependencies
RUN UV_INDEX_URL=https://download.pytorch.org/whl/cpu \
    uv sync --frozen --no-dev --no-install-project

# Remove unnecessary files to reduce image size
RUN find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyo" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pth" -delete 2>/dev/null || true && \
    rm -rf /app/.venv/lib/python3.11/site-packages/torch/test && \
    rm -rf /app/.venv/lib/python3.11/site-packages/torch/include && \
    rm -rf /app/.venv/lib/python3.11/site-packages/torch/share && \
    rm -rf /app/.venv/lib/python3.11/site-packages/*.dist-info/licenses

# Stage 2: Minimal runtime image
FROM python:3.11-slim AS runtime

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --no-log-init --shell /bin/bash appuser

WORKDIR /app

# Copy only the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY src/ ./src/
COPY main.py ./

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Environment setup
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Model path - mount as volume
    INFERENCE_MODEL_PATH=/models \
    INFERENCE_HOST=0.0.0.0 \
    INFERENCE_PORT=8000 \
    INFERENCE_DEVICE=cpu \
    INFERENCE_NUM_THREADS=4 \
    INFERENCE_QUANTIZATION=none \
    INFERENCE_ENABLE_TORCH_COMPILE=false \
    INFERENCE_LOW_CPU_MEM_USAGE=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "main.py"]
