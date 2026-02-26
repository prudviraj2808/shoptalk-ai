# --- Stage 1: Build Stage ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Install build-essential for FAISS/ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev libomp-dev git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies using cache
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

# --- Stage 2: Runtime Stage ---
FROM python:3.12-slim-bookworm
WORKDIR /app

# Install runtime libs for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 libomp5 && rm -rf /var/lib/apt/lists/*

# Copy the environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set paths
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1

# COPY EVERYTHING from the root context into /app
COPY . .

# CRITICAL: Grant permission to the start script
RUN chmod +x start.sh

EXPOSE 8000 5000

# Start via the script
CMD ["./start.sh"]