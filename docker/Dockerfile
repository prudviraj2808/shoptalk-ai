# --- Stage 1: Build Stage ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# 1. Force uv to create the venv in a predictable location
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 2. Install build-essential for FAISS/ML compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev libomp-dev git \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy only dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# 4. Install dependencies (frozen)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# --- Stage 2: Runtime Stage ---
FROM python:3.12-slim-bookworm
WORKDIR /app

# 5. Install runtime libraries for FAISS (smaller than build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 libomp5 && rm -rf /var/lib/apt/lists/*

# 6. COPY YOUR CODE FIRST 
# This ensures the folder structure exists before the venv is moved in
COPY . .

# 7. COPY THE VENV FROM THE BUILDER
# This overwrites any local .venv that might have accidentally slipped in
COPY --from=builder /app/.venv /app/.venv

# 8. Set System Paths
# Putting the venv bin at the start of PATH makes 'uvicorn' accessible
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1

# 9. Fix permissions for the startup script
RUN chmod +x start.sh

EXPOSE 8000

# 10. Start the backend
CMD ["./start.sh"]