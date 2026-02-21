FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Ensure uv-installed binaries are in the PATH
    PATH="/app/.venv/bin:$PATH" \
    # Critical: Ensure Python searches the root for your modules
    PYTHONPATH="/app"

WORKDIR /app

# Install system dependencies
# Added: libopenblas-dev and libomp-dev for FAISS support
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Copy dependency metadata first to leverage Docker caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
# We do this before copying the rest of the code to speed up rebuilds
RUN uv sync --frozen --no-dev

# Copy the actual project structure
# Based on your flat layout, we copy these specific directories
COPY agents ./agents
COPY tools ./tools
COPY utils ./utils
COPY model ./model
COPY main.py .env ./

# Expose ports for both the API (8000) and ADK UI (5000)
EXPOSE 8000
EXPOSE 5000

# Default command: Start the ADK UI pointing to the agents folder
CMD ["uv", "run", "adk", "web", "--host", "0.0.0.0", "--port", "5000", "agents/"]