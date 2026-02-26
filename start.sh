#!/bin/bash
set -e

# 1. Critical: Tell Python to look at the root directory for imports

export PYTHONPATH=$PYTHONPATH:/app

echo "🚀 Starting ShopTalk Combined Services..."

# 2. Start FastAPI (Backend) in the background
# We use 'uv run' to ensure the virtual environment is used
uv run uvicorn main:app --host 0.0.0.0 --port 8000 &

# 3. Start Google ADK UI (Frontend/Testing) in the foreground
# This keeps the container running
echo "🎨 Starting ADK Web UI on port 5000..."
uv run adk web --host 0.0.0.0 --port 5000 agents/