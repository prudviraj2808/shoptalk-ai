#!/bin/bash
set -e

# 1. Critical: Tell Python to look at the root directory for imports
export PYTHONPATH=$PYTHONPATH:/app

echo "🚀 Starting ShopTalk Combined Services..."

# 2. Start FastAPI (Backend) 
# We call 'uvicorn' directly because it's already in the PATH from the .venv
uvicorn main:app --host 0.0.0.0 --port 8000 &

# 3. Start Google ADK UI (Frontend/Testing)
# We call 'adk' directly
echo "🎨 Starting ADK Web UI on port 5000..."
adk web --host 0.0.0.0 --port 5000 agents/