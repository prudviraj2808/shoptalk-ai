#!/bin/bash
# set -e

# 1. Critical: Tell Python to look at the root directory for imports
export PYTHONPATH=$PYTHONPATH:/app

echo "🚀 Starting ShopTalk FastAPI Backend on port 8000..."

# 2. Start FastAPI (Backend) 
# Use the direct path to the venv bin folder. 
# REMOVE the '&' so the container stays alive.
# Use 'exec' so uvicorn catches the 'docker stop' signal.
exec /app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1