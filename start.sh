#!/bin/bash
set -e

export PYTHONPATH=$PYTHONPATH:/app

echo "Starting ShopTalk FastAPI Backend on port 8000..."

exec /app/.venv/bin/uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --timeout-keep-alive 30 \
  --log-level info
  # ✅ workers=1 is correct — GPU state cannot be shared across processes
