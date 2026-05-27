#!/usr/bin/env bash
# FastAPI HTTP + Gemini Live WebSocket tutor (ppt_extract uses LibreOffice in-container).
set -euo pipefail

cd /app

# Run on 8080 only (local + Azure).
PORT="8080"

exec uvicorn app.server:app --host 0.0.0.0 --port "${PORT}"
