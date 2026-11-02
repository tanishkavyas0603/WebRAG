#!/bin/sh
set -e

echo "[STARTUP] Applying database schema migrations..."
alembic upgrade head

echo "[STARTUP] Starting WebRAG FastAPI server on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WORKERS:-2}"
