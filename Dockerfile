# ============================================================
# Stage 1: Build & Dependency Resolution
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install minimal build tools for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for deterministic, ultra-fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY requirements.txt .

# Install dependencies into virtual environment using CPU wheels
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ============================================================
# Stage 2: Production Minimal Runtime
# ============================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Security: create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Configure Python environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FAISS_INDEX_DIR="/app/data/index" \
    PORT=8000

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source, migrations, and config
COPY --chown=appuser:appuser ./app ./app
COPY --chown=appuser:appuser ./alembic ./alembic
COPY --chown=appuser:appuser alembic.ini .
COPY --chown=appuser:appuser entrypoint.sh .

# Create required persistent data directories with non-root ownership
RUN mkdir -p /app/data/index && chown -R appuser:appuser /app/data && chmod +x /app/entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
