# syntax=docker/dockerfile:1.6

# ---------- Builder stage ----------
FROM python:3.12-slim AS builder

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a separate, relocatable venv
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source
COPY app ./app

# ---------- Runtime stage ----------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the prepared virtualenv and application code from the builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app ./app

# Ensure the venv is on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Bind to 0.0.0.0 so the container is reachable from outside
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]