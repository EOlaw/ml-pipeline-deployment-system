# ─── Stage 1: Base image ─────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Stage 2: Dependencies ───────────────────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Stage 3: Application ────────────────────────────────────────────────────
FROM dependencies AS app

# Copy full project
COPY . .

# Ensure runtime directories exist
RUN mkdir -p data/raw data/processed data/external models logs

# Non-root user for security
RUN useradd --create-home --shell /bin/bash mluser \
    && chown -R mluser:mluser /app
USER mluser

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run via Gunicorn (production WSGI server)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "logs/gunicorn_access.log", \
     "--error-logfile", "logs/gunicorn_error.log", \
     "src.api.app:app"]
