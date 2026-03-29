FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    # Install these normally to ensure all sub-dependencies match
    && pip install --no-cache-dir langchain-huggingface huggingface-hub

# ── Application code ──────────────────────────────────────────────────────────
# Copy everything needed for the app to run
COPY agents/   ./agents/
COPY backend/  ./backend/

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Start ─────────────────────────────────────────────────────────────────────
# We use the $PORT variable so Cloud Run can tell the app where to listen
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info