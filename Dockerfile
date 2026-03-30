

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Bake FastEmbed model — download directly to final location
RUN python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/fastembed_cache')"

COPY agents/   ./agents/
COPY backend/  ./backend/

RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

# --timeout-keep-alive 75 is critical — keeps SSE research streams alive
# without it uvicorn defaults to 5s and Cloud Run kills long streams
CMD uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --timeout-keep-alive 75