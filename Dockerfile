FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies - REMOVED --no-deps for stability
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-deps langchain-huggingface

COPY agents/   ./agents/
COPY backend/  ./backend/

# Create a proper home directory for the user so HF models can download
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

# Use the dynamic PORT variable
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1