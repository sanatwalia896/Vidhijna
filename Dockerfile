FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MALLOC_TRIM_THRESHOLD_=65536

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
# poppler-utils  → pdfplumber page rendering
# libglib2.0-0   → Pillow JPEG support
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps langchain-huggingface \
 && pip install --no-cache-dir huggingface-hub

# ── Application code ──────────────────────────────────────────────────────────
COPY agents/   ./agents/
COPY backend/  ./backend/

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd --no-create-home --shell /bin/false appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "65", \
     "--log-level", "warning"]