FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by pdfplumber / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpoppler-cpp-dev \
        poppler-utils \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached until requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/ ./agents/
COPY backend/ ./backend/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
