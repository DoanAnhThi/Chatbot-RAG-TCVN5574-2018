FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create required dirs
RUN mkdir -p /app/app /app/data /app/vectorstore

# Copy source
COPY app /app/app

EXPOSE 8000

# Default command
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000
