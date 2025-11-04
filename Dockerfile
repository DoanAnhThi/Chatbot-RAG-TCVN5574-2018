FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    UVICORN_WORKERS=4

WORKDIR ${APP_HOME}

# System dependencies (keep minimal for a slim runtime image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY vectorstore ./vectorstore

# Ensure runtime directories exist even when baked into the image
RUN mkdir -p ./vectorstore/faiss_index

EXPOSE 8000

# Allow overriding host/port/workers via env vars while defaulting to production-ready values
CMD ["sh", "-c", "uvicorn src.main:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8000} --workers ${UVICORN_WORKERS:-4}"]
