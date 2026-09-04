FROM python:3.11-slim

# GitPython shells out to the real git binary (used by differential scans).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Render injects $PORT. Seed + train once on boot (both steps are idempotent:
# the seed script skips rows it already inserted, and retraining skips when
# there is nothing new to learn), then hand off to uvicorn via exec so signals
# reach the server for graceful shutdown.
CMD ["sh", "-c", "python seed_training_data.py && python -m backend.ml.retraining || true; exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
