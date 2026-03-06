# syntax=docker/dockerfile:1
# ------------------------------------------------------------------
# Multi-arch compatible image (linux/amd64 + linux/arm64 / aarch64)
# python:3.11-slim is officially published for arm64 by Docker Hub.
# ------------------------------------------------------------------
FROM python:3.11-slim

# Ensure Python output is sent straight to the container logs (no buffering)
ENV PYTHONUNBUFFERED=1

# Keep pip quiet and avoid writing .pyc files to the layer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application source
COPY main.py .

# Create a non-root user and the /data directory for the persistent ID cache.
# Mount /data as a Coolify volume so synced_ids.txt survives container restarts.
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /data \
    && chown appuser:appuser /data

# Cache files are bootstrapped automatically on first run.
# Mount /data as a persistent volume so they survive container restarts.

USER appuser

VOLUME ["/data"]

CMD ["python", "main.py"]
