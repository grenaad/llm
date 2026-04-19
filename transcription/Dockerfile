# Stage 1: Build SolidJS frontend with Bun
FROM oven/bun:1 AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/bun.lock* ./
RUN bun install
COPY frontend/ .
RUN bun run build

# Stage 2: Python runtime with CUDA
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libsndfile1 \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy and install Python dependencies with uv (pre-resolved lockfile)
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy backend code
COPY backend/ .

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./static

EXPOSE 7860

CMD ["python3", "main.py"]
