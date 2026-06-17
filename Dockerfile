# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/opt/deep-person-reid \
    YOLO_DRT_MODELS_DIR=/data/models \
    YOLO_DRT_OUTPUT_DIR=/data/output

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch nightly CUDA 12.8 (Blackwell / RTX 50xx support)
RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    python3.11 -m pip install --pre torchvision --no-deps --index-url https://download.pytorch.org/whl/nightly/cu128

COPY requirements-api.txt .
RUN python3.11 -m pip install -r requirements-api.txt
RUN git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git /opt/deep-person-reid

COPY app ./app
COPY api ./api

EXPOSE 8080

CMD ["python3.11", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
