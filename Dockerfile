# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/opt/deep-person-reid \
    YOLO_DRT_MODELS_DIR=/data/models \
    YOLO_DRT_OUTPUT_DIR=/data/output \
    YOLO_DRT_USE_SAM_IDENTITY=true \
    YOLO_DRT_USE_REID=false \
    YOLO_DRT_SAM_OSNET_REENTRY=false \
    YOLO_DRT_USE_OFFLINE_TRACKLET_LINK=true \
    YOLO_DRT_TRACKLET_LINK_USE_REID=true \
    YOLO_DRT_USE_SEG=false \
    YOLO_DRT_GPU_FULL_BATCH=false \
    YOLO_DRT_PRELOAD_VIDEO=false \
    YOLO_DRT_FRAME_SOURCE_MODE=windowed \
    YOLO_DRT_FRAME_STRIDE=0 \
    YOLO_DRT_INFER_BATCH_SIZE=64 \
    YOLO_DRT_REALTIME_MODE=true \
    YOLO_DRT_TRACK_BUFFER=300 \
    YOLO_DRT_USE_TENSORRT=true \
    YOLO_DRT_CROSS_CHECK_ENABLED=true \
    YOLO_DRT_ENCODE_MODE=manual \
    YOLO_DRT_INFERENCE_DEVICE=cuda

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

# Модели (.pt/.pth и опционально TRT/*.engine) — положи в ./models перед docker compose build
COPY models /data/models

COPY app ./app
COPY api ./api
COPY config ./config

EXPOSE 8080

CMD ["python3.11", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
