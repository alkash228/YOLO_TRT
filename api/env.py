"""Build PipelineSettings from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from app.config.settings import (
    MODELS_DIR,
    OSNET_FILENAME,
    OUTPUT_DIR,
    PipelineSettings,
)


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key, "").strip()
    return Path(raw) if raw else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().casefold()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    return float(raw)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    return int(raw)


def load_settings_from_env() -> PipelineSettings:
    models_dir = _env_path("YOLO_DRT_MODELS_DIR", MODELS_DIR)
    output_dir = _env_path("YOLO_DRT_OUTPUT_DIR", OUTPUT_DIR)

    detect = _env_path(
        "YOLO_DRT_DETECT_MODEL",
        models_dir / "YOLO" / "yolo26x-pose.pt",
    )
    cross = _env_path(
        "YOLO_DRT_CROSS_MODEL",
        models_dir / "YOLO" / "Helmet" / "helmet-26m.pt",
    )
    reid = _env_path(
        "YOLO_DRT_REID_MODEL",
        models_dir / "RD" / OSNET_FILENAME,
    )

    strategy = os.environ.get("YOLO_DRT_TRT_ENGINE_STRATEGY", "central").strip() or "central"
    rebuild = os.environ.get("YOLO_DRT_TRT_REBUILD_POLICY", "missing_only").strip() or "missing_only"
    manifest_dir = _env_path("YOLO_DRT_TRT_MANIFEST_DIR", models_dir / "TRT")

    settings = PipelineSettings(
        detect_model=detect,
        cross_check_model=cross,
        reid_model=reid,
        output_dir=output_dir,
        models_dir=models_dir,
        use_seg=_env_bool("YOLO_DRT_USE_SEG", False),
        use_reid=_env_bool("YOLO_DRT_USE_REID", True),
        use_tensorrt=_env_bool("YOLO_DRT_USE_TENSORRT", True),
        cross_check_enabled=_env_bool("YOLO_DRT_CROSS_CHECK_ENABLED", True),
        encode_mode=os.environ.get("YOLO_DRT_ENCODE_MODE", "manual").strip() or "manual",
        default_prompt=os.environ.get("YOLO_DRT_DEFAULT_PROMPT", "person").strip() or "person",
        tensorrt_imgsz=_env_int("YOLO_DRT_TRT_IMGSZ", 640),
        tensorrt_max_batch=_env_int("YOLO_DRT_TRT_MAX_BATCH", 16),
        tensorrt_fp16=_env_bool("YOLO_DRT_TRT_FP16", True),
        tensorrt_workspace_gb=_env_float("YOLO_DRT_TRT_WORKSPACE_GB", 4.0),
        tensorrt_autocast_fast=_env_bool("YOLO_DRT_TRT_AUTOCAST_FAST", False),
        tensorrt_engine_strategy=strategy,
        tensorrt_rebuild_policy=rebuild,
        tensorrt_manifest_dir=manifest_dir,
        inference_device=os.environ.get("YOLO_DRT_INFERENCE_DEVICE", "cuda").strip() or "cuda",
    )
    settings.ensure_dirs()
    return settings
