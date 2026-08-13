"""Build PipelineSettings from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from app.config.settings import (
    MODELS_DIR,
    OUTPUT_DIR,
    SOLIDER_FILENAME,
    UPLOAD_DIR,
    WORK_DIR,
    PipelineSettings,
)


def _env_path(key: str, default: Path | None) -> Path | None:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    return Path(raw)


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
    from app.config.ui_fast_profile import load_ui_fast_pipeline

    models_dir = _env_path("YOLO_DRT_MODELS_DIR", MODELS_DIR) or MODELS_DIR
    output_dir = _env_path("YOLO_DRT_OUTPUT_DIR", OUTPUT_DIR) or OUTPUT_DIR
    work_dir = _env_path("YOLO_DRT_WORK_DIR", WORK_DIR) or WORK_DIR
    upload_dir = _env_path("YOLO_DRT_UPLOAD_DIR", UPLOAD_DIR) or UPLOAD_DIR

    # Person detect (bbox). Helmet / ReID paths are separate roles — never swap them.
    detect = _env_path(
        "YOLO_DRT_DETECT_MODEL",
        models_dir / "YOLO" / "yolo26x.pt",
    ) or (models_dir / "YOLO" / "yolo26x.pt")
    # Accessories-only cross-check model (no ReID / stable_id on these dets).
    cross = _env_path(
        "YOLO_DRT_CROSS_MODEL",
        models_dir / "YOLO" / "Helmet" / "helmet-26m.pt",
    ) or (models_dir / "YOLO" / "Helmet" / "helmet-26m.pt")
    # Person identity weights (Pass2 / SAM re-entry) — not used for helmet boxes.
    reid = _env_path(
        "YOLO_DRT_REID_MODEL",
        models_dir / "RD" / SOLIDER_FILENAME,
    ) or (models_dir / "RD" / SOLIDER_FILENAME)
    sam_model = _env_path("YOLO_DRT_SAM_MODEL", None)

    strategy = os.environ.get("YOLO_DRT_TRT_ENGINE_STRATEGY", "central").strip() or "central"
    rebuild = os.environ.get("YOLO_DRT_TRT_REBUILD_POLICY", "missing_only").strip() or "missing_only"
    manifest_dir = _env_path("YOLO_DRT_TRT_MANIFEST_DIR", models_dir / "TRT") or (
        models_dir / "TRT"
    )

    # Start from baked UI fast profile, then env overrides (compose / Dockerfile).
    baked = load_ui_fast_pipeline()

    settings = PipelineSettings(
        detect_model=detect,
        cross_check_model=cross,
        reid_model=reid,
        reid_backend=(
            os.environ.get(
                "YOLO_DRT_REID_BACKEND",
                str(baked.get("reid_backend", "solider")),
            ).strip()
            or "solider"
        ),
        output_dir=output_dir,
        work_dir=work_dir,
        upload_dir=upload_dir,
        models_dir=models_dir,
        use_seg=_env_bool("YOLO_DRT_USE_SEG", bool(baked.get("use_seg", False))),
        use_sam_identity=_env_bool(
            "YOLO_DRT_USE_SAM_IDENTITY", bool(baked.get("use_sam_identity", True))
        ),
        use_reid=_env_bool("YOLO_DRT_USE_REID", bool(baked.get("use_reid", False))),
        sam_identity_backend=(
            os.environ.get(
                "YOLO_DRT_SAM_IDENTITY_BACKEND",
                str(baked.get("sam_identity_backend", "memory")),
            ).strip()
            or "memory"
        ),
        sam_model=sam_model,
        sam_match_iou=_env_float(
            "YOLO_DRT_SAM_MATCH_IOU", float(baked.get("sam_match_iou", 0.30))
        ),
        sam_osnet_reentry=_env_bool(
            "YOLO_DRT_SAM_OSNET_REENTRY", bool(baked.get("sam_osnet_reentry", True))
        ),
        sam_osnet_reentry_thresh=_env_float(
            "YOLO_DRT_SAM_OSNET_REENTRY_THRESH",
            float(baked.get("sam_osnet_reentry_thresh", 0.65)),
        ),
        sam_osnet_reentry_min_miss=_env_int(
            "YOLO_DRT_SAM_OSNET_REENTRY_MIN_MISS",
            int(baked.get("sam_osnet_reentry_min_miss", 15)),
        ),
        use_offline_tracklet_link=_env_bool(
            "YOLO_DRT_USE_OFFLINE_TRACKLET_LINK",
            bool(baked.get("use_offline_tracklet_link", True)),
        ),
        tracklet_link_max_gap_frames=_env_int(
            "YOLO_DRT_TRACKLET_LINK_MAX_GAP_FRAMES",
            int(baked.get("tracklet_link_max_gap_frames", 0)),
        ),
        tracklet_link_min_sim=_env_float(
            "YOLO_DRT_TRACKLET_LINK_MIN_SIM",
            float(baked.get("tracklet_link_min_sim", 0.65)),
        ),
        tracklet_link_use_reid=_env_bool(
            "YOLO_DRT_TRACKLET_LINK_USE_REID",
            bool(baked.get("tracklet_link_use_reid", True)),
        ),
        tracklet_link_samples_per_tracklet=_env_int(
            "YOLO_DRT_TRACKLET_LINK_SAMPLES_PER_TRACKLET",
            int(baked.get("tracklet_link_samples_per_tracklet", 8)),
        ),
        tracklet_link_spatial_weight=_env_float(
            "YOLO_DRT_TRACKLET_LINK_SPATIAL_WEIGHT", 0.15
        ),
        identity_gallery_enabled=_env_bool(
            "YOLO_DRT_IDENTITY_GALLERY",
            bool(baked.get("identity_gallery_enabled", True)),
        ),
        identity_gallery_min_sim=_env_float(
            "YOLO_DRT_IDENTITY_GALLERY_MIN_SIM",
            float(baked.get("identity_gallery_min_sim", 0.65)),
        ),
        identity_gallery_spill=_env_bool(
            "YOLO_DRT_IDENTITY_GALLERY_SPILL",
            bool(baked.get("identity_gallery_spill", True)),
        ),
        use_tensorrt=_env_bool(
            "YOLO_DRT_USE_TENSORRT", bool(baked.get("use_tensorrt", True))
        ),
        cross_check_enabled=_env_bool(
            "YOLO_DRT_CROSS_CHECK_ENABLED",
            bool(baked.get("cross_check_enabled", True)),
        ),
        cross_check_object_prompt=(
            os.environ.get(
                "YOLO_DRT_CROSS_CHECK_OBJECT_PROMPT",
                str(baked.get("cross_check_object_prompt", "helmet")),
            ).strip()
            or "helmet"
        ),
        cross_check_warning_text=(
            os.environ.get(
                "YOLO_DRT_CROSS_CHECK_WARNING_TEXT",
                str(baked.get("cross_check_warning_text", "NO HELMET")),
            ).strip()
            or "NO HELMET"
        ),
        cross_check_conf=_env_float(
            "YOLO_DRT_CROSS_CHECK_CONF", float(baked.get("cross_check_conf", 0.35))
        ),
        cross_check_min_intersection_px=_env_float(
            "YOLO_DRT_CROSS_CHECK_MIN_INTERSECTION_PX",
            float(baked.get("cross_check_min_intersection_px", 20.0)),
        ),
        cross_check_min_iou=_env_float(
            "YOLO_DRT_CROSS_CHECK_MIN_IOU",
            float(baked.get("cross_check_min_iou", 0.03)),
        ),
        cross_check_helmet_min_conf=_env_float(
            "YOLO_DRT_CROSS_CHECK_HELMET_MIN_CONF",
            float(baked.get("cross_check_helmet_min_conf", 0.30)),
        ),
        cross_check_min_violation_streak=_env_int(
            "YOLO_DRT_CROSS_CHECK_MIN_VIOLATION_STREAK",
            int(baked.get("cross_check_min_violation_streak", 2)),
        ),
        cross_check_verdict_history=_env_int(
            "YOLO_DRT_CROSS_CHECK_VERDICT_HISTORY",
            int(baked.get("cross_check_verdict_history", 5)),
        ),
        cross_check_draw_head_box=_env_bool(
            "YOLO_DRT_CROSS_CHECK_DRAW_HEAD_BOX",
            bool(baked.get("cross_check_draw_head_box", True)),
        ),
        cross_check_draw_boxes=_env_bool(
            "YOLO_DRT_CROSS_CHECK_DRAW_BOXES",
            bool(baked.get("cross_check_draw_boxes", True)),
        ),
        encode_mode=(
            os.environ.get(
                "YOLO_DRT_ENCODE_MODE", str(baked.get("encode_mode", "manual"))
            ).strip()
            or "manual"
        ),
        default_prompt=(
            os.environ.get(
                "YOLO_DRT_DEFAULT_PROMPT", str(baked.get("default_prompt", "person"))
            ).strip()
            or "person"
        ),
        realtime_mode=_env_bool(
            "YOLO_DRT_REALTIME_MODE", bool(baked.get("realtime_mode", True))
        ),
        gpu_full_batch=False,
        infer_batch_size=_env_int(
            "YOLO_DRT_INFER_BATCH_SIZE", int(baked.get("infer_batch_size", 64))
        ),
        gpu_queue_depth=_env_int(
            "YOLO_DRT_GPU_QUEUE_DEPTH", int(baked.get("gpu_queue_depth", 4))
        ),
        decode_prefetch=_env_int(
            "YOLO_DRT_DECODE_PREFETCH", int(baked.get("decode_prefetch", 4))
        ),
        track_buffer=_env_int(
            "YOLO_DRT_TRACK_BUFFER", int(baked.get("track_buffer", 900))
        ),
        frame_stride=_env_int(
            "YOLO_DRT_FRAME_STRIDE", int(baked.get("frame_stride", 0))
        ),
        detect_conf=_env_float(
            "YOLO_DRT_DETECT_CONF", float(baked.get("detect_conf", 0.2))
        ),
        appearance_thresh=_env_float(
            "YOLO_DRT_APPEARANCE_THRESH", float(baked.get("appearance_thresh", 0.55))
        ),
        pose_kpt_conf=_env_float(
            "YOLO_DRT_POSE_KPT_CONF", float(baked.get("pose_kpt_conf", 0.25))
        ),
        draw_pose=_env_bool(
            "YOLO_DRT_DRAW_POSE", bool(baked.get("draw_pose", False))
        ),
        smart_ram_budget=_env_bool(
            "YOLO_DRT_SMART_RAM_BUDGET", bool(baked.get("smart_ram_budget", True))
        ),
        max_process_ram_gb=_env_float(
            "YOLO_DRT_MAX_PROCESS_RAM_GB", float(baked.get("max_process_ram_gb", 10.0))
        ),
        max_window_ram_gb=_env_float("YOLO_DRT_MAX_WINDOW_RAM_GB", 4.0),
        max_preload_ram_gb=_env_float("YOLO_DRT_MAX_PRELOAD_RAM_GB", 12.0),
        frame_source_mode=(
            os.environ.get(
                "YOLO_DRT_FRAME_SOURCE_MODE",
                str(baked.get("frame_source_mode", "windowed")),
            ).strip()
            or "windowed"
        ),
        preload_video=_env_bool(
            "YOLO_DRT_PRELOAD_VIDEO", bool(baked.get("preload_video", False))
        ),
        tensorrt_imgsz=_env_int("YOLO_DRT_TRT_IMGSZ", 640),
        tensorrt_max_batch=_env_int(
            "YOLO_DRT_TRT_MAX_BATCH", int(baked.get("tensorrt_max_batch", 32))
        ),
        tensorrt_fp16=_env_bool(
            "YOLO_DRT_TRT_FP16", bool(baked.get("tensorrt_fp16", True))
        ),
        tensorrt_workspace_gb=_env_float("YOLO_DRT_TRT_WORKSPACE_GB", 2.0),
        tensorrt_autocast_fast=_env_bool(
            "YOLO_DRT_TRT_AUTOCAST_FAST",
            bool(baked.get("tensorrt_autocast_fast", True)),
        ),
        tensorrt_engine_strategy=strategy,
        tensorrt_rebuild_policy=rebuild,
        tensorrt_manifest_dir=manifest_dir,
        tensorrt_central_dir=manifest_dir,
        inference_device=(
            os.environ.get(
                "YOLO_DRT_INFERENCE_DEVICE",
                str(baked.get("inference_device", "cuda")),
            ).strip()
            or "cuda"
        ),
    )
    settings.ensure_dirs()
    return settings
