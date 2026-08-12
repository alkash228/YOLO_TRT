"""Serialize PipelineSettings ↔ JSON (same fields as desktop UI)."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from app.config.settings import DEFAULT_HELMET_MODEL, PipelineSettings

_PATH_FIELDS = frozenset(
    {
        "detect_model",
        "seg_model",
        "reid_model",
        "output_dir",
        "work_dir",
        "upload_dir",
        "cross_check_model",
        "sam_model",
    }
)

# Same fields as MainWindow._collect_settings — everything else stays dataclass default.
UI_PIPELINE_KEYS = frozenset(
    {
        "detect_model",
        "seg_model",
        "reid_model",
        "output_dir",
        "detect_conf",
        "seg_conf",
        "match_iou_min",
        "appearance_thresh",
        "track_buffer",
        "use_seg",
        "use_reid",
        "use_sam_identity",
        "sam_identity_backend",
        "sam_model",
        "sam_match_iou",
        "sam_osnet_reentry",
        "sam_osnet_reentry_thresh",
        "sam_osnet_reentry_min_miss",
        "identity_gallery_enabled",
        "identity_gallery_min_sim",
        "identity_gallery_spill",
        "reid_backend",
        "use_offline_tracklet_link",
        "tracklet_link_max_gap_frames",
        "tracklet_link_min_sim",
        "tracklet_link_samples_per_tracklet",
        "tracklet_link_use_reid",
        "overlay_alpha",
        "draw_boxes",
        "draw_masks",
        "draw_centers",
        "draw_pose",
        "pose_kpt_conf",
        "cross_check_enabled",
        "cross_check_model",
        "cross_check_object_prompt",
        "cross_check_conf",
        "cross_check_min_intersection_px",
        "cross_check_min_iou",
        "cross_check_helmet_min_conf",
        "cross_check_min_violation_streak",
        "cross_check_verdict_history",
        "cross_check_warning_text",
        "cross_check_draw_head_box",
        "cross_check_draw_boxes",
        "gpu_full_batch",
        "infer_batch_size",
        "gpu_queue_depth",
        "decode_prefetch",
        "realtime_mode",
        "frame_stride",
        "use_tensorrt",
        "tensorrt_max_batch",
        "tensorrt_fp16",
        "tensorrt_autocast_fast",
        "inference_device",
        "encode_mode",
        "default_prompt",
        "smart_ram_budget",
        "max_process_ram_gb",
        "max_duration_seconds",
        "frame_source_mode",
        "preload_video",
    }
)


def settings_to_dict(settings: PipelineSettings) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(settings):
        value = getattr(settings, f.name)
        if f.name in _PATH_FIELDS:
            out[f.name] = str(value) if value is not None else None
        else:
            out[f.name] = value
    return out


def settings_from_dict(data: dict[str, Any], *, base: PipelineSettings | None = None) -> PipelineSettings:
    """Merge API/JSON payload onto defaults or an existing settings object."""
    seed = base or PipelineSettings()
    kwargs: dict[str, Any] = {}
    valid_names = {f.name for f in fields(PipelineSettings)}
    for key, value in data.items():
        if key not in valid_names:
            continue
        if key in _PATH_FIELDS:
            kwargs[key] = Path(value) if value else None
        else:
            kwargs[key] = value
    merged = {f.name: getattr(seed, f.name) for f in fields(PipelineSettings)}
    merged.update(kwargs)
    settings = PipelineSettings(**merged)
    settings.ensure_dirs()
    return settings


def settings_from_ui_patch(
    data: dict[str, Any],
    *,
    base: PipelineSettings | None = None,
) -> PipelineSettings:
    """Merge WEB/desktop UI fields onto current settings (preserve TRT/RAM tuning)."""
    ui_data = {k: v for k, v in data.items() if k in UI_PIPELINE_KEYS}
    if "encode_mode" not in ui_data:
        ui_data["encode_mode"] = "manual"
    if ui_data.get("cross_check_enabled") and not ui_data.get("cross_check_model"):
        ui_data["cross_check_model"] = str(DEFAULT_HELMET_MODEL)
    return settings_from_dict(ui_data, base=base or PipelineSettings())
