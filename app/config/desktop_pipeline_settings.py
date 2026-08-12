"""Shared desktop/API pipeline load: baked fast profile wins over stale ui_settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config.settings import RD_DIR, SOLIDER_FILENAME, PipelineSettings
from app.config.ui_fast_profile import load_ui_fast_pipeline

# Keys that must not be poisoned by an old ui_settings.json (leave/return identity).
IDENTITY_BAKE_KEYS: tuple[str, ...] = (
    "use_reid",
    "use_sam_identity",
    "use_offline_tracklet_link",
    "tracklet_link_use_reid",
    "tracklet_link_max_gap_frames",
    "tracklet_link_min_sim",
    "tracklet_link_samples_per_tracklet",
    "sam_osnet_reentry",
    "sam_osnet_reentry_thresh",
    "sam_osnet_reentry_min_miss",
    "identity_gallery_enabled",
    "identity_gallery_min_sim",
    "identity_gallery_spill",
    "reid_backend",
    "track_buffer",
    "gpu_full_batch",
    "preload_video",
    "frame_source_mode",
    "infer_batch_size",
    "realtime_mode",
    "use_seg",
)


def apply_identity_bake(pipeline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Overlay baked identity/speed flags onto a persisted pipeline dict."""
    baked = load_ui_fast_pipeline()
    merged = dict(pipeline or {})
    for key in IDENTITY_BAKE_KEYS:
        if key in baked:
            merged[key] = baked[key]
    return merged


def normalize_reid_paths(settings: PipelineSettings) -> PipelineSettings:
    """Fix stale ui_settings: reid_backend=solider but reid_model still OSNet path."""
    from app.core.reid_engine import resolve_reid_model_path

    fixed = resolve_reid_model_path(settings.reid_backend, settings.reid_model)
    if fixed.resolve() != Path(settings.reid_model).resolve():
        settings.reid_model = fixed
    if str(settings.reid_backend).casefold() == "solider":
        sol = RD_DIR / SOLIDER_FILENAME
        if sol.is_file() and "osnet" in Path(settings.reid_model).name.casefold():
            settings.reid_model = sol
    return settings


def apply_identity_bake_to_settings(settings: PipelineSettings) -> PipelineSettings:
    """Force baked leave/return flags onto an existing settings object."""
    from dataclasses import replace

    baked = load_ui_fast_pipeline()
    kwargs = {k: baked[k] for k in IDENTITY_BAKE_KEYS if k in baked}
    if not kwargs:
        return normalize_reid_paths(settings)
    return normalize_reid_paths(replace(settings, **kwargs))


def load_desktop_pipeline_settings(
    *,
    persisted_pipeline: dict[str, Any] | None = None,
) -> PipelineSettings:
    """Bake UI fast profile, then overlay persisted paths (detect/reid/output/…)."""
    from api.settings_codec import settings_from_dict
    from app.ui.ui_settings_store import load_ui_settings

    base = settings_from_dict(load_ui_fast_pipeline(), base=PipelineSettings())
    pipeline = persisted_pipeline
    if pipeline is None:
        raw = load_ui_settings()
        pipeline = raw.get("pipeline") if isinstance(raw, dict) else None
    if isinstance(pipeline, dict) and pipeline:
        merged = apply_identity_bake(pipeline)
        return normalize_reid_paths(settings_from_dict(merged, base=base))
    return normalize_reid_paths(base)
