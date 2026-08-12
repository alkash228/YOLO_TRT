"""Single pipeline entrypoints for desktop UI and HTTP API (same Init / Run path)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from app.config.settings import PipelineSettings
from app.core.pipeline import (
    build_processor,
    processor_needs_rebuild,
    sync_processor_for_run,
)
from app.core.shared_processor import attach as shared_attach
from app.core.shared_processor import dispose_all as shared_dispose_all
from app.core.shared_processor import get as shared_get
from app.core.shared_processor import replace as shared_replace
from app.core.trt_export import build_all_engines
from app.core.trt_paths import engines_ready
from app.core.video_processor import VideoProcessor

LogFn = Callable[[str], None] | None
StatusFn = Callable[[str], None] | None


def _log(on_log: LogFn, msg: str) -> None:
    if on_log is not None:
        on_log(msg)


def _status(on_status: StatusFn, value: str) -> None:
    if on_status is not None:
        on_status(value)


def validate_settings_for_init(settings: PipelineSettings) -> str | None:
    """Same checks as MainWindow.init_engines before loading models."""
    from app.core.sam_memory_tracker import needs_osnet_embed

    if not settings.detect_model.exists():
        return "Detect model not found"
    if settings.use_seg and not settings.seg_model.exists():
        return "Seg model not found (or disable segmentation)"
    if needs_osnet_embed(settings) and not settings.reid_model.exists():
        return "ReID model not found (or disable ReID / use SAM identity without OSNet re-entry)"
    if settings.cross_check_enabled:
        if not settings.cross_check_model or not Path(settings.cross_check_model).exists():
            return "Cross-check model not found"
    return None


def validate_processor_topology(
    processor: VideoProcessor,
    settings: PipelineSettings,
) -> str | None:
    """Same pre-run checks as MainWindow.run_pipeline."""
    from app.core.sam_memory_tracker import needs_osnet_embed

    if settings.use_seg != (processor.seg_engine is not None):
        return "Segmentation mode changed — re-init engines"
    if needs_osnet_embed(settings) != (processor.reid_engine is not None):
        return "ReID / OSNet mode changed — re-init engines"
    if settings.cross_check_enabled != (processor.cross_check_engine is not None):
        return "Cross-check mode changed — re-init engines"
    return None


def prepare_for_run(
    processor: VideoProcessor,
    settings: PipelineSettings,
) -> str | None:
    """Apply settings to loaded processor exactly like desktop Run."""
    err = validate_processor_topology(processor, settings)
    if err:
        return err
    sync_processor_for_run(processor, settings)
    return None


def resolve_job_prompt(settings: PipelineSettings, prompt: str | None) -> str:
    return (prompt or settings.default_prompt or "person").strip() or "person"


def resolve_job_max_duration(
    settings: PipelineSettings,
    max_duration_seconds: float | None,
) -> float | None:
    max_dur = max_duration_seconds
    if max_dur is None:
        max_dur = getattr(settings, "max_duration_seconds", None)
    if max_dur is not None and float(max_dur) <= 0:
        return None
    return max_dur


def ensure_processor(
    settings: PipelineSettings,
    *,
    holder: str = "api",
    force: bool = False,
    force_reload: bool = False,
    on_log: LogFn = None,
    on_status: StatusFn = None,
    warmup: bool = True,
    auto_build_trt: bool = True,
) -> VideoProcessor | None:
    """
    One Init/bootstrap path for UI and API.
    Reuses shared GPU load when models unchanged.
    force=True alone does NOT wipe a compatible shared processor (avoids duplicate VRAM).
    force_reload=True is for real engine rebuild (e.g. after TRT export).
    auto_build_trt=True: if detect/cross (and OSNet when needed) .engine missing → build now.
    """
    del force  # soft flag kept for call-site compat; hard wipe is force_reload only
    settings.ensure_dirs()
    shared_proc, _shared_settings, shared_holders = shared_get()

    # Compatible shared load → attach only (ignore soft force=True).
    if (
        not force_reload
        and shared_proc is not None
        and not processor_needs_rebuild(shared_proc, settings)
    ):
        # Keep processor.settings == caller's settings (UI batch/flags), not stale copy.
        sync_processor_for_run(shared_proc, settings)
        shared_attach(shared_proc, settings, holder)
        holders = ", ".join(sorted(shared_holders | {holder}))
        if "ui" in shared_holders and holder == "api":
            _log(on_log, "API attached to shared UI processor (no duplicate VRAM)")
        else:
            _log(
                on_log,
                f"Processor ready (reused, holders: {holders}) — no GPU reload",
            )
        return shared_proc

    if shared_proc is not None:
        if shared_holders:
            _log(
                on_log,
                f"Replacing processor (was: {', '.join(sorted(shared_holders))})…",
            )
        shared_dispose_all()

    _log(on_log, "Loading models…")
    _status(on_status, "starting")

    if str(getattr(settings, "inference_device", "cuda") or "cuda").casefold() != "cpu":
        if not torch.cuda.is_available():
            _log(on_log, "CUDA unavailable — GPU inference impossible")
            return None

    engines_map: dict[str, bool] = {}
    if settings.use_tensorrt:
        from app.core.reid_engine import resolve_reid_backend
        from app.core.sam_memory_tracker import needs_osnet_embed

        reid_backend = resolve_reid_backend(
            getattr(settings, "reid_backend", None), settings.reid_model
        )
        need_reid_trt = bool(needs_osnet_embed(settings)) and reid_backend == "osnet"
        _log(
            on_log,
            f"Checking TensorRT engines… detect={Path(settings.detect_model).name} "
            f"imgsz={int(settings.tensorrt_imgsz or 640)} "
            f"batch={int(settings.tensorrt_max_batch)} "
            f"reid_trt={'osnet' if need_reid_trt else 'skip(' + reid_backend + ')'}",
        )
        central = getattr(settings, "tensorrt_central_dir", None) or (
            Path(settings.models_dir) / "TRT"
        )
        engines_kw = dict(
            detect_pt=Path(settings.detect_model),
            cross_pt=Path(settings.cross_check_model) if settings.cross_check_model else None,
            reid_pth=Path(settings.reid_model),
            imgsz=int(settings.tensorrt_imgsz or 640),
            max_batch=int(settings.tensorrt_max_batch),
            fp16=bool(settings.tensorrt_fp16),
            need_cross=bool(settings.cross_check_enabled),
            need_reid=need_reid_trt,
            strategy=str(
                getattr(settings, "tensorrt_engine_strategy", "central") or "central"
            ),
            central_dir=central,
        )
        engines_map = engines_ready(**engines_kw)

        missing = [k for k, ok in engines_map.items() if not ok]
        _log(on_log, f"TensorRT ready map: {engines_map}")
        if missing and auto_build_trt:
            _status(on_status, "building_engines")
            _log(
                on_log,
                "AUTO TensorRT BUILD start — missing: "
                + ", ".join(missing)
                + f" | source detect={Path(settings.detect_model)}",
            )
            build_all_engines(settings, log=on_log)
            engines_map = engines_ready(**engines_kw)
            _log(on_log, f"AUTO TensorRT BUILD done — map: {engines_map}")
            _status(on_status, "starting")
        elif missing:
            _log(
                on_log,
                "TensorRT missing: "
                + ", ".join(missing)
                + " — PyTorch fallback (auto_build_trt=False)",
            )
        else:
            _log(on_log, "TensorRT engines OK")

    try:
        _log(on_log, "Loading detect / cross-check / ReID into GPU...")
        processor = build_processor(settings, warmup=warmup, on_log=on_log)
    except Exception as exc:
        _log(on_log, f"Processor load error: {exc}")
        return None

    shared_replace(processor, settings, holder)
    if settings.use_tensorrt:
        det_trt = getattr(processor.detect_engine, "using_tensorrt", False)
        reid_trt = (
            getattr(processor.reid_engine, "using_tensorrt", False)
            if processor.reid_engine is not None
            else False
        )
        _log(on_log, f"Processor ready | TRT detect={det_trt} reid={reid_trt}")
    else:
        _log(on_log, "Processor ready")
    return processor


def prepare_processor_for_job(
    settings: PipelineSettings,
    processor: VideoProcessor | None,
    *,
    on_log: LogFn = None,
) -> tuple[VideoProcessor | None, str | None]:
    """
    Same as desktop Run prep: loaded processor + current settings, reload only if topology changed.
    """
    if processor is None:
        processor = ensure_processor(settings, holder="api", force=False, on_log=on_log)
        if processor is None:
            return None, "Processor not loaded — Init engines or POST /v1/admin/bootstrap"

        if processor_needs_rebuild(processor, settings):
            _log(
                on_log,
                "Models/modes changed — reloading processor once…",
            )
            processor = ensure_processor(
                settings, holder="api", force_reload=True, on_log=on_log
            )
            if processor is None:
                return None, "Processor reload failed"

    err = prepare_for_run(processor, settings)
    if err:
        return None, err
    return processor, None
