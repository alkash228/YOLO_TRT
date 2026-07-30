"""Engine factory."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from app.config.settings import PipelineSettings
from app.core.batch_utils import effective_speed_tuning
from app.core.detect_engine import DetectEngine
from app.core.reid_engine import ReidEngine
from app.core.seg_engine import SegEngine
from app.core.trt_paths import engines_ready
from app.core.video_processor import VideoProcessor


def _resolve_imgsz(settings: PipelineSettings) -> int:
    tune = effective_speed_tuning(settings)
    imgsz = int(tune.get("imgsz") or 0)
    if imgsz > 0:
        return imgsz
    raw = int(getattr(settings, "infer_imgsz", 0) or 0)
    return raw if raw > 0 else 0


def _trt_kwargs(settings: PipelineSettings) -> dict:
    if not settings.use_tensorrt:
        return {}
    return dict(
        use_tensorrt=True,
        tensorrt_imgsz=int(settings.tensorrt_imgsz or settings.infer_imgsz or 640),
        tensorrt_max_batch=int(settings.tensorrt_max_batch),
        tensorrt_fp16=bool(settings.tensorrt_fp16),
    )


def warmup_engines(
    detect: DetectEngine,
    seg: SegEngine | None,
    reid: ReidEngine | None,
) -> None:
    """Прогрев CUDA перед первым job (убирает пик ~1–3 с на старте)."""
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        detect.track(dummy)
    except Exception:
        pass
    if seg is not None:
        try:
            seg.predict(dummy)
        except Exception:
            pass
    if reid is not None:
        crop = dummy[120:360, 160:480]
        try:
            reid.embed_batch([crop])
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_device(settings: PipelineSettings) -> str:
    dev = str(getattr(settings, "inference_device", "cuda") or "cuda").strip().casefold()
    if dev in ("cpu", "cuda"):
        return dev
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_processor(
    settings: PipelineSettings,
    *,
    warmup: bool = True,
    on_log: Callable[[str], None] | None = None,
) -> VideoProcessor:
    device = _resolve_device(settings)
    if device == "cpu" and on_log:
        on_log("Inference device: CPU (GPU pipeline и TensorRT отключены)")
    half = settings.use_amp and device != "cpu"
    imgsz = _resolve_imgsz(settings)
    use_trt = bool(settings.use_tensorrt) and device == "cuda"
    trt = _trt_kwargs(settings) if use_trt else {}

    if settings.use_tensorrt and device == "cpu" and on_log:
        on_log("TensorRT пропущен: выбран CPU")

    if use_trt:
        ready = engines_ready(
            detect_pt=Path(settings.detect_model),
            cross_pt=Path(settings.cross_check_model) if settings.cross_check_model else None,
            reid_pth=Path(settings.reid_model),
            imgsz=int(settings.tensorrt_imgsz or 640),
            max_batch=int(settings.tensorrt_max_batch),
            fp16=bool(settings.tensorrt_fp16),
            need_cross=bool(settings.cross_check_enabled),
        )
        if on_log:
            on_log(
                f"TensorRT: detect={ready['detect']} reid={ready['reid']} "
                f"cross={ready['cross_check']}"
            )
        if not all(ready.values()):
            if on_log:
                on_log("TensorRT: не все .engine найдены — fallback на PyTorch или Собрать engines")

    detect = DetectEngine(
        settings.detect_model,
        conf=settings.detect_conf,
        device=device,
        half=half,
        imgsz=imgsz,
        **trt,
    )
    if on_log and getattr(detect, "using_tensorrt", False):
        on_log(
            f"Detect TRT: {detect.model_path} (imgsz={detect.imgsz}, batch={detect.trt_max_batch})"
        )
        resolved_b = int(getattr(detect, "trt_max_batch", 0) or 0)
        wanted_b = int(settings.tensorrt_max_batch)
        if resolved_b > 0 and resolved_b < wanted_b:
            on_log(
                f"WARNING: Detect TRT engine is b{resolved_b}, settings.tensorrt_max_batch={wanted_b}. "
                f"Stale/undersized engine — press «Собрать TensorRT engines» for "
                f"…_b{wanted_b}_fp16.engine"
            )
    elif on_log and settings.use_tensorrt:
        on_log(f"Detect PyTorch: {detect.source_model_path}")

    seg = None
    if settings.use_seg:
        seg = SegEngine(
            settings.seg_model,
            conf=settings.seg_conf,
            device=device,
            half=half,
            imgsz=imgsz,
        )
        seg._use_gpu_masks = bool(
            settings.gpu_mask_resize and device == "cuda" and torch.cuda.is_available()
        )

    reid = None
    from app.core.sam_memory_tracker import needs_osnet_embed, uses_sam_identity

    load_osnet = needs_osnet_embed(settings)
    if load_osnet:
        reid = ReidEngine(
            settings.reid_model,
            device=device,
            use_amp=half,
            use_tensorrt=use_trt,
            tensorrt_fp16=bool(settings.tensorrt_fp16),
        )
        if on_log and getattr(reid, "using_tensorrt", False):
            on_log(
                f"ReID TRT: {Path(reid.model_path).name} (batch={reid.trt_max_batch})"
            )
        elif on_log and settings.use_tensorrt:
            on_log("ReID PyTorch: .engine не найден / TRT load failed (см. warning)")
    elif on_log and uses_sam_identity(settings):
        on_log(
            "Identity: SAM masklet memory "
            f"(backend={getattr(settings, 'sam_identity_backend', 'memory')}; OSNet skipped)"
        )
    cross = None
    if settings.cross_check_enabled and settings.cross_check_model is not None:
        p = Path(settings.cross_check_model)
        if p.exists():
            cross = DetectEngine(
                p,
                conf=settings.cross_check_conf,
                device=device,
                half=half,
                imgsz=imgsz,
                **trt,
            )
            if on_log and getattr(cross, "using_tensorrt", False):
                on_log(f"Cross-check TRT: {cross.model_path}")

    if warmup:
        warmup_engines(detect, seg, reid)
    return VideoProcessor(detect, seg, reid, settings, cross_check_engine=cross)


def sync_processor_for_run(processor: VideoProcessor, settings: PipelineSettings) -> None:
    """Push current PipelineSettings onto a loaded processor (desktop Run / API job)."""
    from app.core.sam_memory_tracker import build_identity_tracker

    processor.settings = settings
    processor.detect_engine.conf = settings.detect_conf
    if processor.seg_engine is not None:
        processor.seg_engine.conf = settings.seg_conf
    processor.tracker = build_identity_tracker(settings)


def _engine_weights_path(engine) -> Path | None:
    if engine is None:
        return None
    src = getattr(engine, "source_model_path", None)
    if src:
        return Path(src).resolve()
    return Path(engine.model_path).resolve()


def processor_needs_rebuild(processor: VideoProcessor, settings: PipelineSettings) -> bool:
    """True when model topology changed — need Init/bootstrap, not just sync."""
    from app.core.sam_memory_tracker import needs_osnet_embed

    if settings.use_seg != (processor.seg_engine is not None):
        return True
    want_osnet = needs_osnet_embed(settings)
    if want_osnet != (processor.reid_engine is not None):
        return True
    if settings.cross_check_enabled != (processor.cross_check_engine is not None):
        return True

    det_w = _engine_weights_path(processor.detect_engine)
    if det_w is None or det_w != Path(settings.detect_model).resolve():
        return True

    if settings.use_seg and processor.seg_engine is not None:
        seg_w = _engine_weights_path(processor.seg_engine)
        if seg_w is None or seg_w != Path(settings.seg_model).resolve():
            return True

    if want_osnet and processor.reid_engine is not None:
        reid_w = _engine_weights_path(processor.reid_engine)
        if reid_w is None or reid_w != Path(settings.reid_model).resolve():
            return True

    if settings.cross_check_enabled and settings.cross_check_model:
        if processor.cross_check_engine is None:
            return True
        cross_w = _engine_weights_path(processor.cross_check_engine)
        if cross_w is None or cross_w != Path(settings.cross_check_model).resolve():
            return True

    dev = str(getattr(settings, "inference_device", "cuda") or "cuda").casefold()
    if dev == "cpu" and getattr(processor.detect_engine, "using_tensorrt", False):
        return True
    if settings.use_tensorrt != getattr(processor.detect_engine, "using_tensorrt", False):
        return True
    return False
