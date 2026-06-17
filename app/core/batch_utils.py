"""Batch sizing for GPU inference."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.settings import PipelineSettings


def resolve_infer_batch_size(
    requested: int,
    frame_count: int,
    *,
    gpu_full_batch: bool,
    max_cap: int = 0,
    use_reid: bool = False,
    max_job_batch: int = 0,
) -> int:
    """
    requested <= 0 + gpu_full_batch → весь ролик одним job.
    requested <= 0 без full batch → 16 кадров.
    При ReID (detect=track) job > max_job_batch режется — иначе нет overlap CPU/GPU.
    """
    if frame_count > 0 and gpu_full_batch and int(requested) <= 0:
        req = int(frame_count)
    elif int(requested) <= 0:
        req = 16
    else:
        req = max(1, int(requested))

    if use_reid and int(max_job_batch) > 0:
        req = min(req, int(max_job_batch))
    elif not gpu_full_batch and int(max_cap) > 0:
        req = min(req, int(max_cap))
    if frame_count > 0:
        req = min(req, int(frame_count))
    return max(1, req)


def gpu_predict_chunk_size(
    n_frames: int,
    *,
    gpu_full_batch: bool,
    max_cap: int,
) -> int:
    """Размер YOLO forward (seg/cross/predict). max_cap>0 всегда ограничивает пик VRAM."""
    if int(max_cap) > 0:
        return max(1, int(max_cap))
    if gpu_full_batch:
        return 0
    return max(1, int(n_frames)) if int(n_frames) > 0 else 0


def resolve_batch_prefetch_depth(settings: PipelineSettings) -> int:
    """Глубина очереди async batch loader (decode / slice из RAM)."""
    return max(1, int(getattr(settings, "decode_prefetch", 4)))


def cap_job_batch_long_video(
    job_batch: int,
    n_process: int,
    settings: PipelineSettings,
    *,
    streaming: bool,
    source_frame_count: int = 0,
) -> int:
    """Очень длинные ролики (2+ ч): не держать в RAM гигантский job."""
    cap = int(getattr(settings, "max_job_batch_size", 0)) or 64
    n_src = max(int(source_frame_count), int(n_process))
    # ~2 ч @ 25 fps — только тогда режем job; короткие/средние ролики как раньше.
    long_run = n_process > 5000 or n_src > 5000
    if long_run and streaming:
        return min(max(1, int(job_batch)), max(16, cap))
    return max(1, int(job_batch))


def resolve_use_streaming(
    settings: PipelineSettings,
    *,
    source_frame_count: int,
    width: int,
    height: int,
) -> bool:
    """Preload в RAM, пока влезает; streaming только при неизвестной длине или нехватке RAM."""
    from app.core.frame_pipeline import estimate_video_ram_gb

    if int(source_frame_count) <= 0:
        return True
    if not bool(getattr(settings, "preload_video", True)):
        return True
    if width > 0 and height > 0 and int(source_frame_count) > 0:
        est_gb = estimate_video_ram_gb(width, height, int(source_frame_count))
        if est_gb > float(getattr(settings, "max_preload_ram_gb", 8.0)):
            return True
    return False


def resolve_gpu_queue_depth(
    requested: int,
    *,
    n_jobs: int,
    job_batch_size: int,
    max_job_batch: int = 0,
) -> int:
    """Большие job + глубокая очередь не ускоряют — только держат лишние кадры в RAM."""
    depth = max(1, int(requested))
    if n_jobs > 0:
        depth = min(depth, n_jobs)
    large_job = int(max_job_batch) > 0 and int(job_batch_size) > int(max_job_batch)
    if not large_job and int(job_batch_size) > 96:
        large_job = True
    if large_job:
        depth = min(depth, 3)
    return max(1, depth)


def resolve_reid_embed_chunk(
    setting: int,
    n_crops: int,
    *,
    trt_max_batch: int = 0,
) -> int:
    """
    0 в настройках → без явного чанка (TRT runner сам режет по max batch engine).
    trt_max_batch>0 ограничивает ручной chunk под профиль OSNet engine.
    """
    chunk = int(setting)
    cap = max(0, int(trt_max_batch))
    if cap > 0 and chunk > cap:
        chunk = cap
    if chunk > 0:
        return chunk
    if n_crops <= 64:
        return 0
    return 64


def resolve_frame_stride(settings: PipelineSettings) -> int:
    """1 = каждый кадр; 2+ = пропуск; 0 = авто (realtime → 2, иначе 1)."""
    raw = int(getattr(settings, "frame_stride", 1))
    if raw > 0:
        return max(1, raw)
    if bool(getattr(settings, "realtime_mode", False)):
        return 2
    return 1


def source_frame_indices(source_frame_count: int, stride: int) -> list[int]:
    """Индексы кадров источника для inference (0, stride, 2*stride, …)."""
    n = max(0, int(source_frame_count))
    step = max(1, int(stride))
    return list(range(0, n, step))


def frame_stride_summary(
    source_frame_count: int,
    frame_stride: int,
    processed_count: int | None = None,
) -> dict[str, int | float]:
    """Сводка по пропуску кадров для отчёта и run record."""
    src = max(0, int(source_frame_count))
    stride = max(1, int(frame_stride))
    proc = (
        int(processed_count)
        if processed_count is not None
        else len(source_frame_indices(src, stride))
    )
    skipped = max(0, src - proc)
    pct = round(100.0 * skipped / src, 1) if src > 0 else 0.0
    return {
        "source_frame_count": src,
        "processed_frame_count": proc,
        "frame_stride": stride,
        "frames_skipped": skipped,
        "reduction_pct": pct,
    }


def effective_speed_tuning(settings: PipelineSettings) -> dict[str, int | bool]:
    """Параметры job/YOLO с учётом realtime_mode (~1× длительности видео)."""
    if not bool(getattr(settings, "realtime_mode", False)):
        imgsz = int(getattr(settings, "infer_imgsz", 0) or 0)
        return {
            "infer_batch": int(settings.infer_batch_size),
            "gpu_full_batch": bool(settings.gpu_full_batch),
            "max_job_batch": int(settings.max_job_batch_size),
            "max_infer_batch": int(settings.max_infer_batch_size),
            "imgsz": imgsz,
            "seg_stride": max(1, int(getattr(settings, "seg_stride", 1))),
            "frame_stride": resolve_frame_stride(settings),
        }
    req_batch = int(settings.infer_batch_size)
    return {
        "infer_batch": req_batch if req_batch > 0 else 64,
        "gpu_full_batch": False,
        "max_job_batch": max(96, int(settings.max_job_batch_size)),
        "max_infer_batch": max(48, int(settings.max_infer_batch_size)),
        "imgsz": 512,
        "seg_stride": max(2, int(getattr(settings, "seg_stride", 1))),
        "frame_stride": max(2, resolve_frame_stride(settings)),
    }


def chunk_list(items: list, chunk_size: int):
    if int(chunk_size) <= 0:
        if items:
            yield items
        return
    size = max(1, int(chunk_size))
    for i in range(0, len(items), size):
        yield items[i : i + size]
