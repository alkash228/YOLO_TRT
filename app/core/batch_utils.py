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
    Fixed-size GPU jobs only (YouTube-style). ``gpu_full_batch`` is ignored —
    never expand to the whole video in one job.
    requested <= 0 → 64 (or max_cap if set).
    """
    _ = gpu_full_batch  # deprecated: full-video GPU job removed
    if int(requested) <= 0:
        req = 64
    else:
        req = max(1, int(requested))

    if use_reid and int(max_job_batch) > 0:
        req = min(req, int(max_job_batch))
    if int(max_cap) > 0:
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
    """Chunk size for seg/cross predict_batch. Never 'whole video'."""
    _ = gpu_full_batch
    cap = max(1, int(max_cap) if int(max_cap) > 0 else 64)
    return min(max(1, int(n_frames)), cap)


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
    cap = int(getattr(settings, "max_job_batch_size", 0)) or 200
    n_src = max(int(source_frame_count), int(n_process))
    # ~2 ч @ 25 fps — только тогда режем job; короткие/средние ролики как раньше.
    long_run = n_process > 5000 or n_src > 5000
    if long_run and streaming:
        return min(max(1, int(job_batch)), max(16, cap))
    return max(1, int(job_batch))


def resolve_frame_source(
    settings: PipelineSettings,
    *,
    source_frame_count: int,
    width: int,
    height: int,
    max_preload_ram_gb: float | None = None,
) -> str:
    """
    Источник кадров: preload | stream | windowed.

    Explicit ``preload`` / ``windowed`` / ``stream`` are honored as-is.
    Only ``auto`` may pick: always windowed (YouTube-style batch stream), unless
    preload_video=True and the whole clip fits RAM.
    """
    mode, _reason = resolve_frame_source_ex(
        settings,
        source_frame_count=source_frame_count,
        width=width,
        height=height,
        max_preload_ram_gb=max_preload_ram_gb,
    )
    return mode


def resolve_frame_source_ex(
    settings: PipelineSettings,
    *,
    source_frame_count: int,
    width: int,
    height: int,
    max_preload_ram_gb: float | None = None,
) -> tuple[str, str]:
    """Like resolve_frame_source, but also returns a human reason string."""
    from app.core.frame_pipeline import estimate_video_ram_gb

    requested = str(getattr(settings, "frame_source_mode", "auto") or "auto").casefold().strip()
    if requested in ("windowed", "window"):
        return "windowed", "explicit frame_source_mode=windowed"
    if requested == "stream":
        return "stream", "explicit frame_source_mode=stream"
    if requested == "preload":
        # Never silently demote explicit preload (OOM fallback is handled at call site).
        return "preload", "explicit frame_source_mode=preload"

    # auto only below — YouTube-style: stream windows to GPU, do not preload whole clip
    # unless preload_video is explicitly enabled and the clip fits.
    if int(source_frame_count) <= 0:
        return "stream", "auto: unknown frame count → stream"
    if not bool(getattr(settings, "preload_video", False)):
        return "windowed", "auto: YouTube-style windowed (no full preload)"
    if width > 0 and height > 0:
        est_gb = estimate_video_ram_gb(width, height, int(source_frame_count))
        budget = float(
            max_preload_ram_gb
            if max_preload_ram_gb is not None
            else getattr(settings, "max_preload_ram_gb", 12.0)
        )
        if est_gb > budget:
            return (
                "windowed",
                f"auto: est preload {est_gb:.2f} GB > budget {budget:.2f} GB → windowed",
            )
        return (
            "preload",
            f"auto: preload_video + est {est_gb:.2f} GB ≤ budget {budget:.2f} GB → preload",
        )
    return "windowed", "auto: no size info → windowed"


def resolve_use_streaming(
    settings: PipelineSettings,
    *,
    source_frame_count: int,
    width: int,
    height: int,
    max_preload_ram_gb: float | None = None,
) -> bool:
    """True только для frame_source=stream (OpenCV read in hot path)."""
    return resolve_frame_source(
        settings,
        source_frame_count=source_frame_count,
        width=width,
        height=height,
        max_preload_ram_gb=max_preload_ram_gb,
    ) == "stream"


def resolve_window_frames(
    settings: PipelineSettings,
    *,
    job_batch: int,
    frame_stride: int,
    width: int,
    height: int,
    max_window_ram_gb: float | None = None,
    windows_in_ram: int | None = None,
) -> tuple[int, dict[str, int | float | bool]]:
    """
    infer_per_window = jobs × job_batch (всегда без хвоста GPU, кроме последнего окна ролика).
    source_span = infer_per_window × frame_stride — сколько кадров источника прочитать.

    window_frames=0 → макс. полных job'ов на окно по RAM (считается по infer-кадрам в RAM).
    window_frames>0 → подсказка в кадрах источника, snap вверх к сетке job×stride.
    """
    stride = max(1, int(frame_stride))
    job = max(1, int(job_batch))
    hint_src = int(getattr(settings, "window_frames", 0))
    win_ram = (
        int(windows_in_ram)
        if windows_in_ram is not None
        else int(getattr(settings, "windows_in_ram", 1))
    )
    windows_in_ram = max(1, win_ram)
    ram_budget_gb = float(
        max_window_ram_gb
        if max_window_ram_gb is not None
        else getattr(settings, "max_window_ram_gb", 4.0)
    )
    if ram_budget_gb <= 0:
        ram_budget_gb = float(getattr(settings, "max_preload_ram_gb", 12.0)) / float(windows_in_ram)

    if width > 0 and height > 0 and ram_budget_gb > 0:
        per_window_gb = ram_budget_gb
        bytes_per_frame = width * height * 3
        max_infer_ram = int((per_window_gb * (1024**3)) // max(1, bytes_per_frame))
        max_infer_ram = max(job, (max_infer_ram // job) * job)
    else:
        max_infer_ram = max(job, job * 2)

    if hint_src <= 0:
        infer_per = max_infer_ram
    else:
        hint_infer = max(job, ((int(hint_src) // stride) // job) * job)
        if hint_infer < job:
            hint_infer = job
        infer_per = min(hint_infer, max_infer_ram)
        infer_per = max(infer_per, job)

    jobs_per = infer_per // job
    infer_per = jobs_per * job
    source_span = infer_per * stride
    tail = 0

    return source_span, {
        "hint": hint_src,
        "infer_per_window": infer_per,
        "source_span": source_span,
        "jobs_per_window": jobs_per,
        "job_batch": job,
        "frame_stride": stride,
        "tail_infer_frames": tail,
        "aligned": True,
        "max_infer_ram": max_infer_ram,
    }


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
    if not large_job and int(job_batch_size) > 200:
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
    trt_max_batch>0 ограничивает ручной/auto chunk под профиль OSNet engine.
    """
    chunk = int(setting)
    cap = max(0, int(trt_max_batch))
    if cap > 0 and chunk > cap:
        chunk = cap
    if chunk > 0:
        return chunk
    # Auto: small jobs — one call (TRT self-chunks). Large jobs — chunk at engine
    # max batch when TRT, else 200 for PyTorch host memory.
    if n_crops <= 200 and cap <= 0:
        return 0
    if cap > 0:
        return cap if n_crops > cap else 0
    return 200


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


def _realtime_auto_infer_batch(settings: PipelineSettings) -> int:
    """Авто-batch в realtime: max(tensorrt_max_batch, 64), никогда не 200 втихую."""
    trt_b = int(getattr(settings, "tensorrt_max_batch", 32) or 32)
    return max(64, trt_b)


def effective_speed_tuning(settings: PipelineSettings) -> dict[str, int | bool]:
    """Параметры job/YOLO с учётом realtime_mode (~1× длительности видео)."""
    if not bool(getattr(settings, "realtime_mode", False)):
        imgsz = int(getattr(settings, "infer_imgsz", 0) or 0)
        return {
            "infer_batch": int(settings.infer_batch_size) if int(settings.infer_batch_size) > 0 else 64,
            "gpu_full_batch": False,
            "max_job_batch": int(settings.max_job_batch_size),
            "max_infer_batch": int(settings.max_infer_batch_size),
            "imgsz": imgsz,
            "seg_stride": max(1, int(getattr(settings, "seg_stride", 1))),
            "frame_stride": resolve_frame_stride(settings),
        }
    req_batch = int(settings.infer_batch_size)
    # Explicit >0 always wins. Auto (0) → TRT engine batch, not hardcoded 200.
    infer_batch = req_batch if req_batch > 0 else _realtime_auto_infer_batch(settings)
    return {
        "infer_batch": infer_batch,
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
