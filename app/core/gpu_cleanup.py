"""Release CUDA allocator memory after a video job."""
from __future__ import annotations

import gc
from typing import Any


def dispose_processor(processor: Any) -> None:
    """Unload model weights before replacing VideoProcessor (API bootstrap / Init)."""
    if processor is None:
        return
    for attr in ("detect_engine", "seg_engine", "reid_engine", "cross_check_engine"):
        eng = getattr(processor, attr, None)
        if eng is None:
            continue
        reset = getattr(eng, "reset_session", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
        trt = getattr(eng, "_trt", None)
        if trt is not None:
            close = getattr(trt, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            try:
                eng._trt = None
            except Exception:
                pass
        extractor = getattr(eng, "extractor", None)
        if extractor is not None:
            try:
                eng.extractor = None
            except Exception:
                pass
        for name in ("model", "_model", "engine", "_engine", "session", "predictor"):
            if hasattr(eng, name):
                try:
                    setattr(eng, name, None)
                except Exception:
                    pass
    release_gpu_memory(processor)


def _unload_reid_engine(eng: Any) -> None:
    """Drop OSNet/TRT weights after Pass2; reloaded lazily on next job."""
    if eng is None:
        return
    trt = getattr(eng, "_trt", None)
    if trt is not None:
        close = getattr(trt, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        try:
            eng._trt = None
        except Exception:
            pass
    try:
        eng.extractor = None
    except Exception:
        pass
    try:
        eng.using_tensorrt = False
    except Exception:
        pass


def _unload_trt_detect_engine(eng: Any) -> None:
    """Drop YOLO/TRT predictor; reloaded lazily on next predict()."""
    if eng is None:
        return
    reset = getattr(eng, "reset_session", None)
    if callable(reset):
        try:
            reset(hard=True)
        except TypeError:
            try:
                reset()
            except Exception:
                pass
        except Exception:
            pass
    trt = getattr(eng, "_trt", None)
    if trt is not None:
        close = getattr(trt, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        try:
            eng._trt = None
        except Exception:
            pass
    model = getattr(eng, "model", None)
    if model is not None and hasattr(model, "predictor"):
        try:
            model.predictor = None
        except Exception:
            pass


def soft_cleanup_after_job(
    processor: Any,
    *,
    unload_pass2_reid: bool = False,
    unload_cross_check: bool = False,
    flush_cuda_cache: bool = False,
) -> dict[str, float]:
    """Reset job-scoped state on warm processor. Keep TRT loaded by default.

    Unloading cross-check / ReID TRT after each job reloads .engine and leaks
    TensorRT contexts (~2× slower Pass1 on every repeat). empty_cache() also
    regresses fps_infer across consecutive API runs. Use dispose_processor() for
    full teardown (shutdown / bootstrap reload).
    """
    reserved_before = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            reserved_before = float(torch.cuda.memory_reserved() / (1024 * 1024))
    except Exception:
        pass

    if processor is not None:
        for attr in ("_on_debug_log", "_on_progress_tick"):
            try:
                setattr(processor, attr, None)
            except Exception:
                pass
        timing = getattr(processor, "_stage_timing", None)
        if isinstance(timing, dict):
            timing.clear()
        # Do NOT reset_session on TRT engines here — process_video soft-resets at job start.
        # End-of-job predictor teardown + reload was the main cause of growing Pass1 time.
        if unload_cross_check:
            _unload_trt_detect_engine(getattr(processor, "cross_check_engine", None))
        if unload_pass2_reid:
            settings = getattr(processor, "settings", None)
            if settings is not None and bool(
                getattr(settings, "tracklet_link_use_reid", False)
            ):
                _unload_reid_engine(getattr(processor, "reid_engine", None))
        for name in ("tracker", "motion_tracker"):
            tr = getattr(processor, name, None)
            reset_fn = getattr(tr, "reset", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    pass
    if flush_cuda_cache:
        stats = release_gpu_memory(processor)
    else:
        gc.collect()
        stats = {"before_mb": 0.0, "after_mb": 0.0, "freed_mb": 0.0}
        try:
            import torch

            if torch.cuda.is_available():
                mb = float(torch.cuda.memory_allocated() / (1024 * 1024))
                stats["before_mb"] = mb
                stats["after_mb"] = mb
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            stats["reserved_mb"] = float(
                torch.cuda.memory_reserved() / (1024 * 1024)
            )
            stats["reserved_before_mb"] = reserved_before
            stats["reserved_freed_mb"] = max(
                0.0, reserved_before - float(stats["reserved_mb"])
            )
    except Exception:
        stats["reserved_mb"] = 0.0
        stats["reserved_before_mb"] = reserved_before
        stats["reserved_freed_mb"] = 0.0
    return stats


def release_gpu_memory(*holders: Any) -> dict[str, float]:
    """
    Очищает кэш CUDA после прогона. Модели в processor не трогаем — только VRAM cache.
    Возвращает MB до/после (allocated).
    """
    before = 0.0
    after = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            before = float(torch.cuda.memory_allocated() / (1024 * 1024))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            after = float(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        pass
    gc.collect()
    return {"before_mb": before, "after_mb": after, "freed_mb": max(0.0, before - after)}
