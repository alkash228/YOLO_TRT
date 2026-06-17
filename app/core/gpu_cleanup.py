"""Release CUDA allocator memory after a video job."""
from __future__ import annotations

import gc
from typing import Any


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
