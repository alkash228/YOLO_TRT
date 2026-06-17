"""RAM estimates for frames and inference batches."""
from __future__ import annotations


def frame_bgr_bytes(width: int, height: int, channels: int = 3) -> int:
    return max(0, int(width)) * max(0, int(height)) * max(1, int(channels))


def format_bytes(n: int) -> str:
    n = max(0, int(n))
    if n >= 1024**3:
        return f"{n / (1024**3):.2f} GB"
    if n >= 1024**2:
        return f"{n / (1024**2):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def build_memory_report(
    *,
    width: int,
    height: int,
    effective_batch_size: int,
    frames_total: int = 0,
    preloaded: bool = False,
) -> dict[str, float | int | str | bool]:
    fb = frame_bgr_bytes(width, height)
    bs = max(1, int(effective_batch_size))
    batch_b = fb * bs
    preload_b = fb * max(0, int(frames_total)) if preloaded else 0
    return {
        "width": int(width),
        "height": int(height),
        "frame_bytes": fb,
        "frame_mb": round(fb / (1024**2), 4),
        "frame_human": format_bytes(fb),
        "effective_batch_size": bs,
        "batch_bytes": batch_b,
        "batch_mb": round(batch_b / (1024**2), 4),
        "batch_human": format_bytes(batch_b),
        "preload_bytes": preload_b,
        "preload_mb": round(preload_b / (1024**2), 4),
        "preload_human": format_bytes(preload_b) if preload_b else "—",
        "preloaded": bool(preloaded and preload_b > 0),
        "resolution": f"{width}x{height}",
    }
