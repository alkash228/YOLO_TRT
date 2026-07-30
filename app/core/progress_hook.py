"""Zero/low-overhead progress slot for API jobs (no callback per frame)."""
from __future__ import annotations

import time


class ProgressSlot:
    """
    Infer thread writes here directly; GET /jobs reads and builds UI dict off hot path.
    Updates throttled by frame step + wall time so progress bar does not slow Pass1.
    """

    __slots__ = (
        "current",
        "total",
        "phase",
        "_last_ts",
        "_last_frame",
        "process_rss_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "gpu_device_used_mb",
        "gpu_util_pct",
    )

    def __init__(self) -> None:
        self.current = 0
        self.total = 0
        self.phase = "queued"
        self._last_ts = 0.0
        self._last_frame = -1
        self.process_rss_mb = 0.0
        self.cuda_allocated_mb = 0.0
        self.cuda_reserved_mb = 0.0
        self.gpu_device_used_mb = 0.0
        self.gpu_util_pct = 0.0

    def set_resources(
        self,
        *,
        process_rss_mb: float = 0.0,
        cuda_allocated_mb: float = 0.0,
        cuda_reserved_mb: float = 0.0,
        gpu_device_used_mb: float = 0.0,
        gpu_util_pct: float = 0.0,
    ) -> None:
        self.process_rss_mb = float(process_rss_mb)
        self.cuda_allocated_mb = float(cuda_allocated_mb)
        self.cuda_reserved_mb = float(cuda_reserved_mb)
        self.gpu_device_used_mb = float(gpu_device_used_mb)
        self.gpu_util_pct = float(gpu_util_pct)

    def maybe_update(
        self,
        current: int,
        total: int,
        phase: str,
        *,
        min_frames: int = 8,
        min_sec: float = 0.5,
    ) -> None:
        cur = max(0, int(current))
        tot = max(0, int(total))
        ph = str(phase or "inference")
        now = time.monotonic()
        if (
            cur < tot
            and (cur - self._last_frame) < min_frames
            and (now - self._last_ts) < min_sec
        ):
            return
        self.current = cur
        self.total = tot
        self.phase = ph
        self._last_frame = cur
        self._last_ts = now

    def force(self, current: int, total: int, phase: str) -> None:
        self.current = max(0, int(current))
        self.total = max(0, int(total))
        self.phase = str(phase or "inference")
        self._last_frame = self.current
        self._last_ts = time.monotonic()
