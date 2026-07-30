"""Sample host RAM + CUDA allocator for **this process** (not whole-GPU NVML totals)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from app.core.process_memory import current_process_rss_mb


@dataclass(slots=True)
class ProcessResourceSample:
    t_sec: float
    process_rss_mb: float
    cuda_allocated_mb: float
    cuda_reserved_mb: float


@dataclass
class ProcessResourceMonitor:
    interval_sec: float = 0.5
    _samples: list[ProcessResourceSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _t0: float = 0.0
    _cuda: bool = False
    _baseline_rss_mb: float = 0.0
    _baseline_cuda_alloc_mb: float = 0.0
    _baseline_cuda_reserved_mb: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _latest: ProcessResourceSample | None = None

    def __post_init__(self) -> None:
        try:
            import torch

            self._cuda = bool(torch.cuda.is_available())
        except Exception:
            self._cuda = False

    @property
    def cuda_available(self) -> bool:
        return self._cuda

    def _read_cuda_mb(self) -> tuple[float, float]:
        if not self._cuda:
            return 0.0, 0.0
        try:
            import torch

            return (
                float(torch.cuda.memory_allocated() / (1024 * 1024)),
                float(torch.cuda.memory_reserved() / (1024 * 1024)),
            )
        except Exception:
            return 0.0, 0.0

    def start(self) -> None:
        rss = current_process_rss_mb()
        alloc, reserved = self._read_cuda_mb()
        self._baseline_rss_mb = float(rss or 0.0)
        self._baseline_cuda_alloc_mb = alloc
        self._baseline_cuda_reserved_mb = reserved
        self._samples.clear()
        with self._lock:
            self._latest = None
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(
            target=self._loop, name="yolo-drt-proc-res-mon", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest(self) -> ProcessResourceSample | None:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            rss = current_process_rss_mb()
            alloc, reserved = self._read_cuda_mb()
            sample = ProcessResourceSample(
                t_sec=time.perf_counter() - self._t0,
                process_rss_mb=float(rss or 0.0),
                cuda_allocated_mb=alloc,
                cuda_reserved_mb=reserved,
            )
            self._samples.append(sample)
            with self._lock:
                self._latest = sample

    def summary(self) -> dict[str, float]:
        end_rss = current_process_rss_mb()
        end_alloc, end_reserved = self._read_cuda_mb()
        if self._samples:
            rss_vals = [s.process_rss_mb for s in self._samples if s.process_rss_mb > 0]
            alloc_vals = [s.cuda_allocated_mb for s in self._samples]
            reserved_vals = [s.cuda_reserved_mb for s in self._samples]
            peak_rss = max(rss_vals) if rss_vals else float(end_rss or 0.0)
            peak_alloc = max(alloc_vals) if alloc_vals else end_alloc
            peak_reserved = max(reserved_vals) if reserved_vals else end_reserved
        else:
            peak_rss = float(end_rss or self._baseline_rss_mb)
            peak_alloc = max(end_alloc, self._baseline_cuda_alloc_mb)
            peak_reserved = max(end_reserved, self._baseline_cuda_reserved_mb)

        return {
            "process_rss_baseline_mb": round(self._baseline_rss_mb, 2),
            "process_rss_peak_mb": round(peak_rss, 2),
            "process_rss_end_mb": round(float(end_rss or 0.0), 2),
            "process_rss_delta_peak_mb": round(
                max(0.0, peak_rss - self._baseline_rss_mb), 2
            ),
            "cuda_allocated_baseline_mb": round(self._baseline_cuda_alloc_mb, 2),
            "cuda_allocated_peak_mb": round(peak_alloc, 2),
            "cuda_allocated_end_mb": round(end_alloc, 2),
            "cuda_allocated_delta_peak_mb": round(
                max(0.0, peak_alloc - self._baseline_cuda_alloc_mb), 2
            ),
            "cuda_reserved_peak_mb": round(peak_reserved, 2),
            "cuda_reserved_end_mb": round(end_reserved, 2),
            "samples": float(len(self._samples)),
        }

    def samples_to_dicts(self) -> list[dict[str, float]]:
        return [
            {
                "t_sec": round(s.t_sec, 3),
                "process_rss_mb": round(s.process_rss_mb, 2),
                "cuda_allocated_mb": round(s.cuda_allocated_mb, 2),
                "cuda_reserved_mb": round(s.cuda_reserved_mb, 2),
            }
            for s in self._samples
        ]
