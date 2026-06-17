"""Background NVIDIA GPU utilization sampling during a run."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class GpuSample:
    t_sec: float
    gpu_util_pct: float
    mem_util_pct: float
    mem_used_mb: float


@dataclass
class GpuMonitor:
    interval_sec: float = 0.5
    _samples: list[GpuSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _t0: float = 0.0
    _available: bool = False
    _handle: object | None = None
    _pynvml: object | None = None

    def __post_init__(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._available = True
        except Exception:
            self._pynvml = None
            self._handle = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available:
            return
        self._samples.clear()
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="yolo-drt-gpu-mon", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest(self) -> GpuSample | None:
        return self._samples[-1] if self._samples else None

    def _loop(self) -> None:
        assert self._pynvml is not None and self._handle is not None
        pynvml = self._pynvml
        while not self._stop.wait(self.interval_sec):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._samples.append(
                    GpuSample(
                        t_sec=time.perf_counter() - self._t0,
                        gpu_util_pct=float(util.gpu),
                        mem_util_pct=float(util.memory),
                        mem_used_mb=float(mem.used) / (1024 * 1024),
                    )
                )
            except Exception:
                break

    def summary(self) -> dict[str, float] | None:
        if not self._samples:
            return None
        gpu = [s.gpu_util_pct for s in self._samples]
        mem = [s.mem_used_mb for s in self._samples]
        return {
            "avg_gpu_util_pct": round(sum(gpu) / len(gpu), 2),
            "peak_gpu_util_pct": round(max(gpu), 2),
            "avg_mem_used_mb": round(sum(mem) / len(mem), 2),
            "peak_mem_used_mb": round(max(mem), 2),
            "samples": float(len(self._samples)),
        }

    def samples_to_dicts(self) -> list[dict[str, float]]:
        return [
            {
                "t_sec": round(s.t_sec, 3),
                "gpu_util_pct": s.gpu_util_pct,
                "mem_util_pct": s.mem_util_pct,
                "mem_used_mb": round(s.mem_used_mb, 2),
            }
            for s in self._samples
        ]
