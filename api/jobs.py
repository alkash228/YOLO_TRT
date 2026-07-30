"""In-memory job queue (one GPU job at a time)."""
from __future__ import annotations

import os
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.settings import PipelineSettings
from app.core.gpu_cleanup import soft_cleanup_after_job
from app.core.progress_hook import ProgressSlot
from app.core.schema import ProcessVideoResult
from app.core.video_processor import VideoProcessor


def api_speed_test_enabled() -> bool:
    """YOLO_API_SPEED_TEST=1 → no progress ticks, quiet GET, sync-friendly."""
    return os.environ.get("YOLO_API_SPEED_TEST", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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


def _path_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def stage_video_for_job(
    src: str | Path,
    *,
    work_dir: Path,
    job_id: str,
) -> tuple[str, Path | None]:
    """
    Copy external path videos into work_dir/jobs/<id>/ so windowed decode
    hits the same local disk as typical UI / upload flows.

    Returns (path_to_open, staged_dir_or_None). staged_dir is set only when
    a copy was created (caller may delete after the job).
    """
    src_path = Path(src)
    if not src_path.is_file():
        raise FileNotFoundError(f"Video not found: {src}")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # Already under work (uploads / previous stage) — decode in place.
    if _path_under(src_path, work):
        return str(src_path.resolve()), None

    stage_dir = work / "jobs" / job_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    dest = stage_dir / src_path.name
    if dest.exists() and dest.stat().st_size == src_path.stat().st_size:
        return str(dest.resolve()), stage_dir
    shutil.copy2(src_path, dest)
    return str(dest.resolve()), stage_dir


@dataclass
class JobState:
    job_id: str
    input_path: str
    prompt: str
    max_duration_seconds: float | None = None
    status: str = "queued"
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    debug_logs: deque[str] = field(default_factory=lambda: deque(maxlen=120))
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    infer_started_at: float | None = None
    finished_at: float | None = None
    # Lightweight progress (ints only; no VideoProgress on infer/finalizer threads).
    prog_current: int = 0
    prog_total: int = 0
    prog_phase: str = "queued"
    # Bench / speed-test: no ticks, client should use /v1/jobs/sync (no poll storm).
    bench_mode: bool = False
    progress_slot: ProgressSlot = field(default_factory=ProgressSlot)
    _cancel: threading.Event = field(default_factory=threading.Event)
    _done: threading.Event = field(default_factory=threading.Event)

    def should_stop(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()
        if self.status in ("queued", "running"):
            self.status = "cancelled"

    def tick_progress(self, current: int, total: int, phase: str) -> None:
        """Hot-path safe: plain int/str stores; get_job materializes the dict."""
        self.prog_current = int(current)
        self.prog_total = int(total)
        self.prog_phase = str(phase or "inference")

    def materialize_progress(self) -> dict[str, Any]:
        """Build progress dict for GET /jobs (off the infer hot path)."""
        slot = getattr(self, "progress_slot", None)
        if slot is not None and int(slot.total) > 0:
            current = max(0, int(slot.current))
            total = max(0, int(slot.total))
            phase = str(slot.phase or "inference")
        else:
            total = max(0, int(self.prog_total))
            current = max(0, int(self.prog_current))
            phase = str(self.prog_phase or "inference")
        if total <= 0 and self.progress:
            return dict(self.progress)
        elapsed = 0.0
        if self.started_at is not None:
            end = self.finished_at if self.finished_at is not None else time.time()
            # ETA/fps during inference: exclude upload/staging (more stable between runs).
            infer_t0 = self.infer_started_at or self.started_at
            if str(phase).lower() in ("inference", "finalize", "done", "pass2"):
                elapsed = max(0.0, float(end) - float(infer_t0))
            else:
                elapsed = max(0.0, float(end) - float(self.started_at))
        fps = (current / elapsed) if elapsed > 1e-6 and current > 0 else 0.0
        eta = ((total - current) / fps) if fps > 1e-6 and total > current else 0.0
        percent = round(100.0 * current / total, 2) if total > 0 else 0.0
        return {
            "current": current,
            "total": total,
            "fps": float(fps),
            "eta_seconds": float(eta),
            "elapsed_sec": float(elapsed),
            # CUDA bytes held by this process (torch allocator), not whole-GPU NVML.
            "gpu_mem_mb": float(getattr(self.progress_slot, "cuda_allocated_mb", 0.0) or 0.0),
            "cuda_allocated_mb": float(getattr(self.progress_slot, "cuda_allocated_mb", 0.0) or 0.0),
            "cuda_reserved_mb": float(getattr(self.progress_slot, "cuda_reserved_mb", 0.0) or 0.0),
            "process_rss_mb": float(getattr(self.progress_slot, "process_rss_mb", 0.0) or 0.0),
            "gpu_device_used_mb": float(getattr(self.progress_slot, "gpu_device_used_mb", 0.0) or 0.0),
            "gpu_util_pct": float(getattr(self.progress_slot, "gpu_util_pct", 0.0) or 0.0),
            "instances_current": 0,
            "instances_peak": 0,
            "percent": percent,
            "phase": phase,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker_started = False
        self._processor: VideoProcessor | None = None
        self._settings: PipelineSettings | None = None

    def set_processor(self, processor: VideoProcessor, settings: PipelineSettings) -> None:
        self._processor = processor
        self._settings = settings
        self._ensure_worker()

    def clear_processor(self) -> None:
        with self._lock:
            self._processor = None
            self._settings = None

    def _ensure_worker(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        t = threading.Thread(target=self._worker_loop, name="yolo-api-job-worker", daemon=True)
        t.start()

    def create_job(
        self,
        input_path: str,
        *,
        prompt: str = "person",
        max_duration_seconds: float | None = None,
        bench: bool = False,
    ) -> JobState:
        job_id = uuid.uuid4().hex
        job = JobState(
            job_id=job_id,
            input_path=str(input_path),
            prompt=prompt,
            max_duration_seconds=max_duration_seconds,
            bench_mode=bool(bench) or api_speed_test_enabled(),
        )
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job

    def wait_job(self, job_id: str, *, timeout: float | None = 3600.0) -> JobState | None:
        """Block until job finishes (Event.wait releases GIL — no poll storm)."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        job._done.wait(timeout=timeout)
        with self._lock:
            return self._jobs.get(job_id)

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None and job.status == "running" and not job.bench_mode:
            # Refresh progress dict from atomic counters for pollers.
            job.progress = job.materialize_progress()
        return job

    def cancel(self, job_id: str) -> JobState | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        job.cancel()
        return job

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return
        if job.status == "cancelled":
            job.finished_at = time.time()
            job._done.set()
            return

        # Desktop embedded API: sync via server_state (shared UI processor).
        # Standalone Docker API: fall back to set_processor settings.
        settings: PipelineSettings | None = None
        processor: VideoProcessor | None = None
        prep_err: str | None = None
        try:
            from api.server_state import server_state  # type: ignore

            settings, prep_err = server_state.prepare_processor_for_job()
            with self._lock:
                processor = self._processor
        except ImportError:
            from app.core.pipeline import sync_processor_for_run

            with self._lock:
                settings = self._settings
                processor = self._processor
            if settings is None or processor is None:
                prep_err = "Processor not loaded"
            else:
                try:
                    sync_processor_for_run(processor, settings)
                except Exception as exc:  # noqa: BLE001 — surface as job error
                    prep_err = str(exc)

        if prep_err or settings is None or processor is None:
            job.status = "error"
            job.error = prep_err or "Processor not loaded"
            job.finished_at = time.time()
            job._done.set()
            return

        job.status = "running"
        job.started_at = time.time()
        job.debug_logs.clear()
        job.prog_current = 0
        job.prog_total = 0
        job.prog_phase = "staging"
        job.progress_slot.force(0, 0, "staging")
        if not job.bench_mode:
            job.progress = job.materialize_progress()

        prompt = resolve_job_prompt(settings, job.prompt)
        max_dur = resolve_job_max_duration(settings, job.max_duration_seconds)

        staged_dir: Path | None = None
        video_path = job.input_path
        try:
            video_path, staged_dir = stage_video_for_job(
                job.input_path,
                work_dir=Path(settings.work_dir),
                job_id=job.job_id,
            )
            if staged_dir is not None:
                job.debug_logs.append(f"Staged input → {video_path}")
            job.progress_slot.force(0, 0, "inference")
            if not job.bench_mode:
                job.progress = job.materialize_progress()

            job.infer_started_at = time.time()
            progress_slot = None if job.bench_mode else job.progress_slot
            result: ProcessVideoResult = processor.process_video(
                video_path,
                settings.output_dir,
                prompt,
                max_duration_seconds=max_dur,
                on_progress=None,
                on_debug_log=None,
                on_progress_tick=None,
                progress_slot=progress_slot,
                should_stop=job.should_stop,
            )
            payload = {
                "out_dir": result.out_dir,
                "run_id": result.run_id,
                "frames": result.frames,
                "elapsed_sec": result.elapsed_sec,
                "fps_processed": result.fps_processed,
                "artifacts": result.record.get("artifacts", {}),
                "record": result.record,
            }
            if job.should_stop():
                job.status = "cancelled"
            else:
                job.status = "done"
            job.result = payload
            st = (result.record or {}).get("stats_summary") or {}
            job.debug_logs.append(
                "Timing: "
                f"infer {float(st.get('elapsed_infer_sec') or 0):.1f}s, "
                f"pass2 {float(st.get('elapsed_pass2_sec') or 0):.1f}s, "
                f"wall {float(result.elapsed_sec or 0):.1f}s"
            )
            pm = st.get("process_memory") or {}
            if pm:
                job.debug_logs.append(
                    "Memory (this process): "
                    f"RAM peak {float(pm.get('process_rss_peak_mb') or 0):.0f} MB, "
                    f"CUDA alloc peak {float(pm.get('cuda_allocated_peak_mb') or 0):.0f} MB"
                )
            # Mark progress complete from slot or result frames.
            done_frames = int(result.frames or 0)
            if job.progress_slot.total > 0:
                job.progress_slot.force(
                    job.progress_slot.total, job.progress_slot.total, "done"
                )
            elif done_frames > 0:
                job.progress_slot.force(done_frames, done_frames, "done")
            job.prog_phase = "done"
            job.progress = job.materialize_progress()
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.result = {"error": str(exc)}
            job.prog_phase = "error"
            job.progress = job.materialize_progress()
            job.debug_logs.append(f"Job error: {exc}")
        finally:
            if processor is not None:
                try:
                    gpu_stats = soft_cleanup_after_job(processor)
                    before = float(gpu_stats.get("before_mb") or 0.0)
                    after = float(gpu_stats.get("after_mb") or 0.0)
                    freed = float(gpu_stats.get("freed_mb") or 0.0)
                    reserved = float(gpu_stats.get("reserved_mb") or 0.0)
                    reserved_before = float(gpu_stats.get("reserved_before_mb") or 0.0)
                    reserved_freed = float(gpu_stats.get("reserved_freed_mb") or 0.0)
                    job.debug_logs.append(
                        f"GPU cleanup: warm TRT kept (alloc {before:.0f} MB, "
                        f"reserved {reserved:.0f} MB) — restart API for full VRAM release"
                    )
                except Exception as exc:
                    job.debug_logs.append(f"GPU cleanup failed: {exc}")
            job.finished_at = time.time()
            job._done.set()
            if staged_dir is not None:
                try:
                    shutil.rmtree(staged_dir, ignore_errors=True)
                except OSError:
                    pass


job_manager = JobManager()
