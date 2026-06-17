"""In-memory job queue and GPU worker (one job at a time)."""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.settings import PipelineSettings
from app.core.schema import ProcessVideoResult, VideoProgress
from app.core.video_processor import VideoProcessor


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
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)

    def should_stop(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()
        if self.status in ("queued", "running"):
            self.status = "cancelled"


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

    def _ensure_worker(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        t = threading.Thread(target=self._worker_loop, name="yolo-job-worker", daemon=True)
        t.start()

    def create_job(
        self,
        input_path: str,
        *,
        prompt: str = "person",
        max_duration_seconds: float | None = None,
    ) -> JobState:
        job_id = uuid.uuid4().hex
        job = JobState(
            job_id=job_id,
            input_path=str(input_path),
            prompt=prompt,
            max_duration_seconds=max_duration_seconds,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> JobState | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        job.cancel()
        return job

    def _update_progress(self, job: JobState, prog: VideoProgress) -> None:
        total = max(1, int(prog.total))
        current = int(prog.current)
        percent = round(100.0 * current / total, 2) if total > 0 else 0.0
        job.progress = {
            "current": current,
            "total": int(prog.total),
            "fps": float(prog.fps),
            "eta_seconds": float(prog.eta_seconds),
            "elapsed_sec": float(prog.elapsed_sec),
            "gpu_mem_mb": float(prog.gpu_mem_mb),
            "gpu_util_pct": float(prog.gpu_util_pct),
            "instances_current": int(prog.instances_current),
            "instances_peak": int(prog.instances_peak),
            "percent": percent,
        }

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
            processor = self._processor
            settings = self._settings
        if job is None or processor is None or settings is None:
            return
        if job.status == "cancelled":
            job.finished_at = time.time()
            return

        job.status = "running"
        job.started_at = time.time()

        def on_progress(prog: VideoProgress) -> None:
            self._update_progress(job, prog)

        try:
            result: ProcessVideoResult = processor.process_video(
                job.input_path,
                settings.output_dir,
                job.prompt,
                max_duration_seconds=job.max_duration_seconds,
                on_progress=on_progress,
                should_stop=job.should_stop,
            )
            if job.should_stop():
                job.status = "cancelled"
                job.result = {
                    "out_dir": result.out_dir,
                    "run_id": result.run_id,
                    "frames": result.frames,
                    "elapsed_sec": result.elapsed_sec,
                    "fps_processed": result.fps_processed,
                    "artifacts": result.record.get("artifacts", {}),
                    "record": result.record,
                }
            else:
                job.status = "done"
                job.result = {
                    "out_dir": result.out_dir,
                    "run_id": result.run_id,
                    "frames": result.frames,
                    "elapsed_sec": result.elapsed_sec,
                    "fps_processed": result.fps_processed,
                    "artifacts": result.record.get("artifacts", {}),
                    "record": result.record,
                }
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.result = {"error": str(exc)}
        finally:
            job.finished_at = time.time()


job_manager = JobManager()
