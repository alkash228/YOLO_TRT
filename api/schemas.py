"""Pydantic schemas for the inference API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProgressOut(BaseModel):
    current: int = 0
    total: int = 0
    fps: float = 0.0
    eta_seconds: float = 0.0
    elapsed_sec: float = 0.0
    gpu_mem_mb: float = 0.0
    cuda_allocated_mb: float = 0.0
    cuda_reserved_mb: float = 0.0
    process_rss_mb: float = 0.0
    gpu_device_used_mb: float = 0.0
    gpu_util_pct: float = 0.0
    instances_current: int = 0
    instances_peak: int = 0
    percent: float = 0.0
    phase: str = "inference"


class JobResultOut(BaseModel):
    out_dir: str | None = None
    run_id: str | None = None
    frames: int = 0
    elapsed_sec: float = 0.0
    fps_processed: float = 0.0
    artifacts: dict[str, str | None] = Field(default_factory=dict)
    record: dict[str, Any] | None = None
    error: str | None = None


class JobOut(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "cancelled", "error"]
    progress: ProgressOut
    result: JobResultOut | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    debug_logs: list[str] = Field(default_factory=list)
    error: str | None = None


class JobCreatePathBody(BaseModel):
    input_type: Literal["path"] = "path"
    path: str
    prompt: str = "person"
    max_duration_seconds: float | None = None
    # True → no progress ticks; prefer POST /v1/jobs/sync for fair UI A/B.
    bench: bool = False


class JobCreateResponse(BaseModel):
    job_id: str
    status: str = "queued"


class HealthOut(BaseModel):
    status: Literal["ready", "building_engines", "starting", "stopped", "gpu_missing", "error"]
    engines_ready: dict[str, bool] = Field(default_factory=dict)
    build_logs: list[str] = Field(default_factory=list)
    message: str = ""
    paths: dict[str, str] = Field(default_factory=dict)
    processor: dict[str, Any] = Field(default_factory=dict)


class SettingsOut(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)


class SettingsUpdateBody(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    reload_processor: bool = False
    ui_equivalent: bool = False


class BootstrapBody(BaseModel):
    force: bool = False


class ArtifactItem(BaseModel):
    name: str
    path: str
    size_bytes: int


class ArtifactsOut(BaseModel):
    job_id: str
    out_dir: str
    files: list[ArtifactItem] = Field(default_factory=list)


class RunsOut(BaseModel):
    version: int = 1
    runs: list[dict[str, Any]] = Field(default_factory=list)


class ModelsOut(BaseModel):
    detect: list[str] = Field(default_factory=list)
    seg: list[str] = Field(default_factory=list)
    reid: list[str] = Field(default_factory=list)


class RestartBody(BaseModel):
    mode: Literal["docker", "exit"] = "docker"


class RestartOut(BaseModel):
    ok: bool = True
    mode: str = "docker"
    container: str = ""
    message: str = ""
    docker_status: int | None = None
    fallback_from: str | None = None
    docker_error: str | None = None


class JobSummaryOut(BaseModel):
    job_id: str
    status: str
    status_ru: str = ""
    phase: str = ""
    phase_ru: str = ""
    percent: float = 0.0
    prompt: str = ""
    input_path: str = ""
    input_size_human: str = ""
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_human: str = "—"
    eta_human: str = "—"
    error: str | None = None
    progress: ProgressOut = Field(default_factory=ProgressOut)


class AdminStatusOut(BaseModel):
    service_status: str
    service_status_ru: str = ""
    container_name: str = ""
    docker_sock: bool = False
    uptime_sec: float = 0.0
    uptime_human: str = ""
    last_job: JobSummaryOut | None = None
    active_job: JobSummaryOut | None = None
    recent_jobs: list[JobSummaryOut] = Field(default_factory=list)
    recent_requests: list[dict[str, Any]] = Field(default_factory=list)
    disks: list[dict[str, Any]] = Field(default_factory=list)
    tip: str = ""
