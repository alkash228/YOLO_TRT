"""FastAPI application entrypoint."""
from __future__ import annotations

import os
import shutil
import time
import uuid
import time as _time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.env import load_settings_from_env
from api.jobs import job_manager
from api.schemas import (
    ArtifactsOut,
    ArtifactItem,
    HealthOut,
    JobCreatePathBody,
    JobCreateResponse,
    JobOut,
    JobResultOut,
    ProgressOut,
    RunsOut,
)
from app.config.settings import PipelineSettings
from app.core.pipeline import build_processor
from app.core.run_registry import load_runs_index
from app.core.trt_export import build_all_engines
from app.core.trt_paths import engines_ready, resolve_reid_engine, resolve_yolo_engine
from app.core.video_processor import VideoProcessor

_build_logs: list[str] = []
_app_status: str = "starting"
_engines_ready_map: dict[str, bool] = {}
_settings: PipelineSettings | None = None
_processor: VideoProcessor | None = None


def _log(msg: str) -> None:
    _build_logs.append(msg)
    if len(_build_logs) > 500:
        del _build_logs[: len(_build_logs) - 500]


def _bootstrap() -> None:
    global _app_status, _engines_ready_map, _settings, _processor
    _settings = load_settings_from_env()
    _log(f"Models dir: {_settings.models_dir}")
    _log(f"Output dir: {_settings.output_dir}")
    _log(f"Work dir: {_settings.work_dir}")
    _log(f"Upload dir: {_settings.resolve_upload_dir()}")

    if not torch.cuda.is_available():
        _app_status = "gpu_missing"
        _log("CUDA недоступна — inference невозможен")
        return

    central = Path(_settings.models_dir) / "TRT"
    from app.core.sam_memory_tracker import needs_osnet_embed

    need_osnet = needs_osnet_embed(_settings)
    if _settings.use_tensorrt:
        _engines_ready_map = engines_ready(
            detect_pt=Path(_settings.detect_model),
            cross_pt=Path(_settings.cross_check_model) if _settings.cross_check_model else None,
            reid_pth=Path(_settings.reid_model),
            imgsz=int(_settings.tensorrt_imgsz or 640),
            max_batch=int(_settings.tensorrt_max_batch),
            fp16=bool(_settings.tensorrt_fp16),
            need_cross=bool(_settings.cross_check_enabled),
            need_reid=need_osnet,
            strategy=str(_settings.tensorrt_engine_strategy),
            central_dir=central,
        )
        if not all(_engines_ready_map.values()):
            _app_status = "building_engines"
            _log("Не все TensorRT engines найдены — запуск сборки...")
            build_all_engines(_settings, log=_log)
            _engines_ready_map = engines_ready(
                detect_pt=Path(_settings.detect_model),
                cross_pt=Path(_settings.cross_check_model) if _settings.cross_check_model else None,
                reid_pth=Path(_settings.reid_model),
                imgsz=int(_settings.tensorrt_imgsz or 640),
                max_batch=int(_settings.tensorrt_max_batch),
                fp16=bool(_settings.tensorrt_fp16),
                need_cross=bool(_settings.cross_check_enabled),
                need_reid=need_osnet,
                strategy=str(_settings.tensorrt_engine_strategy),
                central_dir=central,
            )
        if not all(_engines_ready_map.values()):
            _log(f"TensorRT engines не готовы: {_engines_ready_map}")

    _log("Загрузка processor...")
    _processor = build_processor(_settings, warmup=True, on_log=_log)
    job_manager.set_processor(_processor, _settings)
    _app_status = "ready"
    identity = (
        f"sam={_settings.use_sam_identity} reid={_settings.use_reid} "
        f"tracklet_link={_settings.use_offline_tracklet_link}"
    )
    _log(f"Сервис готов ({identity})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _bootstrap()
    except Exception as exc:
        global _app_status
        _app_status = "error"
        _log(f"Bootstrap error: {exc}")

    from app.core.network_urls import print_listen_banner

    api_port = int(os.environ.get("PORT", "8080"))
    print_listen_banner(
        service="YOLO_DRT Docker API",
        host="0.0.0.0",
        port=api_port,
        extra_lines=("Swagger: /docs · Health: /health",),
    )
    yield


app = FastAPI(title="YOLO_DRT API", version="1.0.0", lifespan=lifespan)


def _job_to_out(job) -> JobOut:
    if getattr(job, "bench_mode", False) and job.status == "running":
        return JobOut(
            job_id=job.job_id,
            status=job.status,
            progress=ProgressOut(phase=str(getattr(job, "prog_phase", None) or "inference")),
            result=None,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )
    if job.status == "running" and hasattr(job, "materialize_progress"):
        prog_raw = job.materialize_progress()
    else:
        prog_raw = job.progress or {}
        if not prog_raw and hasattr(job, "materialize_progress"):
            prog_raw = job.materialize_progress()
    progress = ProgressOut(
        current=int(prog_raw.get("current", 0)),
        total=int(prog_raw.get("total", 0)),
        fps=float(prog_raw.get("fps", 0.0)),
        eta_seconds=float(prog_raw.get("eta_seconds", 0.0)),
        elapsed_sec=float(prog_raw.get("elapsed_sec", 0.0)),
        gpu_mem_mb=float(prog_raw.get("gpu_mem_mb", 0.0)),
        cuda_allocated_mb=float(prog_raw.get("cuda_allocated_mb", prog_raw.get("gpu_mem_mb", 0.0))),
        cuda_reserved_mb=float(prog_raw.get("cuda_reserved_mb", 0.0)),
        process_rss_mb=float(prog_raw.get("process_rss_mb", 0.0)),
        gpu_device_used_mb=float(prog_raw.get("gpu_device_used_mb", 0.0)),
        gpu_util_pct=float(prog_raw.get("gpu_util_pct", 0.0)),
        instances_current=int(prog_raw.get("instances_current", 0)),
        instances_peak=int(prog_raw.get("instances_peak", 0)),
        percent=float(prog_raw.get("percent", 0.0)),
        phase=str(prog_raw.get("phase") or "inference"),
    )
    result_out = None
    if job.result is not None or job.error:
        result_out = JobResultOut(
            out_dir=(job.result or {}).get("out_dir"),
            run_id=(job.result or {}).get("run_id"),
            frames=int((job.result or {}).get("frames", 0)),
            elapsed_sec=float((job.result or {}).get("elapsed_sec", 0.0)),
            fps_processed=float((job.result or {}).get("fps_processed", 0.0)),
            artifacts=(job.result or {}).get("artifacts") or {},
            record=(job.result or {}).get("record"),
            error=job.error or (job.result or {}).get("error"),
        )
    return JobOut(
        job_id=job.job_id,
        status=job.status,
        progress=progress,
        result=result_out,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _health_paths() -> dict[str, str]:
    if _settings is None:
        return {}
    s = _settings
    central = Path(s.models_dir) / "TRT"
    strategy = str(s.tensorrt_engine_strategy)
    imgsz = int(s.tensorrt_imgsz or 640)
    max_batch = int(s.tensorrt_max_batch)
    fp16 = bool(s.tensorrt_fp16)
    det_pt = Path(s.detect_model)
    cross_pt = Path(s.cross_check_model) if s.cross_check_model else None
    reid_pth = Path(s.reid_model)
    paths = {
        "models_dir": str(s.models_dir),
        "output_dir": str(s.output_dir),
        "work_dir": str(s.work_dir),
        "upload_dir": str(s.resolve_upload_dir()),
        "trt_strategy": strategy,
        "trt_central_dir": str(central),
        "detect_weights": str(det_pt),
        "cross_weights": str(cross_pt) if cross_pt else "",
        "reid_weights": str(reid_pth),
        "reid_weights_exists": str(reid_pth.is_file()),
        "use_sam_identity": str(bool(s.use_sam_identity)),
        "use_reid": str(bool(s.use_reid)),
        "use_offline_tracklet_link": str(bool(s.use_offline_tracklet_link)),
        "tracklet_link_use_reid": str(bool(s.tracklet_link_use_reid)),
        "sam_osnet_reentry": str(bool(s.sam_osnet_reentry)),
        "cross_check_enabled": str(bool(s.cross_check_enabled)),
        "use_tensorrt": str(bool(s.use_tensorrt)),
        "realtime_mode": str(bool(s.realtime_mode)),
        "encode_mode": str(s.encode_mode),
    }
    det_eng = resolve_yolo_engine(
        det_pt, imgsz=imgsz, max_batch=max_batch, fp16=fp16,
        strategy=strategy, central_dir=central,
    )
    paths["detect_engine"] = str(det_eng)
    paths["detect_engine_exists"] = str(det_eng.exists())
    if cross_pt is not None:
        cross_eng = resolve_yolo_engine(
            cross_pt, imgsz=imgsz, max_batch=max_batch, fp16=fp16,
            strategy=strategy, central_dir=central,
        )
        paths["cross_engine"] = str(cross_eng)
        paths["cross_engine_exists"] = str(cross_eng.exists())
    reid_eng = resolve_reid_engine(
        reid_pth, fp16=fp16, strategy=strategy, central_dir=central,
    )
    paths["reid_engine"] = str(reid_eng)
    paths["reid_engine_exists"] = str(reid_eng.exists())
    return paths


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    msg = ""
    if _app_status == "gpu_missing":
        msg = "CUDA/GPU not available"
    elif _app_status == "error":
        msg = "Bootstrap failed — see build_logs"
    elif _app_status == "ready" and _settings and _settings.use_tensorrt:
        if not all(_engines_ready_map.values()):
            msg = "Some TensorRT engines missing — PyTorch fallback possible"
    return HealthOut(
        status=_app_status,  # type: ignore[arg-type]
        engines_ready=dict(_engines_ready_map),
        build_logs=_build_logs[-80:],
        message=msg,
        paths=_health_paths(),
    )


@app.post("/v1/jobs/upload", response_model=JobCreateResponse)
async def create_job_upload(
    file: UploadFile = File(...),
    prompt: str = Form("person"),
    max_duration_seconds: float | None = Form(None),
) -> JobCreateResponse:
    if _app_status not in ("ready", "building_engines"):
        raise HTTPException(503, f"Service not ready: {_app_status}")
    if _settings is None:
        raise HTTPException(503, "Settings not loaded")

    uploads = _settings.resolve_upload_dir()
    uploads.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = uploads / f"{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job = job_manager.create_job(
        str(dest),
        prompt=prompt,
        max_duration_seconds=max_duration_seconds,
    )
    return JobCreateResponse(job_id=job.job_id, status=job.status)


@app.post("/v1/jobs", response_model=JobCreateResponse)
def create_job_path(body: JobCreatePathBody) -> JobCreateResponse:
    if _app_status not in ("ready", "building_engines"):
        raise HTTPException(503, f"Service not ready: {_app_status}")

    path = Path(body.path)
    if not path.is_file():
        raise HTTPException(400, f"Video not found: {body.path}")

    job = job_manager.create_job(
        str(path.resolve()),
        prompt=body.prompt,
        max_duration_seconds=body.max_duration_seconds,
        bench=bool(getattr(body, "bench", False)),
    )
    return JobCreateResponse(job_id=job.job_id, status=job.status)


@app.post("/v1/jobs/sync", response_model=JobOut)
def create_job_path_sync(body: JobCreatePathBody) -> JobOut:
    """Block until done — no client poll loop (fair UI A/B / GIL isolation)."""
    if _app_status not in ("ready", "building_engines"):
        raise HTTPException(503, f"Service not ready: {_app_status}")
    path = Path(body.path)
    if not path.is_file():
        raise HTTPException(400, f"Video not found: {body.path}")
    job = job_manager.create_job(
        str(path.resolve()),
        prompt=body.prompt,
        max_duration_seconds=body.max_duration_seconds,
        bench=True,
    )
    done = job_manager.wait_job(job.job_id, timeout=7200.0)
    if done is None:
        raise HTTPException(500, "Job disappeared")
    if done.status == "running":
        raise HTTPException(504, "Job timed out")
    return _job_to_out(done)


@app.get("/v1/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return _job_to_out(job)


@app.delete("/v1/jobs/{job_id}", response_model=JobOut)
def cancel_job(job_id: str) -> JobOut:
    job = job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return _job_to_out(job)


@app.get("/v1/jobs/{job_id}/artifacts", response_model=ArtifactsOut)
def list_artifacts(job_id: str) -> ArtifactsOut:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    out_dir = (job.result or {}).get("out_dir")
    if not out_dir:
        raise HTTPException(404, "Artifacts not ready")
    run_dir = Path(out_dir)
    if not run_dir.is_dir():
        raise HTTPException(404, f"Run directory missing: {out_dir}")

    files: list[ArtifactItem] = []
    for p in sorted(run_dir.iterdir()):
        if p.is_file():
            files.append(
                ArtifactItem(
                    name=p.name,
                    path=str(p.resolve()),
                    size_bytes=p.stat().st_size,
                )
            )
    return ArtifactsOut(job_id=job_id, out_dir=str(run_dir.resolve()), files=files)


@app.get("/v1/jobs/{job_id}/artifacts/{filename}")
def download_artifact(job_id: str, filename: str) -> FileResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    out_dir = (job.result or {}).get("out_dir")
    if not out_dir:
        raise HTTPException(404, "Artifacts not ready")
    path = Path(out_dir) / filename
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename)


@app.get("/v1/runs", response_model=RunsOut)
def list_runs() -> RunsOut:
    if _settings is None:
        raise HTTPException(503, "Settings not loaded")
    data = load_runs_index(_settings.output_dir)
    return RunsOut(version=int(data.get("version", 1)), runs=list(data.get("runs", [])))
