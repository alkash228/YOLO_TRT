"""FastAPI application entrypoint."""
from __future__ import annotations

import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from api.admin_html import ADMIN_HTML
from api.admin_ops import (
    container_name,
    disk_info,
    docker_available,
    fetch_container_logs,
    fmt_bytes,
    fmt_sec,
    phase_ru,
    restart_container,
    status_ru,
)
from api.env import load_settings_from_env
from api.jobs import job_manager
from api.request_log import RequestRecord, request_log
from api.schemas import (
    AdminStatusOut,
    ArtifactsOut,
    ArtifactItem,
    HealthOut,
    JobCreatePathBody,
    JobCreateResponse,
    JobOut,
    JobResultOut,
    JobSummaryOut,
    ProgressOut,
    RestartBody,
    RestartOut,
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
_started_at: float = time.time()


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
    from app.core.reid_engine import resolve_reid_backend
    from app.core.sam_memory_tracker import needs_osnet_embed

    # SOLIDER is PyTorch-only — do not require an OSNet-style .engine for it.
    reid_backend = resolve_reid_backend(
        getattr(_settings, "reid_backend", None), _settings.reid_model
    )
    need_reid_trt = bool(needs_osnet_embed(_settings)) and reid_backend == "osnet"
    if _settings.use_tensorrt:
        _engines_ready_map = engines_ready(
            detect_pt=Path(_settings.detect_model),
            cross_pt=Path(_settings.cross_check_model) if _settings.cross_check_model else None,
            reid_pth=Path(_settings.reid_model),
            imgsz=int(_settings.tensorrt_imgsz or 640),
            max_batch=int(_settings.tensorrt_max_batch),
            fp16=bool(_settings.tensorrt_fp16),
            need_cross=bool(_settings.cross_check_enabled),
            need_reid=need_reid_trt,
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
                need_reid=need_reid_trt,
                strategy=str(_settings.tensorrt_engine_strategy),
                central_dir=central,
            )
        if not all(_engines_ready_map.values()):
            _log(f"TensorRT engines не готовы: {_engines_ready_map}")

    _log("Загрузка processor...")
    _processor = build_processor(_settings, warmup=True, on_log=_log)
    job_manager.set_processor(_processor, _settings)
    # Keep api.server_state in sync — jobs.py prefers it when the module exists.
    try:
        from api.server_state import server_state
        from app.core.shared_processor import attach as shared_attach

        server_state.configure(_settings)
        shared_attach(_processor, _settings, "api")
        with server_state._lock:
            server_state.processor = _processor
            server_state.status = "ready"
            server_state.engines_ready = dict(_engines_ready_map)
        job_manager.set_processor(_processor, _settings)
    except Exception as exc:  # noqa: BLE001
        _log(f"server_state sync skipped: {exc}")
    _app_status = "ready"
    identity = (
        f"sam={_settings.use_sam_identity} reid={_settings.use_reid} "
        f"tracklet_link={_settings.use_offline_tracklet_link} "
        f"reid_backend={reid_backend}"
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
        extra_lines=(
            "Swagger: /docs · Health: /health · Admin: /admin",
        ),
    )
    yield


app = FastAPI(title="YOLO_DRT API", version="1.1.0", lifespan=lifespan)


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    err = ""
    status_code = 500
    try:
        response = await call_next(request)
        status_code = int(response.status_code)
        return response
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        raise
    finally:
        path = request.url.path
        # Skip noisy admin polling of static assets if any
        if not path.startswith("/assets"):
            ms = (time.perf_counter() - t0) * 1000.0
            client = ""
            if request.client:
                client = request.client.host or ""
            request_log.add(
                RequestRecord(
                    method=request.method,
                    path=path,
                    status_code=status_code,
                    duration_ms=ms,
                    ts=time.time(),
                    client=client,
                    error=err,
                )
            )


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


def _sync_timeout_sec() -> float:
    raw = os.environ.get("YOLO_DRT_SYNC_TIMEOUT_SEC", "86400").strip() or "86400"
    try:
        return max(60.0, float(raw))
    except ValueError:
        return 86400.0


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
    disk = disk_info(str(uploads))
    free = int(disk.get("free_bytes") or 0)
    if free and free < 2 * (1024**3):
        raise HTTPException(
            507,
            f"Мало места на upload-диске ({disk.get('free_human')}). "
            "Для длинных роликов положи файл в ./videos и используй path "
            "/data/videos/... (без upload).",
        )

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = uploads / f"{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    written = 0
    chunk = 8 * 1024 * 1024
    try:
        with dest.open("wb") as out:
            while True:
                buf = await file.read(chunk)
                if not buf:
                    break
                out.write(buf)
                written += len(buf)
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            507,
            f"Не удалось сохранить upload ({written} bytes): {exc}. "
            "Для 5ч видео используй path /data/videos/... вместо upload.",
        ) from exc

    if written <= 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Пустой файл upload")

    _log(f"Upload saved {fmt_bytes(written)} → {dest}")
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
    done = job_manager.wait_job(job.job_id, timeout=_sync_timeout_sec())
    if done is None:
        raise HTTPException(500, "Job disappeared")
    if done.status == "running":
        raise HTTPException(
            504,
            f"Job timed out after {_sync_timeout_sec():.0f}s "
            "(увеличь YOLO_DRT_SYNC_TIMEOUT_SEC для длинных роликов)",
        )
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


def _job_summary(job) -> JobSummaryOut:
    out = _job_to_out(job)
    prog = out.progress
    size_h = "—"
    try:
        p = Path(job.input_path)
        if p.is_file():
            size_h = fmt_bytes(p.stat().st_size)
    except OSError:
        pass
    elapsed = None
    if job.started_at is not None:
        end = job.finished_at if job.finished_at is not None else time.time()
        elapsed = max(0.0, float(end) - float(job.started_at))
    return JobSummaryOut(
        job_id=job.job_id,
        status=job.status,
        status_ru=status_ru(job.status),
        phase=str(prog.phase or ""),
        phase_ru=phase_ru(str(prog.phase or "")),
        percent=float(prog.percent or 0.0),
        prompt=str(job.prompt or ""),
        input_path=str(job.input_path or ""),
        input_size_human=size_h,
        created_at=float(job.created_at or 0.0),
        started_at=job.started_at,
        finished_at=job.finished_at,
        elapsed_human=fmt_sec(elapsed),
        eta_human=fmt_sec(float(prog.eta_seconds or 0.0) or None),
        error=job.error or (out.result.error if out.result else None),
        progress=prog,
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> HTMLResponse:
    return HTMLResponse(ADMIN_HTML)


@app.get("/v1/admin/status", response_model=AdminStatusOut)
def admin_status() -> AdminStatusOut:
    up = max(0.0, time.time() - _started_at)
    last = job_manager.latest()
    active = job_manager.active_job()
    disks: list[dict[str, Any]] = []
    if _settings is not None:
        for p in (
            str(_settings.work_dir),
            str(_settings.output_dir),
            str(_settings.resolve_upload_dir()),
            "/data/videos",
        ):
            disks.append(disk_info(p))
    tip = (
        "Длинное видео (несколько часов): положи в ./videos на хосте и создай job "
        'POST /v1/jobs {"path":"/data/videos/name.mp4"} — без HTTP upload и без '
        "копирования в volume. Upload годится для коротких роликов."
    )
    return AdminStatusOut(
        service_status=_app_status,
        service_status_ru=status_ru(_app_status),
        container_name=container_name(),
        docker_sock=docker_available(),
        uptime_sec=round(up, 1),
        uptime_human=fmt_sec(up),
        last_job=_job_summary(last) if last else None,
        active_job=_job_summary(active) if active else None,
        recent_jobs=[_job_summary(j) for j in job_manager.list_jobs(limit=15)],
        recent_requests=request_log.recent(40),
        disks=disks,
        tip=tip,
    )


@app.get("/v1/admin/jobs/latest", response_model=JobSummaryOut)
def admin_latest_job() -> JobSummaryOut:
    job = job_manager.active_job() or job_manager.latest()
    if job is None:
        raise HTTPException(404, "Нет jobs")
    return _job_summary(job)


@app.post("/v1/admin/jobs/latest/cancel", response_model=JobSummaryOut)
def admin_cancel_latest() -> JobSummaryOut:
    job = job_manager.active_job() or job_manager.latest()
    if job is None:
        raise HTTPException(404, "Нет jobs для отмены")
    job_manager.cancel(job.job_id)
    refreshed = job_manager.get(job.job_id) or job
    return _job_summary(refreshed)


@app.post("/v1/admin/restart", response_model=RestartOut)
def admin_restart(body: RestartBody | None = None) -> RestartOut:
    mode = (body.mode if body else "docker")
    try:
        info = restart_container(mode=mode)
    except Exception as exc:
        raise HTTPException(500, f"Рестарт не удался: {exc}") from exc
    return RestartOut(**info)


@app.get("/v1/admin/requests")
def admin_requests(limit: int = 50) -> JSONResponse:
    return JSONResponse({"requests": request_log.recent(limit)})


@app.get("/v1/admin/logs")
def admin_container_logs(
    tail: int = 2000,
    timestamps: bool = True,
    download: bool = True,
) -> Response:
    """
    Логи Docker-контейнера как text/plain (.txt).

    Нужен volume /var/run/docker.sock (уже в compose).
    Пример: GET /v1/admin/logs?tail=5000 → скачать yolo-drt-api_logs_….txt
    """
    try:
        filename, text = fetch_container_logs(tail=tail, timestamps=timestamps)
    except Exception as exc:
        # Fallback: хотя бы bootstrap/build_logs из процесса
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        filename = f"{container_name()}_app_logs_{stamp}.txt"
        text = (
            f"# WARN: Docker logs недоступны: {exc}\n"
            f"# Ниже — in-process build_logs (не полный docker log).\n"
            f"# ---\n" + "\n".join(_build_logs[-max(50, min(tail, 2000)) :])
        )
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )
