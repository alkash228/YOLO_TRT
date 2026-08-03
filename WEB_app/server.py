"""WEB_app FastAPI server: static UI + API proxy + violations video build."""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC_DIR = Path(__file__).resolve().parent / "static"


_env_api = os.environ.get("YOLO_DRT_API_URL", "").strip()
WEB_HOST = os.environ.get("WEB_APP_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_APP_PORT", "8088"))

# Prefer Docker :8080 first when both are up (desktop :8765 still wins if only it answers).
_API_PROBE_PORTS = (8080, 8765)
_api_url_cache: str | None = None
_api_url_cache_at: float = 0.0
_API_URL_TTL_SEC = 5.0


def _probe_api(url: str) -> tuple[bool, str]:
    """Return (ok, status_string). ok=True on HTTP 200 /health."""
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{url.rstrip('/')}/health")
            if response.status_code != 200:
                return False, ""
            try:
                status = str((response.json() or {}).get("status") or "")
            except Exception:
                status = ""
            return True, status
    except httpx.HTTPError:
        return False, ""


def _resolve_api_url(explicit: str | None = None) -> str:
    """Resolve API base; re-probes live ports (not frozen at import)."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit.rstrip("/"))
    env = os.environ.get("YOLO_DRT_API_URL", "").strip()
    if env:
        candidates.append(env.rstrip("/"))
    for port in _API_PROBE_PORTS:
        candidates.append(f"http://127.0.0.1:{port}")

    # Dedupe, keep order
    seen: set[str] = set()
    ordered: list[str] = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            ordered.append(u)

    ready_url: str | None = None
    alive_url: str | None = None
    for url in ordered:
        ok, status = _probe_api(url)
        if not ok:
            continue
        if status == "ready":
            return url
        if alive_url is None:
            alive_url = url
        if ready_url is None and status:
            ready_url = url
    # Prefer a live non-ready API over a dead env default (old bat forced :8765).
    return alive_url or ready_url or "http://127.0.0.1:8080"


def api_base() -> str:
    """Live API target with short TTL cache (Docker may start after WEB)."""
    global _api_url_cache, _api_url_cache_at

    now = time.time()
    if _api_url_cache and (now - _api_url_cache_at) < _API_URL_TTL_SEC:
        return _api_url_cache
    url = _resolve_api_url(_env_api or None)
    _api_url_cache = url
    _api_url_cache_at = now
    return url


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Suppress harmless client-abort errors while streaming MP4 previews (Windows)."""
    loop = asyncio.get_running_loop()
    default_handler = loop.get_exception_handler()

    def handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return
        if default_handler is not None:
            default_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)
    yield


app = FastAPI(title="YOLO_DRT WEB_app", version="1.0.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_build_lock = threading.Lock()
_build_jobs: dict[str, dict[str, Any]] = {}

# Docker API returns container paths (/data/output/...); WEB runs on the host.
_DOCKER_OUTPUT_PREFIXES = ("/data/output",)


def _host_output_root() -> Path:
    raw = os.environ.get("YOLO_DRT_HOST_OUTPUT_DIR", "").strip()
    if raw:
        return Path(raw)
    return ROOT / "output"


def _resolve_host_run_dir(run_dir: str | Path) -> Path:
    """Map Docker /data/output/... to the Windows host bind-mount path."""
    raw = str(run_dir or "").strip().replace("\\", "/")
    if not raw:
        return Path(raw)
    p = Path(raw)
    if p.is_dir():
        return p
    for prefix in _DOCKER_OUTPUT_PREFIXES:
        if raw == prefix or raw.startswith(prefix + "/"):
            rel = raw[len(prefix) :].lstrip("/")
            mapped = _host_output_root() / rel if rel else _host_output_root()
            if mapped.is_dir():
                return mapped
            return mapped
    return p


def _run_dir_query(run_dir: str) -> str:
    """Normalize Windows paths for URL query (?run_dir=)."""
    return quote(str(Path(run_dir).resolve()).replace("\\", "/"), safe="/:")


def _video_item(stable_id: int | str, video_name: str, run_dir: str) -> dict[str, str | int]:
    run_q = _run_dir_query(run_dir)
    return {
        "stable_id": int(stable_id) if str(stable_id).isdigit() else stable_id,
        "video_name": video_name,
        "video_url": f"/videos/{quote(video_name)}?run_dir={run_q}",
    }


class BuildVideoBody(BaseModel):
    run_dir: str
    run_id: str
    overlay: dict[str, Any] | None = None


class BuildVideoResponse(BaseModel):
    build_id: str
    status: str
    video_url: str | None = None
    message: str = ""
    info: dict[str, Any] = Field(default_factory=dict)


class WordReportBody(BaseModel):
    run_dir: str
    run_id: str
    stable_id: int
    company: str = ""
    organization: str = ""
    incident_datetime: str = ""


class WordReportResponse(BaseModel):
    ok: bool = True
    download_url: str
    filename: str


class SettingsUpdateProxy(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    reload_processor: bool = True
    ui_equivalent: bool = False


class BootstrapProxy(BaseModel):
    force: bool = False


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/config")
def config() -> dict[str, str]:
    return {"api_url": api_base(), "web_url": f"http://127.0.0.1:{WEB_PORT}"}


@app.get("/access-info")
def access_info() -> dict[str, Any]:
    from app.core.network_urls import lan_ipv4_addresses, listen_urls

    api_target = api_base()
    try:
        api_port = int(api_target.rsplit(":", 1)[-1])
    except ValueError:
        api_port = 8765

    return {
        "web_urls": listen_urls(WEB_HOST, WEB_PORT),
        "api_proxy_target": api_target,
        "api_base_urls": listen_urls("127.0.0.1", api_port),
        "lan_ips": lan_ipv4_addresses(),
        "bind_host": WEB_HOST,
        "web_port": WEB_PORT,
        "api_port": api_port,
    }


@app.get("/proxy/health")
async def proxy_health() -> dict[str, Any]:
    # Bust short cache so UI reconnects when Docker comes up after WEB.
    global _api_url_cache_at
    _api_url_cache_at = 0.0
    base = api_base()
    url = f"{base}/health"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = {**data, "api_proxy_target": base}
            return data
    except httpx.HTTPError as exc:
        raise HTTPException(
            502, f"API unreachable ({base}): {exc}"
        ) from exc


@app.get("/proxy/settings")
async def proxy_get_settings() -> dict[str, Any]:
    """Desktop API has /v1/settings; Docker API uses env — soft-fallback."""
    url = f"{api_base()}/v1/settings"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            if r.status_code == 404:
                return {
                    "settings": {},
                    "docker_mode": True,
                    "message": "Docker API: settings via env / compose (no /v1/settings)",
                }
            if r.status_code >= 400:
                raise HTTPException(r.status_code, r.text)
            return r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"API settings GET failed: {exc}") from exc


@app.put("/proxy/settings")
async def proxy_put_settings(body: SettingsUpdateProxy) -> dict[str, Any]:
    url = f"{api_base()}/v1/settings"
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.put(url, json=body.model_dump())
            if r.status_code == 404:
                # Docker: profile already baked in compose — accept and continue.
                return {
                    "settings": body.settings,
                    "docker_mode": True,
                    "message": "Docker API: settings PUT ignored (use compose env)",
                }
            if r.status_code >= 400:
                raise HTTPException(r.status_code, r.text)
            return r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"API settings PUT failed: {exc}") from exc


@app.get("/proxy/models")
async def proxy_models() -> dict[str, Any]:
    url = f"{api_base()}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            if r.status_code == 404:
                return {"detect": [], "seg": [], "reid": [], "docker_mode": True}
            if r.status_code >= 400:
                raise HTTPException(r.status_code, r.text)
            return r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"API models failed: {exc}") from exc


@app.post("/proxy/admin/bootstrap")
async def proxy_bootstrap(body: BootstrapProxy | None = None) -> dict[str, Any]:
    """Desktop bootstrap; on Docker just return current /health (engines load at start)."""
    url = f"{api_base()}/v1/admin/bootstrap"
    force = bool(body.force) if body is not None else False
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(url, json={"force": force})
            if r.status_code == 404:
                hr = await client.get(f"{api_base()}/health")
                if hr.status_code >= 400:
                    raise HTTPException(hr.status_code, hr.text)
                data = hr.json()
                if isinstance(data, dict):
                    data = {**data, "docker_mode": True, "bootstrap": "skipped"}
                return data
            if r.status_code >= 400:
                raise HTTPException(r.status_code, r.text)
            return r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"API bootstrap failed: {exc}") from exc


@app.post("/proxy/admin/build-trt")
async def proxy_build_trt() -> dict[str, Any]:
    url = f"{api_base()}/v1/admin/build-trt"
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            r = await client.post(url)
            if r.status_code == 404:
                return {
                    "status": "ready",
                    "docker_mode": True,
                    "message": "Docker API: TRT builds at container start (no /v1/admin/build-trt)",
                }
            if r.status_code >= 400:
                raise HTTPException(r.status_code, r.text)
            return r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"API build-trt failed: {exc}") from exc


@app.post("/proxy/jobs/upload")
async def proxy_upload(
    file: UploadFile = File(...),
    prompt: str = Form("person"),
    max_duration_seconds: float | None = Form(None),
) -> dict[str, Any]:
    url = f"{api_base()}/v1/jobs/upload"
    content = await file.read()
    data: dict[str, str] = {"prompt": prompt}
    if max_duration_seconds is not None:
        data["max_duration_seconds"] = str(max_duration_seconds)
    files = {"file": (file.filename or "video.mp4", content, file.content_type or "video/mp4")}
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(url, data=data, files=files)
            if r.status_code >= 400:
                raise HTTPException(r.status_code, r.text)
            return r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"API upload failed: {exc}") from exc


@app.get("/proxy/jobs/{job_id}")
async def proxy_job(job_id: str) -> dict[str, Any]:
    url = f"{api_base()}/v1/jobs/{job_id}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            if r.status_code >= 400:
                raise HTTPException(r.status_code, r.text)
            return r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"API job poll failed: {exc}") from exc


def _run_build(
    build_id: str,
    run_dir: str,
    run_id: str,
    overlay: dict[str, Any] | None,
) -> None:
    job = _build_jobs[build_id]
    try:
        from WEB_app.video_builder import encode_violations_videos_per_id

        def on_progress(done: int, total: int) -> None:
            job["progress"] = {"done": done, "total": total}

        def on_log(msg: str) -> None:
            logs: list[str] = job.setdefault("logs", [])
            logs.append(msg)
            if len(logs) > 100:
                del logs[: len(logs) - 100]

        videos, info = encode_violations_videos_per_id(
            run_dir,
            run_id=run_id,
            overlay_override=overlay,
            on_progress=on_progress,
            on_log=on_log,
        )
        job["status"] = "done"
        job["videos"] = videos
        job["info"] = info
        if videos:
            job["video_path"] = videos[0]["video_path"]
            job["video_name"] = videos[0]["video_name"]
        # Encode is CPU-only — do not empty_cache (regresses next GPU infer job).
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        logs: list[str] = job.setdefault("logs", [])
        logs.append(f"ERROR: {exc}")


@app.post("/build-video", response_model=BuildVideoResponse)
def build_video(body: BuildVideoBody) -> BuildVideoResponse:
    run_dir = _resolve_host_run_dir(body.run_dir)
    if not run_dir.is_dir():
        raise HTTPException(
            400,
            f"Run directory not found: {body.run_dir} (resolved: {run_dir})",
        )

    build_id = uuid.uuid4().hex[:12]
    with _build_lock:
        _build_jobs[build_id] = {
            "status": "running",
            "run_dir": str(run_dir),
            "run_id": body.run_id,
            "progress": {"done": 0, "total": 0},
            "logs": [],
        }
    t = threading.Thread(
        target=_run_build,
        args=(build_id, str(run_dir.resolve()), body.run_id, body.overlay),
        name=f"web-build-{build_id}",
        daemon=True,
    )
    t.start()
    return BuildVideoResponse(build_id=build_id, status="running", message="Encode started")


@app.get("/report/violators")
def report_violators(run_dir: str, run_id: str) -> dict[str, Any]:
    from WEB_app.video_builder import (
        collect_qualified_violator_ids,
        count_presence_by_stable_id,
        count_violations_by_stable_id,
        resolve_run_packets,
    )

    run_path = _resolve_host_run_dir(run_dir)
    if not run_path.is_dir():
        raise HTTPException(
            400,
            f"Run directory not found: {run_dir} (resolved: {run_path})",
        )
    data, _ = resolve_run_packets(run_path, run_id=run_id)
    ids, vcounts, pcounts, threshold = collect_qualified_violator_ids(data, run_path)
    if not ids:
        vcounts = count_violations_by_stable_id(data, run_path)
        pcounts = count_presence_by_stable_id(data, run_path)
        ids = sorted(vcounts.keys(), key=lambda s: (-vcounts[s], s))
        threshold = 0
    violators = [
        {
            "stable_id": int(sid),
            "violation_count": int(vcounts.get(sid, 0)),
            "presence_frames": int(pcounts.get(sid, 0)),
        }
        for sid in ids
    ]
    return {"violators": violators, "threshold": int(threshold)}


@app.get("/report/companies")
def report_companies() -> dict[str, list[str]]:
    from WEB_app.word_report import TEST_COMPANIES

    return {"companies": list(TEST_COMPANIES)}


@app.post("/report/word", response_model=WordReportResponse)
def create_word_report(body: WordReportBody) -> WordReportResponse:
    run_dir = _resolve_host_run_dir(body.run_dir)
    if not run_dir.is_dir():
        raise HTTPException(
            400,
            f"Run directory not found: {body.run_dir} (resolved: {run_dir})",
        )

    from WEB_app.word_report import build_word_report, parse_incident_datetime

    try:
        docx_path = build_word_report(
            run_dir,
            body.run_id,
            int(body.stable_id),
            company=str(body.company or "").strip(),
            organization=str(body.organization or body.company or "").strip(),
            incident_datetime=parse_incident_datetime(body.incident_datetime),
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    run_q = _run_dir_query(str(run_dir.resolve()))
    fname = docx_path.name
    return WordReportResponse(
        download_url=f"/reports/{quote(fname)}?run_dir={run_q}",
        filename=fname,
    )


@app.get("/reports/{filename}")
def download_report(filename: str, run_dir: str) -> FileResponse:
    if ".." in filename or ".." in run_dir:
        raise HTTPException(400, "Invalid path")
    host_run = _resolve_host_run_dir(run_dir)
    path = (host_run / "reports" / filename).resolve()
    if not path.is_file():
        raise HTTPException(404, f"Report not found: {filename}")
    return FileResponse(
        path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.get("/build-video/{build_id}")
def build_status(build_id: str) -> dict[str, Any]:
    job = _build_jobs.get(build_id)
    if job is None:
        raise HTTPException(404, "Build job not found")
    out = dict(job)
    if job.get("status") == "done":
        videos_out: list[dict[str, str | int]] = []
        run_dir = str(job.get("run_dir", ""))
        for v in job.get("videos") or []:
            name = str(v.get("video_name") or "")
            if not name:
                continue
            videos_out.append(
                {
                    **_video_item(v.get("stable_id", 0), name, run_dir),
                    "violation_frames": int(v.get("violation_frames") or 0),
                    "violation_count": int(v.get("violation_count") or 0),
                    "presence_frames": int(v.get("presence_frames") or 0),
                }
            )
        out["videos"] = videos_out
        if videos_out:
            out["video_url"] = videos_out[0]["video_url"]
    if job.get("status") == "error":
        logs = job.get("logs") or []
        if logs and not out.get("error"):
            out["error"] = str(logs[-1])
    return out


@app.get("/videos/{filename}")
def download_video(filename: str, run_dir: str) -> FileResponse:
    if ".." in filename or ".." in run_dir:
        raise HTTPException(400, "Invalid path")
    host_run = _resolve_host_run_dir(run_dir)
    path = (host_run / filename).resolve()
    if not path.is_file():
        raise HTTPException(404, f"Video not found: {filename}")
    return FileResponse(
        path,
        filename=filename,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


def main() -> None:
    import uvicorn

    from app.core.network_urls import print_listen_banner

    print_listen_banner(
        service="YOLO_DRT WEB_app",
        host=WEB_HOST,
        port=WEB_PORT,
        extra_lines=(
            f"API proxy -> {api_base()}",
            "UI ходит через /proxy/* — открывай WEB по адресам выше",
        ),
    )
    uvicorn.run(
        "WEB_app.server:app",
        host=WEB_HOST,
        port=WEB_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
