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
WEB_APP_DIR = Path(__file__).resolve().parent
WEB_OUTPUT_DIR = WEB_APP_DIR / "output"


_env_api = os.environ.get("YOLO_DRT_API_URL", "").strip()
WEB_HOST = os.environ.get("WEB_APP_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_APP_PORT", "8088"))

# Prefer Docker :8080 first when both are up (desktop :8765 still wins if only it answers).
_API_PROBE_PORTS = (8080, 8765)
_api_url_cache: str | None = None
_api_url_cache_at: float = 0.0
_API_URL_TTL_SEC = 5.0
# UI override (POST /proxy/api-url) — wins over auto-probe until cleared.
_user_api_url: str | None = None
_user_api_lock = threading.Lock()


def _normalize_api_url(raw: str) -> str:
    url = (raw or "").strip().rstrip("/")
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    low = url.casefold()
    if not (low.startswith("http://") or low.startswith("https://")):
        raise ValueError("API URL must be http:// or https://")
    # Block obvious junk / SSRF to file schemes etc. (already filtered).
    return url


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

    with _user_api_lock:
        override = _user_api_url
    if override:
        return override

    now = time.time()
    if _api_url_cache and (now - _api_url_cache_at) < _API_URL_TTL_SEC:
        return _api_url_cache
    url = _resolve_api_url(_env_api or None)
    _api_url_cache = url
    _api_url_cache_at = now
    return url


def set_user_api_url(raw: str | None) -> dict[str, Any]:
    """Set or clear UI API override. Empty string clears override (auto-probe)."""
    global _user_api_url, _api_url_cache, _api_url_cache_at

    text = (raw or "").strip()
    if not text:
        with _user_api_lock:
            _user_api_url = None
        _api_url_cache = None
        _api_url_cache_at = 0.0
        resolved = api_base()
        ok, status = _probe_api(resolved)
        return {
            "ok": True,
            "mode": "auto",
            "api_url": resolved,
            "override": None,
            "reachable": ok,
            "status": status,
        }

    url = _normalize_api_url(text)
    ok, status = _probe_api(url)
    with _user_api_lock:
        _user_api_url = url
    _api_url_cache = None
    _api_url_cache_at = 0.0
    return {
        "ok": True,
        "mode": "manual",
        "api_url": url,
        "override": url,
        "reachable": ok,
        "status": status,
    }


def get_api_url_info() -> dict[str, Any]:
    with _user_api_lock:
        override = _user_api_url
    base = api_base()
    ok, status = _probe_api(base)
    return {
        "api_url": base,
        "override": override,
        "mode": "manual" if override else "auto",
        "reachable": ok,
        "status": status,
    }


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
    ensure_web_output_dir()
    yield


app = FastAPI(title="YOLO_DRT WEB_app", version="1.0.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_build_lock = threading.Lock()
_build_jobs: dict[str, dict[str, Any]] = {}

# Docker API returns container paths (/data/output/...); WEB runs on the host.
_DOCKER_OUTPUT_PREFIXES = ("/data/output",)


def _host_output_candidates() -> list[Path]:
    """Host folders that may contain run dirs. Prefer WEB_app/output."""
    out: list[Path] = []
    raw = os.environ.get("YOLO_DRT_HOST_OUTPUT_DIR", "").strip()
    if raw:
        out.append(Path(raw))
    out.append(WEB_OUTPUT_DIR)
    # Legacy: repo root / parent
    out.append(ROOT / "output")
    out.append(ROOT.parent / "output")
    out.append(Path.cwd() / "output")
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def ensure_web_output_dir() -> Path:
    """Canonical output inside WEB_app/ (create if missing)."""
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return WEB_OUTPUT_DIR


def _host_output_root() -> Path:
    ensure_web_output_dir()
    for p in _host_output_candidates():
        if p.is_dir():
            return p
    return WEB_OUTPUT_DIR


def _is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    if name.startswith(".") or name in {"uploads", "input", "reports"}:
        return False
    # Typical run folder or any dir with result/packets
    markers = (
        f"{name}_result.json",
        f"{name}_run_summary.json",
        f"{name}_packets_manifest.json",
    )
    if any((path / m).is_file() for m in markers):
        return True
    if list(path.glob("*_result.json")) or list(path.glob("*_packets_manifest.json")):
        return True
    return False


def list_local_runs(limit: int = 80) -> list[dict[str, Any]]:
    ensure_web_output_dir()
    found: dict[str, dict[str, Any]] = {}
    for root in _host_output_candidates():
        if not root.is_dir():
            continue
        try:
            kids = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            continue
        for child in kids:
            if not _is_run_dir(child):
                continue
            key = str(child.resolve())
            if key in found:
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                mtime = 0.0
            rid = child.name
            for cand in child.glob("*_result.json"):
                rid = cand.name[: -len("_result.json")]
                break
            found[key] = {
                "run_dir": key,
                "run_id": rid,
                "name": child.name,
                "root": str(root.resolve()),
                "mtime": mtime,
            }
            if len(found) >= max(1, int(limit)):
                break
        if len(found) >= max(1, int(limit)):
            break
    rows = list(found.values())
    rows.sort(key=lambda r: float(r.get("mtime") or 0), reverse=True)
    return rows


def _resolve_host_run_dir(run_dir: str | Path) -> Path:
    """Map Docker /data/output/... to the Windows host bind-mount path."""
    raw = str(run_dir or "").strip().replace("\\", "/")
    if not raw:
        return Path(raw)
    p = Path(raw)
    if p.is_dir():
        return p

    rel = ""
    for prefix in _DOCKER_OUTPUT_PREFIXES:
        if raw == prefix or raw.startswith(prefix + "/"):
            rel = raw[len(prefix) :].lstrip("/")
            break

    tried: list[Path] = []
    if rel:
        for root in _host_output_candidates():
            mapped = root / rel if rel else root
            tried.append(mapped)
            if mapped.is_dir():
                return mapped
        # Prefer first candidate in error path (even if missing)
        return tried[0] if tried else (WEB_OUTPUT_DIR / rel)

    # Non-docker path that doesn't exist — still return as-is
    return p


def _run_dir_missing_detail(original: str, resolved: Path) -> str:
    roots = ", ".join(str(r) for r in _host_output_candidates())
    return (
        f"Run directory not found: {original} (resolved: {resolved}). "
        f"Docker пишет прогоны в WEB_app/output. Положи папку прогона туда "
        f"(WEB_app\\output\\{Path(original).name}) или выбери её в UI. "
        f"Искали в: {roots}"
    )


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
    source_video: str | None = None


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
def config() -> dict[str, Any]:
    info = get_api_url_info()
    return {
        "api_url": info["api_url"],
        "web_url": f"http://127.0.0.1:{WEB_PORT}",
        "api_url_mode": info["mode"],
        "api_url_override": info["override"],
    }


@app.get("/proxy/api-url")
def proxy_get_api_url() -> dict[str, Any]:
    return get_api_url_info()


class ApiUrlBody(BaseModel):
    api_url: str = ""


@app.put("/proxy/api-url")
def proxy_put_api_url(body: ApiUrlBody) -> dict[str, Any]:
    try:
        return set_user_api_url(body.api_url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/access-info")
def access_info() -> dict[str, Any]:
    from app.core.network_urls import lan_ipv4_addresses, listen_urls

    api_target = api_base()
    try:
        api_port = int(api_target.rsplit(":", 1)[-1])
    except ValueError:
        api_port = 8765

    with _user_api_lock:
        override = _user_api_url
    return {
        "web_urls": listen_urls(WEB_HOST, WEB_PORT),
        "api_proxy_target": api_target,
        "api_url_mode": "manual" if override else "auto",
        "api_url_override": override,
        "api_base_urls": listen_urls("127.0.0.1", api_port),
        "lan_ips": lan_ipv4_addresses(),
        "bind_host": WEB_HOST,
        "web_port": WEB_PORT,
        "api_port": api_port,
    }


@app.get("/local/output")
def local_output_info() -> dict[str, Any]:
    root = ensure_web_output_dir()
    return {
        "output_dir": str(root.resolve()),
        "candidates": [str(p) for p in _host_output_candidates()],
        "runs": list_local_runs(limit=100),
    }


@app.get("/local/runs")
def local_runs(limit: int = 100) -> dict[str, Any]:
    return {"runs": list_local_runs(limit=limit), "output_dir": str(ensure_web_output_dir().resolve())}


class SelectRunBody(BaseModel):
    run_dir: str
    run_id: str | None = None


@app.post("/local/select-run")
def local_select_run(body: SelectRunBody) -> dict[str, Any]:
    run_path = _resolve_host_run_dir(body.run_dir)
    if not run_path.is_dir():
        # Also accept absolute host path pasted by user
        alt = Path(str(body.run_dir).strip())
        if alt.is_dir():
            run_path = alt
    if not run_path.is_dir():
        raise HTTPException(400, _run_dir_missing_detail(body.run_dir, run_path))
    rid = (body.run_id or "").strip() or run_path.name
    for cand in run_path.glob("*_result.json"):
        rid = cand.name[: -len("_result.json")]
        break
    return {
        "ok": True,
        "run_dir": str(run_path.resolve()),
        "run_id": rid,
        "output_dir": str(ensure_web_output_dir().resolve()),
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
                with _user_api_lock:
                    override = _user_api_url
                data = {
                    **data,
                    "api_proxy_target": base,
                    "api_url_mode": "manual" if override else "auto",
                }
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
    source_video: str | None = None,
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
            source_video=source_video,
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
        raise HTTPException(400, _run_dir_missing_detail(body.run_dir, run_dir))

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
        args=(
            build_id,
            str(run_dir.resolve()),
            body.run_id,
            body.overlay,
            (body.source_video or "").strip() or None,
        ),
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
        raise HTTPException(400, _run_dir_missing_detail(run_dir, run_path))
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
        raise HTTPException(400, _run_dir_missing_detail(body.run_dir, run_dir))

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
