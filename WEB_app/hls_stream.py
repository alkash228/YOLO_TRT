"""On-the-fly H.264 HLS so Chrome can play HEVC sources without VLC."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from app.core.ffmpeg_utils import (
    _popen_kwargs,
    resolve_encode_codec,
    resolve_ffmpeg_exe,
)

_lock = threading.Lock()
_job: dict[str, Any] | None = None


def _stop_unlocked() -> None:
    global _job
    if not _job:
        return
    proc: subprocess.Popen[bytes] | None = _job.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=4)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    log_f = _job.get("log")
    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass
    _job = None


def stop_hls() -> None:
    with _lock:
        _stop_unlocked()


def hls_dir_for(run_dir: Path, run_id: str) -> Path:
    return Path(run_dir) / "reports" / f"{run_id}_hls"


def active_hls_dir(run_id: str) -> Path | None:
    with _lock:
        if _job and str(_job.get("run_id") or "") == str(run_id):
            return Path(_job["dir"])
    return None


def _encoder_args(exe: str) -> list[str]:
    codec = resolve_encode_codec("auto", ffmpeg_exe=exe)
    if codec == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-rc",
            "vbr",
            "-cq",
            "32",
            "-b:v",
            "0",
            "-gpu",
            "0",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-crf",
        "28",
        "-threads",
        "0",
    ]


def start_hls(
    *,
    run_dir: Path,
    run_id: str,
    source: str,
    start_sec: float = 0.0,
    duration_sec: float = 0.0,
) -> dict[str, Any]:
    """
    Progressive H.264 HLS on the host (NVENC if possible).

    First .ts in a few seconds; ffmpeg keeps encoding ahead while the browser plays.
    duration_sec <= 0 → from start_sec to end of file (Stop kills ffmpeg).
    """
    exe = resolve_ffmpeg_exe()
    src = Path(source)
    if not src.is_file():
        raise FileNotFoundError(str(src))
    start = max(0.0, float(start_sec))
    duration = float(duration_sec)
    out = hls_dir_for(Path(run_dir), str(run_id))
    with _lock:
        _stop_unlocked()
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        playlist = out / "index.m3u8"
        segment = str(out / "seg%05d.ts")
        log_path = out / "ffmpeg.log"
        cmd = [
            exe,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-ss",
            f"{start:.3f}",
            "-hwaccel",
            "auto",
            "-i",
            str(src),
        ]
        if duration > 0.5:
            cmd.extend(["-t", f"{duration:.3f}"])
        cmd.extend(
            [
                "-an",
                "-vf",
                r"scale=trunc(min(1280\,iw)/2)*2:-2",
                *_encoder_args(exe),
                "-g",
                "48",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "hls",
                "-hls_time",
                "2",
                "-hls_list_size",
                "0",
                "-hls_playlist_type",
                "event",
                "-hls_flags",
                "independent_segments+append_list",
                "-hls_segment_filename",
                segment,
                str(playlist),
            ]
        )
        log_f = log_path.open("wb")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=log_f,
            **_popen_kwargs(),
        )
        global _job
        _job = {
            "proc": proc,
            "log": log_f,
            "dir": out,
            "run_id": str(run_id),
            "start_sec": start,
            "duration_sec": duration,
            "codec": resolve_encode_codec("auto", ffmpeg_exe=exe),
        }

    deadline = time.time() + 25.0
    while time.time() < deadline:
        if playlist.is_file() and playlist.stat().st_size > 24 and list(out.glob("*.ts")):
            break
        if proc.poll() is not None:
            err = ""
            try:
                err = log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
            except OSError:
                err = f"ffmpeg exit {proc.returncode}"
            raise RuntimeError(err or f"ffmpeg exit {proc.returncode}")
        time.sleep(0.25)
    else:
        raise RuntimeError("HLS не стартовал за 25с — проверь ffmpeg/NVENC в логе reports/*_hls/ffmpeg.log")

    return {
        "ok": True,
        "start_sec": start,
        "duration_sec": duration,
        "codec": _job["codec"] if _job else "libx264",
        "playlist_url": f"/overlay/hls/{run_id}/index.m3u8",
    }
