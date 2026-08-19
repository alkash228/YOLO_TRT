"""Compact box overlay for WEB preview and download (no pixel re-encode)."""
from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from WEB_app.video_builder import iter_run_packets, load_run_metadata

from app.core.video_encode import resolve_run_source_video


def _verdict_is_violation(verdict: Any) -> bool:
    if verdict is None:
        return False
    if isinstance(verdict, dict):
        return not bool(verdict.get("ok", True))
    return not bool(getattr(verdict, "ok", True))


def _packet_boxes(packet: Any, *, video_w: int, video_h: int) -> list[dict[str, int]]:
    n = int(getattr(packet, "n_inst", 0) or 0)
    if n <= 0:
        return []
    meta = list(getattr(packet, "instance_meta", None) or [])
    sids = getattr(packet, "stable_ids", None)
    verdicts = list(getattr(packet, "cross_check_verdicts", None) or [])
    src_h = src_w = 0
    mask_hw = getattr(packet, "mask_hw", None)
    if mask_hw:
        src_h, src_w = int(mask_hw[0]), int(mask_hw[1])
    sx = (float(video_w) / float(src_w)) if src_w > 0 and video_w > 0 else 1.0
    sy = (float(video_h) / float(src_h)) if src_h > 0 and video_h > 0 else 1.0
    out: list[dict[str, int]] = []
    for i in range(n):
        sid = int(sids[i]) if sids is not None and i < len(sids) else i + 1
        if i >= len(meta) or not isinstance(meta[i], dict):
            continue
        bb = meta[i].get("bbox_xywh")
        if bb is None or len(bb) < 4:
            continue
        x, y, w, h = [float(v) for v in bb[:4]]
        if w <= 0 or h <= 0:
            continue
        viol = i < len(verdicts) and _verdict_is_violation(verdicts[i])
        out.append(
            {
                "id": sid,
                "x": int(round(x * sx)),
                "y": int(round(y * sy)),
                "w": max(1, int(round(w * sx))),
                "h": max(1, int(round(h * sy))),
                "v": 1 if viol else 0,
            }
        )
    return out


def build_overlay_timeline(
    run_dir: str | Path,
    run_id: str,
    *,
    source_video: str | None = None,
) -> dict[str, Any]:
    """Keyframe boxes from packets (instance_meta only — no RLE inflate)."""
    run_path = Path(run_dir)
    rid = str(run_id)
    meta = load_run_metadata(run_path, rid, source_video=source_video)
    data = meta["packets_data"]
    fps = float(meta.get("fps") or data.get("fps") or 25.0) or 25.0
    width = int(meta.get("width") or data.get("width") or 0)
    height = int(meta.get("height") or data.get("height") or 0)
    recorded = str(meta.get("input_path") or meta.get("recorded_input_path") or "")
    source = resolve_run_source_video(
        run_path,
        recorded,
        run_id=rid,
        override=source_video,
    )
    events: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for packet in iter_run_packets(data, run_path):
        fi = int(packet.frame_idx)
        t = fi / fps
        if prev is not None:
            prev["t1"] = t
            events.append(prev)
        if width <= 0 or height <= 0:
            mh = getattr(packet, "mask_hw", None)
            if mh:
                height, width = int(mh[0]), int(mh[1])
        prev = {
            "n": fi,
            "t0": t,
            "t1": t + (1.0 / fps),
            "boxes": _packet_boxes(packet, video_w=width, video_h=height),
        }
    if prev is not None:
        events.append(prev)

    src_name = Path(source).name if source else f"{rid}_source.mp4"
    return {
        "run_id": rid,
        "fps": fps,
        "width": width,
        "height": height,
        "video": src_name,
        "source_path": source,
        "events": events,
    }


def _ass_time(sec: float) -> str:
    s = max(0.0, float(sec))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    rem = s - h * 3600 - m * 60
    return f"{h}:{m:02d}:{rem:05.2f}"


def timeline_to_ass(timeline: dict[str, Any]) -> str:
    w = max(1, int(timeline.get("width") or 1920))
    h = max(1, int(timeline.get("height") or 1080))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {w}",
        f"PlayResY: {h}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Box,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for ev in timeline.get("events") or []:
        t0 = _ass_time(float(ev.get("t0") or 0.0))
        t1 = _ass_time(float(ev.get("t1") or 0.0))
        if t1 <= t0:
            continue
        for box in ev.get("boxes") or []:
            x, y = int(box["x"]), int(box["y"])
            bw, bh = int(box["w"]), int(box["h"])
            viol = int(box.get("v") or 0) == 1
            color = "&H0000FF&" if viol else "&H00C800&"
            sid = int(box.get("id") or 0)
            label = f"NO HELMET ID {sid}" if viol else f"ID {sid}"
            draw = (
                f"{{\\an7\\pos({x},{y})\\p1\\bord3\\c{color}\\1a&H90&}}"
                f"m 0 0 l {bw} 0 l {bw} {bh} l 0 {bh}{{\\p0}}"
            )
            text = (
                f"{{\\an7\\pos({x},{max(0, y - 32)})\\bord2\\c{color}\\fs28}}"
                f"{label}"
            )
            lines.append(f"Dialogue: 0,{t0},{t1},Box,,0,0,0,,{draw}")
            lines.append(f"Dialogue: 1,{t0},{t1},Box,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def write_ass_file(timeline: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(timeline_to_ass(timeline), encoding="utf-8")
    return path


def mux_overlay_mkv(
    source_video: str,
    ass_path: Path,
    out_mkv: Path,
) -> Path:
    """Attach ASS as a subtitle track. Video stream is copied — no HEVC re-encode."""
    from app.core.ffmpeg_utils import _popen_kwargs, resolve_ffmpeg_exe

    exe = resolve_ffmpeg_exe()
    out_mkv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-f",
        "ass",
        "-i",
        str(ass_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map",
        "1:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:s",
        "ass",
        str(out_mkv),
    ]
    def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            args,
            capture_output=True,
            timeout=600,
            check=False,
            **_popen_kwargs(),
        )

    proc = _run(cmd)
    if proc.returncode != 0:
        cmd_no_audio = [
            exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_video),
            "-f",
            "ass",
            "-i",
            str(ass_path),
            "-map",
            "0:v:0",
            "-map",
            "1:0",
            "-c:v",
            "copy",
            "-c:s",
            "ass",
            str(out_mkv),
        ]
        proc = _run(cmd_no_audio)
    if proc.returncode != 0 or not out_mkv.is_file() or out_mkv.stat().st_size <= 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"ffmpeg mux failed: {err or proc.returncode}")
    return out_mkv


_STANDALONE_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YOLO_DRT overlay</title>
  <style>
    body { margin: 0; background: #0f172a; color: #e2e8f0; font-family: Segoe UI, system-ui, sans-serif; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 1rem; }
    .stage { position: relative; background: #000; border-radius: 10px; overflow: hidden; }
    video, canvas { display: block; width: 100%; height: auto; }
    canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }
    p { color: #94a3b8; font-size: 0.9rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="stage">
      <video id="v" controls playsinline></video>
      <canvas id="c"></canvas>
    </div>
    <p>Положи это HTML рядом с файлом <code id="vn"></code> и открой в Chrome/Edge. Боксы рисуются поверх, видео не перекодировалось.</p>
  </div>
  <script>
    const OVERLAY = __OVERLAY_JSON__;
    const video = document.getElementById("v");
    const canvas = document.getElementById("c");
    const ctx = canvas.getContext("2d");
    const events = OVERLAY.events || [];
    document.getElementById("vn").textContent = OVERLAY.video || "video.mp4";
    video.src = OVERLAY.video || "video.mp4";
    let idx = 0;
    function findEvent(t) {
      if (!events.length) return null;
      while (idx + 1 < events.length && events[idx + 1].t0 <= t) idx += 1;
      while (idx > 0 && events[idx].t0 > t) idx -= 1;
      const ev = events[idx];
      if (t < ev.t0 - 0.05) return null;
      return ev;
    }
    function draw() {
      const w = video.videoWidth || OVERLAY.width || 1280;
      const h = video.videoHeight || OVERLAY.height || 720;
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const ev = findEvent(video.currentTime);
      if (!ev) return;
      const sx = canvas.width / Math.max(1, OVERLAY.width || canvas.width);
      const sy = canvas.height / Math.max(1, OVERLAY.height || canvas.height);
      for (const b of ev.boxes || []) {
        const x = b.x * sx, y = b.y * sy, bw = b.w * sx, bh = b.h * sy;
        ctx.lineWidth = b.v ? 4 : 2;
        ctx.strokeStyle = b.v ? "#ef4444" : "#22c55e";
        ctx.strokeRect(x, y, bw, bh);
        ctx.font = "16px Segoe UI";
        ctx.fillStyle = b.v ? "#ef4444" : "#22c55e";
        ctx.fillText(b.v ? ("NO HELMET ID " + b.id) : ("ID " + b.id), x, Math.max(16, y - 6));
      }
    }
    video.addEventListener("timeupdate", draw);
    video.addEventListener("seeked", draw);
    video.addEventListener("play", () => {
      const loop = () => { draw(); if (!video.paused && !video.ended) requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    });
  </script>
</body>
</html>
"""


def write_standalone_player(timeline: dict[str, Any], path: Path) -> Path:
    payload = {
        "run_id": timeline.get("run_id"),
        "fps": timeline.get("fps"),
        "width": timeline.get("width"),
        "height": timeline.get("height"),
        "video": timeline.get("video"),
        "events": timeline.get("events"),
    }
    html = _STANDALONE_HTML.replace(
        "__OVERLAY_JSON__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_player_zip(timeline: dict[str, Any], path: Path) -> Path:
    html_name = f"{timeline.get('run_id') or 'run'}_overlay_player.html"
    html_path = path.with_name(html_name)
    write_standalone_player(timeline, html_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(html_path, arcname=html_name)
        readme = (
            "1. Скачай исходное видео в ту же папку (имя как в плеере).\n"
            "2. Открой HTML в Chrome или Edge.\n"
            "Боксы рисуются поверх, файл видео не перекодировался.\n"
            "Для одного файла с боксами открой MKV в VLC.\n"
        )
        zf.writestr("README.txt", readme)
    return path
