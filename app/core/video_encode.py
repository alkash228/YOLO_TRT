"""Encode annotated MP4 from saved FramePackets (manual / on-demand)."""
from __future__ import annotations

import os
import pickle
import queue
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.ffmpeg_utils import (
    build_encode_writer_args,
    decode_vf_filter,
    describe_encode_plan,
    format_encode_plan,
    list_hwaccel_decode_profiles,
    resolve_ffmpeg_exe,
)
from app.core.frame_pipeline import AsyncVideoWriter, FramePacket, materialize_packet_for_render, post_process_frame
from app.core.schema import prompt_id_lookup_from_prompt


PACKETS_SUFFIX = "_packets.pkl"
_PLACEHOLDER_MAX = 2  # legacy packets stored 1×1 stub frames


def resolve_encode_codec(requested: str = "auto", *, ffmpeg_exe: str | None = None) -> str:
    from app.core.ffmpeg_utils import resolve_encode_codec as _resolve

    return _resolve(requested, ffmpeg_exe=ffmpeg_exe)


def build_encode_writer_args(
    *,
    codec: str = "auto",
    preset: str = "fast",
    crf: int = 23,
    ffmpeg_exe: str | None = None,
) -> tuple[str, list[str], str]:
    from app.core.ffmpeg_utils import build_encode_writer_args as _build

    return _build(codec=codec, preset=preset, crf=crf, ffmpeg_exe=ffmpeg_exe)


@dataclass(frozen=True, slots=True)
class _EncodeJob:
    src_i: int
    carry: FramePacket
    frame_bgr: np.ndarray | None


class _FfmpegHwVideoReader:
    """Decode source video via ffmpeg with GPU hwaccel when available."""

    def __init__(
        self,
        path: str,
        *,
        width: int,
        height: int,
        prefetch: int = 24,
        ffmpeg_exe: str | None = None,
        use_hwaccel: bool = True,
        hw_name: str = "",
        hw_args: list[str] | None = None,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._frame_bytes = self._width * self._height * 3
        exe = ffmpeg_exe or resolve_ffmpeg_exe()
        if use_hwaccel and hw_args is None:
            profiles = list_hwaccel_decode_profiles(self._width, self._height, ffmpeg_exe=exe)
            if not profiles:
                raise OSError("No working ffmpeg hwaccel profile for bgr24 decode")
            self._hw_name, hw_args = profiles[0]
        else:
            self._hw_name = hw_name or "cpu"
            hw_args = list(hw_args or [])
        vf = decode_vf_filter(self._hw_name, self._width, self._height, hw_args=hw_args)
        from app.core.ffmpeg_utils import _popen_kwargs as popen_kwargs

        cmd = [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            *hw_args,
            "-i",
            str(path),
            "-an",
            "-sn",
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs(),
        )
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(4, prefetch))
        self._done = False
        self._err: Exception | None = None
        self._hold_first: np.ndarray | None = None
        self._thread = threading.Thread(target=self._loop, name="yolo-drt-decode", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._proc.stdout is not None
        try:
            while True:
                raw = self._proc.stdout.read(self._frame_bytes)
                if not raw or len(raw) < self._frame_bytes:
                    self._q.put(None)
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(self._height, self._width, 3)
                self._q.put(frame.copy())
        except Exception as exc:
            self._err = exc
            self._q.put(None)
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
            err = (self._proc.stderr.read() if self._proc.stderr else b"").decode(errors="replace").strip()
            rc = self._proc.wait(timeout=30)
            if rc != 0 and self._err is None:
                self._err = RuntimeError(err or f"ffmpeg decode exit {rc}")

    def read(self) -> np.ndarray | None:
        if self._err is not None:
            raise self._err
        if self._hold_first is not None:
            frame = self._hold_first
            self._hold_first = None
            return frame
        frame = self._q.get()
        if frame is None:
            self._done = True
        return frame

    def close(self, *, raise_on_error: bool = True) -> None:
        self._thread.join(timeout=120.0)
        if raise_on_error and self._err is not None:
            raise self._err


def _open_decode_reader(
    path: str,
    *,
    width: int,
    height: int,
    prefetch: int,
    ffmpeg_exe: str,
    on_log: Callable[[str], None] | None = None,
) -> _FfmpegHwVideoReader | _PrefetchedVideoReader | None:
    profiles = list_hwaccel_decode_profiles(width, height, ffmpeg_exe=ffmpeg_exe)
    last_err: Exception | None = None
    for name, hw_args in profiles:
        reader: _FfmpegHwVideoReader | None = None
        try:
            reader = _FfmpegHwVideoReader(
                path,
                width=width,
                height=height,
                prefetch=prefetch,
                ffmpeg_exe=ffmpeg_exe,
                hw_name=name,
                hw_args=hw_args,
            )
            frame = reader.read()
            if frame is not None and reader._err is None:
                reader._hold_first = frame
                if on_log:
                    on_log(f"Decode: ffmpeg {name}")
                return reader
        except (OSError, RuntimeError) as exc:
            last_err = exc
        if reader is not None:
            try:
                reader.close(raise_on_error=False)
            except Exception:
                pass
    try:
        reader = _PrefetchedVideoReader(path, prefetch=prefetch)
        if on_log:
            on_log("Decode: OpenCV (CPU fallback)")
        return reader
    except OSError as exc:
        last_err = exc
    if last_err is not None:
        raise last_err
    return None


class _PrefetchedVideoReader:
    """CPU fallback decode (OpenCV) when ffmpeg hwaccel fails."""

    def __init__(self, path: str, *, prefetch: int = 24) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise OSError(f"Cannot open video: {path}")
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(4, prefetch))
        self._err: Exception | None = None
        self._thread = threading.Thread(target=self._loop, name="yolo-drt-decode-cv2", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    self._q.put(None)
                    break
                self._q.put(frame)
        except Exception as exc:
            self._err = exc
            self._q.put(None)
        finally:
            self._cap.release()

    def read(self) -> np.ndarray | None:
        if self._err is not None:
            raise self._err
        return self._q.get()

    def close(self) -> None:
        self._thread.join(timeout=120.0)
        if self._err is not None:
            raise self._err


@dataclass(frozen=True, slots=True)
class _RenderContext:
    target_w: int
    target_h: int
    prompt_lookup: dict[str, int]
    overlay: dict[str, Any]


def _render_encode_job(job: _EncodeJob, ctx: _RenderContext) -> tuple[int, np.ndarray]:
    from app.core.instance_serialize import serialize_frame_instances

    work = materialize_packet_for_render(
        replace(job.carry, frame_idx=job.src_i),
        _packet_frame_bgr(replace(job.carry, frame_idx=job.src_i), job.frame_bgr),
        height=ctx.target_h,
        width=ctx.target_w,
    )
    result = post_process_frame(
        work,
        serialize_fn=serialize_frame_instances,
        prompt_id_lookup=ctx.prompt_lookup,
        overlay_alpha=float(ctx.overlay.get("overlay_alpha", 0.45)),
        draw_boxes=bool(ctx.overlay.get("draw_boxes", True)),
        draw_masks=bool(ctx.overlay.get("draw_masks", True)),
        draw_centers=bool(ctx.overlay.get("draw_centers", True)),
        draw_pose=bool(ctx.overlay.get("draw_pose", True)),
        pose_kpt_conf=float(ctx.overlay.get("pose_kpt_conf", 0.25)),
        cross_check_enabled=bool(ctx.overlay.get("cross_check_enabled", False)),
        cross_check_draw_head_box=bool(ctx.overlay.get("cross_check_draw_head_box", True)),
        cross_check_draw_boxes=bool(ctx.overlay.get("cross_check_draw_boxes", True)),
        pose_point_radius=int(ctx.overlay.get("pose_point_radius", 4) or 4),
        pose_line_thickness=int(ctx.overlay.get("pose_line_thickness", 2) or 2),
    )
    rgb = _rgb_to_size(result.rgb, ctx.target_w, ctx.target_h)
    return job.src_i, rgb


def packets_path_for_run(run_dir: Path, run_id: str) -> Path:
    return Path(run_dir) / f"{run_id}{PACKETS_SUFFIX}"


def save_run_packets(
    path: Path,
    *,
    run_id: str,
    packets: list[FramePacket],
    fps: float,
    input_path: str,
    prompt: str,
    overlay: dict[str, Any],
    width: int = 0,
    height: int = 0,
    frame_stride: int = 1,
    source_frame_count: int = 0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(packets, key=lambda p: p.frame_idx)
    if (width <= 0 or height <= 0) and ordered:
        h, w = ordered[0].frame_bgr.shape[:2]
        height, width = int(h), int(w)
    payload = {
        "version": 2,
        "run_id": run_id,
        "fps": float(fps),
        "input_path": str(input_path),
        "prompt": str(prompt),
        "overlay": dict(overlay),
        "width": int(width),
        "height": int(height),
        "frame_stride": max(1, int(frame_stride)),
        "source_frame_count": int(source_frame_count) if source_frame_count > 0 else len(ordered),
        "packets": ordered,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_run_packets(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "packets" not in data:
        raise ValueError(f"Invalid packets file: {path}")
    return data


def find_packets_in_run(run_dir: Path) -> Path | None:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    from app.core.packet_spill import MANIFEST_SUFFIX

    manifests = sorted(
        run_dir.glob(f"*{MANIFEST_SUFFIX}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if manifests:
        return manifests[0]
    matches = sorted(run_dir.glob(f"*{PACKETS_SUFFIX}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def infer_run_id(run_dir: Path, run_id: str | None = None) -> str:
    if run_id:
        return str(run_id)
    try:
        data, _ = resolve_run_packets(run_dir, run_id=None)
        rid = data.get("run_id")
        if rid:
            return str(rid)
    except FileNotFoundError:
        pass
    return Path(run_dir).name


def resolve_run_packets(run_dir: Path, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
    """Load legacy single pkl or chunked manifest + run_dir for spill."""
    run_dir = Path(run_dir)
    found = find_packets_in_run(run_dir)
    if found is None:
        raise FileNotFoundError(f"No packets in {run_dir}")
    if found.name.endswith(".json"):
        from app.core.packet_spill import load_packets_manifest

        return load_packets_manifest(found), run_dir
    return load_run_packets(found), run_dir


def _packet_frame_bgr(
    packet: FramePacket,
    frame_bgr: np.ndarray | None,
) -> np.ndarray:
    fb = packet.frame_bgr
    if fb is not None and fb.shape[0] > _PLACEHOLDER_MAX and fb.shape[1] > _PLACEHOLDER_MAX:
        return fb
    if frame_bgr is not None and frame_bgr.size > 0:
        return frame_bgr
    if fb is not None and fb.size > 0:
        if fb.shape[0] <= _PLACEHOLDER_MAX and fb.shape[1] <= _PLACEHOLDER_MAX:
                raise ValueError(
                    f"Frame {packet.frame_idx}: нет пикселей исходного видео. "
                    "На этой машине нет файла по input_path из прогона — "
                    "положи оригинал как {run_id}_source.mp4 в папку прогона "
                    "или укажи локальный путь к ролику в WEB."
                )
        return fb
    raise ValueError(
        f"Frame {packet.frame_idx}: no image data (check input video path in run manifest)"
    )


_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


def _source_search_dirs(run_dir: Path) -> list[Path]:
    """Candidate folders when a run is copied to another machine without absolute paths."""
    run_dir = Path(run_dir).resolve()
    dirs: list[Path] = [run_dir, run_dir.parent]
    env_dirs = os.environ.get("YOLO_DRT_VIDEOS") or os.environ.get("YOLO_SOURCE_DIRS") or ""
    for part in env_dirs.replace(";", os.pathsep).split(os.pathsep):
        part = part.strip().strip('"')
        if part:
            dirs.append(Path(part))
    here = Path(__file__).resolve()
    # .../app/core/video_encode.py → repo root (parent of app/)
    repo = here.parents[2]
    dirs.extend(
        [
            repo / "videos",
            repo / "WEB_app" / "videos",
            repo / "YOLO_DOCKER" / "videos",
            Path.cwd() / "videos",
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        try:
            key = str(d.resolve()) if d.exists() else str(d)
        except OSError:
            key = str(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def resolve_run_source_video(
    run_dir: Path | str,
    input_path: str | None = None,
    *,
    run_id: str | None = None,
    override: str | None = None,
) -> str | None:
    """
    Locate source pixels for encode/report.

    Order: explicit override → existing absolute path → {run_id}_source.* in run →
    basename of recorded path in run / videos dirs → any *_source.* in run.
    """
    run_dir = Path(run_dir)
    if override:
        ov = Path(override.strip().strip('"'))
        if ov.is_file():
            return str(ov.resolve())
    if input_path:
        path = Path(str(input_path).strip().strip('"'))
        if path.is_file():
            return str(path.resolve())

    rid = (run_id or "").strip()
    if rid:
        for candidate in sorted(run_dir.glob(f"{rid}_source.*")):
            if candidate.is_file() and candidate.suffix.lower() in _VIDEO_SUFFIXES:
                return str(candidate.resolve())

    basename = Path(str(input_path or "")).name
    if basename and Path(basename).suffix.lower() in _VIDEO_SUFFIXES:
        for folder in _source_search_dirs(run_dir):
            hit = folder / basename
            if hit.is_file():
                return str(hit.resolve())
            # Docker-style names sometimes land as {stem}_source{suffix} after copy
            stem = Path(basename).stem
            for alt in folder.glob(f"{stem}_source.*"):
                if alt.is_file():
                    return str(alt.resolve())

    for pattern in ("*_source.mp4", "*_source.mkv", "*_source.mov", "*_source.webm"):
        matches = sorted(
            (p for p in run_dir.glob(pattern) if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return str(matches[0].resolve())
    return None


def source_video_missing_message(
    run_dir: Path | str,
    input_path: str | None = None,
    *,
    run_id: str | None = None,
) -> str:
    run_dir = Path(run_dir)
    rid = (run_id or "").strip() or "RUNID"
    recorded = str(input_path or "").strip() or "(не записан)"
    name = Path(recorded).name if recorded and recorded != "(не записан)" else "video.mp4"
    return (
        "Исходное видео для сборки клипов не найдено (прогон с другой машины?). "
        f"В метаданных путь: {recorded}. "
        f"Положи оригинал как {rid}_source{Path(name).suffix or '.mp4'} в папку прогона "
        f"или файл {name} в videos/, либо укажи локальный путь к ролику в WEB."
    )


def _resolve_encode_input_path(
    input_path: str | None,
    run_dir: Path,
    *,
    run_id: str | None = None,
    override: str | None = None,
) -> str | None:
    return resolve_run_source_video(
        run_dir,
        input_path,
        run_id=run_id,
        override=override,
    )


def _resolve_target_size(
    *,
    width: int,
    height: int,
    input_path: str | None,
) -> tuple[int, int]:
    if width > 0 and height > 0:
        return int(width), int(height)
    if input_path:
        cap = cv2.VideoCapture(str(input_path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if w > 0 and h > 0:
                return w, h
    raise ValueError("Cannot determine video frame size (set width/height in run metadata)")


def _rgb_to_size(rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w == target_w and h == target_h:
        return np.ascontiguousarray(rgb)
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


class _ChunkPacketLookup:
    """One spill chunk in RAM at a time."""

    def __init__(self, manifest: dict[str, Any], run_dir: Path) -> None:
        self._entries = sorted(
            list(manifest.get("chunks") or []),
            key=lambda e: int(e.get("start_frame", 0)),
        )
        self._run_dir = Path(run_dir)
        self._chunk_idx = -1
        self._map: dict[int, FramePacket] = {}

    def keyframe_at(self, src_i: int) -> FramePacket | None:
        self._ensure_chunk_for(int(src_i))
        return self._map.get(int(src_i))

    def _ensure_chunk_for(self, src_i: int) -> None:
        if not self._entries:
            return
        if self._chunk_idx < 0:
            self._load_chunk(0)
        while (
            self._chunk_idx + 1 < len(self._entries)
            and src_i >= int(self._entries[self._chunk_idx + 1].get("start_frame", 0))
        ):
            self._load_chunk(self._chunk_idx + 1)

    def _load_chunk(self, idx: int) -> None:
        entry = self._entries[idx]
        chunk_path = self._run_dir / str(entry.get("path") or "")
        if not chunk_path.is_file():
            raise FileNotFoundError(f"Missing spill chunk: {chunk_path}")
        with chunk_path.open("rb") as f:
            data = pickle.load(f)
        self._map = {int(p.frame_idx): p for p in list(data.get("packets") or [])}
        self._chunk_idx = idx
        del data


class _SortedPacketLookup:
    """Legacy single list / pkl — sorted keyframes, pointer scan (no full timeline)."""

    def __init__(self, packets: list[FramePacket]) -> None:
        self._ordered = sorted(packets, key=lambda p: p.frame_idx)
        self._ptr = 0

    def keyframe_at(self, src_i: int) -> FramePacket | None:
        src_i = int(src_i)
        while self._ptr < len(self._ordered) and self._ordered[self._ptr].frame_idx < src_i:
            self._ptr += 1
        if self._ptr < len(self._ordered) and self._ordered[self._ptr].frame_idx == src_i:
            return self._ordered[self._ptr]
        return None


def _encode_timeline_streaming(
    lookup: _ChunkPacketLookup | _SortedPacketLookup,
    *,
    video_path: Path,
    fps: float,
    prompt: str,
    overlay: dict[str, Any],
    input_path: str | None,
    width: int,
    height: int,
    source_frame_count: int,
    frame_stride: int = 1,
    post_workers: int = 6,
    encode_preset: str = "fast",
    encode_crf: int = 23,
    encode_codec: str = "auto",
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    encode_src_indices: set[int] | None = None,
    run_id: str | None = None,
    render_job_fn: Callable[[_EncodeJob, _RenderContext], tuple[int, np.ndarray]] | None = None,
) -> Path:
    if source_frame_count <= 0:
        raise ValueError("source_frame_count required for streaming encode")

    render_one = render_job_fn or _render_encode_job
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    recorded_input = input_path
    resolved_input = resolve_run_source_video(
        video_path.parent,
        input_path,
        run_id=run_id,
    )
    if resolved_input:
        input_path = resolved_input
    target_w, target_h = _resolve_target_size(width=width, height=height, input_path=input_path)
    prompt_lookup = prompt_id_lookup_from_prompt(prompt)
    n_src = int(source_frame_count)
    encode_total = len(encode_src_indices) if encode_src_indices else n_src
    workers = max(1, int(post_workers))
    render_ctx = _RenderContext(
        target_w=target_w,
        target_h=target_h,
        prompt_lookup=prompt_lookup,
        overlay=overlay,
    )
    codec, ffmpeg_params, ffmpeg_exe = build_encode_writer_args(
        codec=encode_codec,
        preset=encode_preset,
        crf=encode_crf,
    )
    if on_log:
        plan = describe_encode_plan(
            codec=encode_codec,
            preset=encode_preset,
            crf=encode_crf,
            width=target_w,
            height=target_h,
        )
        on_log(format_encode_plan(plan))
    writer = AsyncVideoWriter(
        str(video_path),
        float(fps),
        codec=codec,
        ffmpeg_params=ffmpeg_params,
        ffmpeg_exe=ffmpeg_exe,
        queue_size=max(32, workers * 4),
    )
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="yolo-drt-encode-render")

    reader: _FfmpegHwVideoReader | _PrefetchedVideoReader | None = None
    cap: cv2.VideoCapture | None = None
    clip_mode = encode_src_indices is not None
    # Full-timeline encode: stream-decode every frame. Clip mode uses grab-skip
    # to needed indices only (long videos otherwise look "stuck on decode").
    if (not clip_mode) and resolved_input and Path(resolved_input).is_file():
        try:
            reader = _open_decode_reader(
                str(resolved_input),
                width=target_w,
                height=target_h,
                prefetch=max(16, workers * 3),
                ffmpeg_exe=ffmpeg_exe,
                on_log=on_log,
            )
        except (OSError, RuntimeError):
            cap = cv2.VideoCapture(str(resolved_input))
            if not cap.isOpened():
                cap = None
                if on_log:
                    on_log("Decode: none (packets only)")
            elif on_log:
                on_log("Decode: OpenCV (sync)")
    elif clip_mode and resolved_input and Path(resolved_input).is_file():
        if on_log:
            on_log(
                f"Decode: grab-skip to {len(encode_src_indices or [])} violation frames "
                f"(not full {n_src}-frame scan)"
            )
    elif on_log:
        on_log(
            source_video_missing_message(
                video_path.parent,
                recorded_input,
                run_id=run_id,
            )
        )

    carry: FramePacket | None = None
    stride = max(1, int(frame_stride))
    pending: dict[int, Future[tuple[int, np.ndarray]]] = {}
    next_write: int | None = 0 if clip_mode else None
    written = 0
    out_seq = 0
    max_inflight = max(8, workers * 3)

    def _read_frame_bgr() -> np.ndarray | None:
        if reader is not None:
            return reader.read()
        if cap is not None:
            ok, frame = cap.read()
            return frame if ok else None
        return None

    def _flush_ready(block: bool = False) -> None:
        nonlocal next_write, written
        if next_write is None:
            return
        while next_write in pending and (pending[next_write].done() or block):
            _, rgb = pending.pop(next_write).result()
            writer.submit(rgb)
            written += 1
            next_write += 1
            if on_progress is not None:
                on_progress(written, encode_total)

    def _submit_clip_frame(src_i: int, frame_bgr: np.ndarray | None) -> None:
        nonlocal out_seq, next_write
        keyframe = lookup.keyframe_at(src_i)
        if keyframe is None:
            return
        carry_local = keyframe
        seq = out_seq
        out_seq += 1
        if next_write is None:
            next_write = seq
        fb_copy = frame_bgr.copy() if frame_bgr is not None else None
        pending[seq] = executor.submit(
            render_one,
            _EncodeJob(src_i=src_i, carry=replace(carry_local), frame_bgr=fb_copy),
            render_ctx,
        )
        _flush_ready(block=False)
        while len(pending) >= max_inflight:
            if next_write is None or next_write not in pending:
                break
            _flush_ready(block=True)

    try:
        if clip_mode:
            needed = sorted({int(i) for i in (encode_src_indices or set())})
            encode_total = max(1, len(needed))
            if resolved_input and Path(resolved_input).is_file():
                from app.core.frame_io import iter_selected_bgr_frames

                def _on_scan(cur: int, last: int) -> None:
                    if on_log and cur % 900 == 0:
                        on_log(f"Decode scan {cur}/{last} (grab-skip)")

                for src_i, frame_bgr in iter_selected_bgr_frames(
                    str(resolved_input),
                    needed,
                    on_scan=_on_scan,
                ):
                    _submit_clip_frame(src_i, frame_bgr)
            else:
                for src_i in needed:
                    _submit_clip_frame(src_i, None)
        else:
            for src_i in range(n_src):
                frame_bgr = _read_frame_bgr()
                if frame_bgr is None and (reader is not None or cap is not None):
                    break

                keyframe = lookup.keyframe_at(src_i)
                if keyframe is not None:
                    carry = keyframe
                elif carry is not None and (int(src_i) - int(carry.frame_idx)) >= stride:
                    carry = None
                if carry is None:
                    continue

                seq = src_i
                if next_write is None:
                    next_write = seq

                fb_copy = frame_bgr.copy() if frame_bgr is not None else None
                job = _EncodeJob(src_i=src_i, carry=replace(carry), frame_bgr=fb_copy)
                pending[seq] = executor.submit(render_one, job, render_ctx)

                _flush_ready(block=False)
                while len(pending) >= max_inflight:
                    if next_write is None or next_write not in pending:
                        break
                    _flush_ready(block=True)

        while pending:
            if next_write is None:
                next_write = min(pending.keys())
            _flush_ready(block=True)
    finally:
        if reader is not None:
            reader.close()
        if cap is not None:
            cap.release()
        executor.shutdown(wait=True)
        writer.close()

    if written <= 0:
        raise ValueError("No frames encoded (missing packets or source video)")
    return video_path


def encode_packets_to_video(
    packets: list[FramePacket],
    *,
    video_path: Path,
    fps: float,
    prompt: str,
    overlay: dict[str, Any],
    post_workers: int = 6,
    encode_preset: str = "fast",
    encode_crf: int = 23,
    encode_codec: str = "auto",
    input_path: str | None = None,
    width: int = 0,
    height: int = 0,
    frame_stride: int = 1,
    source_frame_count: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    if not packets:
        raise ValueError("No frames to encode")
    ordered = sorted(packets, key=lambda p: p.frame_idx)
    n_src = int(source_frame_count) if source_frame_count > 0 else max(p.frame_idx for p in ordered) + 1
    lookup = _SortedPacketLookup(ordered)
    return _encode_timeline_streaming(
        lookup,
        video_path=video_path,
        fps=fps,
        prompt=prompt,
        overlay=overlay,
        input_path=input_path,
        width=width,
        height=height,
        source_frame_count=n_src,
        frame_stride=frame_stride,
        post_workers=post_workers,
        encode_preset=encode_preset,
        encode_crf=encode_crf,
        encode_codec=encode_codec,
        on_progress=on_progress,
        on_log=on_log,
    )


def encode_manifest_to_video(
    manifest: dict[str, Any],
    *,
    run_dir: Path,
    video_path: Path,
    overlay_override: dict[str, Any] | None = None,
    post_workers: int = 6,
    encode_preset: str = "fast",
    encode_crf: int = 23,
    encode_codec: str = "auto",
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    overlay = dict(manifest.get("overlay") or {})
    if overlay_override:
        overlay.update(overlay_override)
    lookup = _ChunkPacketLookup(manifest, Path(run_dir))
    return _encode_timeline_streaming(
        lookup,
        video_path=video_path,
        fps=float(manifest.get("fps") or 25.0),
        prompt=str(manifest.get("prompt") or "person"),
        overlay=overlay,
        input_path=str(manifest.get("input_path") or "") or None,
        width=int(manifest.get("width") or 0),
        height=int(manifest.get("height") or 0),
        source_frame_count=int(manifest.get("source_frame_count") or 0),
        frame_stride=int(manifest.get("frame_stride") or 1),
        post_workers=post_workers,
        encode_preset=encode_preset,
        encode_crf=encode_crf,
        encode_codec=encode_codec,
        on_progress=on_progress,
        on_log=on_log,
    )


def encode_run_folder(
    run_dir: Path,
    *,
    run_id: str | None = None,
    video_path: Path | str | None = None,
    post_workers: int = 6,
    encode_preset: str = "fast",
    encode_crf: int = 23,
    encode_codec: str = "auto",
    overlay_override: dict[str, Any] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    data, base_dir = resolve_run_packets(run_dir, run_id=run_id)
    rid = infer_run_id(run_dir, run_id)
    if video_path is None:
        video_path = run_dir / f"{rid}_annotated.mp4"
    else:
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = dict(data.get("overlay") or {})
    if overlay_override:
        overlay.update(overlay_override)

    encode_kwargs = dict(
        post_workers=post_workers,
        encode_preset=encode_preset,
        encode_crf=encode_crf,
        encode_codec=encode_codec,
        on_progress=on_progress,
        on_log=on_log,
    )

    if data.get("format") == "chunked" and "chunks" in data:
        encode_manifest_to_video(
            data,
            run_dir=base_dir,
            video_path=video_path,
            overlay_override=overlay_override,
            **encode_kwargs,
        )
    else:
        encode_packets_to_video(
            list(data["packets"]),
            video_path=video_path,
            fps=float(data.get("fps") or 25.0),
            prompt=str(data.get("prompt") or "person"),
            overlay=overlay,
            input_path=str(data.get("input_path") or "") or None,
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            frame_stride=int(data.get("frame_stride") or 1),
            source_frame_count=int(data.get("source_frame_count") or 0),
            **encode_kwargs,
        )

    input_path = str(data.get("input_path") or "")
    if input_path:
        from app.core.ffmpeg_utils import mux_audio_if_possible

        mux_audio_if_possible(input_path, video_path)
    return video_path


# --- legacy helpers (tests / external imports) ---

def expand_packets_to_timeline(
    packets: list[FramePacket],
    *,
    source_frame_count: int,
    frame_stride: int = 1,
    frame_source: list[np.ndarray] | None = None,
) -> list[FramePacket]:
    """Hold-forward expand (legacy; prefer streaming encode)."""
    stride = max(1, int(frame_stride))
    if not packets:
        return []
    ordered = sorted(packets, key=lambda p: p.frame_idx)
    n_src = max(int(source_frame_count), max(p.frame_idx for p in ordered) + 1)
    by_idx = {p.frame_idx: p for p in ordered}
    out: list[FramePacket] = []
    carry: FramePacket | None = None
    for i in range(n_src):
        if i in by_idx:
            carry = by_idx[i]
        elif carry is not None and (i - int(carry.frame_idx)) >= stride:
            carry = None
        if carry is None:
            continue
        stub = replace(carry, frame_idx=i)
        fb = _packet_frame_bgr(stub, frame_source[i] if frame_source and i < len(frame_source) else None)
        out.append(materialize_packet_for_render(replace(carry, frame_idx=i), fb))
    return out
