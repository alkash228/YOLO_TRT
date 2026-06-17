"""Encode annotated MP4 from saved FramePackets (manual / on-demand)."""
from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np

from app.core.frame_pipeline import FramePacket, OrderedPostExecutor, materialize_packet_for_render, post_process_frame
from app.core.schema import prompt_id_lookup_from_prompt


PACKETS_SUFFIX = "_packets.pkl"
_PLACEHOLDER_MAX = 2  # legacy packets stored 1×1 stub frames


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
    matches = sorted(run_dir.glob(f"*{PACKETS_SUFFIX}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _load_source_frames(input_path: str, n_expected: int) -> list[np.ndarray] | None:
    p = Path(input_path)
    if not p.exists():
        return None
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max(0, n_expected) or n_expected <= 0:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames or None


def _packet_frame_bgr(
    packet: FramePacket,
    frame_source: list[np.ndarray] | None,
) -> np.ndarray:
    fb = packet.frame_bgr
    if fb is not None and fb.shape[0] > _PLACEHOLDER_MAX and fb.shape[1] > _PLACEHOLDER_MAX:
        return fb
    if frame_source is not None and 0 <= packet.frame_idx < len(frame_source):
        return frame_source[packet.frame_idx]
    if fb is not None and fb.size > 0:
        return fb
    raise ValueError(f"Frame {packet.frame_idx}: no image data (re-run inference or check input video)")


def _resolve_target_size(
    packets: list[FramePacket],
    frame_source: list[np.ndarray] | None,
    width: int,
    height: int,
) -> tuple[int, int]:
    if width > 0 and height > 0:
        return int(width), int(height)
    for packet in packets:
        try:
            fb = _packet_frame_bgr(packet, frame_source)
            h, w = fb.shape[:2]
            if h > _PLACEHOLDER_MAX and w > _PLACEHOLDER_MAX:
                return int(w), int(h)
        except ValueError:
            continue
    if frame_source:
        h, w = frame_source[0].shape[:2]
        return int(w), int(h)
    raise ValueError("Cannot determine video frame size")


def _rgb_to_size(rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w == target_w and h == target_h:
        return np.ascontiguousarray(rgb)
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _clone_packet_for_timeline(
    src: FramePacket,
    frame_idx: int,
    frame_source: list[np.ndarray] | None,
) -> FramePacket:
    stub = replace(src, frame_idx=frame_idx)
    fb = _packet_frame_bgr(stub, frame_source)
    return materialize_packet_for_render(replace(src, frame_idx=frame_idx), fb)


def expand_packets_to_timeline(
    packets: list[FramePacket],
    *,
    source_frame_count: int,
    frame_stride: int = 1,
    frame_source: list[np.ndarray] | None = None,
) -> list[FramePacket]:
    """Hold-forward: пропущенные кадры получают последний inference-результат."""
    if not packets:
        return []
    ordered = sorted(packets, key=lambda p: p.frame_idx)
    n_src = max(int(source_frame_count), max(p.frame_idx for p in ordered) + 1)
    by_idx = {p.frame_idx: p for p in ordered}
    stride = max(1, int(frame_stride))

    if stride <= 1 and len(by_idx) >= n_src:
        return [_clone_packet_for_timeline(by_idx[i], i, frame_source) for i in range(n_src)]

    out: list[FramePacket] = []
    carry: FramePacket | None = None
    for i in range(n_src):
        if i in by_idx:
            carry = by_idx[i]
        if carry is None:
            continue
        out.append(_clone_packet_for_timeline(carry, i, frame_source))
    return out


def encode_packets_to_video(
    packets: list[FramePacket],
    *,
    video_path: Path,
    fps: float,
    prompt: str,
    overlay: dict[str, Any],
    post_workers: int = 6,
    input_path: str | None = None,
    width: int = 0,
    height: int = 0,
    frame_stride: int = 1,
    source_frame_count: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    if not packets:
        raise ValueError("No frames to encode")
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_lookup = prompt_id_lookup_from_prompt(prompt)
    ordered = sorted(packets, key=lambda p: p.frame_idx)
    stride = max(1, int(frame_stride))
    n_src = int(source_frame_count) if source_frame_count > 0 else max(p.frame_idx for p in ordered) + 1
    frame_source = _load_source_frames(input_path, n_src) if input_path else None
    timeline = expand_packets_to_timeline(
        ordered,
        source_frame_count=n_src,
        frame_stride=stride,
        frame_source=frame_source,
    )
    if not timeline:
        raise ValueError("No frames to encode after timeline expansion")
    n = len(timeline)
    target_w, target_h = _resolve_target_size(timeline, frame_source, width, height)

    def _post_fn(packet: FramePacket, _src=None) -> Any:
        from app.core.video_processor import VideoProcessor

        work = materialize_packet_for_render(
            packet,
            _packet_frame_bgr(packet, frame_source),
            height=target_h,
            width=target_w,
        )
        return post_process_frame(
            work,
            serialize_fn=VideoProcessor.serialize_frame_instances,
            prompt_id_lookup=prompt_lookup,
            overlay_alpha=float(overlay.get("overlay_alpha", 0.45)),
            draw_boxes=bool(overlay.get("draw_boxes", True)),
            draw_masks=bool(overlay.get("draw_masks", True)),
            draw_centers=bool(overlay.get("draw_centers", True)),
            draw_pose=bool(overlay.get("draw_pose", True)),
            pose_kpt_conf=float(overlay.get("pose_kpt_conf", 0.25)),
            cross_check_enabled=bool(overlay.get("cross_check_enabled", False)),
            cross_check_draw_head_box=bool(overlay.get("cross_check_draw_head_box", True)),
            cross_check_draw_boxes=bool(overlay.get("cross_check_draw_boxes", True)),
        )

    post_exec = OrderedPostExecutor(workers=post_workers, process_fn=_post_fn)
    for packet in timeline:
        post_exec.submit(packet)
    results = sorted(post_exec.finish(), key=lambda r: r.frame_idx)
    post_exec.shutdown()

    writer = imageio.get_writer(
        str(video_path),
        fps=float(fps),
        codec="libx264",
        ffmpeg_params=["-crf", "18", "-preset", "medium"],
    )
    try:
        for i, result in enumerate(results):
            rgb = _rgb_to_size(result.rgb, target_w, target_h)
            writer.append_data(rgb)
            if on_progress is not None:
                on_progress(i + 1, n)
    finally:
        writer.close()
    return video_path


def encode_run_folder(
    run_dir: Path,
    *,
    run_id: str | None = None,
    post_workers: int = 6,
    overlay_override: dict[str, Any] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    pkl = find_packets_in_run(run_dir)
    if pkl is None:
        raise FileNotFoundError(f"No {PACKETS_SUFFIX} in {run_dir}")
    data = load_run_packets(pkl)
    rid = str(run_id or data.get("run_id") or run_dir.name)
    video_path = run_dir / f"{rid}_annotated.mp4"
    input_path = str(data.get("input_path") or "")
    overlay = dict(data.get("overlay") or {})
    if overlay_override:
        overlay.update(overlay_override)
    encode_packets_to_video(
        list(data["packets"]),
        video_path=video_path,
        fps=float(data.get("fps") or 25.0),
        prompt=str(data.get("prompt") or "person"),
        overlay=overlay,
        post_workers=post_workers,
        input_path=input_path or None,
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        frame_stride=int(data.get("frame_stride") or 1),
        source_frame_count=int(data.get("source_frame_count") or 0),
        on_progress=on_progress,
    )
    if input_path:
        from app.core.video_processor import VideoProcessor

        VideoProcessor._mux_audio_if_possible(input_path, video_path)
    return video_path
