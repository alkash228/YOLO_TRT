"""JSON schema helpers and run statistics."""
from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class RunStats:
    frame: int
    total_frames: int
    fps: float
    eta_sec: float
    gpu_mem_mb: float
    instances_current: int
    instances_peak: int
    id_switches: int
    reid_recoveries: int
    detect_ms: float
    seg_ms: float
    reid_ms: float
    total_ms: float
    elapsed_sec: float = 0.0
    gpu_util_pct: float = 0.0


@dataclass(slots=True)
class VideoProgress:
    current: int
    total: int
    fps: float
    eta_seconds: float
    gpu_mem_mb: float
    elapsed_sec: float = 0.0
    gpu_util_pct: float = 0.0
    instances_current: int = 0
    instances_peak: int = 0
    stats: RunStats | None = None
    phase: str = "inference"


@dataclass(slots=True)
class ProcessVideoResult:
    out_dir: str
    run_id: str
    elapsed_sec: float
    fps_processed: float
    frames: int
    record: dict[str, Any]
    packets_path: str | None = None
    video_path: str | None = None
    has_video: bool = False


def video_data_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_result_json(
    path: Path,
    payload: dict[str, Any],
    *,
    frame_iter: Iterator[dict[str, Any]] | None = None,
    on_progress: Callable[[int], None] | None = None,
    progress_every: int = 5000,
) -> None:
    """Write result JSON; stream frames[] from spill when frame_iter is set."""
    if frame_iter is None:
        Path(path).write_text(video_data_json_dumps(payload), encoding="utf-8")
        return

    out = Path(path)
    with out.open("w", encoding="utf-8") as f:
        f.write("{\n")
        body_keys = [k for k in payload if k != "frames"]
        for i, key in enumerate(body_keys):
            val = json.dumps(payload[key], ensure_ascii=False, indent=2)
            f.write(f'  {json.dumps(key, ensure_ascii=False)}: ')
            f.write(textwrap.indent(val, "  ").lstrip())
            f.write(",\n" if i < len(body_keys) - 1 else ",\n")
        f.write('  "frames": [\n')
        first = True
        count = 0
        for frame in frame_iter:
            count += 1
            if on_progress and progress_every > 0 and count % progress_every == 0:
                on_progress(count)
            if not first:
                f.write(",\n")
            first = False
            fr = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
            f.write("    ")
            f.write(fr)
        if on_progress and count > 0:
            on_progress(count)
        f.write("\n  ]\n}\n")


def label_slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.strip().casefold())
    return s.strip("_")


def prompt_id_lookup_from_prompt(prompt: str) -> dict[str, int]:
    from app.core.prompt_utils import parse_prompt_segments

    lookup: dict[str, int] = {}
    for i, seg in enumerate(parse_prompt_segments(prompt), start=1):
        key = seg.strip().casefold()
        if key and key not in lookup:
            lookup[key] = i
    return lookup


def build_result_payload(
    *,
    schema: str,
    input_path: str,
    prompt: str,
    fps: float,
    width: int,
    height: int,
    frames_written: int,
    elapsed_sec: float,
    frames: list[dict[str, Any]],
    models: dict[str, str],
    draw_boxes: bool,
    draw_masks: bool,
    draw_centers: bool,
    draw_pose: bool = True,
    run_id: str,
    stats_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "media_type": "video",
        "input_path": input_path,
        "prompt": prompt,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "frames_written": int(frames_written),
        "elapsed_sec": float(elapsed_sec),
        "models": dict(models),
        "draw_boxes": bool(draw_boxes),
        "draw_masks": bool(draw_masks),
        "draw_centers": bool(draw_centers),
        "draw_pose": bool(draw_pose),
        "job_id": run_id,
        "frames": frames,
    }
    if stats_summary:
        payload["stats_summary"] = stats_summary
    return payload


def stats_summary_from_counters(
    *,
    id_switches: int,
    reid_recoveries: int,
    instances_peak: int,
    avg_detect_ms: float,
    avg_seg_ms: float,
    avg_reid_ms: float,
    avg_total_ms: float,
) -> dict[str, Any]:
    return {
        "id_switches": int(id_switches),
        "reid_recoveries": int(reid_recoveries),
        "instances_peak": int(instances_peak),
        "avg_detect_ms": round(avg_detect_ms, 3),
        "avg_seg_ms": round(avg_seg_ms, 3),
        "avg_reid_ms": round(avg_reid_ms, 3),
        "avg_total_ms": round(avg_total_ms, 3),
    }


def run_stats_to_dict(stats: RunStats) -> dict[str, Any]:
    return asdict(stats)
