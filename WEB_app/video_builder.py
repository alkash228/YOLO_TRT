"""Build violations-only MP4 from a YOLO_DRT run folder — one video per stable ReID ID."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from app.core.frame_pipeline import FramePacket
from app.core.video_encode import (
    _ChunkPacketLookup,
    _EncodeJob,
    _RenderContext,
    _SortedPacketLookup,
    _encode_timeline_streaming,
    _render_encode_job,
    infer_run_id,
    resolve_run_packets,
)

_OVERLAY_KEYS = (
    "overlay_alpha",
    "draw_boxes",
    "draw_masks",
    "draw_centers",
    "draw_pose",
    "pose_kpt_conf",
    "cross_check_enabled",
    "cross_check_draw_head_box",
    "cross_check_draw_boxes",
)


def load_run_metadata(
    run_dir: Path,
    run_id: str,
    *,
    source_video: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    result_path = run_dir / f"{run_id}_result.json"
    summary_path = run_dir / f"{run_id}_run_summary.json"
    meta: dict[str, Any] = {}
    if result_path.is_file():
        try:
            meta = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = {}

    data, _ = resolve_run_packets(run_dir, run_id=run_id)
    saved_overlay = dict(data.get("overlay") or {})

    pipeline = meta.get("pipeline") or summary.get("pipeline") or {}
    if not pipeline and isinstance(meta.get("record"), dict):
        pipeline = meta["record"].get("pipeline") or {}

    cross_enabled = bool(
        pipeline.get("cross_check_enabled")
        or saved_overlay.get("cross_check_enabled")
    )
    models = meta.get("models") or summary.get("models") or {}
    cross_model = str(models.get("cross_check") or "")
    if cross_model and cross_model.casefold() not in ("disabled", "none", ""):
        cross_enabled = True

    violations = 0
    report_path = run_dir / f"{run_id}_report.txt"
    if report_path.is_file():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if "Cross-check violations" in line:
                try:
                    violations = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
                break

    source_frame_count = int(
        data.get("source_frame_count")
        or meta.get("source_frame_count")
        or summary.get("source_frames")
        or 0
    )
    frame_stride = int(
        data.get("frame_stride")
        or (pipeline.get("frame_stride") if isinstance(pipeline, dict) else 0)
        or (summary.get("stats_summary") or meta.get("stats_summary") or {}).get("frame_stride")
        or 1
    )
    if source_frame_count <= 0:
        max_idx = -1
        for packet in iter_run_packets(data, run_dir):
            max_idx = max(max_idx, int(packet.frame_idx))
        if max_idx >= 0:
            source_frame_count = max_idx + 1

    recorded = str(
        data.get("input_path")
        or meta.get("input_path")
        or summary.get("input_path")
        or meta.get("input_video")
        or ""
    )
    from app.core.video_encode import resolve_run_source_video, source_video_missing_message

    rid = infer_run_id(run_dir, run_id)
    input_path = (
        resolve_run_source_video(
            run_dir,
            recorded,
            run_id=rid,
            override=source_video,
        )
        or ""
    )

    return {
        "fps": float(
            data.get("fps") or meta.get("fps") or summary.get("video_fps") or 25.0
        ),
        "input_path": input_path,
        "recorded_input_path": recorded,
        "source_missing_hint": (
            ""
            if input_path
            else source_video_missing_message(run_dir, recorded, run_id=rid)
        ),
        "prompt": str(data.get("prompt") or meta.get("prompt") or summary.get("prompt") or "person"),
        "width": int(data.get("width") or meta.get("width") or summary.get("width") or 0),
        "height": int(data.get("height") or meta.get("height") or summary.get("height") or 0),
        "source_frame_count": source_frame_count,
        "frame_stride": max(1, frame_stride),
        "cross_check_enabled": cross_enabled,
        "cross_check_violations": violations,
        "format": data.get("format"),
        "packets_data": data,
        "saved_overlay": saved_overlay,
    }


def resolve_overlay(
    meta: dict[str, Any],
    *,
    overlay_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Overlay for encode: saved run snapshot, then user override from WEB/API settings."""
    overlay: dict[str, Any] = {
        "overlay_alpha": 0.45,
        "draw_boxes": True,
        "draw_masks": True,
        "draw_centers": True,
        "draw_pose": True,
        "pose_kpt_conf": 0.25,
        "cross_check_enabled": bool(meta.get("cross_check_enabled", True)),
        "cross_check_draw_head_box": True,
        "cross_check_draw_boxes": False,
    }
    saved = meta.get("saved_overlay") or {}
    if isinstance(saved, dict):
        for key in _OVERLAY_KEYS:
            if key in saved:
                overlay[key] = saved[key]
    if overlay_override:
        for key in _OVERLAY_KEYS:
            if key in overlay_override and overlay_override[key] is not None:
                overlay[key] = overlay_override[key]
    # Violations clip: only the target person — never draw helmet boxes of others.
    overlay["cross_check_draw_boxes"] = False
    return overlay


def iter_run_packets(data: dict[str, Any], run_dir: Path) -> Iterator[FramePacket]:
    if data.get("format") == "chunked" and data.get("chunks"):
        from app.core.packet_spill import iter_spilled_packets

        yield from iter_spilled_packets(data, run_dir=run_dir)
        return
    for packet in data.get("packets") or []:
        yield packet


def find_packet_at_frame(
    data: dict[str, Any], run_dir: Path, frame_idx: int
) -> FramePacket | None:
    target = int(frame_idx)
    for packet in iter_run_packets(data, Path(run_dir)):
        if int(packet.frame_idx) == target:
            return packet
    return None


def collect_stable_ids(data: dict[str, Any], run_dir: Path) -> list[int]:
    """All unique stable ReID IDs seen in the run."""
    ids: set[int] = set()
    for packet in iter_run_packets(data, Path(run_dir)):
        n = int(packet.n_inst)
        if n <= 0 or packet.stable_ids is None:
            continue
        for i in range(min(n, len(packet.stable_ids))):
            ids.add(int(packet.stable_ids[i]))
    return sorted(ids)


def collect_violator_stable_ids(data: dict[str, Any], run_dir: Path) -> list[int]:
    """Unique stable ReID IDs that violated cross-check (no helmet) at least once."""
    ids: set[int] = set()
    for packet in iter_run_packets(data, Path(run_dir)):
        verdicts = packet.cross_check_verdicts or []
        n = int(packet.n_inst)
        if n <= 0 or not verdicts:
            continue
        for i in range(min(n, len(verdicts))):
            if verdicts[i].ok:
                continue
            if packet.stable_ids is None or i >= len(packet.stable_ids):
                continue
            ids.add(int(packet.stable_ids[i]))
    return sorted(ids)


def count_violations_by_stable_id(data: dict[str, Any], run_dir: Path) -> dict[int, int]:
    """Cross-check NO HELMET keyframes per stable ReID id."""
    counts: dict[int, int] = {}
    for packet in iter_run_packets(data, Path(run_dir)):
        verdicts = packet.cross_check_verdicts or []
        n = int(packet.n_inst)
        if n <= 0 or not verdicts or packet.stable_ids is None:
            continue
        for i in range(min(n, len(verdicts))):
            if verdicts[i].ok:
                continue
            if i >= len(packet.stable_ids):
                break
            sid = int(packet.stable_ids[i])
            counts[sid] = counts.get(sid, 0) + 1
    return counts


def count_presence_by_stable_id(data: dict[str, Any], run_dir: Path) -> dict[int, int]:
    """Keyframes where stable_id appears (helmet ok or not)."""
    counts: dict[int, int] = {}
    for packet in iter_run_packets(data, Path(run_dir)):
        n = int(packet.n_inst)
        if n <= 0 or packet.stable_ids is None:
            continue
        for i in range(min(n, len(packet.stable_ids))):
            sid = int(packet.stable_ids[i])
            counts[sid] = counts.get(sid, 0) + 1
    return counts


def collect_qualified_violator_ids(
    data: dict[str, Any],
    run_dir: Path,
    *,
    min_violation_frames: int = 3,
    min_violation_ratio: float = 0.08,
    min_relative_to_max: float = 0.25,
) -> tuple[list[int], dict[int, int], dict[int, int], int]:
    """
    Violators worth a clip: enough NO HELMET hits vs presence and vs worst offender.

    Returns (qualified_ids desc by count, violation_counts, presence_counts, threshold_used).
    """
    vcounts = count_violations_by_stable_id(data, run_dir)
    pcounts = count_presence_by_stable_id(data, run_dir)
    if not vcounts:
        return [], vcounts, pcounts, min_violation_frames

    max_v = max(vcounts.values())
    threshold = max(int(min_violation_frames), int(max_v * min_relative_to_max))

    qualified: list[int] = []
    for sid, vc in vcounts.items():
        if vc < threshold:
            continue
        pc = pcounts.get(sid, 0)
        if pc > 0 and (vc / pc) < float(min_violation_ratio):
            continue
        qualified.append(sid)

    qualified.sort(key=lambda s: (-vcounts[s], s))
    return qualified, vcounts, pcounts, threshold


def collect_violator_stable_ids_by_count(data: dict[str, Any], run_dir: Path) -> list[int]:
    """Violators sorted by violation keyframe count (desc), then stable_id."""
    ids, _, _, _ = collect_qualified_violator_ids(data, run_dir)
    if ids:
        return ids
    counts = count_violations_by_stable_id(data, run_dir)
    if not counts:
        return []
    return sorted(counts.keys(), key=lambda sid: (-counts[sid], sid))


def collect_violation_frame_indices(
    data: dict[str, Any], run_dir: Path, target_sid: int
) -> list[int]:
    """Source frame indices where target_sid has cross_check ok=False."""
    frames: list[int] = []
    for packet in iter_run_packets(data, Path(run_dir)):
        if filter_violator_single_id(packet, target_sid).n_inst > 0:
            frames.append(int(packet.frame_idx))
    return sorted(set(frames))


def collect_presence_frame_indices(
    data: dict[str, Any], run_dir: Path, target_sid: int
) -> list[int]:
    """All source keyframes where target_sid is tracked (helmet ok or not)."""
    frames: list[int] = []
    for packet in iter_run_packets(data, Path(run_dir)):
        if filter_person_single_id(packet, target_sid).n_inst > 0:
            frames.append(int(packet.frame_idx))
    return sorted(set(frames))


def filter_person_single_id(packet: FramePacket, target_sid: int) -> FramePacket:
    """Keep one tracked person by stable ReID id (any cross-check state)."""
    n = int(packet.n_inst)
    if n <= 0 or packet.stable_ids is None:
        return _empty_instances(packet)

    keep_idx: int | None = None
    for i in range(min(n, len(packet.stable_ids))):
        if int(packet.stable_ids[i]) == int(target_sid):
            keep_idx = i
            break

    if keep_idx is None:
        return _empty_instances(packet)

    i = keep_idx
    stack = packet.stack[i : i + 1] if packet.stack is not None and packet.stack.size else packet.stack
    stable_ids = packet.stable_ids[i : i + 1]
    scores = packet.scores[i : i + 1] if packet.scores is not None else packet.scores
    labels = [packet.labels[i]] if packet.labels and i < len(packet.labels) else []
    kpts = [packet.keypoints_list[i]] if packet.keypoints_list and i < len(packet.keypoints_list) else []
    verdicts = packet.cross_check_verdicts or []
    vlist = [verdicts[i]] if i < len(verdicts) else []
    masks_rle = [packet.masks_rle[i]] if packet.masks_rle and i < len(packet.masks_rle) else None
    instance_meta = (
        [packet.instance_meta[i]]
        if packet.instance_meta and i < len(packet.instance_meta)
        else None
    )

    return replace(
        packet,
        stack=stack,
        stable_ids=stable_ids,
        scores=scores,
        labels=labels,
        n_inst=1,
        keypoints_list=kpts,
        cross_check_verdicts=vlist,
        cross_check_accessories=[],
        masks_rle=masks_rle,
        instance_meta=instance_meta,
    )


def filter_violator_single_id(packet: FramePacket, target_sid: int) -> FramePacket:
    """Keep at most one instance: target stable_id AND cross_check_ok=False."""
    verdicts = packet.cross_check_verdicts or []
    n = int(packet.n_inst)
    if n <= 0 or not verdicts or packet.stable_ids is None:
        return _empty_instances(packet)

    keep_idx: int | None = None
    for i in range(min(n, len(verdicts))):
        if i >= len(packet.stable_ids):
            break
        sid = int(packet.stable_ids[i])
        if sid != int(target_sid):
            continue
        if verdicts[i].ok:
            return _empty_instances(packet)
        keep_idx = i
        break

    if keep_idx is None:
        return _empty_instances(packet)

    i = keep_idx
    stack = packet.stack[i : i + 1] if packet.stack is not None and packet.stack.size else packet.stack
    stable_ids = packet.stable_ids[i : i + 1]
    scores = packet.scores[i : i + 1] if packet.scores is not None else packet.scores
    labels = [packet.labels[i]] if packet.labels and i < len(packet.labels) else []
    kpts = [packet.keypoints_list[i]] if packet.keypoints_list and i < len(packet.keypoints_list) else []
    vlist = [verdicts[i]]
    masks_rle = [packet.masks_rle[i]] if packet.masks_rle and i < len(packet.masks_rle) else None
    instance_meta = (
        [packet.instance_meta[i]]
        if packet.instance_meta and i < len(packet.instance_meta)
        else None
    )

    return replace(
        packet,
        stack=stack,
        stable_ids=stable_ids,
        scores=scores,
        labels=labels,
        n_inst=1,
        keypoints_list=kpts,
        cross_check_verdicts=vlist,
        cross_check_accessories=[],
        masks_rle=masks_rle,
        instance_meta=instance_meta,
    )


def _empty_instances(packet: FramePacket) -> FramePacket:
    if packet.n_inst > 0 and packet.stack is not None and packet.stack.ndim == 3:
        h, w = int(packet.stack.shape[1]), int(packet.stack.shape[2])
    elif packet.frame_bgr is not None and packet.frame_bgr.size > 0:
        h, w = int(packet.frame_bgr.shape[0]), int(packet.frame_bgr.shape[1])
    elif packet.mask_hw:
        h, w = int(packet.mask_hw[0]), int(packet.mask_hw[1])
    else:
        h, w = 480, 640
    return replace(
        packet,
        stack=np.zeros((0, h, w), dtype=np.uint8),
        stable_ids=np.zeros((0,), dtype=np.int64),
        scores=np.zeros((0,), dtype=np.float32),
        labels=[],
        n_inst=0,
        keypoints_list=[],
        cross_check_verdicts=[],
        cross_check_accessories=[],
        masks_rle=None,
        mask_hw=None,
        instance_meta=None,
    )


def _make_person_render_fn(target_sid: int):
    """Smooth per-person clip: update on keyframes, hold-forward between stride gaps."""
    last_draw: FramePacket | None = None

    def _render(job: _EncodeJob, ctx: _RenderContext) -> tuple[int, np.ndarray]:
        nonlocal last_draw
        carry = job.carry
        is_key = int(carry.frame_idx) == int(job.src_i)
        if is_key:
            filtered = filter_person_single_id(carry, target_sid)
            if filtered.n_inst > 0:
                last_draw = filtered
            else:
                last_draw = None

        if last_draw is None:
            empty = _empty_instances(replace(carry, frame_idx=job.src_i))
            return _render_encode_job(replace(job, carry=empty), ctx)

        work = replace(last_draw, frame_idx=job.src_i)
        return _render_encode_job(replace(job, carry=work, frame_bgr=job.frame_bgr), ctx)

    return _render


def _make_violator_render_fn(target_sid: int):
    """Draw overlay only on violation keyframes for target_sid (no hold-forward)."""

    def _render(job: _EncodeJob, ctx: _RenderContext) -> tuple[int, np.ndarray]:
        carry = job.carry
        is_key = int(carry.frame_idx) == int(job.src_i)
        if not is_key:
            empty = _empty_instances(replace(carry, frame_idx=job.src_i))
            return _render_encode_job(
                replace(job, carry=empty, frame_bgr=job.frame_bgr), ctx
            )
        filtered = filter_violator_single_id(carry, target_sid)
        if filtered.n_inst <= 0:
            empty = _empty_instances(replace(carry, frame_idx=job.src_i))
            return _render_encode_job(
                replace(job, carry=empty, frame_bgr=job.frame_bgr), ctx
            )
        return _render_encode_job(
            replace(job, carry=filtered, frame_bgr=job.frame_bgr), ctx
        )

    return _render


def _encode_one_violator(
    *,
    lookup: _ChunkPacketLookup | _SortedPacketLookup,
    run_dir: Path,
    run_id: str,
    stable_id: int,
    meta: dict[str, Any],
    overlay: dict[str, Any],
    video_path: Path,
    post_workers: int,
    encode_preset: str,
    encode_crf: int,
    encode_codec: str,
    on_progress: Callable[[int, int], None] | None,
    on_log: Callable[[str], None] | None,
    violation_frames: set[int] | None = None,
    violations_only: bool = True,
) -> Path:
    import app.core.video_encode as ve

    render_fn = (
        _make_violator_render_fn(stable_id)
        if violations_only
        else _make_person_render_fn(stable_id)
    )
    orig_render = ve._render_encode_job
    try:
        ve._render_encode_job = render_fn
        path = _encode_timeline_streaming(
            lookup,
            video_path=video_path,
            fps=float(meta["fps"]),
            prompt=str(meta["prompt"]),
            overlay=overlay,
            input_path=str(meta["input_path"]) or None,
            width=int(meta["width"]),
            height=int(meta["height"]),
            source_frame_count=int(meta["source_frame_count"]),
            frame_stride=int(meta.get("frame_stride") or 1),
            post_workers=post_workers,
            encode_preset=encode_preset,
            encode_crf=encode_crf,
            encode_codec=encode_codec,
            on_progress=on_progress,
            on_log=on_log,
            encode_src_indices=violation_frames,
            run_id=str(run_id),
        )
    finally:
        ve._render_encode_job = orig_render

    # Violation clips are short — skip full-length audio mux.
    if not violation_frames:
        input_path = str(meta["input_path"] or "")
        if input_path:
            from app.core.ffmpeg_utils import mux_audio_if_possible

            mux_audio_if_possible(input_path, path)

    return path


def encode_violations_videos_per_id(
    run_dir: Path | str,
    *,
    run_id: str | None = None,
    overlay_override: dict[str, Any] | None = None,
    source_video: str | None = None,
    post_workers: int = 6,
    encode_preset: str = "fast",
    encode_crf: int = 23,
    encode_codec: str = "auto",
    min_violation_frames: int = 3,
    min_violation_ratio: float = 0.08,
    min_relative_to_max: float = 0.25,
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    One short MP4 per qualified violator: only NO HELMET keyframes for that stable_id.
    Skips IDs with too few helmet violations vs presence / vs max offender.
    """
    run_dir = Path(run_dir)
    rid = infer_run_id(run_dir, run_id)
    meta = load_run_metadata(run_dir, rid, source_video=source_video)
    data = meta["packets_data"]
    overlay = resolve_overlay(meta, overlay_override=overlay_override)

    if not meta.get("input_path"):
        raise ValueError(meta.get("source_missing_hint") or "Source video not found for this run")
    if on_log:
        on_log(f"Source video: {meta['input_path']}")

    if not meta.get("cross_check_enabled"):
        raise ValueError("Cross-check was disabled for this run")

    violator_ids, violation_counts, presence_counts, threshold = collect_qualified_violator_ids(
        data,
        run_dir,
        min_violation_frames=min_violation_frames,
        min_violation_ratio=min_violation_ratio,
        min_relative_to_max=min_relative_to_max,
    )
    if not violator_ids:
        raw = count_violations_by_stable_id(data, run_dir)
        if not raw:
            raise ValueError("No helmet violations (NO HELMET) found in run packets")
        detail = ", ".join(f"id{sid}={n}" for sid, n in sorted(raw.items(), key=lambda x: -x[1]))
        raise ValueError(
            f"No violators passed threshold (>={threshold} NO HELMET frames, "
            f"ratio>={min_violation_ratio:.0%} of presence). Raw counts: {detail}"
        )

    if on_log:
        on_log(
            f"Qualified violators (threshold>={threshold} frames): "
            + ", ".join(
                f"id{sid}={violation_counts[sid]}/{presence_counts.get(sid, '?')}"
                for sid in violator_ids
            )
        )

    def _make_lookup() -> _ChunkPacketLookup | _SortedPacketLookup:
        if data.get("format") == "chunked" and "chunks" in data:
            return _ChunkPacketLookup(data, run_dir)
        return _SortedPacketLookup(list(data.get("packets") or []))

    frame_sets: dict[int, set[int]] = {}
    violation_sets: dict[int, set[int]] = {}
    for sid in violator_ids:
        presence = set(collect_presence_frame_indices(data, run_dir, sid))
        violations = set(collect_violation_frame_indices(data, run_dir, sid))
        if not presence:
            continue
        frame_sets[sid] = presence
        violation_sets[sid] = violations
    total_steps = sum(len(frames) for frames in frame_sets.values()) or 1
    done_steps = 0

    outputs: list[dict[str, Any]] = []
    for idx, sid in enumerate(violator_ids):
        encode_frames = frame_sets.get(sid)
        if not encode_frames:
            continue
        n_presence = len(encode_frames)
        n_v = len(violation_sets.get(sid, ()))
        if on_log:
            on_log(
                f"Encode violator stable_id={sid} "
                f"({idx + 1}/{len(violator_ids)}, {n_presence} frames, {n_v} NO HELMET)…"
            )
        out_path = run_dir / f"{rid}_violation_id{sid}.mp4"

        def _inner_progress(done: int, total: int, *, _base: int = done_steps) -> None:
            if on_progress is not None:
                on_progress(_base + done, total_steps)

        path = _encode_one_violator(
            lookup=_make_lookup(),
            run_dir=run_dir,
            run_id=rid,
            stable_id=sid,
            meta=meta,
            overlay=overlay,
            video_path=out_path,
            post_workers=post_workers,
            encode_preset=encode_preset,
            encode_crf=encode_crf,
            encode_codec=encode_codec,
            on_progress=lambda d, t, b=done_steps: _inner_progress(d, t, _base=b),
            on_log=on_log,
            violation_frames=encode_frames,
            violations_only=False,
        )
        done_steps += n_presence
        outputs.append(
            {
                "stable_id": sid,
                "violation_frames": n_v,
                "violation_count": violation_counts.get(sid, n_v),
                "presence_frames": n_presence,
                "video_path": str(path.resolve()),
                "video_name": path.name,
            }
        )

    info = {
        "run_id": rid,
        "cross_check_enabled": meta["cross_check_enabled"],
        "cross_check_violations": meta["cross_check_violations"],
        "violator_count": len(outputs),
        "violation_threshold": threshold,
        "stable_ids": [o["stable_id"] for o in outputs],
        "violation_counts": {str(k): v for k, v in violation_counts.items()},
        "presence_counts": {str(k): v for k, v in presence_counts.items()},
        "overlay_used": overlay,
        "videos": outputs,
    }
    return outputs, info


def encode_person_videos_per_id(
    run_dir: Path | str,
    *,
    run_id: str | None = None,
    overlay_override: dict[str, Any] | None = None,
    post_workers: int = 6,
    encode_preset: str = "fast",
    encode_crf: int = 23,
    encode_codec: str = "auto",
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    One MP4 per stable ReID id (full timeline, any cross-check state).
    Prefer encode_violations_videos_per_id for NO HELMET review clips.
    """
    run_dir = Path(run_dir)
    rid = infer_run_id(run_dir, run_id)
    meta = load_run_metadata(run_dir, rid)
    data = meta["packets_data"]
    overlay = resolve_overlay(meta, overlay_override=None)

    person_ids = collect_stable_ids(data, run_dir)
    if not person_ids:
        raise ValueError("No tracked persons (stable ReID ids) found in run packets")

    def _make_lookup() -> _ChunkPacketLookup | _SortedPacketLookup:
        if data.get("format") == "chunked" and "chunks" in data:
            return _ChunkPacketLookup(data, run_dir)
        return _SortedPacketLookup(list(data.get("packets") or []))

    n_ids = len(person_ids)
    n_frames = max(1, int(meta["source_frame_count"]))
    total_steps = n_ids * n_frames
    done_steps = 0

    outputs: list[dict[str, Any]] = []
    for idx, sid in enumerate(person_ids):
        if on_log:
            on_log(f"Encode person stable_id={sid} ({idx + 1}/{n_ids})…")
        out_path = run_dir / f"{rid}_person_id{sid}.mp4"

        def _inner_progress(done: int, total: int, *, _base: int = done_steps) -> None:
            if on_progress is not None:
                on_progress(_base + done, total_steps)

        path = _encode_one_violator(
            lookup=_make_lookup(),
            run_dir=run_dir,
            run_id=rid,
            stable_id=sid,
            meta=meta,
            overlay=overlay,
            video_path=out_path,
            post_workers=post_workers,
            encode_preset=encode_preset,
            encode_crf=encode_crf,
            encode_codec=encode_codec,
            on_progress=lambda d, t, b=done_steps: _inner_progress(d, t, _base=b),
            on_log=on_log,
            violation_frames=None,
            violations_only=False,
        )
        done_steps += n_frames
        outputs.append(
            {
                "stable_id": sid,
                "video_path": str(path.resolve()),
                "video_name": path.name,
            }
        )

    info = {
        "run_id": rid,
        "cross_check_enabled": meta["cross_check_enabled"],
        "cross_check_violations": meta["cross_check_violations"],
        "person_count": len(outputs),
        "stable_ids": person_ids,
        "overlay_used": overlay,
        "videos": outputs,
    }
    return outputs, info


def encode_violations_video(
    run_dir: Path | str,
    *,
    run_id: str | None = None,
    video_path: Path | str | None = None,
    overlay_override: dict[str, Any] | None = None,
    post_workers: int = 6,
    encode_preset: str = "fast",
    encode_crf: int = 23,
    encode_codec: str = "auto",
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Legacy single-file API: returns first violator video only."""
    outputs, info = encode_violations_videos_per_id(
        run_dir,
        run_id=run_id,
        overlay_override=overlay_override,
        post_workers=post_workers,
        encode_preset=encode_preset,
        encode_crf=encode_crf,
        encode_codec=encode_codec,
        on_progress=on_progress,
        on_log=on_log,
    )
    first = outputs[0]
    path = Path(first["video_path"])
    if video_path is not None:
        dest = Path(video_path)
        if dest.resolve() != path.resolve():
            import shutil

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            path = dest
    return path, info
