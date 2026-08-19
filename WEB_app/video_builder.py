"""Build per-violator MP4s from a YOLO_DRT run folder — one video per stable ReID ID.

Qualification uses NO HELMET counts; clip timeline is the full presence of that ID
(all keyframes where the person appears), not a sparse NO HELMET montage.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
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


def web_debug_show_all_dets() -> bool:
    """Debug period: force boxes/masks/helmet overlay knobs. Off: YOLO_DRT_WEB_DEBUG_OVERLAY=0.

    Per-ID violation/person clips still draw only the target stable_id (no bystanders).
    """
    return os.environ.get("YOLO_DRT_WEB_DEBUG_OVERLAY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
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
    scan_source_frames: bool = True,
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
    if source_frame_count <= 0 and scan_source_frames:
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
    debug_all = web_debug_show_all_dets()
    overlay: dict[str, Any] = {
        "overlay_alpha": 0.45,
        "draw_boxes": True,
        "draw_masks": False,
        "draw_centers": False,
        # Pose is optional; person∩helmet does not require keypoints.
        "draw_pose": False,
        "pose_kpt_conf": 0.25,
        "cross_check_enabled": bool(meta.get("cross_check_enabled", True)),
        "cross_check_draw_head_box": True,
        "cross_check_draw_boxes": True,
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
    if debug_all:
        # Debug: force target overlay knobs (boxes / helmet / head). Per-ID render
        # still filters to one stable_id — do not use this to stamp bystanders.
        # Do not force pose skeletons (no-op when keypoints absent anyway).
        overlay["draw_boxes"] = True
        overlay["draw_masks"] = True
        overlay["cross_check_enabled"] = True
        overlay["cross_check_draw_head_box"] = True
        overlay["cross_check_draw_boxes"] = True
        overlay["pose_point_radius"] = 6
        overlay["pose_line_thickness"] = 3
    else:
        # Production violation clips: don't spam accessory boxes of bystanders.
        overlay["cross_check_draw_boxes"] = False
    return overlay


def stamp_debug_hud_rgb(
    rgb: np.ndarray,
    packet: FramePacket,
    *,
    draw_pose: bool,
) -> np.ndarray:
    """Visible proof that new WEB debug render ran (not a cached old clip)."""
    kpts = packet.keypoints_list or []
    n_kpt = sum(
        1
        for k in kpts
        if k is not None and getattr(k, "size", 0) > 0
    )
    n_helm = len(packet.cross_check_accessories or [])
    n_fail = sum(
        1
        for v in (packet.cross_check_verdicts or [])
        if _verdict_is_violation(v)
    )
    pose_state = "ON" if draw_pose and n_kpt > 0 else ("off*" if draw_pose else "off")
    label = (
        f"DBG person+helmet  people={int(packet.n_inst)}  "
        f"helmets={n_helm}  NO_HELMET={n_fail}  pose={pose_state}"
    )
    out = np.ascontiguousarray(rgb.copy())
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    cv2.rectangle(bgr, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.putText(
        bgr,
        label,
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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


def _verdict_is_violation(verdict: Any) -> bool:
    if verdict is None:
        return False
    if isinstance(verdict, dict):
        return not bool(verdict.get("ok", True))
    return not bool(getattr(verdict, "ok", True))


def diagnose_packet_violations(data: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Stats used when encode finds zero NO HELMET — distinguish empty vs wrong folder."""
    n_packets = 0
    n_with_people = 0
    n_with_verdicts = 0
    n_ok = 0
    n_fail = 0
    n_empty_verdicts = 0
    for packet in iter_run_packets(data, Path(run_dir)):
        n_packets += 1
        n = int(packet.n_inst)
        if n > 0:
            n_with_people += 1
        verdicts = packet.cross_check_verdicts or []
        if n > 0 and not verdicts:
            n_empty_verdicts += 1
        if not verdicts:
            continue
        n_with_verdicts += 1
        for i in range(min(n if n > 0 else len(verdicts), len(verdicts))):
            if _verdict_is_violation(verdicts[i]):
                n_fail += 1
            else:
                n_ok += 1
    chunks = list(Path(run_dir).glob("*_packets_chunk_*.pkl"))
    manifests = list(Path(run_dir).glob("*_packets_manifest.json"))
    return {
        "run_dir": str(Path(run_dir).resolve()),
        "packets": n_packets,
        "packets_with_people": n_with_people,
        "packets_with_verdicts": n_with_verdicts,
        "packets_people_but_no_verdicts": n_empty_verdicts,
        "verdict_ok": n_ok,
        "verdict_no_helmet": n_fail,
        "chunk_files": len(chunks),
        "has_manifest": bool(manifests),
        "format": data.get("format"),
    }


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
    """Unique person stable_ids that violated cross-check (no helmet) at least once."""
    ids: set[int] = set()
    for packet in iter_run_packets(data, Path(run_dir)):
        verdicts = packet.cross_check_verdicts or []
        n = int(packet.n_inst)
        if n <= 0 or not verdicts:
            continue
        for i in range(min(n, len(verdicts))):
            if not _verdict_is_violation(verdicts[i]):
                continue
            if packet.stable_ids is None or i >= len(packet.stable_ids):
                continue
            ids.add(int(packet.stable_ids[i]))
    return sorted(ids)


def count_violations_by_stable_id(data: dict[str, Any], run_dir: Path) -> dict[int, int]:
    """Cross-check NO HELMET keyframes per person stable_id."""
    counts: dict[int, int] = {}
    for packet in iter_run_packets(data, Path(run_dir)):
        verdicts = packet.cross_check_verdicts or []
        n = int(packet.n_inst)
        if n <= 0 or not verdicts or packet.stable_ids is None:
            continue
        for i in range(min(n, len(verdicts))):
            if not _verdict_is_violation(verdicts[i]):
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
    min_violation_ratio: float = 0.0,
    min_relative_to_max: float = 0.0,
) -> tuple[list[int], dict[int, int], dict[int, int], int]:
    """
    Gate for writing a per-ID clip: primarily NO HELMET frame count.

    INCLUDE if viol_frames >= min_violation_frames (default 3).
    Optional extras (off by default — they used to fight “many NO HELMET ⇒ in”):
      - min_relative_to_max > 0: also require >= max_v * that fraction
      - min_violation_ratio > 0: also require viol/presence >= that ratio

    Returns (qualified_ids desc by count, violation_counts, presence_counts, threshold_used).
    """
    vcounts = count_violations_by_stable_id(data, run_dir)
    pcounts = count_presence_by_stable_id(data, run_dir)
    if not vcounts:
        return [], vcounts, pcounts, int(min_violation_frames)

    threshold = max(1, int(min_violation_frames))
    if float(min_relative_to_max) > 0:
        max_v = max(vcounts.values())
        threshold = max(threshold, int(max_v * float(min_relative_to_max)))

    qualified: list[int] = []
    for sid, vc in vcounts.items():
        if vc < threshold:
            continue
        if float(min_violation_ratio) > 0:
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
    # Keep helmet accessory dets so stills/clips can draw person∩helmet without pose.
    accessories = list(packet.cross_check_accessories or [])

    return replace(
        packet,
        stack=stack,
        stable_ids=stable_ids,
        scores=scores,
        labels=labels,
        n_inst=1,
        keypoints_list=kpts,
        cross_check_verdicts=vlist,
        cross_check_accessories=accessories,
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
        if not _verdict_is_violation(verdicts[i]):
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
    accessories = list(packet.cross_check_accessories or [])

    return replace(
        packet,
        stack=stack,
        stable_ids=stable_ids,
        scores=scores,
        labels=labels,
        n_inst=1,
        keypoints_list=kpts,
        cross_check_verdicts=vlist,
        cross_check_accessories=accessories,
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


def _instance_index_for_sid(packet: FramePacket, target_sid: int) -> int | None:
    if packet.stable_ids is None or packet.n_inst <= 0:
        return None
    for i in range(min(int(packet.n_inst), len(packet.stable_ids))):
        if int(packet.stable_ids[i]) == int(target_sid):
            return i
    return None


def _bbox_xywh_for_sid(
    packet: FramePacket,
    target_sid: int,
    *,
    target_w: int | None = None,
    target_h: int | None = None,
) -> tuple[int, int, int, int] | None:
    i = _instance_index_for_sid(packet, target_sid)
    if i is None:
        return None
    bbox: tuple[int, int, int, int] | None = None
    meta = packet.instance_meta or []
    if i < len(meta):
        bb = meta[i].get("bbox_xywh") if isinstance(meta[i], dict) else None
        if bb is not None and len(bb) >= 4:
            x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            if w > 0 and h > 0:
                bbox = (x, y, w, h)
    if bbox is None and packet.stack is not None and packet.stack.ndim == 3 and i < int(
        packet.stack.shape[0]
    ):
        m = (packet.stack[i] > 127).astype(np.uint8)
        if m.any():
            x, y, w, h = cv2.boundingRect(m)
            if w > 0 and h > 0:
                bbox = (int(x), int(y), int(w), int(h))
    if bbox is None:
        return None
    if target_w is None or target_h is None or target_w <= 0 or target_h <= 0:
        return bbox
    src_h = src_w = 0
    if packet.mask_hw:
        src_h, src_w = int(packet.mask_hw[0]), int(packet.mask_hw[1])
    elif packet.stack is not None and packet.stack.ndim == 3 and packet.stack.shape[0] > 0:
        src_h, src_w = int(packet.stack.shape[1]), int(packet.stack.shape[2])
    elif packet.frame_bgr is not None and packet.frame_bgr.size > 0:
        src_h, src_w = int(packet.frame_bgr.shape[0]), int(packet.frame_bgr.shape[1])
    if src_w <= 0 or src_h <= 0 or (src_w == target_w and src_h == target_h):
        return bbox
    x, y, w, h = bbox
    sx = float(target_w) / float(src_w)
    sy = float(target_h) / float(src_h)
    return (
        int(round(x * sx)),
        int(round(y * sy)),
        max(1, int(round(w * sx))),
        max(1, int(round(h * sy))),
    )


def stamp_all_people_boxes_rgb(
    rgb: np.ndarray,
    packet: FramePacket,
    *,
    target_w: int,
    target_h: int,
    focus_sid: int | None = None,
) -> np.ndarray:
    """
    Draw every tracked person from instance_meta / stack bboxes.
    Used in debug when mask overlay silently skips (empty stack / size mismatch).
    """
    n = int(packet.n_inst or 0)
    if n <= 0:
        return rgb
    out = np.ascontiguousarray(rgb.copy())
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    tw, th = int(target_w), int(target_h)
    for i in range(n):
        sid = (
            int(packet.stable_ids[i])
            if packet.stable_ids is not None and i < len(packet.stable_ids)
            else i + 1
        )
        if focus_sid is not None and int(sid) == int(focus_sid):
            continue  # focus callout drawn separately
        bbox = _bbox_xywh_for_sid(packet, sid, target_w=tw, target_h=th)
        if bbox is None:
            # Try index-based meta when stable_ids missing/duplicate edge cases.
            meta = packet.instance_meta or []
            if i < len(meta) and isinstance(meta[i], dict):
                bb = meta[i].get("bbox_xywh")
                if bb is not None and len(bb) >= 4:
                    x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                    if w > 0 and h > 0:
                        src_h = src_w = 0
                        if packet.mask_hw:
                            src_h, src_w = int(packet.mask_hw[0]), int(packet.mask_hw[1])
                        if src_w > 0 and src_h > 0 and (src_w != tw or src_h != th):
                            sx = float(tw) / float(src_w)
                            sy = float(th) / float(src_h)
                            bbox = (
                                int(round(x * sx)),
                                int(round(y * sy)),
                                max(1, int(round(w * sx))),
                                max(1, int(round(h * sy))),
                            )
                        else:
                            bbox = (x, y, w, h)
        if bbox is None:
            continue
        x, y, w, h = bbox
        color = (80, 200, 80)  # green-ish BGR for helmeted / other people
        verdicts = packet.cross_check_verdicts or []
        if i < len(verdicts) and _verdict_is_violation(verdicts[i]):
            color = (0, 165, 255)  # orange other violators
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            bgr,
            f"ID {sid}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def highlight_no_helmet_on_rgb(
    rgb: np.ndarray,
    packet: FramePacket,
    target_sid: int,
    *,
    target_w: int,
    target_h: int,
    frame_bgr: np.ndarray | None = None,
    force: bool = False,
) -> np.ndarray:
    """Thick red callout on the NO HELMET target (RGB image)."""
    del frame_bgr  # kept for call-site compatibility
    i = _instance_index_for_sid(packet, target_sid)
    if i is None:
        return rgb
    verdicts = packet.cross_check_verdicts or []
    is_viol = i < len(verdicts) and _verdict_is_violation(verdicts[i])
    if not force and not is_viol:
        return rgb

    bbox = _bbox_xywh_for_sid(
        packet, target_sid, target_w=int(target_w), target_h=int(target_h)
    )
    if bbox is None:
        return rgb

    out = np.ascontiguousarray(rgb.copy())
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    # Outer black + thick red so the no-helmet target is obvious among all dets.
    cv2.rectangle(bgr, (x - 4, y - 4), (x2 + 4, y2 + 4), (0, 0, 0), 6)
    cv2.rectangle(bgr, (x, y), (x2, y2), (0, 0, 255), 4)
    label = f"NO HELMET  ID {int(target_sid)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, min(1.4, w / 220.0))
    thickness = 2 if scale < 1.0 else 3
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 8
    ty = max(th + pad * 2, y - 10)
    cv2.rectangle(
        bgr,
        (x, ty - th - pad * 2),
        (x + tw + pad * 2, ty),
        (0, 0, 180),
        -1,
    )
    cv2.putText(
        bgr,
        label,
        (x + pad, ty - pad),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def stamp_helmet_accessories_rgb(
    rgb: np.ndarray,
    packet: FramePacket,
    *,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Draw helmet / accessory boxes from cross_check_accessories (no pose required)."""
    from app.core.cross_check import draw_cross_check_detections
    from app.core.frame_pipeline import _scale_cross_check_geometry

    accessories = list(packet.cross_check_accessories or [])
    if not accessories:
        return rgb

    tw, th = int(target_w), int(target_h)
    src_h = src_w = 0
    if packet.mask_hw:
        src_h, src_w = int(packet.mask_hw[0]), int(packet.mask_hw[1])
    elif packet.stack is not None and packet.stack.ndim == 3 and packet.stack.shape[0] > 0:
        src_h, src_w = int(packet.stack.shape[1]), int(packet.stack.shape[2])
    if src_h > 0 and src_w > 0 and (src_h != th or src_w != tw):
        _, accessories = _scale_cross_check_geometry(
            None, accessories, float(tw) / float(src_w), float(th) / float(src_h)
        )

    ih, iw = int(rgb.shape[0]), int(rgb.shape[1])
    if (ih, iw) != (th, tw) and ih > 0 and iw > 0:
        _, accessories = _scale_cross_check_geometry(
            None, accessories, float(iw) / float(tw), float(ih) / float(th)
        )

    out = np.ascontiguousarray(rgb.copy())
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    bgr = draw_cross_check_detections(bgr, accessories, color=(0, 200, 255))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def stamp_all_poses_rgb(
    rgb: np.ndarray,
    packet: FramePacket,
    *,
    target_w: int,
    target_h: int,
    pose_kpt_conf: float = 0.25,
    pose_point_radius: int = 6,
    pose_line_thickness: int = 3,
) -> np.ndarray:
    """
    Optional YOLO pose skeletons for every instance.
    No-op when keypoints_list is empty (person-detect / no-pose runs).
    """
    from app.core.exporter import pose_overlay_colors
    from app.core.frame_pipeline import _scale_keypoints_list
    from app.core.pose_utils import draw_pose_on_frame

    kpts = list(packet.keypoints_list or [])
    if not kpts or not any(
        k is not None and getattr(k, "size", 0) > 0 for k in kpts
    ):
        return rgb

    tw, th = int(target_w), int(target_h)
    src_h = src_w = 0
    if packet.mask_hw:
        src_h, src_w = int(packet.mask_hw[0]), int(packet.mask_hw[1])
    elif packet.stack is not None and packet.stack.ndim == 3 and packet.stack.shape[0] > 0:
        src_h, src_w = int(packet.stack.shape[1]), int(packet.stack.shape[2])
    if src_h > 0 and src_w > 0 and (src_h != th or src_w != tw):
        kpts = _scale_keypoints_list(kpts, float(tw) / float(src_w), float(th) / float(src_h))

    # Also scale if image size differs from requested target (letterbox / resize).
    ih, iw = int(rgb.shape[0]), int(rgb.shape[1])
    if (ih, iw) != (th, tw) and ih > 0 and iw > 0:
        kpts = _scale_keypoints_list(kpts, float(iw) / float(tw), float(ih) / float(th))

    out = np.ascontiguousarray(rgb.copy())
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    # High-contrast BGR colors so bystander skeletons are not lost on concrete/sun.
    palette = [
        (0, 255, 255),  # cyan
        (255, 0, 255),  # magenta
        (0, 255, 0),  # green
        (255, 255, 0),  # yellow
        (255, 128, 0),  # orange
        (255, 0, 128),  # pink
    ]
    n = max(int(packet.n_inst or 0), len(kpts), 1)
    colors = [palette[i % len(palette)] for i in range(n)]
    # Keep default distinct palette as fallback length pad.
    colors = colors + pose_overlay_colors(n)[len(colors) :]
    bgr = draw_pose_on_frame(
        bgr,
        kpts,
        colors,
        kpt_conf=float(pose_kpt_conf),
        draw_skeleton=True,
        point_radius=int(max(4, pose_point_radius)),
        line_thickness=int(max(3, pose_line_thickness)),
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _render_full_scene_highlight(
    job: _EncodeJob,
    ctx: _RenderContext,
    carry: FramePacket,
    target_sid: int,
    *,
    force_highlight: bool,
) -> tuple[int, np.ndarray]:
    work = replace(carry, frame_idx=job.src_i)
    src_i, rgb = _render_encode_job(
        replace(job, carry=work, frame_bgr=job.frame_bgr), ctx
    )
    # Person boxes first — primary cue for person∩helmet (no pose required).
    rgb = stamp_all_people_boxes_rgb(
        rgb,
        work,
        target_w=ctx.target_w,
        target_h=ctx.target_h,
        focus_sid=target_sid,
    )
    # Helmet accessory boxes from cross-check model.
    rgb = stamp_helmet_accessories_rgb(
        rgb,
        work,
        target_w=ctx.target_w,
        target_h=ctx.target_h,
    )
    # Optional pose: only when enabled AND keypoints exist (else no-op).
    if bool(ctx.overlay.get("draw_pose", False)):
        rgb = stamp_all_poses_rgb(
            rgb,
            work,
            target_w=ctx.target_w,
            target_h=ctx.target_h,
            pose_kpt_conf=float(ctx.overlay.get("pose_kpt_conf", 0.25)),
            pose_point_radius=int(ctx.overlay.get("pose_point_radius", 6) or 6),
            pose_line_thickness=int(ctx.overlay.get("pose_line_thickness", 3) or 3),
        )
    rgb = highlight_no_helmet_on_rgb(
        rgb,
        work,
        target_sid,
        target_w=ctx.target_w,
        target_h=ctx.target_h,
        frame_bgr=job.frame_bgr,
        force=force_highlight,
    )
    rgb = stamp_debug_hud_rgb(
        rgb,
        work,
        draw_pose=bool(ctx.overlay.get("draw_pose", False)),
    )
    return src_i, rgb


def _render_target_only(
    job: _EncodeJob,
    ctx: _RenderContext,
    filtered: FramePacket,
    target_sid: int,
    *,
    debug_all: bool,
    force_highlight: bool = False,
) -> tuple[int, np.ndarray]:
    """Draw only target stable_id; highlight NO HELMET; never stamp bystanders."""
    work = replace(filtered, frame_idx=job.src_i)
    src_i, rgb = _render_encode_job(
        replace(job, carry=work, frame_bgr=job.frame_bgr), ctx
    )
    if debug_all:
        rgb = stamp_helmet_accessories_rgb(
            rgb,
            work,
            target_w=ctx.target_w,
            target_h=ctx.target_h,
        )
        if bool(ctx.overlay.get("draw_pose", False)):
            rgb = stamp_all_poses_rgb(
                rgb,
                work,
                target_w=ctx.target_w,
                target_h=ctx.target_h,
                pose_kpt_conf=float(ctx.overlay.get("pose_kpt_conf", 0.25)),
                pose_point_radius=int(ctx.overlay.get("pose_point_radius", 6) or 6),
                pose_line_thickness=int(ctx.overlay.get("pose_line_thickness", 3) or 3),
            )
    rgb = highlight_no_helmet_on_rgb(
        rgb,
        work,
        target_sid,
        target_w=ctx.target_w,
        target_h=ctx.target_h,
        frame_bgr=job.frame_bgr,
        force=force_highlight,
    )
    if debug_all:
        rgb = stamp_debug_hud_rgb(
            rgb,
            work,
            draw_pose=bool(ctx.overlay.get("draw_pose", False)),
        )
    return src_i, rgb


def _make_person_render_fn(target_sid: int):
    """Per-ID clip: target only, hold-forward between stride gaps; highlight NO HELMET."""
    last_draw: FramePacket | None = None
    debug_all = web_debug_show_all_dets()

    def _render(job: _EncodeJob, ctx: _RenderContext) -> tuple[int, np.ndarray]:
        nonlocal last_draw
        carry = job.carry
        is_key = int(carry.frame_idx) == int(job.src_i)
        if is_key:
            filtered = filter_person_single_id(carry, target_sid)
            last_draw = filtered if filtered.n_inst > 0 else None

        if last_draw is None:
            empty = _empty_instances(replace(carry, frame_idx=job.src_i))
            return _render_encode_job(replace(job, carry=empty), ctx)

        return _render_target_only(
            job, ctx, last_draw, target_sid, debug_all=debug_all, force_highlight=False
        )

    return _render


def _make_violator_render_fn(target_sid: int):
    """Draw overlay on violation keyframes for target_sid only (no bystanders)."""
    debug_all = web_debug_show_all_dets()

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
        return _render_target_only(
            job, ctx, filtered, target_sid, debug_all=debug_all, force_highlight=True
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
    if on_log and web_debug_show_all_dets():
        on_log(
            "WEB debug overlay ON: target-only boxes/masks/helmet + NO HELMET "
            f"(id={stable_id}; no bystanders)"
        )
    # Force rewrite so players/browsers cannot keep a stale same-name MP4.
    try:
        if Path(video_path).is_file():
            Path(video_path).unlink()
    except OSError:
        pass

    render_fn = (
        _make_violator_render_fn(stable_id)
        if violations_only
        else _make_person_render_fn(stable_id)
    )
    path = _encode_timeline_streaming(
        lookup,
        video_path=video_path,
        fps=float(meta["fps"]),
        prompt=str(meta["prompt"]),
        overlay=overlay,
        input_path=str(meta["input_path"] or "") or None,
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
        render_job_fn=render_fn,
    )

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
    min_violation_ratio: float = 0.0,
    min_relative_to_max: float = 0.0,
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    One MP4 per ID with enough NO HELMET frames (gate: min_violation_frames).

    Clip timeline = all presence keyframes for that stable_id (grab-skip), not a
    sparse NO HELMET montage. Overlay draws only that ID.
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
    qualified_set = set(violator_ids)
    if on_log:
        extras = []
        if float(min_relative_to_max) > 0:
            extras.append(f"min_relative_to_max={min_relative_to_max}")
        if float(min_violation_ratio) > 0:
            extras.append(f"min_violation_ratio={min_violation_ratio}")
        extra_s = f" (+{', '.join(extras)})" if extras else ""
        on_log(
            f"Qualification gate: min_violation_frames={threshold} "
            f"(main knob; many NO HELMET ⇒ INCLUDE){extra_s}"
        )
        for sid in sorted(violation_counts.keys(), key=lambda s: (-violation_counts[s], s)):
            vc = int(violation_counts[sid])
            decision = "INCLUDE" if sid in qualified_set else "SKIP"
            on_log(f"id{sid} viol_frames={vc} → {decision}")

    if not violator_ids:
        raw = count_violations_by_stable_id(data, run_dir)
        diag = diagnose_packet_violations(data, run_dir)
        report_n = int(meta.get("cross_check_violations") or 0)
        if not raw:
            raise ValueError(
                "No helmet violations (NO HELMET) found in run packets. "
                f"diag={diag}; report.txt violations={report_n}. "
                + (
                    "Report says violations but packets have 0 — wrong/incomplete run folder "
                    "(missing *_packets_chunk_*.pkl). Copy full run from Docker output bind."
                    if report_n > 0
                    else (
                        "Packets have people but empty cross_check_verdicts — helmet engine "
                        "likely did not run in Docker."
                        if int(diag.get("packets_people_but_no_verdicts") or 0) > 0
                        else "In this run the model marked everyone as OK (helmet on) "
                        "or no people were tracked."
                    )
                )
            )
        detail = ", ".join(f"id{sid}={n}" for sid, n in sorted(raw.items(), key=lambda x: -x[1]))
        raise ValueError(
            f"No violators passed gate (>= {threshold} NO HELMET frames). Raw counts: {detail}"
        )

    def _make_lookup() -> _ChunkPacketLookup | _SortedPacketLookup:
        if data.get("format") == "chunked" and "chunks" in data:
            return _ChunkPacketLookup(data, run_dir)
        return _SortedPacketLookup(list(data.get("packets") or []))

    frame_sets: dict[int, set[int]] = {}
    violation_sets: dict[int, set[int]] = {}
    presence_sets: dict[int, set[int]] = {}
    encode_is_presence: dict[int, bool] = {}
    for sid in violator_ids:
        presence = set(collect_presence_frame_indices(data, run_dir, sid))
        violations = set(collect_violation_frame_indices(data, run_dir, sid))
        # Full presence timeline for this ID; fall back to violation frames if empty.
        use_presence = bool(presence)
        use_frames = presence if use_presence else violations
        if not use_frames:
            continue
        frame_sets[sid] = use_frames
        violation_sets[sid] = violations
        presence_sets[sid] = presence
        encode_is_presence[sid] = use_presence
    total_steps = sum(len(frames) for frames in frame_sets.values()) or 1
    done_steps = 0

    outputs: list[dict[str, Any]] = []
    for idx, sid in enumerate(violator_ids):
        encode_frames = frame_sets.get(sid)
        if not encode_frames:
            continue
        n_encode = len(encode_frames)
        n_v = len(violation_sets.get(sid, ()))
        n_presence = len(presence_sets.get(sid, ())) or int(presence_counts.get(sid, 0))
        if on_log:
            if encode_is_presence.get(sid, False):
                on_log(
                    f"id{sid}: presence_frames={n_encode} violation_frames={n_v} "
                    f"→ encoding all presence ({idx + 1}/{len(violator_ids)})"
                )
            else:
                on_log(
                    f"id{sid}: presence_frames={n_presence} violation_frames={n_v} "
                    f"→ encoding violation frames fallback ({idx + 1}/{len(violator_ids)})"
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
        done_steps += n_encode
        outputs.append(
            {
                "stable_id": sid,
                "violation_frames": n_v,
                "violation_count": violation_counts.get(sid, n_v),
                "presence_frames": n_presence if n_presence > 0 else n_encode,
                "encode_frames": n_encode,
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
        "min_violation_frames": int(min_violation_frames),
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
    One MP4 per stable ReID id (full source timeline, any cross-check state).
    For violator review clips prefer encode_violations_videos_per_id
    (qualify by NO HELMET count; encode full presence of that ID).
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
