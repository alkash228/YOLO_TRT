"""Keyframe boxes for the in-browser player (same bboxes as Word stills)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from WEB_app.video_builder import (
    _bbox_xywh_for_sid,
    _verdict_is_violation,
    iter_run_packets,
    load_run_metadata,
)

from app.core.video_encode import resolve_run_source_video


def _helmet_boxes(packet: Any, *, video_w: int, video_h: int) -> list[dict[str, int]]:
    acc = list(getattr(packet, "cross_check_accessories", None) or [])
    if not acc:
        return []
    src_h = src_w = 0
    mask_hw = getattr(packet, "mask_hw", None)
    if mask_hw:
        src_h, src_w = int(mask_hw[0]), int(mask_hw[1])
    sx = (float(video_w) / float(src_w)) if src_w > 0 and video_w > 0 else 1.0
    sy = (float(video_h) / float(src_h)) if src_h > 0 and video_h > 0 else 1.0
    out: list[dict[str, int]] = []
    for det in acc:
        xyxy = getattr(det, "xyxy", None)
        if xyxy is None and isinstance(det, dict):
            xyxy = det.get("xyxy")
        if xyxy is None:
            continue
        try:
            vals = list(xyxy)
        except TypeError:
            continue
        if len(vals) < 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in vals[:4]]
        x0 *= sx
        x1 *= sx
        y0 *= sy
        y1 *= sy
        w, h = x1 - x0, y1 - y0
        if w <= 1 or h <= 1:
            continue
        out.append(
            {
                "id": 0,
                "x": int(round(x0)),
                "y": int(round(y0)),
                "w": max(1, int(round(w))),
                "h": max(1, int(round(h))),
                "v": 0,
                "k": "h",
            }
        )
    return out


def _packet_boxes(packet: Any, *, video_w: int, video_h: int) -> list[dict[str, int]]:
    n = int(getattr(packet, "n_inst", 0) or 0)
    sids = getattr(packet, "stable_ids", None)
    verdicts = list(getattr(packet, "cross_check_verdicts", None) or [])
    out: list[dict[str, int]] = []
    tw = max(1, int(video_w))
    th = max(1, int(video_h))
    meta = list(getattr(packet, "instance_meta", None) or [])
    for i in range(max(0, n)):
        sid = int(sids[i]) if sids is not None and i < len(sids) else i + 1
        bbox = _bbox_xywh_for_sid(packet, sid, target_w=tw, target_h=th)
        if bbox is None and i < len(meta) and isinstance(meta[i], dict):
            bb = meta[i].get("bbox_xywh")
            if bb is not None and len(bb) >= 4:
                x, y, bw, bh = [float(v) for v in bb[:4]]
                if bw > 0 and bh > 0:
                    src_h = src_w = 0
                    mh = getattr(packet, "mask_hw", None)
                    if mh:
                        src_h, src_w = int(mh[0]), int(mh[1])
                    if src_w > 0 and src_h > 0 and (src_w != tw or src_h != th):
                        bbox = (
                            int(round(x * tw / src_w)),
                            int(round(y * th / src_h)),
                            max(1, int(round(bw * tw / src_w))),
                            max(1, int(round(bh * th / src_h))),
                        )
                    else:
                        bbox = (int(x), int(y), int(bw), int(bh))
        if bbox is None:
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        viol = i < len(verdicts) and _verdict_is_violation(verdicts[i])
        out.append(
            {
                "id": sid,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "v": 1 if viol else 0,
                "k": "p",
            }
        )
    out.extend(_helmet_boxes(packet, video_w=tw, video_h=th))
    return out


def build_overlay_timeline(
    run_dir: str | Path,
    run_id: str,
    *,
    source_video: str | None = None,
    t0: float | None = None,
    t1: float | None = None,
) -> dict[str, Any]:
    """Boxes from packets. Optional [t0, t1] window so hour-long runs stay small."""
    run_path = Path(run_dir)
    rid = str(run_id)
    meta = load_run_metadata(
        run_path, rid, source_video=source_video, scan_source_frames=False
    )
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
    win0 = None if t0 is None else max(0.0, float(t0))
    win1 = None if t1 is None else max(0.0, float(t1))
    events: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    hold: dict[str, Any] | None = None
    for packet in iter_run_packets(data, run_path):
        fi = int(packet.frame_idx)
        t = fi / fps
        if win1 is not None and t > win1 + 0.05:
            break
        if width <= 0 or height <= 0:
            mh = getattr(packet, "mask_hw", None)
            if mh:
                height, width = int(mh[0]), int(mh[1])
        boxes = _packet_boxes(packet, video_w=max(1, width), video_h=max(1, height))
        row = {
            "n": fi,
            "t0": t,
            "t1": t + (1.0 / fps),
            "boxes": boxes,
        }
        if win0 is not None and t < win0 - 0.05:
            hold = row
            continue
        if prev is None and hold is not None:
            hold["t1"] = t
            events.append(hold)
            hold = None
        if prev is not None:
            prev["t1"] = t
            events.append(prev)
        prev = row
    if prev is not None:
        if win1 is not None:
            prev["t1"] = max(float(prev["t1"]), min(win1, float(prev["t0"]) + 2.0))
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
