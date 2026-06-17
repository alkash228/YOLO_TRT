"""Shared per-frame detection prep for GPU worker and CPU finalize."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config.settings import PipelineSettings
from app.core.detect_engine import DetectItem
from app.core.fusion import inherit_motion_ids, merge_seg_fallback_detections
from app.core.motion_tracker import MotionTracker
from app.core.prompt_utils import label_match
from app.core.reid_engine import ReidEngine
from app.core.seg_engine import SegItem


@dataclass(slots=True)
class PreparedFrame:
    valid_dets: list[DetectItem]
    segments: list[SegItem]
    cross_raw: list[DetectItem]


def prepare_batch_frames(
    *,
    detections: list[list[DetectItem]],
    segments: list[list[SegItem]],
    cross_detections: list[list[DetectItem]],
    frames_bgr: list[np.ndarray],
    terms: list[str],
    settings: PipelineSettings,
    motion_tracker: MotionTracker | None,
    use_motion_tracker: bool,
) -> tuple[list[PreparedFrame], list[np.ndarray], list[tuple[int, int]]]:
    """Filter/merge detections and collect ReID crops (shared GPU+CPU path)."""
    use_seg = bool(settings.use_seg)
    use_reid = bool(settings.use_reid)
    n = len(frames_bgr)
    out: list[PreparedFrame] = []
    all_crops: list[np.ndarray] = []
    crop_map: list[tuple[int, int]] = []

    for fi in range(n):
        dets_raw = detections[fi] if fi < len(detections) else []
        if use_motion_tracker and motion_tracker is not None:
            dets_raw = motion_tracker.assign(dets_raw)

        segs_raw = segments[fi] if fi < len(segments) else []
        cross_raw = cross_detections[fi] if fi < len(cross_detections) else []

        detections_f = [d for d in dets_raw if label_match(d.label, terms)]
        segments_f = [s for s in segs_raw if label_match(s.label, terms)]
        if use_seg:
            detections_f = merge_seg_fallback_detections(
                detections_f,
                segments_f,
                match_iou_min=settings.seg_fallback_iou_min,
            )
        if use_reid:
            detections_f = inherit_motion_ids(detections_f)

        valid_dets: list[DetectItem] = []
        if use_reid:
            frame = frames_bgr[fi]
            for det in detections_f:
                crop = ReidEngine.crop_from_bbox(frame, det.xyxy)
                if crop is not None and crop.size > 0:
                    crop_map.append((fi, len(valid_dets)))
                    all_crops.append(crop)
                    valid_dets.append(det)
        else:
            valid_dets = detections_f

        out.append(
            PreparedFrame(
                valid_dets=valid_dets,
                segments=segments_f,
                cross_raw=cross_raw,
            )
        )

    return out, all_crops, crop_map
