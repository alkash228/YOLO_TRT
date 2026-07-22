"""Match detect boxes to seg masks via IoU."""
from __future__ import annotations

import numpy as np

from app.core.detect_engine import DetectItem
from app.core.seg_engine import SegItem


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a.tolist()]
    bx0, by0, bx1, by1 = [float(v) for v in b.tolist()]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_rect_mask(xyxy: np.ndarray, h: int, w: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in xyxy.tolist()]
    xa = int(np.clip(np.floor(x0), 0, max(0, w - 1)))
    ya = int(np.clip(np.floor(y0), 0, max(0, h - 1)))
    xb = int(np.clip(np.ceil(x1), 0, max(0, w - 1)))
    yb = int(np.clip(np.ceil(y1), 0, max(0, h - 1)))
    out = np.zeros((h, w), dtype=np.uint8)
    if xb > xa and yb > ya:
        out[ya : yb + 1, xa : xb + 1] = 255
    return out


def dedupe_detections(
    detections: list[DetectItem],
    *,
    iou_min: float = 0.55,
) -> list[DetectItem]:
    """Drop overlapping same-label boxes (seg-fallback dupes confuse identity)."""
    if len(detections) <= 1:
        return list(detections)
    order = sorted(range(len(detections)), key=lambda i: detections[i].conf, reverse=True)
    keep: list[int] = []
    for i in order:
        det = detections[i]
        duplicate = False
        for j in keep:
            ref = detections[j]
            if det.label.casefold() != ref.label.casefold():
                continue
            if box_iou(det.xyxy, ref.xyxy) >= iou_min:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    keep.sort()
    return [detections[i] for i in keep]


def merge_seg_fallback_detections(
    detections: list[DetectItem],
    segments: list[SegItem],
    match_iou_min: float = 0.25,
) -> list[DetectItem]:
    """Add seg boxes that detect missed (common with partial/occluded people)."""
    if not segments:
        return list(detections)
    merged = list(detections)
    for seg in segments:
        covered = False
        for det in detections:
            if det.label.casefold() != seg.label.casefold():
                continue
            if box_iou(det.xyxy, seg.xyxy) >= match_iou_min:
                covered = True
                break
        if not covered:
            merged.append(
                DetectItem(
                    xyxy=seg.xyxy.copy(),
                    cls_id=seg.cls_id,
                    label=seg.label,
                    conf=seg.conf,
                    motion_id=-1,
                )
            )
    return merged


def inherit_motion_ids(
    detections: list[DetectItem],
    *,
    iou_min: float = 0.25,
) -> list[DetectItem]:
    """Copy motion_id from overlapping tracked boxes (e.g. seg-fallback dets)."""
    refs = [d for d in detections if d.motion_id >= 0]
    if not refs:
        return detections
    out: list[DetectItem] = []
    for det in detections:
        if det.motion_id >= 0:
            out.append(det)
            continue
        best_iou = 0.0
        best_mid = -1
        for ref in refs:
            iou = box_iou(det.xyxy, ref.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_mid = ref.motion_id
        if best_mid >= 0 and best_iou >= iou_min:
            out.append(
                DetectItem(
                    xyxy=det.xyxy.copy(),
                    cls_id=det.cls_id,
                    label=det.label,
                    conf=det.conf,
                    motion_id=int(best_mid),
                    keypoints=det.keypoints.copy() if det.keypoints is not None else None,
                )
            )
        else:
            out.append(det)
    return out


def match_detections_to_segments(
    detections: list[DetectItem],
    segments: list[SegItem],
    match_iou_min: float = 0.5,
    frame_h: int = 0,
    frame_w: int = 0,
) -> list[np.ndarray]:
    """For each detection return matched seg mask or bbox fallback."""
    if not detections:
        return []
    n_det = len(detections)
    n_seg = len(segments)
    masks: list[np.ndarray] = []
    used_seg: set[int] = set()

    pairs: list[tuple[float, int, int]] = []
    for di, det in enumerate(detections):
        for si, seg in enumerate(segments):
            if det.label.casefold() != seg.label.casefold():
                continue
            iou = box_iou(det.xyxy, seg.xyxy)
            if iou >= match_iou_min:
                pairs.append((iou, di, si))
    pairs.sort(key=lambda x: x[0], reverse=True)

    det_to_seg: dict[int, int] = {}
    for _iou, di, si in pairs:
        if di in det_to_seg or si in used_seg:
            continue
        det_to_seg[di] = si
        used_seg.add(si)

    for di, det in enumerate(detections):
        si = det_to_seg.get(di)
        if si is not None:
            masks.append(segments[si].mask_u8)
        else:
            masks.append(bbox_rect_mask(det.xyxy, frame_h, frame_w))
    return masks
