"""Cross-model intersection rules (e.g. pose head ∩ helmet)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.detect_engine import DetectItem
from app.core.fusion import box_iou

# COCO head keypoints: nose, eyes, ears.
HEAD_KEYPOINT_IDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(slots=True)
class CrossCheckVerdict:
    ok: bool
    warning: str
    head_xyxy: np.ndarray | None
    best_intersection_px: float
    best_iou: float


def box_intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a.tolist()]
    bx0, by0, bx1, by1 = [float(v) for v in b.tolist()]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    return float(iw * ih)


def head_bbox_from_keypoints(
    kpts: np.ndarray | None,
    person_xyxy: np.ndarray,
    *,
    kpt_conf: float = 0.25,
    pad: float = 0.35,
    frame_w: int = 0,
    frame_h: int = 0,
) -> np.ndarray:
    """Head region from pose keypoints; fallback = upper part of person box."""
    xs: list[float] = []
    ys: list[float] = []
    if kpts is not None and isinstance(kpts, np.ndarray) and kpts.ndim == 2 and kpts.shape[1] >= 3:
        for idx in HEAD_KEYPOINT_IDS:
            if idx >= int(kpts.shape[0]):
                continue
            x, y, c = float(kpts[idx, 0]), float(kpts[idx, 1]), float(kpts[idx, 2])
            if c >= kpt_conf:
                xs.append(x)
                ys.append(y)
    if xs and ys:
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bw = max(8.0, x1 - x0)
        bh = max(8.0, y1 - y0)
        x0 -= pad * bw
        x1 += pad * bw
        y0 -= pad * bh
        y1 += pad * bh
    else:
        px0, py0, px1, py1 = [float(v) for v in person_xyxy.tolist()]
        ph = max(1.0, py1 - py0)
        pw = max(1.0, px1 - px0)
        x0 = px0 + pw * 0.15
        x1 = px1 - pw * 0.15
        y0 = py0
        y1 = py0 + ph * 0.35

    if frame_w > 0:
        x0 = max(0.0, min(x0, float(frame_w - 1)))
        x1 = max(0.0, min(x1, float(frame_w - 1)))
    if frame_h > 0:
        y0 = max(0.0, min(y0, float(frame_h - 1)))
        y1 = max(0.0, min(y1, float(frame_h - 1)))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def evaluate_head_helmet_cross(
    person: DetectItem,
    accessories: list[DetectItem],
    *,
    kpt_conf: float = 0.25,
    min_intersection_px: float = 1.0,
    min_iou: float = 0.0,
    frame_w: int = 0,
    frame_h: int = 0,
    ok_text: str = "OK",
    warn_text: str = "NO HELMET",
) -> CrossCheckVerdict:
    head = head_bbox_from_keypoints(
        person.keypoints,
        person.xyxy,
        kpt_conf=kpt_conf,
        frame_w=frame_w,
        frame_h=frame_h,
    )
    best_ia = 0.0
    best_iou = 0.0
    for acc in accessories:
        ia = box_intersection_area(head, acc.xyxy)
        iou = box_iou(head, acc.xyxy)
        best_ia = max(best_ia, ia)
        best_iou = max(best_iou, iou)

    ok = best_ia >= min_intersection_px
    if min_iou > 0:
        ok = ok or best_iou >= min_iou
    return CrossCheckVerdict(
        ok=ok,
        warning="" if ok else warn_text,
        head_xyxy=head,
        best_intersection_px=best_ia,
        best_iou=best_iou,
    )


def evaluate_cross_check_batch(
    persons: list[DetectItem],
    accessories: list[DetectItem],
    *,
    kpt_conf: float = 0.25,
    min_intersection_px: float = 1.0,
    min_iou: float = 0.0,
    frame_w: int = 0,
    frame_h: int = 0,
    warn_text: str = "NO HELMET",
    helmet_min_conf: float = 0.0,
) -> list[CrossCheckVerdict]:
    """helmet_min_conf: ignore accessory detections below this score."""
    filtered_acc = (
        [a for a in accessories if float(a.conf) >= float(helmet_min_conf)]
        if helmet_min_conf > 0
        else accessories
    )
    return [
        evaluate_head_helmet_cross(
            p,
            filtered_acc,
            kpt_conf=kpt_conf,
            min_intersection_px=min_intersection_px,
            min_iou=min_iou,
            frame_w=frame_w,
            frame_h=frame_h,
            warn_text=warn_text,
        )
        for p in persons
    ]


def smooth_helmet_verdicts(
    stable_ids: np.ndarray,
    verdicts: list[CrossCheckVerdict],
    history: dict[int, list[bool]],
    *,
    min_violation_streak: int = 2,
    history_len: int = 5,
) -> list[CrossCheckVerdict]:
    """
    Temporal filter per stable_id: need consecutive NO HELMET frames before violation.
    Helmet detected → OK immediately (no flicker «был в каске → без» на одном кадре).
    """
    from dataclasses import replace

    min_streak = max(1, int(min_violation_streak))
    max_hist = max(min_streak + 1, int(history_len))
    out: list[CrossCheckVerdict] = []
    for i, verdict in enumerate(verdicts):
        raw_ok = bool(verdict.ok)
        sid = int(stable_ids[i]) if i < len(stable_ids) else i
        h = history.setdefault(sid, [])
        h.append(raw_ok)
        if len(h) > max_hist:
            del h[: len(h) - max_hist]
        if raw_ok:
            smoothed_ok = True
        else:
            streak = 0
            for ok in reversed(h):
                if ok:
                    break
                streak += 1
            smoothed_ok = streak < min_streak
        if smoothed_ok == raw_ok:
            out.append(verdict)
        else:
            out.append(
                replace(
                    verdict,
                    ok=smoothed_ok,
                    warning="" if smoothed_ok else verdict.warning,
                )
            )
    return out


@dataclass(slots=True)
class CrossCheckDetection:
    label: str
    conf: float
    xyxy: np.ndarray


def draw_cross_check_detections(
    frame_bgr: np.ndarray,
    detections: list[CrossCheckDetection],
    *,
    color: tuple[int, int, int] = (0, 200, 255),
) -> np.ndarray:
    """Draw bounding boxes from the cross-check (secondary) model."""
    out = frame_bgr
    for det in detections:
        x0, y0, x1, y1 = [int(v) for v in det.xyxy.tolist()]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
        text = f"{det.label} {det.conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
        ty = max(th + 4, y0 - 4)
        cv2.rectangle(out, (x0, ty - th - 4), (x0 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, text, (x0 + 3, ty), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def draw_cross_check_overlay(
    frame_bgr: np.ndarray,
    verdicts: list[CrossCheckVerdict],
    *,
    draw_head_box: bool = True,
) -> np.ndarray:
    out = frame_bgr
    for verdict in verdicts:
        if verdict.head_xyxy is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in verdict.head_xyxy.tolist()]
        if draw_head_box:
            color = (0, 200, 80) if verdict.ok else (0, 0, 255)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
        if not verdict.ok and verdict.warning:
            tx = max(0, x0)
            ty = max(24, y0 - 8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = verdict.warning
            (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
            cv2.rectangle(out, (tx, ty - th - 6), (tx + tw + 8, ty + 4), (0, 0, 200), -1)
            cv2.putText(out, text, (tx + 4, ty), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out
