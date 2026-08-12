"""Cross-model intersection rules (person bbox ∩ helmet bbox).

Helmet detections are accessories only: used for OK / NO HELMET verdicts and
overlay drawing. They are never tracked, never embedded for ReID, and never
receive their own stable_id — identity belongs to the person DetectItem.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.detect_engine import DetectItem
from app.core.fusion import box_iou

# Deprecated: previously used for pose-head ∩ helmet. Kept for import compat.
HEAD_KEYPOINT_IDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(slots=True)
class CrossCheckVerdict:
    ok: bool
    warning: str
    # Highlight region for overlay/JSON (legacy name head_xyxy):
    # OK → person∩best-helmet; NO HELMET → upper portion of person bbox.
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


def box_intersection_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    ax0, ay0, ax1, ay1 = [float(v) for v in a.tolist()]
    bx0, by0, bx1, by1 = [float(v) for v in b.tolist()]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return np.array([ix0, iy0, ix1, iy1], dtype=np.float32)


def person_upper_bbox(
    person_xyxy: np.ndarray,
    *,
    frame_w: int = 0,
    frame_h: int = 0,
    top_frac: float = 0.35,
    side_inset: float = 0.15,
) -> np.ndarray:
    """Upper person band used as overlay/warn anchor when no helmet match."""
    px0, py0, px1, py1 = [float(v) for v in person_xyxy.tolist()]
    ph = max(1.0, py1 - py0)
    pw = max(1.0, px1 - px0)
    x0 = px0 + pw * side_inset
    x1 = px1 - pw * side_inset
    y0 = py0
    y1 = py0 + ph * top_frac
    if frame_w > 0:
        x0 = max(0.0, min(x0, float(frame_w - 1)))
        x1 = max(0.0, min(x1, float(frame_w - 1)))
    if frame_h > 0:
        y0 = max(0.0, min(y0, float(frame_h - 1)))
        y1 = max(0.0, min(y1, float(frame_h - 1)))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def head_bbox_from_keypoints(
    kpts: np.ndarray | None,
    person_xyxy: np.ndarray,
    *,
    kpt_conf: float = 0.25,
    pad: float = 0.35,
    frame_w: int = 0,
    frame_h: int = 0,
) -> np.ndarray:
    """Deprecated: pose-head region. Prefer person_upper_bbox / person∩helmet."""
    del kpts, kpt_conf, pad
    return person_upper_bbox(person_xyxy, frame_w=frame_w, frame_h=frame_h)


def evaluate_person_helmet_cross(
    person: DetectItem,
    accessories: list[DetectItem],
    *,
    min_intersection_px: float = 1.0,
    min_iou: float = 0.0,
    frame_w: int = 0,
    frame_h: int = 0,
    ok_text: str = "OK",
    warn_text: str = "NO HELMET",
) -> CrossCheckVerdict:
    """OK only if a helmet bbox intersects this person (area and/or IoU).

    Missing helmet association is always NO HELMET (ok=False):
    - empty accessories (model found nothing, or all filtered by conf/prompt)
    - helmets elsewhere in the frame but none overlapping this person

    There is no second "no_helmet" classifier — absence of a qualifying
    helmet detection is itself the violation.
    """
    del ok_text
    person_box = person.xyxy
    best_ia = 0.0
    best_iou = 0.0
    best_acc: DetectItem | None = None
    miss_ia = 0.0
    miss_iou = 0.0

    for acc in accessories:
        ia = box_intersection_area(person_box, acc.xyxy)
        iou = float(box_iou(person_box, acc.xyxy))
        qualifies = ia >= float(min_intersection_px)
        if float(min_iou) > 0:
            qualifies = qualifies or iou >= float(min_iou)
        if qualifies:
            if best_acc is None or ia > best_ia or (ia == best_ia and iou > best_iou):
                best_ia = ia
                best_iou = iou
                best_acc = acc
        elif ia > miss_ia or (ia == miss_ia and iou > miss_iou):
            miss_ia = ia
            miss_iou = iou

    if best_acc is not None:
        highlight = box_intersection_xyxy(person_box, best_acc.xyxy)
        if highlight is None:
            highlight = np.asarray(best_acc.xyxy, dtype=np.float32).copy()
        return CrossCheckVerdict(
            ok=True,
            warning="",
            head_xyxy=highlight,
            best_intersection_px=best_ia,
            best_iou=best_iou,
        )

    # No qualifying helmet on/near this person → NO HELMET (feeds violation streak).
    return CrossCheckVerdict(
        ok=False,
        warning=warn_text,
        head_xyxy=person_upper_bbox(person_box, frame_w=frame_w, frame_h=frame_h),
        best_intersection_px=miss_ia,
        best_iou=miss_iou,
    )


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
    """Deprecated alias → person bbox ∩ helmet (keypoints ignored)."""
    del kpt_conf
    return evaluate_person_helmet_cross(
        person,
        accessories,
        min_intersection_px=min_intersection_px,
        min_iou=min_iou,
        frame_w=frame_w,
        frame_h=frame_h,
        ok_text=ok_text,
        warn_text=warn_text,
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
    """Per-person helmet verdicts; empty/filtered accessories → all NO HELMET.

    helmet_min_conf: ignore accessory detections below this score (raises
    "no det" rate — those persons still get ok=False / NO HELMET).

    kpt_conf is ignored (API compat); rule is person bbox ∩ helmet bbox.
    """
    del kpt_conf
    filtered_acc = (
        [a for a in accessories if float(a.conf) >= float(helmet_min_conf)]
        if helmet_min_conf > 0
        else list(accessories)
    )
    return [
        evaluate_person_helmet_cross(
            p,
            filtered_acc,
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
    raw_ok=False (missing/non-overlapping helmet) always appends to the streak.
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
    """Draw highlight box (person∩helmet or person-top) + NO HELMET warning."""
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
