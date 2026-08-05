"""YOLO pose keypoints: skeleton, overlay, JSON."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# COCO person 17 keypoints (YOLO pose default).
COCO17_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO17_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def keypoint_names_for_count(k: int) -> list[str]:
    if k == len(COCO17_KEYPOINT_NAMES):
        return list(COCO17_KEYPOINT_NAMES)
    return [f"kpt_{i}" for i in range(k)]


def skeleton_for_count(k: int) -> list[tuple[int, int]]:
    if k == 17:
        return list(COCO17_SKELETON)
    return []


def keypoints_to_json(kpts: np.ndarray, kpt_conf_min: float = 0.0) -> list[dict[str, Any]]:
    """(K, 3) x,y,conf -> list for result JSON."""
    if kpts is None or not isinstance(kpts, np.ndarray) or kpts.ndim != 2 or kpts.shape[1] < 3:
        return []
    k = int(kpts.shape[0])
    names = keypoint_names_for_count(k)
    out: list[dict[str, Any]] = []
    for i in range(k):
        x, y, c = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
        if c < kpt_conf_min:
            continue
        out.append(
            {
                "id": i,
                "name": names[i],
                "x": round(x, 2),
                "y": round(y, 2),
                "conf": round(c, 4),
            }
        )
    return out


def draw_pose_on_frame(
    frame_bgr: np.ndarray,
    keypoints_list: list[np.ndarray | None],
    colors: list[tuple[int, int, int]],
    *,
    kpt_conf: float = 0.25,
    draw_skeleton: bool = True,
    point_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw all visible keypoints + skeleton per instance."""
    if not keypoints_list:
        return frame_bgr
    out = frame_bgr
    h, w = out.shape[:2]
    for idx, kpts in enumerate(keypoints_list):
        if kpts is None or not isinstance(kpts, np.ndarray) or kpts.ndim != 2 or kpts.shape[0] == 0:
            continue
        color = colors[idx % len(colors)] if colors else (0, 255, 0)
        k = int(kpts.shape[0])
        skeleton = skeleton_for_count(k) if draw_skeleton else []
        pts: list[tuple[int, int] | None] = []
        for i in range(k):
            x, y, c = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
            if c >= kpt_conf:
                xi = int(round(x))
                yi = int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    pts.append((xi, yi))
                    cv2.circle(out, (xi, yi), point_radius, color, -1, lineType=cv2.LINE_AA)
                    # Dark outline so skeleton stays readable on busy construction frames.
                    cv2.circle(out, (xi, yi), point_radius + 1, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                else:
                    pts.append(None)
            else:
                pts.append(None)
        for a, b in skeleton:
            if a >= len(pts) or b >= len(pts):
                continue
            pa, pb = pts[a], pts[b]
            if pa is not None and pb is not None:
                cv2.line(out, pa, pb, (0, 0, 0), line_thickness + 2, lineType=cv2.LINE_AA)
                cv2.line(out, pa, pb, color, line_thickness, lineType=cv2.LINE_AA)
    return out
