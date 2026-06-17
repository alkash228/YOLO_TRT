"""Export and compositing helpers."""
from __future__ import annotations

import cv2
import numpy as np


def _distinct_bgr_colors(n: int) -> list[tuple[int, int, int]]:
    if n <= 0:
        return []
    colors: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = int(((i * 37) % 180))
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def _draw_label_bgr(
    image_bgr: np.ndarray,
    text: str,
    x: int,
    y_above: int,
    color_bgr: tuple[int, int, int],
    font_scale: float = 0.52,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), _baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = image_bgr.shape[:2]
    y_text = int(np.clip(y_above, th + 2, h - 2))
    x_text = int(np.clip(x, 0, max(0, w - tw - 2)))
    cv2.putText(image_bgr, text, (x_text, y_text), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image_bgr, text, (x_text, y_text), font, font_scale, color_bgr, thickness, cv2.LINE_AA)


def pose_overlay_colors(n: int) -> list[tuple[int, int, int]]:
    return _distinct_bgr_colors(n)


def overlay_instance_masks_on_frame(
    frame: np.ndarray,
    stack: np.ndarray,
    alpha: float = 0.45,
    object_ids: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    prompt_labels: list[str] | None = None,
    draw_boxes: bool = False,
    draw_masks: bool = False,
    draw_centers: bool = False,
) -> np.ndarray:
    """Renderer for instance stack: optional mask tint + boxes/labels."""
    if stack is None or stack.size == 0 or stack.ndim != 3 or stack.shape[0] == 0:
        return frame.copy()
    n = int(stack.shape[0])
    colors = _distinct_bgr_colors(n)
    out_u8 = frame.copy()
    for i in range(n):
        m = (stack[i] > 0).astype(np.uint8)
        if not m.any():
            continue
        if draw_masks:
            tint = np.zeros_like(out_u8, dtype=np.float32)
            tint[:, :, 0] = float(colors[i][0])
            tint[:, :, 1] = float(colors[i][1])
            tint[:, :, 2] = float(colors[i][2])
            mask3 = m.astype(bool)[:, :, None]
            base = out_u8.astype(np.float32)
            mixed = base * (1.0 - float(alpha)) + tint * float(alpha)
            out_u8 = np.where(mask3, mixed, base).astype(np.uint8)
        x, y, mw, mh = cv2.boundingRect(m)
        if draw_boxes:
            cv2.rectangle(out_u8, (x, y), (x + mw, y + mh), colors[i], 2)
        oid = int(object_ids[i]) if object_ids is not None and i < len(object_ids) else i + 1
        if scores is not None and i < len(scores):
            label = f"ID {oid}  p={float(scores[i]):.2f}"
        else:
            label = f"ID {oid}"
        if prompt_labels and i < len(prompt_labels) and prompt_labels[i]:
            frag = prompt_labels[i]
            if len(frag) > 22:
                frag = frag[:21] + "\u2026"
            label = f"{label}  {frag}"
        if draw_boxes:
            _draw_label_bgr(out_u8, label, x, y - 6, colors[i], font_scale=0.5)
        if draw_centers and m.any():
            cx = int(x + mw // 2)
            cy = int(y + mh // 2)
            cv2.circle(out_u8, (cx, cy), 3, colors[i], -1)
    return out_u8
