"""Torch-free frame I/O helpers (WEB encode / Word report)."""
from __future__ import annotations

from collections.abc import Callable, Iterator

import cv2
import numpy as np


def read_frame_bgr_sequential(
    input_path: str,
    frame_idx: int,
    *,
    max_skip: int = 500_000,
) -> np.ndarray | None:
    """Read one frame by sequential decode (HEVC-safe; no POS_FRAMES seek)."""
    target = max(0, int(frame_idx))
    if target > int(max_skip):
        return None
    # Prefer grab-skip (decode without BGR convert) until the target frame.
    try:
        for idx, frame in iter_selected_bgr_frames(str(input_path), [target]):
            if idx == target:
                return frame
    except RuntimeError:
        return None
    return None


def read_frame_bgr_smart(
    input_path: str,
    frame_idx: int,
    *,
    max_skip: int = 500_000,
) -> np.ndarray | None:
    """
    Fast path: CAP_PROP_POS_FRAMES seek when the backend lands near the target.
    Fallback: HEVC-safe grab-skip sequential decode.
    """
    target = max(0, int(frame_idx))
    if target > int(max_skip):
        return None
    path = str(input_path)
    if target == 0:
        return read_frame_bgr_sequential(path, 0, max_skip=max_skip)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or -1)
            # After reading frame N, POS is often N+1.
            if pos < 0 or abs(pos - (target + 1)) <= 2 or abs(pos - target) <= 2:
                return frame
    finally:
        cap.release()
    return read_frame_bgr_sequential(path, target, max_skip=max_skip)

def iter_selected_bgr_frames(
    input_path: str,
    indices: list[int] | set[int],
    *,
    on_scan: Callable[[int, int], None] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_idx, bgr) for selected indices only.

    Uses grab() to skip unused frames (cheaper than full read) and avoids
    CAP_PROP_POS_FRAMES so HEVC streams stay valid. Still must walk the file
    sequentially up to the last needed index.
    """
    needed = sorted({max(0, int(i)) for i in indices})
    if not needed:
        return
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    last = needed[-1]
    try:
        src_i = 0
        for target in needed:
            while src_i < target:
                if not cap.grab():
                    return
                src_i += 1
                if on_scan is not None and src_i % 300 == 0:
                    on_scan(src_i, last)
            if not cap.grab():
                return
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                return
            yield target, frame
            src_i = target + 1
            if on_scan is not None:
                on_scan(src_i, last)
    finally:
        cap.release()
