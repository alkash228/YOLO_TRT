"""Сжатое представление бинарной маски в JSON (RLE row-major, порядок C)."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    from numba import njit

    _NUMBA_RLE_AVAILABLE = True
except ImportError:
    _NUMBA_RLE_AVAILABLE = False
    njit = None  # type: ignore[misc, assignment]


def _env_mask_rle_numba_enabled() -> bool:
    if not _NUMBA_RLE_AVAILABLE:
        return False
    return os.getenv("YOLO_DRT_MASK_RLE_NUMBA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


if _NUMBA_RLE_AVAILABLE:

    @njit(cache=True)
    def _rle_flat_counts_numba(flat: np.ndarray) -> np.ndarray:
        n = flat.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int64)
        tmp = np.empty(n + 2, dtype=np.int64)
        nr = 0
        cur = flat[0]
        run = np.int64(1)
        for i in range(1, n):
            if flat[i] == cur:
                run += 1
            else:
                tmp[nr] = run
                nr += 1
                cur = flat[i]
                run = 1
        tmp[nr] = run
        nr += 1
        if flat[0] == 1:
            out = np.empty(nr + 1, dtype=np.int64)
            out[0] = 0
            out[1 : nr + 1] = tmp[:nr]
            return out
        return tmp[:nr].copy()


def _rle_flat_counts_python(flat: np.ndarray) -> list[int]:
    n = int(flat.size)
    counts: list[int] = []
    cur = int(flat[0])
    run = 1
    for i in range(1, n):
        v = int(flat[i])
        if v == cur:
            run += 1
        else:
            counts.append(run)
            cur = v
            run = 1
    counts.append(run)
    if int(flat[0]) == 1:
        counts.insert(0, 0)
    return counts


def binary_mask_to_rle_row_major(mask_bool: np.ndarray) -> dict[str, Any]:
    if mask_bool.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = int(mask_bool.shape[0]), int(mask_bool.shape[1])
    flat = mask_bool.reshape(-1).astype(np.uint8, copy=False)
    n = int(flat.size)
    if n == 0:
        return {"height": h, "width": w, "format": "rle_row_major", "order": "C", "counts": []}

    if _env_mask_rle_numba_enabled():
        arr = _rle_flat_counts_numba(np.ascontiguousarray(flat))
        counts = [int(x) for x in arr.tolist()]
    else:
        counts = _rle_flat_counts_python(flat)

    return {
        "height": h,
        "width": w,
        "format": "rle_row_major",
        "order": "C",
        "counts": counts,
    }


def mask_u8_to_rle_dict(mask_u8: np.ndarray, foreground_threshold: int = 127) -> dict[str, Any]:
    if mask_u8.ndim != 2:
        raise ValueError("mask must be 2D")
    return binary_mask_to_rle_row_major(mask_u8 > foreground_threshold)


def rle_row_major_to_bool(enc: dict[str, Any]) -> np.ndarray:
    h = int(enc["height"])
    w = int(enc["width"])
    counts = enc.get("counts")
    if not isinstance(counts, list):
        return np.zeros((h, w), dtype=bool)
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for ln in counts:
        n = int(ln)
        if n > 0:
            end = min(pos + n, flat.size)
            if end > pos:
                flat[pos:end] = val
                pos = end
        val = 1 - val
    return flat.reshape(h, w).astype(bool)


def rle_list_to_stack_u8(masks_rle: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """Восстановить stack (N,H,W) uint8 из RLE-словарей (0/255, как в inference)."""
    if not masks_rle:
        return np.zeros((0, height, width), dtype=np.uint8)
    planes = [(rle_row_major_to_bool(enc).astype(np.uint8) * 255) for enc in masks_rle]
    return np.stack(planes, axis=0)
