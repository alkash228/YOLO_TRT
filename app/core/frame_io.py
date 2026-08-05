"""Torch-free frame I/O helpers (WEB encode / Word report)."""
from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator

import cv2
import numpy as np

_HEVC_FOURCC = {
    "hevc",
    "hvc1",
    "hev1",
    "x265",
    "hev ",
    "h265",
}


def _fourcc_to_str(fourcc: int) -> str:
    if fourcc <= 0:
        return ""
    chars: list[str] = []
    for i in range(4):
        c = (int(fourcc) >> (8 * i)) & 0xFF
        if 32 <= c < 127:
            chars.append(chr(c))
        else:
            chars.append("?")
    return "".join(chars)


def is_hevc_video(input_path: str, cap: cv2.VideoCapture | None = None) -> bool:
    """True when container/codec looks like H.265 — OpenCV POS_FRAMES seek corrupts it."""
    own = False
    if cap is None:
        cap = cv2.VideoCapture(str(input_path))
        own = True
    try:
        if not cap.isOpened():
            return "hevc" in str(input_path).casefold() or str(input_path).lower().endswith(
                (".h265", ".hevc")
            )
        code = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC) or 0)).casefold().strip()
        if code in _HEVC_FOURCC or "hevc" in code or "h265" in code or "hvc" in code:
            return True
        # Some builds report 0 fourcc; sniff with ffprobe when available.
    finally:
        if own:
            cap.release()
    try:
        from app.core.ffmpeg_utils import resolve_ffmpeg_exe

        exe = resolve_ffmpeg_exe().replace("ffmpeg", "ffprobe")
        # resolve_ffmpeg_exe returns ffmpeg path — derive ffprobe sibling
        from pathlib import Path

        ff = Path(resolve_ffmpeg_exe())
        probe = ff.with_name("ffprobe.exe" if ff.suffix.lower() == ".exe" else "ffprobe")
        if not probe.is_file():
            probe = ff.with_name("ffprobe")
        if probe.is_file():
            proc = subprocess.run(
                [
                    str(probe),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "csv=p=0",
                    str(input_path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            name = (proc.stdout or "").strip().casefold()
            if "hevc" in name or "h265" in name:
                return True
    except Exception:
        pass
    return False


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
    try:
        for idx, frame in iter_selected_bgr_frames(str(input_path), [target]):
            if idx == target:
                return frame
    except RuntimeError:
        return None
    return None


def read_frame_bgr_ffmpeg(
    input_path: str,
    frame_idx: int,
    *,
    max_skip: int = 500_000,
) -> np.ndarray | None:
    """
    Exact frame via ffmpeg select filter (no OpenCV seek).
    Decodes from the start up to frame_idx — correct for HEVC, no POC spam.
    """
    target = max(0, int(frame_idx))
    if target > int(max_skip):
        return None
    try:
        from app.core.ffmpeg_utils import resolve_ffmpeg_exe

        exe = resolve_ffmpeg_exe()
    except Exception:
        return None
    # select=eq(n\,N) is 0-based; vsync vfr emits only selected frames.
    vf = f"select=eq(n\\,{target})"
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=max(60, target // 100 + 30),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame if frame is not None and frame.size > 0 else None


def read_frame_bgr_smart(
    input_path: str,
    frame_idx: int,
    *,
    max_skip: int = 500_000,
) -> np.ndarray | None:
    """
    HEVC: never OpenCV-seek (causes POC errors + corrupt frames).
    H.264: optional POS_FRAMES seek; fallback sequential / ffmpeg.
    """
    target = max(0, int(frame_idx))
    if target > int(max_skip):
        return None
    path = str(input_path)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return read_frame_bgr_ffmpeg(path, target, max_skip=max_skip)
    hevc = is_hevc_video(path, cap)
    try:
        if hevc or target == 0:
            cap.release()
            cap = None
            # Prefer ffmpeg for mid/late frames on HEVC (cleaner than OpenCV grab storm).
            if hevc and target > 0:
                hit = read_frame_bgr_ffmpeg(path, target, max_skip=max_skip)
                if hit is not None:
                    return hit
            return read_frame_bgr_sequential(path, target, max_skip=max_skip)

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or -1)
            if abs(pos - (target + 1)) <= 2 or abs(pos - target) <= 2:
                return frame
    finally:
        if cap is not None:
            cap.release()

    hit = read_frame_bgr_ffmpeg(path, target, max_skip=max_skip)
    if hit is not None:
        return hit
    return read_frame_bgr_sequential(path, target, max_skip=max_skip)


def iter_selected_bgr_frames(
    input_path: str,
    indices: list[int] | set[int],
    *,
    on_scan: Callable[[int, int], None] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_idx, bgr) for selected indices only.

    Uses grab() to skip unused frames and never CAP_PROP_POS_FRAMES
    (HEVC-safe). Walks the file sequentially up to the last needed index.
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
