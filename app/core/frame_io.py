"""Torch-free frame I/O helpers (WEB encode / Word report)."""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path

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


def _ffprobe_exe() -> str | None:
    try:
        from app.core.ffmpeg_utils import resolve_ffmpeg_exe

        ff = Path(resolve_ffmpeg_exe())
    except Exception:
        found = shutil.which("ffprobe")
        return found
    names = ("ffprobe.exe", "ffprobe") if ff.suffix.lower() == ".exe" else ("ffprobe", "ffprobe.exe")
    for name in names:
        cand = ff.with_name(name)
        if cand.is_file():
            return str(cand)
    return shutil.which("ffprobe")


@lru_cache(maxsize=64)
def probe_video_fps(input_path: str) -> float:
    """Best-effort fps for timestamp seeks. Never raises."""
    path = str(input_path)
    cap = cv2.VideoCapture(path)
    try:
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if 1.0 <= fps <= 240.0:
                return fps
    finally:
        cap.release()
    probe = _ffprobe_exe()
    if not probe:
        return 25.0
    try:
        proc = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 25.0
    for line in (proc.stdout or "").splitlines():
        text = line.strip()
        if not text or text in {"0/0", "N/A"}:
            continue
        if "/" in text:
            num, den = text.split("/", 1)
            try:
                fps = float(num) / float(den)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        else:
            try:
                fps = float(text)
            except ValueError:
                continue
        if 1.0 <= fps <= 240.0:
            return fps
    return 25.0


def _decode_mjpeg_pipe(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame if frame is not None and frame.size > 0 else None


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


def _ffmpeg_still_cmd(
    extra: list[str],
    *,
    timeout_sec: float,
) -> np.ndarray | None:
    try:
        from app.core.ffmpeg_utils import _popen_kwargs, resolve_ffmpeg_exe

        exe = resolve_ffmpeg_exe()
        kwargs = _popen_kwargs()
    except Exception:
        return None
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        *extra,
        "-frames:v",
        "1",
        "-an",
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
            timeout=max(15.0, float(timeout_sec)),
            check=False,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return _decode_mjpeg_pipe(proc.stdout or b"")


def read_frame_bgr_ffmpeg(
    input_path: str,
    frame_idx: int,
    *,
    max_skip: int = 500_000,
    fps: float | None = None,
) -> np.ndarray | None:
    """
    One still via ffmpeg timestamp seek (GOP-local decode).

    Does not walk the file from frame 0 — that is what made Word reports
    crawl on long HEVC. Overlay may be ~1 GOP off if timestamps drift;
    reports only need a recognizable NO-HELMET still.
    """
    target = max(0, int(frame_idx))
    if target > int(max_skip):
        return None
    path = str(input_path)
    rate = float(fps) if fps and fps > 1.0 else probe_video_fps(path)
    if rate < 1.0:
        rate = 25.0
    t = target / rate
    # Input -ss jumps near a keyframe; output -ss finishes the last few seconds.
    pad = 4.0
    input_ss = max(0.0, t - pad)
    output_ss = max(0.0, t - input_ss)
    hit = _ffmpeg_still_cmd(
        [
            "-ss",
            f"{input_ss:.3f}",
            "-i",
            path,
            "-ss",
            f"{output_ss:.3f}",
        ],
        timeout_sec=45.0,
    )
    if hit is not None:
        return hit
    # Keyframe-only (faster, slightly less exact).
    return _ffmpeg_still_cmd(
        ["-ss", f"{t:.3f}", "-i", path],
        timeout_sec=30.0,
    )


def read_frame_bgr_smart(
    input_path: str,
    frame_idx: int,
    *,
    max_skip: int = 500_000,
    fps: float | None = None,
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
        return read_frame_bgr_ffmpeg(path, target, max_skip=max_skip, fps=fps)
    hevc = is_hevc_video(path, cap)
    try:
        if hevc or target == 0:
            cap.release()
            cap = None
            if hevc and target > 0:
                hit = read_frame_bgr_ffmpeg(path, target, max_skip=max_skip, fps=fps)
                if hit is not None:
                    return hit
                # Sequential from t=0 on a 4h HEVC file is worse than a missing still.
                if target > 150:
                    return None
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

    hit = read_frame_bgr_ffmpeg(path, target, max_skip=max_skip, fps=fps)
    if hit is not None:
        return hit
    if target > 150:
        return None
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
