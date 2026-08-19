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


@lru_cache(maxsize=64)
def probe_video_codec(input_path: str) -> str:
    """Video codec_name (lowercase). ffprobe, else ffmpeg -i stderr. No OpenCV."""
    probe = _ffprobe_exe()
    if probe:
        try:
            proc = subprocess.run(
                [
                    probe,
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
            if name:
                return name
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        from app.core.ffmpeg_utils import resolve_ffmpeg_exe

        exe = resolve_ffmpeg_exe()
        proc = subprocess.run(
            [exe, "-hide_banner", "-i", str(input_path)],
            capture_output=True,
            timeout=20,
            check=False,
        )
        text = ((proc.stderr or b"") + (proc.stdout or b"")).decode("utf-8", errors="replace").casefold()
        if "video: hevc" in text or "video: h265" in text:
            return "hevc"
        if "video: h264" in text or "video: avc" in text:
            return "h264"
        for token in ("hevc", "h265", "h264", "av1", "mpeg4"):
            if f"video: {token}" in text:
                return token
    except Exception:
        pass
    return ""


def is_hevc_video(input_path: str, cap: cv2.VideoCapture | None = None) -> bool:
    """True when container/codec looks like H.265 — OpenCV POS_FRAMES seek corrupts it."""
    path = str(input_path)
    low = path.casefold()
    if low.endswith((".h265", ".hevc")) or "hevc" in Path(path).name.casefold():
        return True
    codec = probe_video_codec(path)
    if codec:
        return "hevc" in codec or "h265" in codec
    if cap is not None and cap.isOpened():
        code = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC) or 0)).casefold().strip()
        if code in _HEVC_FOURCC or "hevc" in code or "h265" in code or "hvc" in code:
            return True
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
    """Best-effort fps for timestamp seeks. Never opens OpenCV (slow on long HEVC)."""
    path = str(input_path)
    probe = _ffprobe_exe()
    if probe:
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
            proc = None
        if proc is not None:
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


@lru_cache(maxsize=64)
def probe_video_wh(input_path: str) -> tuple[int, int]:
    """Display width/height. Never opens OpenCV."""
    path = str(input_path)
    probe = _ffprobe_exe()
    if not probe:
        return (0, 0)
    try:
        proc = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return (0, 0)
    text = (proc.stdout or "").strip().replace(";", ",")
    if not text:
        return (0, 0)
    parts = text.split(",")[0].split("x") if "x" in text.split(",")[0] else text.split(",")
    if len(parts) < 2:
        return (0, 0)
    try:
        w, h = int(float(parts[0])), int(float(parts[1]))
    except (TypeError, ValueError):
        return (0, 0)
    if w <= 0 or h <= 0:
        return (0, 0)
    return (w, h)


def _decode_image_pipe(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame if frame is not None and frame.size > 0 else None


def _ffmpeg_raw_still(
    extra: list[str],
    *,
    width: int,
    height: int,
    timeout_sec: float,
) -> np.ndarray | None:
    """One still as raw BGR24 — no JPEG/PNG re-encode (that smeared Word photos)."""
    tw, th = int(width), int(height)
    if tw <= 0 or th <= 0:
        return None
    frame_bytes = tw * th * 3
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
        "-sws_flags",
        "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp",
        "-vf",
        "format=bgr24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
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
    raw = proc.stdout or b""
    if len(raw) < frame_bytes:
        return None
    return np.frombuffer(raw[:frame_bytes], dtype=np.uint8).reshape(th, tw, 3).copy()


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
    vcodec: str = "mjpeg",
) -> np.ndarray | None:
    try:
        from app.core.ffmpeg_utils import _popen_kwargs, resolve_ffmpeg_exe

        exe = resolve_ffmpeg_exe()
        kwargs = _popen_kwargs()
    except Exception:
        return None
    codec = str(vcodec or "mjpeg").strip().lower() or "mjpeg"
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
        codec,
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
    return _decode_image_pipe(proc.stdout or b"")


def read_frame_bgr_ffmpeg(
    input_path: str,
    frame_idx: int,
    *,
    max_skip: int = 500_000,
    fps: float | None = None,
    accurate: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray | None:
    """
    One still via ffmpeg timestamp seek (GOP-local decode).

    accurate=True (Word): GOP-local dual -ss + raw BGR24. image2pipe mjpeg/png
    was smearing the photo while Python boxes stayed sharp.
    """
    target = max(0, int(frame_idx))
    if target > int(max_skip):
        return None
    path = str(input_path)
    rate = float(fps) if fps and fps > 1.0 else probe_video_fps(path)
    if rate < 1.0:
        rate = 25.0
    t = target / rate
    if not accurate:
        return _ffmpeg_still_cmd(
            ["-ss", f"{t:.3f}", "-i", path],
            timeout_sec=20.0,
        )
    tw, th = probe_video_wh(path)
    if tw <= 0 or th <= 0:
        tw = int(width or 0)
        th = int(height or 0)
    if tw > 0 and th > 0:
        # Input -ss ≈ keyframe, output -ss reconstructs the GOP. Raw BGR, no JPEG.
        pad = 4.0
        in_ss = max(0.0, t - pad)
        out_ss = t - in_ss
        extra = ["-ss", f"{in_ss:.3f}", "-i", path]
        if out_ss > 0.02:
            extra.extend(["-ss", f"{out_ss:.3f}"])
        hit = _ffmpeg_raw_still(extra, width=tw, height=th, timeout_sec=40.0)
        if hit is not None:
            return hit
    extra = ["-i", path, "-ss", f"{t:.3f}"]
    hit = _ffmpeg_still_cmd(extra, timeout_sec=40.0, vcodec="png")
    if hit is not None:
        return hit
    return _ffmpeg_still_cmd(extra, timeout_sec=40.0, vcodec="mjpeg")


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
    if is_hevc_video(path):
        return read_frame_bgr_ffmpeg(path, target, max_skip=max_skip, fps=fps)

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


def iter_seek_selected_bgr_frames(
    input_path: str,
    indices: list[int] | set[int],
    *,
    fps: float,
    width: int,
    height: int,
    on_scan: Callable[[int, int], None] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield selected (frame_idx, bgr) by seeking near the first index.

    Decodes only the [min, max] window — not from t=0. Used for short WEB clips
    so HEVC hour-long files do not grab-skip from the start.
    """
    needed = sorted({max(0, int(i)) for i in indices})
    if not needed:
        return
    first, last = needed[0], needed[-1]
    needed_set = set(needed)
    rate = float(fps) if fps and fps > 1.0 else probe_video_fps(str(input_path))
    if rate < 1.0:
        rate = 25.0
    tw, th = int(width), int(height)
    if tw <= 0 or th <= 0:
        return
    frame_bytes = tw * th * 3
    t0 = first / rate
    n_decode = last - first + 1
    pad = 2.5
    in_ss = max(0.0, t0 - pad)
    out_ss = t0 - in_ss
    try:
        from app.core.ffmpeg_utils import _popen_kwargs, resolve_ffmpeg_exe

        exe = resolve_ffmpeg_exe()
        kwargs = _popen_kwargs()
    except Exception:
        return
    extra: list[str] = ["-ss", f"{in_ss:.3f}", "-i", str(input_path)]
    if out_ss > 0.02:
        extra.extend(["-ss", f"{out_ss:.3f}"])
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        *extra,
        "-frames:v",
        str(max(1, n_decode)),
        "-an",
        "-vf",
        f"scale={tw}:{th}:flags=fast_bilinear,format=bgr24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        assert proc.stdout is not None
        src_i = first
        while src_i <= last:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            if src_i in needed_set:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(th, tw, 3).copy()
                yield src_i, frame
                if on_scan is not None:
                    on_scan(src_i, last)
            src_i += 1
    finally:
        if proc is not None:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=8)
            except Exception:
                pass
