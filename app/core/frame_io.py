"""Torch-free frame I/O helpers (WEB encode / Word report)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
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


def _even_wh(width: int, height: int) -> tuple[int, int]:
    tw, th = int(width), int(height)
    if tw <= 0 or th <= 0:
        return (0, 0)
    return (tw - (tw % 2), th - (th % 2))


def _keyframe_pts_at_or_before(input_path: str, t: float, *, lookback: float = 30.0) -> float:
    """Last keyframe PTS <= t. Mid-GOP HEVC decode is the smeared Word photo."""
    probe = _ffprobe_exe()
    target = max(0.0, float(t))
    start = max(0.0, target - max(2.0, float(lookback)))
    if not probe:
        return start
    cmd = [
        probe,
        "-v",
        "error",
        "-skip_frame",
        "nokey",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pts_time,pkt_pts_time,best_effort_timestamp_time",
        "-read_intervals",
        f"{start:.3f}%{target + 0.08:.3f}",
        "-of",
        "csv=p=0",
        str(input_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return start
    best: float | None = None
    for line in (proc.stdout or "").splitlines():
        for part in line.strip().replace(",", " ").replace("|", " ").split():
            try:
                pts = float(part)
            except ValueError:
                continue
            if pts < 0.0 or pts > target + 0.08:
                continue
            if best is None or pts > best:
                best = pts
    return best if best is not None else start


def _ffmpeg_exe_kwargs() -> tuple[str, dict[str, object]] | None:
    try:
        from app.core.ffmpeg_utils import _popen_kwargs, resolve_ffmpeg_exe

        return resolve_ffmpeg_exe(), _popen_kwargs()  # type: ignore[return-value]
    except Exception:
        return None


def _reshape_bgr24(raw: bytes, width: int, height: int) -> np.ndarray | None:
    tw, th = int(width), int(height)
    if tw <= 0 or th <= 0 or not raw:
        return None
    need = tw * th * 3
    if len(raw) >= need:
        return np.frombuffer(raw[:need], dtype=np.uint8).reshape(th, tw, 3).copy()
    # ffmpeg sometimes emits a different height; keep stride = width.
    if len(raw) >= tw * 3 and len(raw) % (tw * 3) == 0:
        got_h = len(raw) // (tw * 3)
        if got_h > 0:
            return np.frombuffer(raw, dtype=np.uint8).reshape(got_h, tw, 3).copy()
    return None


def _chroma_strength(bgr: np.ndarray | None) -> float:
    """Mean |R-G|+|B-G|. Near 0 → luma-only / gray HEVC still."""
    if bgr is None or bgr.size < 64:
        return 0.0
    s = bgr[::8, ::8]
    if s.size < 16:
        s = bgr
    b = s[:, :, 0].astype(np.int16)
    g = s[:, :, 1].astype(np.int16)
    r = s[:, :, 2].astype(np.int16)
    return float(np.abs(b - g).mean() + np.abs(r - g).mean())


def _is_almost_gray(bgr: np.ndarray | None) -> bool:
    return _chroma_strength(bgr) < 8.0


def _still_vf(width: int, height: int, *, expand_tv: bool = False) -> str:
    """8-bit 4:2:0 → BGR with chroma. neighbor+10-bit HEVC dropped chroma (gray photo)."""
    tw, th = int(width), int(height)
    scale = (
        f"scale={tw}:{th}:flags=bicubic+accurate_rnd+full_chroma_int+full_chroma_inp"
    )
    if expand_tv:
        scale += ":in_range=tv:out_range=pc"
    return f"format=yuv420p,{scale},format=bgr24"


def _ffmpeg_raw_still(
    extra: list[str],
    *,
    width: int,
    height: int,
    timeout_sec: float,
    expand_tv: bool = False,
) -> np.ndarray | None:
    """One still as raw BGR24 — no JPEG/PNG re-encode (that smeared Word photos)."""
    tw, th = _even_wh(width, height)
    if tw <= 0 or th <= 0:
        return None
    resolved = _ffmpeg_exe_kwargs()
    if resolved is None:
        return None
    exe, kwargs = resolved
    cmd = [
        exe,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        *extra,
        "-frames:v",
        "1",
        "-an",
        "-vf",
        _still_vf(tw, th, expand_tv=expand_tv),
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
    return _reshape_bgr24(proc.stdout or b"", tw, th)


def _ffmpeg_png_file_still(
    extra: list[str],
    *,
    timeout_sec: float,
) -> np.ndarray | None:
    """Lossless PNG on disk after yuv420p — keeps chroma; not an image2pipe JPEG."""
    resolved = _ffmpeg_exe_kwargs()
    if resolved is None:
        return None
    exe, kwargs = resolved
    tmp = tempfile.NamedTemporaryFile(prefix="yolo_still_", suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    cmd = [
        exe,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        *extra,
        "-frames:v",
        "1",
        "-an",
        "-vf",
        "format=yuv420p,format=rgb24",
        "-update",
        "1",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=max(15.0, float(timeout_sec)),
            check=False,
            **kwargs,
        )
        if proc.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size < 64:
            return None
        frame = cv2.imread(str(tmp_path), cv2.IMREAD_COLOR)
        if frame is None or frame.size <= 0:
            return None
        return frame
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _best_still(*candidates: np.ndarray | None) -> np.ndarray | None:
    ranked = [c for c in candidates if c is not None and c.size > 0]
    if not ranked:
        return None
    ranked.sort(key=_chroma_strength, reverse=True)
    return ranked[0]


def _ffmpeg_bmp_still(
    extra: list[str],
    *,
    timeout_sec: float,
) -> np.ndarray | None:
    """Lossless BMP on disk — last resort, never JPEG/PNG pipe."""
    resolved = _ffmpeg_exe_kwargs()
    if resolved is None:
        return None
    exe, kwargs = resolved
    tmp = tempfile.NamedTemporaryFile(prefix="yolo_still_", suffix=".bmp", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    cmd = [
        exe,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        *extra,
        "-frames:v",
        "1",
        "-an",
        "-vf",
        "format=yuv420p,format=bgr24",
        "-update",
        "1",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=max(15.0, float(timeout_sec)),
            check=False,
            **kwargs,
        )
        if proc.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size < 64:
            return None
        frame = cv2.imread(str(tmp_path), cv2.IMREAD_COLOR)
        if frame is None or frame.size <= 0:
            return None
        return frame
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


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

    accurate=True (Word): seek to the real keyframe before t, then decode that
    GOP to raw BGR24. A short pad (4s) lands mid-GOP on camera HEVC → smeared
    photo with sharp Python boxes. Never JPEG/PNG pipe on this path.
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
    key = _keyframe_pts_at_or_before(path, t, lookback=30.0)
    in_ss = max(0.0, key)
    out_ss = max(0.0, t - in_ss)
    extra = ["-ss", f"{in_ss:.3f}", "-i", path]
    if out_ss > 0.01:
        extra.extend(["-ss", f"{out_ss:.3f}"])
    a: np.ndarray | None = None
    if tw > 0 and th > 0:
        a = _ffmpeg_raw_still(extra, width=tw, height=th, timeout_sec=50.0)
        if a is not None and not _is_almost_gray(a):
            return a
        b = _ffmpeg_raw_still(
            extra, width=tw, height=th, timeout_sec=50.0, expand_tv=True
        )
        colored = _best_still(a, b)
        if colored is not None and not _is_almost_gray(colored):
            return colored
    png = _ffmpeg_png_file_still(extra, timeout_sec=50.0)
    if png is not None and not _is_almost_gray(png):
        return png
    bmp = _ffmpeg_bmp_still(extra, timeout_sec=50.0)
    hit = _best_still(a, png, bmp)
    if hit is not None:
        return hit
    extra_slow = ["-i", path, "-ss", f"{t:.3f}"]
    if tw > 0 and th > 0:
        hit = _ffmpeg_raw_still(
            extra_slow, width=tw, height=th, timeout_sec=90.0, expand_tv=True
        )
        if hit is not None:
            return hit
    return _best_still(
        _ffmpeg_png_file_still(extra_slow, timeout_sec=90.0),
        _ffmpeg_bmp_still(extra_slow, timeout_sec=90.0),
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
