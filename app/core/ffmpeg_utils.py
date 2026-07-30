"""FFmpeg discovery, NVENC detection, and GPU decode helpers."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Any

_NVENC_PRESET = {
    "ultrafast": "p1",
    "superfast": "p2",
    "veryfast": "p3",
    "faster": "p4",
    "fast": "p4",
    "medium": "p5",
    "slow": "p6",
}


def resolve_ffmpeg_exe() -> str:
    """Best ffmpeg binary: explicit env → system PATH (NVENC) → imageio bundled."""
    env = os.getenv("IMAGEIO_FFMPEG_EXE", "").strip()
    if env and _is_valid_ffmpeg(env):
        return env

    system = shutil.which("ffmpeg")
    if system and _ffmpeg_has_encoder(system, "h264_nvenc"):
        return system

    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and _is_valid_ffmpeg(bundled):
            return bundled
    except Exception:
        pass

    if system and _is_valid_ffmpeg(system):
        return system
    raise RuntimeError("ffmpeg not found — install ffmpeg or imageio[ffmpeg]")


@lru_cache(maxsize=8)
def _is_valid_ffmpeg(exe: str) -> bool:
    try:
        subprocess.run(
            [exe, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            **_popen_kwargs(),
        )
        return True
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False


@lru_cache(maxsize=16)
def _ffmpeg_has_encoder(exe: str, encoder: str) -> bool:
    if not _is_valid_ffmpeg(exe):
        return False
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            **_popen_kwargs(),
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        needle = encoder.casefold()
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0].casefold() == needle:
                return True
    except OSError:
        return False
    return False


def resolve_encode_codec(requested: str = "auto", *, ffmpeg_exe: str | None = None) -> str:
    req = (requested or "auto").strip().lower()
    exe = ffmpeg_exe or resolve_ffmpeg_exe()
    if req in ("h264_nvenc", "nvenc"):
        return "h264_nvenc"
    if req == "libx264":
        return "libx264"
    if _ffmpeg_has_encoder(exe, "h264_nvenc"):
        try:
            from imageio_ffmpeg._io import ffmpeg_test_encoder

            if ffmpeg_test_encoder("h264_nvenc"):
                return "h264_nvenc"
        except Exception:
            pass
        if _nvenc_smoke_test(exe):
            return "h264_nvenc"
    return "libx264"


def _nvenc_smoke_test(exe: str) -> bool:
    """Quick NVENC encode test (null output) when imageio test is unavailable."""
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=0.04",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, **_popen_kwargs())
        return proc.returncode == 0
    except OSError:
        return False


def build_encode_writer_args(
    *,
    codec: str = "auto",
    preset: str = "fast",
    crf: int = 23,
    ffmpeg_exe: str | None = None,
) -> tuple[str, list[str], str]:
    """Return (codec, ffmpeg_params, ffmpeg_exe)."""
    exe = ffmpeg_exe or resolve_ffmpeg_exe()
    resolved = resolve_encode_codec(codec, ffmpeg_exe=exe)
    if resolved == "h264_nvenc":
        nv_preset = _NVENC_PRESET.get(preset.strip().lower(), "p4")
        params = [
            "-preset",
            nv_preset,
            "-rc",
            "vbr",
            "-cq",
            str(int(crf)),
            "-b:v",
            "0",
            "-gpu",
            "0",
        ]
        return resolved, params, exe
    return "libx264", ["-crf", str(int(crf)), "-preset", preset, "-threads", "0"], exe


def decode_vf_filter(
    hw_name: str,
    width: int,
    height: int,
    *,
    hw_args: list[str] | None = None,
) -> str:
    """Video filter chain for raw BGR24 pipe after hwaccel decode."""
    w, h = int(width), int(height)
    needs_hwdownload = bool(
        hw_args
        and "-hwaccel_output_format" in hw_args
        and "cuda" in hw_args
    )
    if needs_hwdownload or hw_name == "cuda_hw":
        return f"hwdownload,format=nv12,scale={w}:{h},format=bgr24"
    return f"scale={w}:{h},format=bgr24"


def list_hwaccel_decode_profiles(
    width: int,
    height: int,
    *,
    ffmpeg_exe: str | None = None,
) -> list[tuple[str, list[str]]]:
    """Ordered decode profiles that pass bgr24 pipe smoke test."""
    exe = ffmpeg_exe or resolve_ffmpeg_exe()
    w, h = int(width), int(height)
    if sys.platform == "win32":
        candidates: list[tuple[str, list[str]]] = [
            ("d3d11va", ["-hwaccel", "d3d11va"]),
            ("dxva2", ["-hwaccel", "dxva2"]),
            ("cuda", ["-hwaccel", "cuda"]),
            ("cuda_hw", ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]),
            ("auto", ["-hwaccel", "auto"]),
        ]
    else:
        candidates = [
            ("cuda", ["-hwaccel", "cuda"]),
            ("cuda_hw", ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]),
            ("vaapi", ["-hwaccel", "vaapi"]),
            ("auto", ["-hwaccel", "auto"]),
        ]
    ok: list[tuple[str, list[str]]] = []
    for name, hw_args in candidates:
        vf = decode_vf_filter(name, w, h, hw_args=hw_args)
        if _decode_profile_smoke_test(exe, hw_args, vf, w, h):
            ok.append((name, hw_args))
    return ok


def pick_hwaccel_decode_args(
    *,
    ffmpeg_exe: str | None = None,
    width: int = 0,
    height: int = 0,
) -> tuple[list[str], str]:
    """Return ffmpeg input args for hardware decode, or ([], 'cpu')."""
    if width > 0 and height > 0:
        profiles = list_hwaccel_decode_profiles(width, height, ffmpeg_exe=ffmpeg_exe)
        if profiles:
            name, args = profiles[0]
            return args, name
    exe = ffmpeg_exe or resolve_ffmpeg_exe()
    # Legacy probe without dimensions (encode plan logging only).
    for name, args in (
        ("d3d11va", ["-hwaccel", "d3d11va"]),
        ("dxva2", ["-hwaccel", "dxva2"]),
        ("cuda", ["-hwaccel", "cuda"]),
        ("auto", ["-hwaccel", "auto"]),
    ):
        if _hwaccel_smoke_test(exe, args):
            return args, name
    return [], "cpu"


def _decode_profile_smoke_test(
    exe: str,
    hwaccel_args: list[str],
    vf: str,
    width: int,
    height: int,
) -> bool:
    """Test full decode→bgr24 pipe (1 frame) — catches hwdownload format errors."""
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        *hwaccel_args,
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={max(64, width)}x{max(64, height)}:d=0.04",
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, **_popen_kwargs())
        if proc.returncode != 0:
            return False
        need = max(64, width) * max(64, height) * 3
        return len(proc.stdout or b"") >= need
    except OSError:
        return False


def _hwaccel_smoke_test(exe: str, hwaccel_args: list[str]) -> bool:
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        *hwaccel_args,
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=0.04",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, **_popen_kwargs())
        return proc.returncode == 0
    except OSError:
        return False


def describe_encode_plan(
    *,
    codec: str = "auto",
    preset: str = "fast",
    crf: int = 23,
    hw_decode: bool = True,
    width: int = 0,
    height: int = 0,
) -> dict[str, Any]:
    exe = resolve_ffmpeg_exe()
    resolved, params, _ = build_encode_writer_args(codec=codec, preset=preset, crf=crf, ffmpeg_exe=exe)
    if hw_decode and width > 0 and height > 0:
        profiles = list_hwaccel_decode_profiles(width, height, ffmpeg_exe=exe)
        if profiles:
            hw_name, hw_args = profiles[0]
        else:
            hw_name, hw_args = "cpu", []
    elif hw_decode:
        hw_args, hw_name = pick_hwaccel_decode_args(ffmpeg_exe=exe)
    else:
        hw_name, hw_args = "cpu", []
    return {
        "ffmpeg": exe,
        "codec": resolved,
        "ffmpeg_params": params,
        "hwaccel": hw_name,
        "hwaccel_args": hw_args,
    }


def format_encode_plan(plan: dict[str, Any]) -> str:
    codec = plan.get("codec", "?")
    hw = plan.get("hwaccel", "?")
    exe = Path_str(plan.get("ffmpeg", ""))
    return f"MP4 encode: {codec} | decode: {hw} | ffmpeg: {exe}"


def Path_str(p: Any) -> str:
    try:
        from pathlib import Path

        return Path(str(p)).name
    except Exception:
        return str(p)


def _popen_kwargs() -> dict[str, Any]:
    if not sys.platform.startswith("win"):
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {"startupinfo": startupinfo, "creationflags": 0x00000200}
