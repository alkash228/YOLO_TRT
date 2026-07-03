"""Spill compact inference packets to disk per window (bounded RAM)."""
from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.frame_pipeline import FramePacket
from app.core.video_encode import PACKETS_SUFFIX

CHUNK_SUFFIX = "_packets_chunk_"
MANIFEST_SUFFIX = "_packets_manifest.json"


def spill_chunk_path(run_dir: Path, run_id: str, chunk_idx: int) -> Path:
    return Path(run_dir) / f"{run_id}{CHUNK_SUFFIX}{chunk_idx:05d}.pkl"


def manifest_path_for_run(run_dir: Path, run_id: str) -> Path:
    return Path(run_dir) / f"{run_id}{MANIFEST_SUFFIX}"


class PacketSpillWriter:
    """Write one pickle chunk per window; finalize with manifest JSON."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self._run_dir = Path(run_dir)
        self._run_id = str(run_id)
        self._chunk_idx = 0
        self._chunks: list[dict[str, Any]] = []
        self._total_packets = 0
        self._run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def chunk_count(self) -> int:
        return self._chunk_idx

    @property
    def total_packets(self) -> int:
        return self._total_packets

    def write_chunk(
        self,
        packets: list[FramePacket],
        *,
        start_frame: int,
        end_frame: int,
    ) -> Path | None:
        if not packets:
            return None
        ordered = sorted(packets, key=lambda p: p.frame_idx)
        path = spill_chunk_path(self._run_dir, self._run_id, self._chunk_idx)
        payload = {
            "version": 1,
            "run_id": self._run_id,
            "chunk_idx": self._chunk_idx,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "packets": ordered,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._chunks.append(
            {
                "path": path.name,
                "chunk_idx": self._chunk_idx,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "packet_count": len(ordered),
            }
        )
        self._total_packets += len(ordered)
        self._chunk_idx += 1
        return path

    def finalize(
        self,
        *,
        fps: float,
        input_path: str,
        prompt: str,
        overlay: dict[str, Any],
        width: int = 0,
        height: int = 0,
        frame_stride: int = 1,
        source_frame_count: int = 0,
        window_frames: int = 0,
    ) -> Path:
        manifest = {
            "version": 1,
            "run_id": self._run_id,
            "format": "chunked",
            "fps": float(fps),
            "input_path": str(input_path),
            "prompt": str(prompt),
            "overlay": dict(overlay),
            "width": int(width),
            "height": int(height),
            "frame_stride": max(1, int(frame_stride)),
            "source_frame_count": int(source_frame_count),
            "window_frames": int(window_frames),
            "total_packets": self._total_packets,
            "chunks": self._chunks,
        }
        path = manifest_path_for_run(self._run_dir, self._run_id)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def load_packets_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "chunks" not in data:
        raise ValueError(f"Invalid packets manifest: {path}")
    return data


def find_packets_manifest(run_dir: Path, run_id: str | None = None) -> Path | None:
    run_dir = Path(run_dir)
    if run_id:
        p = manifest_path_for_run(run_dir, run_id)
        return p if p.is_file() else None
    matches = sorted(
        run_dir.glob(f"*{MANIFEST_SUFFIX}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def iter_spilled_packets(manifest: dict[str, Any], *, run_dir: Path) -> Iterator[FramePacket]:
    run_dir = Path(run_dir)
    for entry in manifest.get("chunks") or []:
        rel = str(entry.get("path") or "")
        chunk_path = run_dir / rel
        if not chunk_path.is_file():
            raise FileNotFoundError(f"Missing spill chunk: {chunk_path}")
        with chunk_path.open("rb") as f:
            data = pickle.load(f)
        for packet in data.get("packets") or []:
            yield packet


def load_all_spilled_packets(manifest: dict[str, Any], *, run_dir: Path) -> list[FramePacket]:
    return sorted(iter_spilled_packets(manifest, run_dir=run_dir), key=lambda p: p.frame_idx)


def merge_spill_to_single_pkl(
    manifest: dict[str, Any],
    *,
    run_dir: Path,
    out_path: Path | None = None,
) -> Path:
    """Merge chunked spill into one legacy _packets.pkl (reads one chunk at a time)."""
    run_dir = Path(run_dir)
    run_id = str(manifest.get("run_id") or run_dir.name)
    if out_path is None:
        out_path = run_dir / f"{run_id}{PACKETS_SUFFIX}"
    all_packets: list[FramePacket] = []
    for entry in manifest.get("chunks") or []:
        chunk_path = run_dir / str(entry.get("path") or "")
        with chunk_path.open("rb") as f:
            data = pickle.load(f)
        all_packets.extend(list(data.get("packets") or []))
    all_packets.sort(key=lambda p: p.frame_idx)
    payload = {
        "version": 2,
        "run_id": run_id,
        "fps": float(manifest.get("fps") or 25.0),
        "input_path": str(manifest.get("input_path") or ""),
        "prompt": str(manifest.get("prompt") or "person"),
        "overlay": dict(manifest.get("overlay") or {}),
        "width": int(manifest.get("width") or 0),
        "height": int(manifest.get("height") or 0),
        "frame_stride": max(1, int(manifest.get("frame_stride") or 1)),
        "source_frame_count": int(manifest.get("source_frame_count") or 0),
        "packets": all_packets,
    }
    with Path(out_path).open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return Path(out_path)
