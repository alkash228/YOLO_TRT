"""Scan models/ folders for available weights."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import RD_DIR, YOLO_DIR, YOLO_SEG_DIR


@dataclass
class ModelScanResult:
    detect: list[Path] = field(default_factory=list)
    seg: list[Path] = field(default_factory=list)
    reid: list[Path] = field(default_factory=list)


def _unique_sorted(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in sorted(paths, key=lambda x: x.name.casefold()):
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def scan_detect_models(yolo_dir: Path | None = None) -> list[Path]:
    root = yolo_dir or YOLO_DIR
    if not root.exists():
        return []
    found: list[Path] = []
    for p in root.rglob("*.pt"):
        if p.is_file():
            found.append(p)
    return _unique_sorted(found)


def scan_seg_models(
    yolo_dir: Path | None = None,
    seg_dir: Path | None = None,
) -> list[Path]:
    yroot = yolo_dir or YOLO_DIR
    sroot = seg_dir or YOLO_SEG_DIR
    found: list[Path] = []
    if sroot.exists():
        for p in sroot.rglob("*.pt"):
            if p.is_file():
                found.append(p)
    if yroot.exists():
        for p in yroot.glob("*.pt"):
            if p.is_file() and "seg" in p.stem.casefold():
                found.append(p)
    return _unique_sorted(found)


def scan_reid_models(rd_dir: Path | None = None) -> list[Path]:
    root = rd_dir or RD_DIR
    if not root.exists():
        return []
    found = [p for p in root.rglob("*.pth") if p.is_file()]
    return _unique_sorted(found)


def scan_all_models(
    yolo_dir: Path | None = None,
    seg_dir: Path | None = None,
    rd_dir: Path | None = None,
) -> ModelScanResult:
    return ModelScanResult(
        detect=scan_detect_models(yolo_dir),
        seg=scan_seg_models(yolo_dir, seg_dir),
        reid=scan_reid_models(rd_dir),
    )


def pick_default(path: Path, options: list[Path]) -> Path:
    if path in options:
        return path
    resolved = str(path.resolve()) if path.exists() else str(path)
    for opt in options:
        if str(opt.resolve()) == resolved or opt.name == path.name:
            return opt
    return options[0] if options else path
