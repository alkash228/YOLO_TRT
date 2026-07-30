"""Paths and naming for TensorRT engines (YOLO + OSNet ReID)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import TRT_DIR

MANIFEST_NAME = "engines_manifest.json"
INSTRUCTIONS_NAME = "TENSORRT_INSTRUCTIONS.md"


@dataclass(slots=True)
class TrtEngineRecord:
    role: str
    source: str
    engine: str
    imgsz: int
    max_batch: int
    fp16: bool
    built_at: str
    notes: str = ""


def yolo_engine_name(stem: str, imgsz: int, max_batch: int, fp16: bool) -> str:
    prec = "fp16" if fp16 else "fp32"
    return f"{stem}_i{imgsz}_b{max_batch}_{prec}.engine"


def reid_engine_name(stem: str, fp16: bool) -> str:
    prec = "fp16" if fp16 else "fp32"
    return f"{stem}_256x128_{prec}.engine"


def reid_onnx_name(stem: str) -> str:
    return f"{stem}_256x128.onnx"


def engine_root_for_model(
    model_path: Path,
    *,
    strategy: str = "central",
    central_dir: Path | None = None,
) -> Path:
    if strategy.strip().casefold() == "colocate":
        return model_path.parent
    return central_dir or TRT_DIR


def find_yolo_engine(
    pt_path: Path,
    *,
    imgsz: int,
    max_batch: int,
    fp16: bool,
    trt_dir: Path | None = None,
) -> Path:
    """Exact name first; иначе manifest или ближайший batch ≤ запрошенного."""
    root = trt_dir or TRT_DIR
    exact = root / yolo_engine_name(pt_path.stem, imgsz, max_batch, fp16)
    if exact.exists():
        return exact

    src = str(pt_path.resolve())
    best: tuple[int, Path] | None = None
    for rec in load_manifest(root):
        try:
            same_src = Path(rec.source).resolve() == Path(src)
        except OSError:
            same_src = str(rec.source) == src
        if not same_src or int(rec.imgsz) != int(imgsz) or bool(rec.fp16) != bool(fp16):
            continue
        eng = Path(rec.engine)
        if not eng.is_file():
            eng = root / eng.name
        if eng.is_file() and int(rec.max_batch) <= int(max_batch):
            if best is None or int(rec.max_batch) > best[0]:
                best = (int(rec.max_batch), eng)

    prec = "fp16" if fp16 else "fp32"
    pattern = f"{pt_path.stem}_i{imgsz}_b*_{prec}.engine"
    for eng in root.glob(pattern):
        try:
            part = eng.stem.split("_b", 1)[1].split("_", 1)[0]
            b = int(part)
        except (IndexError, ValueError):
            continue
        if b <= int(max_batch) and (best is None or b > best[0]):
            best = (b, eng)

    if best is not None:
        return best[1]
    return exact


def resolve_yolo_engine(
    pt_path: Path,
    *,
    imgsz: int,
    max_batch: int,
    fp16: bool,
    trt_dir: Path | None = None,
    strategy: str = "central",
    central_dir: Path | None = None,
) -> Path:
    root = trt_dir or engine_root_for_model(
        pt_path, strategy=strategy, central_dir=central_dir
    )
    exact = root / yolo_engine_name(pt_path.stem, imgsz, max_batch, fp16)
    if exact.exists():
        return exact
    return find_yolo_engine(
        pt_path, imgsz=imgsz, max_batch=max_batch, fp16=fp16, trt_dir=root
    )


def resolve_reid_engine(
    pth_path: Path,
    *,
    fp16: bool,
    trt_dir: Path | None = None,
    strategy: str = "central",
    central_dir: Path | None = None,
) -> Path:
    root = trt_dir or engine_root_for_model(
        pth_path, strategy=strategy, central_dir=central_dir
    )
    return root / reid_engine_name(pth_path.stem, fp16)


def resolve_reid_onnx(
    pth_path: Path,
    *,
    trt_dir: Path | None = None,
    strategy: str = "central",
    central_dir: Path | None = None,
) -> Path:
    root = trt_dir or engine_root_for_model(
        pth_path, strategy=strategy, central_dir=central_dir
    )
    return root / reid_onnx_name(pth_path.stem)


def manifest_path(trt_dir: Path | None = None) -> Path:
    return (trt_dir or TRT_DIR) / MANIFEST_NAME


def instructions_path(trt_dir: Path | None = None) -> Path:
    return (trt_dir or TRT_DIR) / INSTRUCTIONS_NAME


def load_manifest(trt_dir: Path | None = None) -> list[TrtEngineRecord]:
    path = manifest_path(trt_dir)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [TrtEngineRecord(**item) for item in raw.get("engines", [])]
    except Exception:
        return []


def save_manifest(records: list[TrtEngineRecord], trt_dir: Path | None = None) -> Path:
    root = trt_dir or TRT_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = manifest_path(root)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "engines": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def engines_ready(
    *,
    detect_pt: Path,
    cross_pt: Path | None,
    reid_pth: Path,
    imgsz: int,
    max_batch: int,
    fp16: bool,
    need_cross: bool,
    need_reid: bool = True,
    strategy: str = "central",
    central_dir: Path | None = None,
) -> dict[str, bool]:
    det = resolve_yolo_engine(
        detect_pt,
        imgsz=imgsz,
        max_batch=max_batch,
        fp16=fp16,
        strategy=strategy,
        central_dir=central_dir,
    )
    out = {
        "detect": det.exists(),
        "reid": True,
        "cross_check": True,
    }
    if need_reid:
        reid = resolve_reid_engine(
            reid_pth,
            fp16=fp16,
            strategy=strategy,
            central_dir=central_dir,
        )
        out["reid"] = reid.exists()
    if need_cross and cross_pt is not None:
        cross = resolve_yolo_engine(
            cross_pt,
            imgsz=imgsz,
            max_batch=max_batch,
            fp16=fp16,
            strategy=strategy,
            central_dir=central_dir,
        )
        out["cross_check"] = cross.exists()
    return out
