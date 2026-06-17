"""YOLO model class catalog — read supported classes from weights."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class YoloClassInfo:
    class_id: int
    name: str


@dataclass
class ModelClassCatalog:
    model_path: Path
    model_kind: str
    dataset: str
    classes: list[YoloClassInfo] = field(default_factory=list)

    @property
    def class_count(self) -> int:
        return len(self.classes)

    def names(self) -> list[str]:
        return [c.name for c in self.classes]


@dataclass(slots=True)
class DualModelCatalog:
    detect: ModelClassCatalog
    seg: ModelClassCatalog

    @property
    def in_sync(self) -> bool:
        return self.detect.names() == self.seg.names()

    @property
    def all_classes(self) -> list[YoloClassInfo]:
        return list(self.detect.classes)


def _read_model_names(model_path: Path) -> dict[int, str]:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    from ultralytics import YOLO

    model = YOLO(str(p))
    raw = model.names
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    return {int(i): str(n) for i, n in enumerate(raw)}


def load_catalog(model_path: str | Path, model_kind: str = "detect") -> ModelClassCatalog:
    path = Path(model_path)
    names_map = _read_model_names(path)
    classes = [
        YoloClassInfo(class_id=cid, name=names_map[cid])
        for cid in sorted(names_map.keys())
    ]
    return ModelClassCatalog(
        model_path=path,
        model_kind=model_kind,
        dataset="COCO",
        classes=classes,
    )


def load_dual_catalog(
    detect_path: str | Path,
    seg_path: str | Path,
) -> DualModelCatalog:
    return DualModelCatalog(
        detect=load_catalog(detect_path, "detect"),
        seg=load_catalog(seg_path, "seg"),
    )


def filter_classes(
    classes: list[YoloClassInfo],
    *,
    query: str = "",
) -> list[YoloClassInfo]:
    q = query.strip().casefold()
    if not q:
        return list(classes)
    return [
        c
        for c in classes
        if q in c.name.casefold() or q == str(c.class_id)
    ]


def prompt_from_class_names(names: list[str]) -> str:
    cleaned = [n.strip() for n in names if n.strip()]
    return ".".join(dict.fromkeys(cleaned))


def merge_prompt(existing: str, names: list[str]) -> str:
    from app.core.prompt_utils import prompt_terms

    terms = list(prompt_terms(existing))
    for n in names:
        t = n.strip().casefold()
        if t and t not in terms:
            terms.append(t)
    return ".".join(terms)
