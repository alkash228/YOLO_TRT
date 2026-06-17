"""YOLO26x-seg instance segmentation with batched GPU mask resize."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.batch_utils import chunk_list


@dataclass(slots=True)
class SegItem:
    xyxy: np.ndarray
    cls_id: int
    label: str
    conf: float
    mask_u8: np.ndarray


class SegEngine:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        device: str | None = None,
        half: bool = True,
        imgsz: int = 0,
    ) -> None:
        from ultralytics import YOLO

        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Seg model not found: {p}")
        self.model_path = str(p)
        self.conf = float(conf)
        self.half = bool(half)
        self.imgsz = int(imgsz) if int(imgsz) > 0 else None
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(self.model_path)
        self._use_gpu_masks = device.startswith("cuda")

    def _yolo_kwargs(self) -> dict:
        kw: dict = dict(
            conf=max(0.001, self.conf),
            verbose=False,
            device=self.device,
            half=self.half,
        )
        if self.imgsz is not None:
            kw["imgsz"] = self.imgsz
        return kw

    def predict(self, frame_bgr: np.ndarray) -> list[SegItem]:
        results = self.model.predict(source=frame_bgr, **self._yolo_kwargs())
        res = results[0] if results else None
        if res is None:
            return []
        h, w = frame_bgr.shape[:2]
        return self._parse_single_result(res, h, w)

    def predict_batch(
        self,
        frames_bgr: list[np.ndarray],
        *,
        max_batch: int = 0,
    ) -> list[list[SegItem]]:
        if not frames_bgr:
            return []
        cap = int(max_batch)
        out: list[list[SegItem]] = []
        for chunk in chunk_list(frames_bgr, cap):
            if len(chunk) == 1:
                out.append(self.predict(chunk[0]))
                continue
            results = self.model.predict(source=chunk, batch=len(chunk), **self._yolo_kwargs())
            if not isinstance(results, list):
                results = [results]
            for i, res in enumerate(results):
                h, w = chunk[i].shape[:2]
                out.append(self._parse_single_result(res, h, w))
        return out

    def _parse_single_result(self, res, h: int, w: int) -> list[SegItem]:
        if res is None or res.boxes is None or len(res.boxes) == 0:
            return []

        xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        cls = res.boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
        conf = res.boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        name_map = res.names if isinstance(res.names, dict) else {}

        masks_u8: list[np.ndarray] | None = None
        if res.masks is not None and res.masks.data is not None:
            masks_u8 = self._masks_to_u8_batch(res.masks.data, h, w)

        out: list[SegItem] = []
        for i in range(int(xyxy.shape[0])):
            label = str(name_map.get(int(cls[i]), str(int(cls[i]))))
            if masks_u8 is not None and i < len(masks_u8):
                m = masks_u8[i]
            else:
                m = self._bbox_mask(xyxy[i], h, w)
            out.append(
                SegItem(
                    xyxy=xyxy[i].copy(),
                    cls_id=int(cls[i]),
                    label=label,
                    conf=float(conf[i]),
                    mask_u8=m,
                )
            )
        return out

    def _masks_to_u8_batch(self, masks_t, h: int, w: int) -> list[np.ndarray]:
        """Resize masks on GPU, single D2H transfer per frame batch."""
        if self._use_gpu_masks:
            try:
                import torch
                import torch.nn.functional as F

                m = masks_t
                if m.device.type == "cpu":
                    m = m.cuda(non_blocking=True)
                if m.ndim == 3:
                    m = m.unsqueeze(1).float()
                mh, mw = int(m.shape[-2]), int(m.shape[-1])
                if mh != h or mw != w:
                    m = F.interpolate(m, size=(h, w), mode="nearest")
                m = (m.squeeze(1) > 0.5).byte()
                arr = m.cpu().numpy()
                return [(arr[i] * 255).astype(np.uint8, copy=False) for i in range(int(arr.shape[0]))]
            except Exception:
                pass

        masks_np = masks_t.detach().cpu().numpy()
        out: list[np.ndarray] = []
        for i in range(int(masks_np.shape[0])):
            m = (masks_np[i] > 0.5).astype(np.uint8) * 255
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            out.append(m)
        return out

    @staticmethod
    def _bbox_mask(xyxy: np.ndarray, h: int, w: int) -> np.ndarray:
        x0, y0, x1, y1 = [float(v) for v in xyxy.tolist()]
        xa = int(np.clip(np.floor(x0), 0, max(0, w - 1)))
        ya = int(np.clip(np.floor(y0), 0, max(0, h - 1)))
        xb = int(np.clip(np.ceil(x1), 0, max(0, w - 1)))
        yb = int(np.clip(np.ceil(y1), 0, max(0, h - 1)))
        out = np.zeros((h, w), dtype=np.uint8)
        if xb > xa and yb > ya:
            out[ya : yb + 1, xa : xb + 1] = 255
        return out
