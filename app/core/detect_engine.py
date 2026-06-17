"""YOLO26x detect + BoT-SORT track / batched predict."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.batch_utils import chunk_list
from app.core.trt_paths import resolve_yolo_engine


@dataclass(slots=True)
class DetectItem:
    xyxy: np.ndarray
    cls_id: int
    label: str
    conf: float
    motion_id: int
    keypoints: np.ndarray | None = None  # (K, 3) x, y, conf — YOLO pose


class DetectEngine:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        device: str | None = None,
        half: bool = True,
        imgsz: int = 0,
        use_tensorrt: bool = False,
        tensorrt_imgsz: int = 640,
        tensorrt_max_batch: int = 16,
        tensorrt_fp16: bool = True,
        tensorrt_engine_strategy: str = "central",
        tensorrt_central_dir: Path | None = None,
    ) -> None:
        from ultralytics import YOLO

        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Detect model not found: {p}")
        load_path = p
        self.using_tensorrt = False
        if use_tensorrt:
            eng = resolve_yolo_engine(
                p,
                imgsz=int(tensorrt_imgsz),
                max_batch=int(tensorrt_max_batch),
                fp16=bool(tensorrt_fp16),
                strategy=tensorrt_engine_strategy,
                central_dir=tensorrt_central_dir,
            )
            if eng.exists():
                load_path = eng
                self.using_tensorrt = True
        self.model_path = str(load_path)
        self.source_model_path = str(p)
        self.trt_max_batch = int(tensorrt_max_batch) if self.using_tensorrt else 0
        self.trt_imgsz = int(tensorrt_imgsz) if self.using_tensorrt else 0
        self._trt_names: dict[int, str] = {}
        self.conf = float(conf)
        self.half = bool(half)
        requested_imgsz = int(imgsz) if int(imgsz) > 0 else None
        if self.using_tensorrt:
            self.imgsz = self.trt_imgsz
            if bool(tensorrt_fp16):
                self.half = True
        else:
            self.imgsz = requested_imgsz
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(self.model_path)
        self.is_pose = str(getattr(self.model, "task", "") or "").casefold() == "pose"
        if self.using_tensorrt:
            self._setup_trt_runtime(p)

    def _setup_trt_runtime(self, source_pt: Path) -> None:
        """TRT engine без metadata: end2end, kpt_shape, names из исходного .pt."""
        kpt_shape = (17, 3)
        try:
            from ultralytics import YOLO

            src = YOLO(str(source_pt))
            self._trt_names = {int(k): str(v) for k, v in dict(src.names).items()}
            if self.is_pose:
                head = src.model.model[-1] if hasattr(src.model, "model") else None
                if head is not None and hasattr(head, "kpt_shape"):
                    kpt_shape = tuple(int(x) for x in head.kpt_shape)
        except Exception:
            self._trt_names = {0: "person"} if self.is_pose else {0: "object"}

        def _on_predict_start(predictor) -> None:
            backend = predictor.model
            backend.end2end = True
            # Ultralytics TRT без metadata отдаёт class0..N — промпт person/helmet не матчится.
            backend.names = self._trt_names
            if self.is_pose:
                backend.kpt_shape = kpt_shape

        self.model.add_callback("on_predict_start", _on_predict_start)

    @property
    def model_task(self) -> str:
        return str(getattr(self.model, "task", "detect") or "detect")

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

    def _pad_trt_batch(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Fixed-batch TRT engine: дополнить до trt_max_batch (повтор последнего кадра)."""
        n = len(frames)
        cap = self.trt_max_batch
        if n <= 0 or cap <= 0:
            return frames, n
        if n >= cap:
            return frames[:cap], cap
        return frames + [frames[-1]] * (cap - n), n

    def track(self, frame_bgr: np.ndarray) -> list[DetectItem]:
        if self.using_tensorrt:
            rows = self.track_batch([frame_bgr])
            return rows[0] if rows else []
        results = self.model.track(source=frame_bgr, persist=True, **self._yolo_kwargs())
        return self._parse_results(results)

    def track_batch(self, frames_bgr: list[np.ndarray]) -> list[list[DetectItem]]:
        """BoT-SORT track; для TRT — батчами фиксированного размера (engine batch=N)."""
        if not frames_bgr:
            return []
        if not self.using_tensorrt:
            return [self.track(f) for f in frames_bgr]

        cap = self.trt_max_batch
        out: list[list[DetectItem]] = []
        for start in range(0, len(frames_bgr), cap):
            chunk = frames_bgr[start : start + cap]
            padded, n_real = self._pad_trt_batch(chunk)
            results = self.model.track(
                source=padded,
                persist=True,
                batch=cap,
                **self._yolo_kwargs(),
            )
            if not isinstance(results, list):
                results = [results]
            out.extend(self._parse_single_result(res) for res in results[:n_real])
        return out

    def predict(self, frame_bgr: np.ndarray) -> list[DetectItem]:
        """Single-frame detect (no tracker) — for auxiliary cross-check models."""
        if self.using_tensorrt:
            rows = self.predict_batch([frame_bgr])
            return rows[0] if rows else []
        results = self.model.predict(source=frame_bgr, **self._yolo_kwargs())
        return self._parse_results(results)

    def predict_batch(
        self,
        frames_bgr: list[np.ndarray],
        *,
        max_batch: int = 0,
    ) -> list[list[DetectItem]]:
        """Batched predict; max_batch<=0 = entire list in one GPU forward."""
        if not frames_bgr:
            return []
        if self.using_tensorrt:
            cap = self.trt_max_batch
            out: list[list[DetectItem]] = []
            for start in range(0, len(frames_bgr), cap):
                chunk = frames_bgr[start : start + cap]
                padded, n_real = self._pad_trt_batch(chunk)
                results = self.model.predict(source=padded, batch=cap, **self._yolo_kwargs())
                if not isinstance(results, list):
                    results = [results]
                out.extend(self._parse_single_result(res) for res in results[:n_real])
            return out

        cap = int(max_batch)
        out: list[list[DetectItem]] = []
        for chunk in chunk_list(frames_bgr, cap):
            if len(chunk) == 1:
                out.append(self.predict(chunk[0]))
                continue
            results = self.model.predict(source=chunk, batch=len(chunk), **self._yolo_kwargs())
            if not isinstance(results, list):
                results = [results]
            out.extend(self._parse_single_result(res) for res in results)
        return out

    def reset_session(self) -> None:
        pred = getattr(self.model, "predictor", None)
        if pred is not None:
            try:
                for tr in getattr(pred, "trackers", None) or []:
                    reset_fn = getattr(tr, "reset", None)
                    if callable(reset_fn):
                        reset_fn()
            except Exception:
                pass
        if hasattr(self.model, "predictor"):
            try:
                self.model.predictor = None
            except Exception:
                pass
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _parse_results(self, results) -> list[DetectItem]:
        res = results[0] if results else None
        if res is None:
            return []
        return self._parse_single_result(res)

    def _parse_single_result(self, res) -> list[DetectItem]:
        if res is None or res.boxes is None or len(res.boxes) == 0 or res.boxes.xyxy is None:
            return []

        xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        cls = res.boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
        conf = res.boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        ids_t = res.boxes.id
        ids = (
            ids_t.detach().cpu().numpy().astype(np.int64, copy=False)
            if ids_t is not None
            else np.full(len(xyxy), -1, dtype=np.int64)
        )
        name_map = res.names if isinstance(res.names, dict) else {}
        if self.using_tensorrt and self._trt_names:
            name_map = {**self._trt_names, **name_map}

        kpts_all: np.ndarray | None = None
        if res.keypoints is not None and res.keypoints.data is not None:
            kpts_all = res.keypoints.data.detach().cpu().numpy().astype(np.float32, copy=False)

        out: list[DetectItem] = []
        for i in range(int(xyxy.shape[0])):
            label = str(name_map.get(int(cls[i]), str(int(cls[i]))))
            kpts_i: np.ndarray | None = None
            if kpts_all is not None and i < int(kpts_all.shape[0]):
                kpts_i = kpts_all[i].copy()
            out.append(
                DetectItem(
                    xyxy=xyxy[i].copy(),
                    cls_id=int(cls[i]),
                    label=label,
                    conf=float(conf[i]),
                    motion_id=int(ids[i]),
                    keypoints=kpts_i,
                )
            )
        return out
