"""OSNet AIN x1.0 ReID feature extractor."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch


class ReidEngine:
    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        use_amp: bool = True,
        image_size: tuple[int, int] = (256, 128),
        use_tensorrt: bool = False,
        tensorrt_fp16: bool = True,
        tensorrt_engine_strategy: str = "central",
        tensorrt_central_dir: Path | None = None,
    ) -> None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"ReID model not found: {p}")
        self.model_path = str(p)
        self.use_amp = bool(use_amp)
        self.image_size = image_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.using_tensorrt = False
        self.trt_max_batch = 0
        self._trt = None
        self.extractor = None

        if use_tensorrt:
            from app.core.trt_paths import resolve_reid_engine

            eng = resolve_reid_engine(
                p,
                fp16=bool(tensorrt_fp16),
                strategy=tensorrt_engine_strategy,
                central_dir=tensorrt_central_dir,
            )
            if eng.exists():
                try:
                    from app.core.trt_runtime import ReidTrtRunner

                    self._trt = ReidTrtRunner(eng, device=device)
                    self.using_tensorrt = True
                    self.trt_max_batch = int(self._trt.max_batch)
                    return
                except Exception:
                    self._trt = None

        try:
            from torchreid.utils import FeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "torchreid not installed. Run: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            ) from exc

        self.extractor = FeatureExtractor(
            model_name="osnet_ain_x1_0",
            model_path=self.model_path,
            device=self.device,
            image_size=self.image_size,
        )

    @staticmethod
    def crop_from_bbox(frame_bgr: np.ndarray, xyxy: np.ndarray, pad: float = 0.05) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = [float(v) for v in xyxy.tolist()]
        bw = x1 - x0
        bh = y1 - y0
        x0 = max(0.0, x0 - pad * bw)
        y0 = max(0.0, y0 - pad * bh)
        x1 = min(float(w - 1), x1 + pad * bw)
        y1 = min(float(h - 1), y1 + pad * bh)
        xa, ya, xb, yb = int(x0), int(y0), int(x1), int(y1)
        if xb <= xa or yb <= ya:
            return None
        return frame_bgr[ya:yb, xa:xb].copy()

    def embed_batch(self, crops_bgr: list[np.ndarray], *, cuda_stream=None) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        if self._trt is not None:
            return self._trt.embed_batch(crops_bgr, cuda_stream=cuda_stream)
        rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
        ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.use_amp and self.device.startswith("cuda")
            else nullcontext()
        )
        with ctx:
            feats = self.extractor(rgb_crops)
        if isinstance(feats, torch.Tensor):
            arr = feats.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(feats, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return (arr / norms).astype(np.float32, copy=False)
