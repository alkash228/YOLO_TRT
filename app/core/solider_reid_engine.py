"""SOLIDER Swin-S ReID embedder for offline Pass2 tracklet linking."""
from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

_log = logging.getLogger(__name__)

# MSMT17 identity count used when the public Swin-S ReID checkpoint was trained.
_MSMT17_NUM_CLASSES = 1041
_DEFAULT_H = 384
_DEFAULT_W = 128
_PIXEL_MEAN = (0.5, 0.5, 0.5)
_PIXEL_STD = (0.5, 0.5, 0.5)


def _ensure_solider_on_path() -> Path:
    root = Path(__file__).resolve().parents[1] / "third_party" / "solider_reid"
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def _build_swin_small_cfg(*, semantic_weight: float = 0.2) -> SimpleNamespace:
    return SimpleNamespace(
        MODEL=SimpleNamespace(
            NAME="transformer",
            PRETRAIN_PATH="",
            PRETRAIN_CHOICE="self",
            TRANSFORMER_TYPE="swin_small_patch4_window7_224",
            DROP_PATH=0.1,
            DROP_OUT=0.0,
            ATT_DROP_RATE=0.0,
            REDUCE_FEAT_DIM=False,
            FEAT_DIM=512,
            DROPOUT_RATE=0.0,
            ID_LOSS_TYPE="softmax",
            JPM=False,
            SEMANTIC_WEIGHT=float(semantic_weight),
        ),
        INPUT=SimpleNamespace(
            SIZE_TRAIN=(_DEFAULT_H, _DEFAULT_W),
            SIZE_TEST=(_DEFAULT_H, _DEFAULT_W),
            PIXEL_MEAN=list(_PIXEL_MEAN),
            PIXEL_STD=list(_PIXEL_STD),
        ),
        TEST=SimpleNamespace(NECK_FEAT="before", FEAT_NORM="yes"),
    )


class SoliderReidEngine:
    """PyTorch-only SOLIDER Swin-S embeddings (no TensorRT)."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        use_amp: bool = True,
        image_size: tuple[int, int] = (_DEFAULT_H, _DEFAULT_W),
        semantic_weight: float = 0.2,
        num_classes: int = _MSMT17_NUM_CLASSES,
    ) -> None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"SOLIDER ReID model not found: {p}")
        self.source_model_path = str(p)
        self.model_path = str(p)
        self.use_amp = bool(use_amp)
        # (H, W) matching SOLIDER SIZE_TEST.
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.using_tensorrt = False
        self.trt_max_batch = 0
        self._trt = None
        self.extractor = None
        self.semantic_weight = float(semantic_weight)

        _ensure_solider_on_path()
        from model.make_model import make_model  # type: ignore

        cfg = _build_swin_small_cfg(semantic_weight=self.semantic_weight)
        self.model = make_model(
            cfg,
            num_class=int(num_classes),
            camera_num=0,
            view_num=0,
            semantic_weight=self.semantic_weight,
        )
        loaded = self.model.load_param(str(p))
        self.model.to(self.device)
        self.model.eval()
        self.feat_dim = int(self.model.in_planes)
        _log.info(
            "SOLIDER Swin-S loaded from %s (dim=%d, keys=%d, device=%s)",
            p.name,
            self.feat_dim,
            loaded,
            self.device,
        )

    def _preprocess(self, crops_bgr: list[np.ndarray]) -> torch.Tensor:
        h, w = self.image_size
        mean = np.asarray(_PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(_PIXEL_STD, dtype=np.float32).reshape(1, 1, 3)
        batch = np.empty((len(crops_bgr), 3, h, w), dtype=np.float32)
        for i, crop in enumerate(crops_bgr):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = resized.astype(np.float32) / 255.0
            arr = (arr - mean) / std
            batch[i] = arr.transpose(2, 0, 1)
        return torch.from_numpy(batch)

    @torch.inference_mode()
    def embed_batch(self, crops_bgr: list[np.ndarray], *, cuda_stream=None) -> np.ndarray:
        del cuda_stream  # unused; API parity with ReidEngine
        if not crops_bgr:
            return np.zeros((0, int(self.feat_dim)), dtype=np.float32)
        tensor = self._preprocess(crops_bgr).to(self.device, non_blocking=True)
        ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.use_amp and str(self.device).startswith("cuda")
            else nullcontext()
        )
        with ctx:
            feats, _ = self.model(tensor)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        arr = feats.detach().float().cpu().numpy().astype(np.float32, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return (arr / norms).astype(np.float32, copy=False)
