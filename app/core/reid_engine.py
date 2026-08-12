"""OSNet AIN x1.0 ReID feature extractor."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

_log = logging.getLogger(__name__)

# COCO-17 torso anchors for ReID crop (shoulders + hips).
_TORSO_KPT_IDS: tuple[int, ...] = (5, 6, 11, 12)
# Min crop size after pad (OSNet is 256x128; tiny crops are noise).
_MIN_CROP_H = 48
_MIN_CROP_W = 24
# Reject extreme aspect ratios (sitting side-view / bad boxes).
_MIN_ASPECT = 0.35  # h/w
_MAX_ASPECT = 5.0
# Fraction of bbox that may touch the frame edge before reject (0 = any edge touch ok).
_EDGE_MARGIN_PX = 2


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
        self.source_model_path = str(p)
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
            import inspect

            from app.core.trt_paths import resolve_reid_engine

            resolve_kw: dict = {"fp16": bool(tensorrt_fp16)}
            sig = inspect.signature(resolve_reid_engine)
            if "strategy" in sig.parameters:
                resolve_kw["strategy"] = str(tensorrt_engine_strategy or "central")
                resolve_kw["central_dir"] = tensorrt_central_dir
            eng = resolve_reid_engine(p, **resolve_kw)
            if eng.exists():
                try:
                    from app.core.trt_runtime import ReidTrtRunner

                    self._trt = ReidTrtRunner(eng, device=device)
                    self.using_tensorrt = True
                    self.model_path = str(eng)
                    self.trt_max_batch = int(self._trt.max_batch)
                    return
                except Exception as exc:
                    self._trt = None
                    _log.warning(
                        "ReID TensorRT load failed (%s); falling back to PyTorch: %s",
                        eng,
                        exc,
                    )
            else:
                _log.warning(
                    "ReID TensorRT engine not found at %s; falling back to PyTorch",
                    eng,
                )

        try:
            from torchreid.utils import FeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "torchreid not installed. Run: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            ) from exc

        self.extractor = FeatureExtractor(
            model_name="osnet_ain_x1_0",
            model_path=self.source_model_path,
            device=self.device,
            image_size=self.image_size,
        )

    @staticmethod
    def torso_bbox_from_keypoints(
        keypoints: np.ndarray | None,
        person_xyxy: np.ndarray,
        *,
        kpt_conf: float = 0.25,
        pad: float = 0.20,
        min_valid: int = 3,
    ) -> np.ndarray | None:
        """
        Build a torso box from COCO shoulders+hips when confidence is good.
        Returns None when pose is too weak (caller should use full padded bbox).
        """
        if (
            keypoints is None
            or not isinstance(keypoints, np.ndarray)
            or keypoints.ndim != 2
            or keypoints.shape[1] < 3
        ):
            return None
        xs: list[float] = []
        ys: list[float] = []
        for idx in _TORSO_KPT_IDS:
            if idx >= int(keypoints.shape[0]):
                continue
            x, y, c = float(keypoints[idx, 0]), float(keypoints[idx, 1]), float(keypoints[idx, 2])
            if c >= kpt_conf:
                xs.append(x)
                ys.append(y)
        if len(xs) < min_valid:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bw = max(8.0, x1 - x0)
        bh = max(8.0, y1 - y0)
        # Expand slightly; clamp inside person bbox so we stay on the subject.
        x0 -= pad * bw
        x1 += pad * bw
        y0 -= pad * bh
        y1 += pad * bh
        px0, py0, px1, py1 = [float(v) for v in person_xyxy.tolist()]
        x0 = max(px0, x0)
        y0 = max(py0, y0)
        x1 = min(px1, x1)
        y1 = min(py1, y1)
        if x1 - x0 < 8.0 or y1 - y0 < 8.0:
            return None
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    @staticmethod
    def crop_quality_ok(
        frame_h: int,
        frame_w: int,
        xyxy: np.ndarray,
        crop: np.ndarray | None,
        *,
        min_h: int = _MIN_CROP_H,
        min_w: int = _MIN_CROP_W,
        reject_edge: bool = False,
    ) -> bool:
        """Reject crops that would poison the ReID gallery (tiny / extreme / edge)."""
        if crop is None or crop.size == 0:
            return False
        ch, cw = int(crop.shape[0]), int(crop.shape[1])
        if ch < min_h or cw < min_w:
            return False
        aspect = ch / max(1.0, float(cw))
        if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
            return False
        if reject_edge:
            x0, y0, x1, y1 = [float(v) for v in xyxy.tolist()]
            m = float(_EDGE_MARGIN_PX)
            if x0 <= m or y0 <= m or x1 >= float(frame_w) - 1.0 - m or y1 >= float(frame_h) - 1.0 - m:
                return False
        return True

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

    @classmethod
    def crop_for_reid(
        cls,
        frame_bgr: np.ndarray,
        xyxy: np.ndarray,
        keypoints: np.ndarray | None = None,
        *,
        pad: float = 0.05,
        kpt_conf: float = 0.25,
        use_torso: bool = True,
        reject_edge: bool = False,
        torso_pad: float = 0.20,
        torso_min_area_ratio: float = 0.12,
    ) -> np.ndarray | None:
        """
        Prefer pose torso crop when shoulders/hips are confident and large enough;
        otherwise padded full-body bbox (better discrimination for same-PPE workers).
        Returns None when the crop fails the bad-crop filter (caller skips embed / gallery).
        """
        h, w = frame_bgr.shape[:2]
        box = xyxy
        if use_torso:
            torso = cls.torso_bbox_from_keypoints(
                keypoints, xyxy, kpt_conf=kpt_conf, pad=torso_pad
            )
            if torso is not None:
                px0, py0, px1, py1 = [float(v) for v in xyxy.tolist()]
                person_area = max(1.0, (px1 - px0) * (py1 - py0))
                tw = float(torso[2] - torso[0])
                th = float(torso[3] - torso[1])
                torso_area = max(1.0, tw * th)
                # Tiny torso relative to person → vest/helmet blob; use full body.
                if (
                    torso_area >= float(torso_min_area_ratio) * person_area
                    and th >= float(_MIN_CROP_H) * 0.75
                ):
                    box = torso
        crop = cls.crop_from_bbox(frame_bgr, box, pad=pad)
        if not cls.crop_quality_ok(h, w, box, crop, reject_edge=reject_edge):
            return None
        return crop

    @property
    def feat_dim(self) -> int:
        return 512

    def embed_batch(self, crops_bgr: list[np.ndarray], *, cuda_stream=None) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, int(self.feat_dim)), dtype=np.float32)
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


def _backend_from_weight_name(model_path: str | Path) -> str | None:
    name = Path(model_path).name.casefold()
    if "solider" in name or "swin_small" in name or "swin_base" in name or "swin_tiny" in name:
        return "solider"
    if "osnet" in name:
        return "osnet"
    return None


def resolve_reid_backend(backend: str | None, model_path: str | Path) -> str:
    """
    Return ``osnet`` or ``solider``.

    Explicit ``reid_backend`` wins when weights path is ambiguous. If the filename
    clearly disagrees (e.g. settings=solider but path is OSNet .pth), filename wins
    unless we can retarget weights in ``resolve_reid_model_path``.
    """
    b = (backend or "auto").strip().casefold()
    from_path = _backend_from_weight_name(model_path)
    if b == "auto" or b not in ("osnet", "solider"):
        return from_path or "osnet"
    return b


def resolve_reid_model_path(backend: str | None, model_path: str | Path) -> Path:
    """
    Align weight file with backend. Stale ui_settings often keep OSNet path while
    ``reid_backend=solider`` — that must not load OSNet bytes into SoliderReidEngine.
    """
    from app.config.settings import OSNET_FILENAME, RD_DIR, SOLIDER_FILENAME

    p = Path(model_path)
    kind = resolve_reid_backend(backend, p)
    name = p.name.casefold()
    if kind == "solider":
        if (not p.is_file()) or ("osnet" in name):
            alt = RD_DIR / SOLIDER_FILENAME
            if alt.is_file():
                return alt
        return p
    if kind == "osnet":
        if (not p.is_file()) or ("solider" in name) or ("swin_" in name):
            alt = RD_DIR / OSNET_FILENAME
            if alt.is_file():
                return alt
        return p
    return p


def create_reid_engine(
    model_path: str | Path,
    *,
    backend: str | None = None,
    device: str | None = None,
    use_amp: bool = True,
    use_tensorrt: bool = False,
    tensorrt_fp16: bool = True,
    **kwargs,
):
    """
    Factory for Pass2 / live ReID.
    SOLIDER is always PyTorch-only (TRT path stays OSNet-specific).
    """
    path = resolve_reid_model_path(backend, model_path)
    kind = resolve_reid_backend(backend, path)
    # If settings say solider but we still only have an OSNet file, do not pretend.
    if kind == "solider" and "osnet" in path.name.casefold():
        kind = "osnet"
    if Path(model_path).resolve() != path.resolve():
        _log.warning(
            "ReID weights retargeted for backend=%s: %s -> %s",
            kind,
            Path(model_path).name,
            path.name,
        )
    if kind == "solider":
        from app.core.solider_reid_engine import SoliderReidEngine

        return SoliderReidEngine(
            path,
            device=device,
            use_amp=use_amp,
        )
    return ReidEngine(
        path,
        device=device,
        use_amp=use_amp,
        use_tensorrt=bool(use_tensorrt),
        tensorrt_fp16=bool(tensorrt_fp16),
        **kwargs,
    )
