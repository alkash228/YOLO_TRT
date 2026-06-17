"""TensorRT inference runtime for OSNet ReID."""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_reid_crops(crops_bgr: list[np.ndarray]) -> np.ndarray:
    """NCHW float32, как torchreid (256×128)."""
    tensors: list[np.ndarray] = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        tensors.append(np.transpose(x, (2, 0, 1)))
    return np.stack(tensors, axis=0).astype(np.float32, copy=False)


class ReidTrtRunner:
    """OSNet engine через TensorRT + PyTorch CUDA buffers."""

    def __init__(self, engine_path: str | Path, device: str = "cuda") -> None:
        import tensorrt as trt
        import torch

        self._torch = torch
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger = trt.Logger(trt.Logger.ERROR)
        data = Path(engine_path).read_bytes()
        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(data)
        if self._engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        self._input_name = self._engine.get_tensor_name(0)
        self._output_name = self._engine.get_tensor_name(1)
        self._lock = threading.Lock()
        self._max_batch = self._read_max_batch()

    def _read_max_batch(self) -> int:
        try:
            _min, _opt, max_shape = self._engine.get_tensor_profile_shape(self._input_name, 0)
            return max(1, int(max_shape[0]))
        except Exception:
            return 16

    @property
    def max_batch(self) -> int:
        return self._max_batch

    def embed_batch(self, crops_bgr: list[np.ndarray], *, cuda_stream=None) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        cap = self._max_batch
        if len(crops_bgr) <= cap:
            return self._embed_once(crops_bgr, cuda_stream=cuda_stream)
        parts: list[np.ndarray] = []
        for start in range(0, len(crops_bgr), cap):
            parts.append(self._embed_once(crops_bgr[start : start + cap], cuda_stream=cuda_stream))
        return np.concatenate(parts, axis=0)

    def _embed_once(self, crops_bgr: list[np.ndarray], *, cuda_stream=None) -> np.ndarray:
        torch = self._torch
        batch = preprocess_reid_crops(crops_bgr)
        n = int(batch.shape[0])
        if n > self._max_batch:
            raise RuntimeError(
                f"ReID TRT batch {n} exceeds engine limit {self._max_batch}"
            )
        stream = cuda_stream if cuda_stream is not None else torch.cuda.current_stream()
        with self._lock:
            if not self._context.set_input_shape(self._input_name, (n, 3, 256, 128)):
                raise RuntimeError(
                    f"ReID TRT set_input_shape failed for batch={n} (max={self._max_batch})"
                )
            in_shape = tuple(self._context.get_tensor_shape(self._input_name))
            if len(in_shape) >= 1 and int(in_shape[0]) != n:
                raise RuntimeError(
                    f"ReID TRT input shape {in_shape} != requested batch {n}"
                )
            out_shape = tuple(self._context.get_tensor_shape(self._output_name))
            if len(out_shape) >= 1 and (out_shape[0] < 0 or int(out_shape[0]) != n):
                out_shape = (n,) + tuple(out_shape[1:])
            with torch.cuda.stream(stream):
                inp = torch.from_numpy(batch).to(device=self._device, dtype=torch.float32)
                out_t = torch.empty(out_shape, device=self._device, dtype=torch.float32)
                self._context.set_tensor_address(self._input_name, int(inp.data_ptr()))
                self._context.set_tensor_address(self._output_name, int(out_t.data_ptr()))
                ok = self._context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")
            stream.synchronize()
            arr = out_t.detach().cpu().numpy().astype(np.float32, copy=True)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] != n:
            raise RuntimeError(
                f"ReID TRT returned {arr.shape[0]} embeddings for {n} crops (shape={arr.shape})"
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return (arr / norms).astype(np.float32, copy=False)
