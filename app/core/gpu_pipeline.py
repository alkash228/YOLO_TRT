"""Dedicated GPU worker + batched YOLO inference (pipeline steps 2–6)."""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from app.config.settings import PipelineSettings
from app.core.batch_prepare import PreparedFrame, prepare_batch_frames
from app.core.batch_utils import effective_speed_tuning, gpu_predict_chunk_size, resolve_reid_embed_chunk
from app.core.detect_engine import DetectEngine, DetectItem
from app.core.reid_engine import ReidEngine
from app.core.seg_engine import SegEngine, SegItem

_STOP = object()


@dataclass(slots=True)
class FrameBatchJob:
    frame_indices: list[int]
    frames_bgr: list[np.ndarray]


@dataclass(slots=True)
class GpuBatchResult:
    frame_indices: list[int]
    detections: list[list[DetectItem]]
    segments: list[list[SegItem]]
    cross_detections: list[list[DetectItem]]
    detect_ms: float
    seg_ms: float
    cross_ms: float
    # Заполняется в GPU-потоке при ReID — CPU только tracker/fusion/post.
    prepared_frames: list[PreparedFrame] | None = None
    reid_embeddings: np.ndarray | None = None
    reid_crop_map: list[tuple[int, int]] | None = None
    reid_ms: float = 0.0


@dataclass(slots=True)
class ReidEmbedJob:
    result: GpuBatchResult
    frames_bgr: list[np.ndarray]


class ReidEmbedWorker:
    """
    Отдельный CUDA-поток для OSNet embed.
    Пока ReID считает батч N, GpuInferWorker уже делает detect/seg на батче N+1.
    """

    def __init__(
        self,
        reid_engine: ReidEngine,
        settings: PipelineSettings,
        prompt_terms: list[str],
        *,
        queue_depth: int = 3,
    ) -> None:
        self._reid = reid_engine
        self._settings = settings
        self._terms = list(prompt_terms)
        depth = max(1, int(queue_depth))
        # Без maxsize: иначе main блокируется на submit, пока не poll() — deadlock.
        self._in_q: queue.Queue[ReidEmbedJob | object] = queue.Queue()
        self._out_q: queue.Queue[tuple[GpuBatchResult, list[np.ndarray]] | BaseException] = (
            queue.Queue()
        )
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="yolo-drt-reid-gpu", daemon=True)
        self._cuda_stream = None
        try:
            import torch

            if torch.cuda.is_available():
                self._cuda_stream = torch.cuda.Stream()
        except Exception:
            self._cuda_stream = None

    def start(self) -> None:
        self._thread.start()

    def submit(self, result: GpuBatchResult, frames_bgr: list[np.ndarray]) -> None:
        if self._error is not None:
            raise self._error
        self._in_q.put(ReidEmbedJob(result=result, frames_bgr=frames_bgr))

    def poll(self) -> list[tuple[GpuBatchResult, list[np.ndarray]]]:
        if self._error is not None:
            raise self._error
        out: list[tuple[GpuBatchResult, list[np.ndarray]]] = []
        while True:
            try:
                item = self._out_q.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, BaseException):
                self._error = item
                raise item
            out.append(item)
        return out

    def get_result(self, timeout: float | None = None) -> tuple[GpuBatchResult, list[np.ndarray]]:
        if self._error is not None:
            raise self._error
        item = self._out_q.get(timeout=timeout)
        if isinstance(item, BaseException):
            self._error = item
            raise item
        return item

    def close(self) -> None:
        self._in_q.put(_STOP)
        self._thread.join(timeout=600.0)
        if self._cuda_stream is not None:
            try:
                self._cuda_stream.synchronize()
            except Exception:
                pass
        if self._error is not None:
            raise self._error

    def drain(self) -> list[tuple[GpuBatchResult, list[np.ndarray]]]:
        batches = self.poll()
        if not batches:
            return []
        return batches

    def finish(self) -> list[tuple[GpuBatchResult, list[np.ndarray]]]:
        self.close()
        batches: list[tuple[GpuBatchResult, list[np.ndarray]]] = []
        while True:
            try:
                item = self._out_q.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, BaseException):
                self._error = item
                raise item
            batches.append(item)
        return batches

    def _embed_crops(self, all_crops: list[np.ndarray]) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        trt_cap = self._reid.trt_max_batch if getattr(self._reid, "using_tensorrt", False) else 0
        chunk = resolve_reid_embed_chunk(
            self._settings.reid_embed_chunk,
            len(all_crops),
            trt_max_batch=trt_cap,
        )
        stream = self._cuda_stream
        if chunk <= 0:
            embeddings = self._reid.embed_batch(all_crops, cuda_stream=stream)
        else:
            emb_parts: list[np.ndarray] = []
            for i in range(0, len(all_crops), chunk):
                emb_parts.append(
                    self._reid.embed_batch(all_crops[i : i + chunk], cuda_stream=stream)
                )
            embeddings = (
                np.concatenate(emb_parts, axis=0)
                if emb_parts
                else np.zeros((0, 512), dtype=np.float32)
            )
        if embeddings.shape[0] != len(all_crops):
            raise RuntimeError(
                f"ReID embed count mismatch: got {embeddings.shape[0]} for {len(all_crops)} crops"
            )
        return embeddings, (time.perf_counter() - t0) * 1000.0

    def _run(self) -> None:
        try:
            import torch
            from contextlib import nullcontext

            stream_ctx = (
                torch.cuda.stream(self._cuda_stream)
                if self._cuda_stream is not None
                else nullcontext()
            )
            while True:
                item = self._in_q.get()
                if item is _STOP:
                    break
                assert isinstance(item, ReidEmbedJob)
                result = item.result
                frames = item.frames_bgr
                prepared, all_crops, crop_map = prepare_batch_frames(
                    detections=result.detections,
                    segments=result.segments,
                    cross_detections=result.cross_detections,
                    frames_bgr=frames,
                    terms=self._terms,
                    settings=self._settings,
                    motion_tracker=None,
                    use_motion_tracker=False,
                )
                reid_embeddings: np.ndarray | None = None
                reid_ms = 0.0
                if all_crops:
                    with stream_ctx:
                        reid_embeddings, reid_ms = self._embed_crops(all_crops)
                    if self._cuda_stream is not None:
                        self._cuda_stream.synchronize()
                enriched = GpuBatchResult(
                    frame_indices=list(result.frame_indices),
                    detections=result.detections,
                    segments=result.segments,
                    cross_detections=result.cross_detections,
                    detect_ms=result.detect_ms,
                    seg_ms=result.seg_ms,
                    cross_ms=result.cross_ms,
                    prepared_frames=prepared,
                    reid_embeddings=reid_embeddings,
                    reid_crop_map=crop_map if all_crops else None,
                    reid_ms=reid_ms,
                )
                self._out_q.put((enriched, frames))
        except BaseException as exc:
            self._error = exc
            self._out_q.put(exc)


def chunk_frame_jobs(
    frames: list[np.ndarray],
    *,
    batch_size: int,
    frame_indices: list[int] | None = None,
) -> list[FrameBatchJob]:
    """Split frames into inference batches; frame_indices = source video indices."""
    bs = max(1, int(batch_size))
    if frame_indices is None:
        frame_indices = list(range(len(frames)))
    if len(frames) != len(frame_indices):
        raise ValueError("frames and frame_indices length mismatch")
    jobs: list[FrameBatchJob] = []
    for start in range(0, len(frames), bs):
        chunk = frames[start : start + bs]
        indices = frame_indices[start : start + len(chunk)]
        jobs.append(FrameBatchJob(frame_indices=indices, frames_bgr=chunk))
    return jobs


class MemoryBatchJobs(Iterator[FrameBatchJob]):
    """Background thread: slice preloaded frames into batch jobs (bounded queue)."""

    def __init__(
        self,
        *,
        all_frames: list[np.ndarray],
        process_indices: list[int],
        batch_size: int,
        queue_depth: int = 4,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> None:
        self._all_frames = all_frames
        self._process_indices = list(process_indices)
        self._batch_size = max(1, int(batch_size))
        self._should_stop = should_stop
        self._should_pause = should_pause
        self._done = object()
        self._error: BaseException | None = None
        depth = max(1, int(queue_depth))
        self._q: queue.Queue[FrameBatchJob | object] = queue.Queue(maxsize=depth)
        self._thread = threading.Thread(
            target=self._run, name="yolo-drt-batch-memory", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._thread.join(timeout=600.0)
        if self._error is not None:
            raise self._error

    def __iter__(self) -> MemoryBatchJobs:
        return self

    def __next__(self) -> FrameBatchJob:
        if self._error is not None:
            raise self._error
        item = self._q.get()
        if item is self._done:
            if self._error is not None:
                raise self._error
            raise StopIteration
        assert isinstance(item, FrameBatchJob)
        return item

    def _run(self) -> None:
        try:
            bs = self._batch_size
            indices = self._process_indices
            for start in range(0, len(indices), bs):
                if self._should_stop and self._should_stop():
                    break
                while self._should_pause and self._should_pause():
                    time.sleep(0.05)
                    if self._should_stop and self._should_stop():
                        break
                chunk_idx = indices[start : start + bs]
                frames = [self._all_frames[i] for i in chunk_idx]
                self._q.put(FrameBatchJob(frame_indices=chunk_idx, frames_bgr=frames))
        except BaseException as exc:
            self._error = exc
        finally:
            self._q.put(self._done)


class PrefetchBatchJobs(Iterator[FrameBatchJob]):
    """Background thread: read video and assemble batch jobs ahead of GPU consumption."""

    def __init__(
        self,
        *,
        cap: cv2.VideoCapture,
        n_frames_total: int,
        batch_size: int,
        queue_depth: int = 4,
        frame_stride: int = 1,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> None:
        self._cap = cap
        self._n_frames_total = int(n_frames_total)
        self._batch_size = max(1, int(batch_size))
        self._frame_stride = max(1, int(frame_stride))
        self._should_stop = should_stop
        self._should_pause = should_pause
        self._done = object()
        self._error: BaseException | None = None
        # queue_depth — legacy param; очередь без лимита (см. комментарий выше).
        self._q: queue.Queue[FrameBatchJob | object] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="yolo-drt-batch-prefetch", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._thread.join(timeout=600.0)
        if self._error is not None:
            raise self._error

    def __iter__(self) -> PrefetchBatchJobs:
        return self

    def __next__(self) -> FrameBatchJob:
        if self._error is not None:
            raise self._error
        item = self._q.get()
        if item is self._done:
            if self._error is not None:
                raise self._error
            raise StopIteration
        assert isinstance(item, FrameBatchJob)
        return item

    def _run(self) -> None:
        try:
            bs = self._batch_size
            buf_frames: list[np.ndarray] = []
            buf_indices: list[int] = []
            idx = 0
            while True:
                if self._n_frames_total > 0 and idx >= self._n_frames_total:
                    break
                if self._should_stop and self._should_stop():
                    break
                while self._should_pause and self._should_pause():
                    time.sleep(0.05)
                    if self._should_stop and self._should_stop():
                        break
                ok, frame = self._cap.read()
                if not ok:
                    break
                take = idx % self._frame_stride == 0
                idx += 1
                if not take:
                    continue
                buf_frames.append(frame)
                buf_indices.append(idx - 1)
                if len(buf_frames) >= bs:
                    self._q.put(FrameBatchJob(frame_indices=buf_indices, frames_bgr=buf_frames))
                    buf_frames = []
                    buf_indices = []
            if buf_frames:
                self._q.put(FrameBatchJob(frame_indices=buf_indices, frames_bgr=buf_frames))
        except BaseException as exc:
            self._error = exc
        finally:
            self._q.put(self._done)


def make_async_batch_jobs(
    *,
    all_frames: list[np.ndarray] | None,
    cap: cv2.VideoCapture | None,
    n_frames_total: int,
    process_indices: list[int],
    batch_size: int,
    queue_depth: int,
    frame_stride: int = 1,
    should_stop: Callable[[], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> Iterator[FrameBatchJob]:
    """Bounded async batch source: video decode or RAM slice, queue_depth slots."""
    depth = max(1, int(queue_depth))
    bs = max(1, int(batch_size))
    if all_frames is not None:
        feeder = MemoryBatchJobs(
            all_frames=all_frames,
            process_indices=process_indices,
            batch_size=bs,
            queue_depth=depth,
            should_stop=should_stop,
            should_pause=should_pause,
        )
        feeder.start()
        return feeder
    if cap is not None:
        feeder = PrefetchBatchJobs(
            cap=cap,
            n_frames_total=int(n_frames_total),
            batch_size=bs,
            queue_depth=depth,
            frame_stride=max(1, int(frame_stride)),
            should_stop=should_stop,
            should_pause=should_pause,
        )
        feeder.start()
        return feeder
    return iter(())


class GpuInferWorker:
    """
    Single CUDA thread: detect (+batch) → seg (+batch) → cross-check (+batch).
    Overlaps with CPU post via bounded input queue (step 2).
    """

    def __init__(
        self,
        detect_engine: DetectEngine,
        seg_engine: SegEngine | None,
        cross_engine: DetectEngine | None,
        settings: PipelineSettings,
    ) -> None:
        self._detect = detect_engine
        self._seg = seg_engine
        self._cross = cross_engine
        self._settings = settings
        self._use_seg = bool(settings.use_seg and seg_engine is not None)
        self._use_cross = bool(settings.cross_check_enabled and cross_engine is not None)
        # Stable identity (SAM or OSNet) needs BoT-SORT track; predict_batch alone jumps IDs.
        from app.core.sam_memory_tracker import uses_stable_identity

        if uses_stable_identity(settings):
            self._detect_use_track = True
        elif settings.gpu_full_batch:
            self._detect_use_track = False
        else:
            self._detect_use_track = not bool(settings.use_batch_detect)
        depth = max(1, int(settings.gpu_queue_depth))
        self._in_q: queue.Queue[FrameBatchJob | object] = queue.Queue()
        self._out_q: queue.Queue[GpuBatchResult | BaseException] = queue.Queue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="yolo-drt-gpu", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, job: FrameBatchJob) -> None:
        if self._error is not None:
            raise self._error
        self._in_q.put(job)

    def get_result(self, timeout: float | None = None) -> GpuBatchResult:
        if self._error is not None:
            raise self._error
        item = self._out_q.get(timeout=timeout)
        if isinstance(item, BaseException):
            self._error = item
            raise item
        return item

    def close(self) -> None:
        self._in_q.put(_STOP)
        self._thread.join(timeout=600.0)
        if self._error is not None:
            raise self._error

    def _loop(self) -> None:
        try:
            while True:
                job = self._in_q.get()
                if job is _STOP:
                    break
                assert isinstance(job, FrameBatchJob)
                result = self._infer_batch(job)
                self._out_q.put(result)
        except BaseException as exc:
            self._error = exc
            self._out_q.put(exc)

    def _infer_batch(self, job: FrameBatchJob) -> GpuBatchResult:
        frames = job.frames_bgr
        n = len(frames)
        if n == 0:
            return GpuBatchResult(
                frame_indices=[],
                detections=[],
                segments=[],
                cross_detections=[],
                detect_ms=0.0,
                seg_ms=0.0,
                cross_ms=0.0,
                prepared_frames=None,
                reid_embeddings=None,
                reid_crop_map=None,
                reid_ms=0.0,
            )

        t_det = time.perf_counter()
        max_batch = gpu_predict_chunk_size(
            n,
            gpu_full_batch=bool(self._settings.gpu_full_batch),
            max_cap=self._settings.max_infer_batch_size,
        )
        if self._detect_use_track:
            t_det = time.perf_counter()
            if getattr(self._detect, "using_tensorrt", False):
                detections = self._detect.track_batch(frames)
            else:
                detections = []
                for frame in frames:
                    detections.append(self._detect.track(frame))
            detect_ms = (time.perf_counter() - t_det) * 1000.0
        else:
            trt_cap = (
                self._detect.trt_max_batch
                if getattr(self._detect, "using_tensorrt", False)
                else max_batch
            )
            detections = self._detect.predict_batch(frames, max_batch=trt_cap or max_batch)
            detect_ms = (time.perf_counter() - t_det) * 1000.0

        segments: list[list[SegItem]] = [[] for _ in range(n)]
        seg_ms = 0.0
        if self._use_seg and self._seg is not None:
            t_seg = time.perf_counter()
            stride = max(1, int(effective_speed_tuning(self._settings)["seg_stride"]))
            if stride <= 1:
                segments = self._seg.predict_batch(frames, max_batch=max_batch)
            else:
                segments = [[] for _ in range(n)]
                idxs = [i for i in range(n) if i % stride == 0]
                if idxs:
                    sub_frames = [frames[i] for i in idxs]
                    sub_segs = self._seg.predict_batch(sub_frames, max_batch=max_batch)
                    for j, fi in enumerate(idxs):
                        segments[fi] = sub_segs[j]
            seg_ms = (time.perf_counter() - t_seg) * 1000.0

        cross_detections: list[list[DetectItem]] = [[] for _ in range(n)]
        cross_ms = 0.0
        if self._use_cross and self._cross is not None:
            t_cross = time.perf_counter()
            cross_cap = (
                self._cross.trt_max_batch
                if getattr(self._cross, "using_tensorrt", False)
                else max_batch
            )
            cross_detections = self._cross.predict_batch(frames, max_batch=cross_cap or max_batch)
            cross_ms = (time.perf_counter() - t_cross) * 1000.0

        return GpuBatchResult(
            frame_indices=list(job.frame_indices),
            detections=detections,
            segments=segments,
            cross_detections=cross_detections,
            detect_ms=detect_ms,
            seg_ms=seg_ms,
            cross_ms=cross_ms,
            prepared_frames=None,
            reid_embeddings=None,
            reid_crop_map=None,
            reid_ms=0.0,
        )
