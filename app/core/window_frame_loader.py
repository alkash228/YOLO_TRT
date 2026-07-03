"""Windowed video decode: bounded RAM preload with optional decode-ahead."""
from __future__ import annotations

import gc
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.gpu_pipeline import FrameBatchJob, chunk_frame_jobs

_STOP = object()


@dataclass(slots=True)
class VideoWindow:
    """Infer-ready window: frames_bgr[i] ↔ frame_indices[i] (global source idx)."""

    start_frame: int
    end_frame: int
    frame_indices: list[int]
    frames_bgr: list[np.ndarray]

    @property
    def count(self) -> int:
        return len(self.frames_bgr)


@dataclass(slots=True)
class PendingWindowSpill:
    """Window metadata kept until all GPU/finalizer jobs for the window complete."""

    window_num: int
    start_frame: int
    end_frame: int
    infer_count: int


def load_video_window(
    input_path: str,
    start_frame: int,
    infer_target: int,
    *,
    frame_stride: int = 1,
    total_frames: int = 0,
) -> VideoWindow:
    """
    Decode until infer_target inference-кадров или конец ролика.
    При stride>1 в RAM только кадры для infer (без «лишних» между stride).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    start = max(0, int(start_frame))
    target = max(1, int(infer_target))
    stride = max(1, int(frame_stride))
    total = max(0, int(total_frames))

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames: list[np.ndarray] = []
    indices: list[int] = []
    src_idx = start
    try:
        while len(frames) < target:
            if total > 0 and src_idx >= total:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx % stride == 0:
                indices.append(src_idx)
                frames.append(frame)
            src_idx += 1
    finally:
        cap.release()

    return VideoWindow(
        start_frame=start,
        end_frame=src_idx,
        frame_indices=indices,
        frames_bgr=frames,
    )


class WindowPrefetcher:
    """
    Background decode of the next window while GPU processes the current one.
    infer_per_window — ровно N×job_batch infer-кадров (без хвоста), кроме финала ролика.
    """

    def __init__(
        self,
        input_path: str,
        *,
        infer_per_window: int,
        frame_stride: int,
        total_frames: int,
        windows_ahead: int = 1,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> None:
        self._input_path = input_path
        self._infer_per_window = max(1, int(infer_per_window))
        self._frame_stride = max(1, int(frame_stride))
        self._total_frames = max(0, int(total_frames))
        self._should_stop = should_stop
        self._should_pause = should_pause
        self._next_start = 0
        self._done = False
        self._error: BaseException | None = None
        depth = max(1, int(windows_ahead) + 1)
        self._q: queue.Queue[VideoWindow | object] = queue.Queue(maxsize=depth)
        self._thread = threading.Thread(
            target=self._run,
            name="yolo-drt-window-prefetch",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    @property
    def is_done(self) -> bool:
        return self._done

    def close(self) -> None:
        self._send_stop(force=True)
        self._thread.join(timeout=600.0)
        if self._error is not None:
            raise self._error

    def _send_stop(self, *, force: bool = False) -> None:
        """Ensure EOF sentinel is queued; never block forever on a full bounded queue."""
        if not force and self._done:
            return
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                self._q.put(_STOP, timeout=0.05)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    time.sleep(0.001)
        try:
            self._q.put_nowait(_STOP)
        except queue.Full:
            pass

    def next_window(self) -> VideoWindow | None:
        if self._error is not None:
            raise self._error
        # Prefetcher finished but _STOP not yet visible — don't block the GPU loop forever.
        if self._done:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return None
        else:
            item = self._q.get()
        if item is _STOP:
            return None
        assert isinstance(item, VideoWindow)
        return item

    def _run(self) -> None:
        try:
            while self._total_frames <= 0 or self._next_start < self._total_frames:
                if self._should_stop and self._should_stop():
                    break
                while self._should_pause and self._should_pause():
                    time.sleep(0.05)
                    if self._should_stop and self._should_stop():
                        break
                if self._total_frames > 0 and self._next_start >= self._total_frames:
                    break
                window = load_video_window(
                    self._input_path,
                    self._next_start,
                    self._infer_per_window,
                    frame_stride=self._frame_stride,
                    total_frames=self._total_frames,
                )
                if not window.frames_bgr:
                    break
                self._q.put(window)
                self._next_start = window.end_frame
                if window.count < self._infer_per_window:
                    break
        except BaseException as exc:
            self._error = exc
        finally:
            self._done = True
            self._send_stop(force=True)


class WindowedBatchJobs(Iterator[FrameBatchJob]):
    """
    Один непрерывный поток batch-job'ов через все окна.
    GPU worker стартует один раз на весь ролик, не на каждое окно.
    """

    def __init__(
        self,
        prefetcher: WindowPrefetcher,
        *,
        batch_size: int,
        on_window_start: Callable[[int, VideoWindow], None] | None = None,
        on_window_end: Callable[[int, VideoWindow], None] | None = None,
    ) -> None:
        self._prefetcher = prefetcher
        self._batch_size = max(1, int(batch_size))
        self._on_window_start = on_window_start
        self._on_window_end = on_window_end
        self._window_num = 0
        self._current_window: VideoWindow | None = None
        self._job_iter: Iterator[FrameBatchJob] | None = None
        self._pending_spill: PendingWindowSpill | None = None

    @property
    def has_pending_spill(self) -> bool:
        return self._pending_spill is not None

    @property
    def pending_spill_info(self) -> PendingWindowSpill | None:
        return self._pending_spill

    def try_finish_pending_window(self, *, ready: bool) -> bool:
        """Spill window after all in-flight GPU/finalizer work for it is done."""
        if not ready or self._pending_spill is None or self._on_window_end is None:
            return False
        info = self._pending_spill
        self._pending_spill = None
        window = VideoWindow(
            start_frame=info.start_frame,
            end_frame=info.end_frame,
            frame_indices=[],
            frames_bgr=[],
        )
        self._on_window_end(info.window_num, window)
        return True

    def force_finish_pending_window(self) -> bool:
        return self.try_finish_pending_window(ready=True)

    def __iter__(self) -> WindowedBatchJobs:
        return self

    def __next__(self) -> FrameBatchJob:
        while True:
            if self._job_iter is not None:
                try:
                    return next(self._job_iter)
                except StopIteration:
                    if self._current_window is not None:
                        self._pending_spill = PendingWindowSpill(
                            window_num=self._window_num,
                            start_frame=int(self._current_window.start_frame),
                            end_frame=int(self._current_window.end_frame),
                            infer_count=len(self._current_window.frame_indices),
                        )
                    self._release_window()
                    self._job_iter = None
                    self._window_num += 1

            window = self._prefetcher.next_window()
            if window is None or not window.frames_bgr:
                raise StopIteration

            self._current_window = window
            if self._on_window_start is not None:
                self._on_window_start(self._window_num, window)

            win_process = list(window.frame_indices)
            if not win_process:
                self._release_window()
                self._window_num += 1
                continue

            self._job_iter = iter(
                chunk_frame_jobs(
                    window.frames_bgr,
                    batch_size=self._batch_size,
                    frame_indices=win_process,
                )
            )

    def _release_window(self) -> None:
        if self._current_window is not None:
            self._current_window.frames_bgr.clear()
            self._current_window = None
        gc.collect()

    def close(self) -> None:
        self._prefetcher.close()


def window_process_indices(
    start_frame: int,
    end_frame: int,
    frame_stride: int,
) -> list[int]:
    """Global source indices in [start, end) to run inference on."""
    step = max(1, int(frame_stride))
    start = max(0, int(start_frame))
    end = max(start, int(end_frame))
    return list(range(start, end, step))
