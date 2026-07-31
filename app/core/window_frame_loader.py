"""Windowed video decode: bounded RAM preload with optional decode-ahead."""
from __future__ import annotations

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


def load_video_window_from_cap(
    cap: cv2.VideoCapture,
    start_frame: int,
    infer_target: int,
    *,
    frame_stride: int = 1,
    total_frames: int = 0,
    current_pos: int | None = None,
) -> tuple[VideoWindow, int]:
    """
    Decode until infer_target inference frames (or EOF).

    Prefer sequential reads. CAP_PROP_POS_FRAMES seek is avoided when possible —
    on HEVC (H.265) random seeks cause «Could not find ref with POC / Error
    constructing the frame RPS» storms and broken frames.

    Returns (window, new_pos_after_read).
    """
    start = max(0, int(start_frame))
    target = max(1, int(infer_target))
    stride = max(1, int(frame_stride))
    total = max(0, int(total_frames))
    pos = int(current_pos) if current_pos is not None else start

    # Catch up by reading forward (keeps HEVC reference chain). Seek only if we
    # somehow need to go backwards (should not happen in windowed prefetch).
    if pos > start:
        # Already past requested start — reopen caller should avoid this.
        pos = start
    while pos < start:
        ok, _ = cap.read()
        if not ok:
            break
        pos += 1
    if pos < start:
        # EOF before start
        return (
            VideoWindow(start_frame=start, end_frame=pos, frame_indices=[], frames_bgr=[]),
            pos,
        )

    frames: list[np.ndarray] = []
    indices: list[int] = []
    src_idx = pos
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

    return (
        VideoWindow(
            start_frame=start,
            end_frame=src_idx,
            frame_indices=indices,
            frames_bgr=frames,
        ),
        src_idx,
    )


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

    For start_frame>0 on HEVC this still may warn if seek is used as fallback;
    WindowPrefetcher uses a persistent capture without seek.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    start = max(0, int(start_frame))
    try:
        if start > 0:
            # Last resort for one-shot loads. Prefetch path avoids this.
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            window, _ = load_video_window_from_cap(
                cap,
                start,
                infer_target,
                frame_stride=frame_stride,
                total_frames=total_frames,
                current_pos=start,
            )
        else:
            window, _ = load_video_window_from_cap(
                cap,
                0,
                infer_target,
                frame_stride=frame_stride,
                total_frames=total_frames,
                current_pos=0,
            )
        return window
    finally:
        cap.release()


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
        self.decode_sec: float = 0.0
        # +2: next window can sit in queue while GPU finishes the current one; room for _STOP.
        depth = max(2, int(windows_ahead) + 2)
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
        """Queue EOF sentinel; never discard decoded VideoWindow buffers."""
        if not force and self._done:
            return
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            try:
                self._q.put(_STOP, timeout=0.05)
                return
            except queue.Full:
                # Consumer still busy on previous window — wait, do not drop queued windows.
                time.sleep(0.01)
        # _done is already True; next_window() treats empty queue as EOF.

    def next_window(self) -> VideoWindow | None:
        if self._error is not None:
            raise self._error
        while True:
            try:
                if self._done:
                    item = self._q.get_nowait()
                else:
                    item = self._q.get(timeout=0.25)
            except queue.Empty:
                if self._done:
                    return None
                continue
            if item is _STOP:
                return None
            assert isinstance(item, VideoWindow)
            return item

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        pos = 0
        try:
            # One capture for the whole video — sequential decode only.
            # Re-open + CAP_PROP_POS_FRAMES per window breaks HEVC (POC / RPS errors).
            cap = cv2.VideoCapture(self._input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self._input_path}")

            while self._total_frames <= 0 or self._next_start < self._total_frames:
                if self._should_stop and self._should_stop():
                    break
                while self._should_pause and self._should_pause():
                    time.sleep(0.05)
                    if self._should_stop and self._should_stop():
                        break
                if self._total_frames > 0 and self._next_start >= self._total_frames:
                    break
                t_dec = time.perf_counter()
                window, pos = load_video_window_from_cap(
                    cap,
                    self._next_start,
                    self._infer_per_window,
                    frame_stride=self._frame_stride,
                    total_frames=self._total_frames,
                    current_pos=pos,
                )
                self.decode_sec += max(0.0, time.perf_counter() - t_dec)
                if not window.frames_bgr:
                    break
                self._q.put(window)
                self._next_start = window.end_frame
                if window.count < self._infer_per_window:
                    break
        except BaseException as exc:
            self._error = exc
        finally:
            if cap is not None:
                cap.release()
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
        # Do NOT gc.collect() here — mid-pipeline full GC stalls the GIL and
        # doubles wall time when uvicorn/Jupyter also contend for it.
        if self._current_window is not None:
            self._current_window.frames_bgr.clear()
            self._current_window = None

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
