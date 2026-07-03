"""Background CPU finalize — tracker/fusion параллельно со следующим GPU job."""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

_STOP = object()


class AsyncBatchFinalizer:
    def __init__(
        self,
        finalize_fn: Callable,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self._finalize_fn = finalize_fn
        self._should_stop = should_stop
        self._in_q: queue.Queue[object] = queue.Queue()
        self._out_q: queue.Queue[list | object] = queue.Queue()
        self._error: BaseException | None = None
        self._processing = False
        self._thread = threading.Thread(target=self._run, name="yolo-drt-finalize", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, result, frames_bgr: list) -> None:
        if self._error is not None:
            raise self._error
        self._in_q.put((result, frames_bgr))

    def pending_depth(self) -> int:
        """Job'ы в очереди на CPU finalize (держат frames_bgr в RAM)."""
        return self._in_q.qsize()

    def is_busy(self) -> bool:
        return (
            self._processing
            or self.pending_depth() > 0
            or self._out_q.qsize() > 0
        )

    def drain(self) -> list[list]:
        if self._error is not None:
            raise self._error
        batches: list[list] = []
        while True:
            try:
                item = self._out_q.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                break
            batches.append(item)
        return batches

    def flush_pending(self, *, timeout_sec: float = 300.0) -> list[list]:
        """Drain in-flight finalize jobs without shutting down the worker."""
        if self._error is not None:
            raise self._error
        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        batches: list[list] = []
        while time.monotonic() < deadline:
            batches.extend(self.drain())
            if not self._processing and self.pending_depth() <= 0:
                batches.extend(self.drain())
                if not self._processing and self.pending_depth() <= 0:
                    break
            time.sleep(0.005)
        return batches

    def close(self) -> None:
        self._in_q.put(_STOP)
        self._thread.join(timeout=600.0)
        if self._error is not None:
            raise self._error

    def finish(self) -> list[list]:
        self.close()
        batches: list[list] = []
        while True:
            try:
                item = self._out_q.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                continue
            batches.append(item)
        return batches

    def _run(self) -> None:
        try:
            while True:
                item = self._in_q.get()
                if item is _STOP:
                    break
                if self._should_stop and self._should_stop():
                    continue
                result, frames_bgr = item
                self._processing = True
                try:
                    packets = self._finalize_fn(result, frames_bgr)
                    self._out_q.put(packets)
                finally:
                    self._processing = False
        except BaseException as exc:
            self._error = exc
        finally:
            self._out_q.put(_STOP)
