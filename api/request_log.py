"""Ring buffer of recent API request timings (human-readable admin view)."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RequestRecord:
    method: str
    path: str
    status_code: int
    duration_ms: float
    ts: float
    client: str = ""
    error: str = ""


class RequestLog:
    def __init__(self, maxlen: int = 200) -> None:
        self._items: deque[RequestRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, rec: RequestRecord) -> None:
        with self._lock:
            self._items.append(rec)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._items)[-limit:]
        out: list[dict[str, Any]] = []
        for r in reversed(rows):
            out.append(
                {
                    "method": r.method,
                    "path": r.path,
                    "status_code": r.status_code,
                    "duration_ms": round(r.duration_ms, 1),
                    "duration_human": _fmt_ms(r.duration_ms),
                    "ts": r.ts,
                    "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r.ts)),
                    "client": r.client,
                    "error": r.error,
                }
            )
        return out


def _fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    sec = ms / 1000.0
    if sec < 60:
        return f"{sec:.2f} s"
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins}m {rem:.1f}s"


request_log = RequestLog()
