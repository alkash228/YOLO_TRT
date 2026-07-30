"""Host RAM sampling for the current process (Windows/Linux)."""
from __future__ import annotations

import gc
import os
import sys


def current_process_rss_gb(*, collect_gc: bool = False) -> float | None:
    """Working Set / RSS of this process in GB, or None if unavailable."""
    mb = current_process_rss_mb(collect_gc=collect_gc)
    if mb is None:
        return None
    return float(mb) / 1024.0


def current_process_rss_mb(*, collect_gc: bool = False) -> float | None:
    """Working Set / RSS of this process in MB."""
    if collect_gc:
        gc.collect()
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            access = 0x0400 | 0x0010  # QUERY_INFORMATION | VM_READ
            handle = kernel32.OpenProcess(access, False, os.getpid())
            if not handle:
                return None
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    return None
                return float(counters.WorkingSetSize) / (1024**2)
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(rss) / (1024**2)
        return float(rss) / 1024.0
    except Exception:
        return None


def current_process_peak_rss_mb() -> float | None:
    """Peak working set (Windows) or current RSS peak hint (Linux ru_maxrss), MB."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            access = 0x0400 | 0x0010
            handle = kernel32.OpenProcess(access, False, os.getpid())
            if not handle:
                return None
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    return None
                return float(counters.PeakWorkingSetSize) / (1024**2)
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    return current_process_rss_mb()
