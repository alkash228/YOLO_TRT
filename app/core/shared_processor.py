"""Single VideoProcessor per process — UI and API share one GPU model load."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from app.config.settings import PipelineSettings
from app.core.gpu_cleanup import dispose_processor

ClearUiRef = Callable[[], None]

_lock = threading.RLock()
_processor: Any | None = None
_settings: PipelineSettings | None = None
_holders: set[str] = set()
_ui_clear_ref: ClearUiRef | None = None


def register_ui_clear_ref(cb: ClearUiRef | None) -> None:
    global _ui_clear_ref
    _ui_clear_ref = cb


def get() -> tuple[Any | None, PipelineSettings | None, frozenset[str]]:
    with _lock:
        return _processor, _settings, frozenset(_holders)


def status() -> dict[str, Any]:
    """Diagnostics for /health — detect duplicate GPU loads."""
    proc, settings, holders = get()
    out: dict[str, Any] = {
        "loaded": proc is not None,
        "holders": sorted(holders),
        "holder_count": len(holders),
    }
    if settings is not None:
        out["use_tensorrt"] = bool(settings.use_tensorrt)
        out["detect_model"] = str(settings.detect_model)
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_allocated_mb"] = round(
                float(torch.cuda.memory_allocated() / (1024 * 1024)), 1
            )
            out["cuda_reserved_mb"] = round(
                float(torch.cuda.memory_reserved() / (1024 * 1024)), 1
            )
    except Exception:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        out["nvml_mem_used_mb"] = round(float(mem.used) / (1024 * 1024), 1)
    except Exception:
        pass
    if out.get("nvml_mem_used_mb") and float(out["nvml_mem_used_mb"]) > 6500 and proc is not None:
        out["duplicate_warning"] = (
            "NVML VRAM >6.5 GB with one processor — likely a second Python process "
            "also loaded models (e.g. `python -m api.main` + desktop UI Init). "
            "Stop the extra process."
        )
    return out


def attach(processor: Any, settings: PipelineSettings, holder: str) -> None:
    """Register holder on an existing or new processor instance."""
    with _lock:
        global _processor, _settings
        if _processor is not None and _processor is not processor:
            _dispose_locked()
        _processor = processor
        _settings = settings
        _holders.add(holder)


def replace(processor: Any, settings: PipelineSettings, holder: str) -> None:
    """Unload any previous processor, then set a new one for holder."""
    global _processor, _settings, _holders
    with _lock:
        _dispose_locked()
        _processor = processor
        _settings = settings
        _holders = {holder}


def detach(holder: str, *, dispose_if_last: bool = True) -> None:
    with _lock:
        _holders.discard(holder)
        if dispose_if_last and not _holders:
            _dispose_locked()


def dispose_all() -> None:
    with _lock:
        _dispose_locked()


def _dispose_locked() -> None:
    global _processor, _settings
    old = _processor
    had_ui = "ui" in _holders
    _processor = None
    _settings = None
    _holders.clear()
    if old is not None:
        dispose_processor(old)
    if had_ui and _ui_clear_ref is not None:
        _ui_clear_ref()
