"""Shared API server state (settings, processor, logs)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import torch

from api.jobs import job_manager
from api.settings_codec import settings_from_dict, settings_from_ui_patch, settings_to_dict
from app.config.settings import PipelineSettings
from app.core.gpu_cleanup import release_gpu_memory
from app.core.pipeline import processor_needs_rebuild, sync_processor_for_run
from app.core.pipeline_runtime import ensure_processor, prepare_processor_for_job as runtime_prepare_job
from app.core.shared_processor import attach as shared_attach
from app.core.shared_processor import detach as shared_detach
from app.core.shared_processor import get as shared_get
from app.core.trt_export import build_all_engines
from app.core.trt_paths import engines_ready, resolve_reid_engine, resolve_yolo_engine


def _trt_central(settings: PipelineSettings) -> Path:
    """Docker: /data/models/TRT (env models_dir). Never use /app/models/TRT constant."""
    custom = getattr(settings, "tensorrt_central_dir", None)
    if custom is not None:
        return Path(custom)
    return Path(settings.models_dir) / "TRT"


class ServerState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.settings: PipelineSettings | None = None
        self.processor = None
        self.status: str = "stopped"
        self.engines_ready: dict[str, bool] = {}
        self.build_logs: list[str] = []
        self._on_log: Callable[[str], None] | None = None
        self._bootstrap_lock = threading.Lock()

    def set_log_callback(self, cb: Callable[[str], None] | None) -> None:
        self._on_log = cb

    def _log(self, msg: str) -> None:
        with self._lock:
            self.build_logs.append(msg)
            if len(self.build_logs) > 500:
                del self.build_logs[: len(self.build_logs) - 500]
        if self._on_log is not None:
            self._on_log(msg)

    def configure(self, settings: PipelineSettings) -> None:
        with self._lock:
            from app.config.desktop_pipeline_settings import (
                apply_identity_bake_to_settings,
            )

            settings = apply_identity_bake_to_settings(settings)
            settings.ensure_dirs()
            self.settings = settings

    def get_settings_dict(self) -> dict:
        with self._lock:
            if self.settings is None:
                return {}
            return settings_to_dict(self.settings)

    def update_settings(
        self,
        patch: dict,
        *,
        reload_processor: bool = True,
        ui_equivalent: bool = False,
    ) -> PipelineSettings:
        with self._lock:
            from app.config.desktop_pipeline_settings import (
                apply_identity_bake_to_settings,
            )

            if ui_equivalent:
                current = settings_from_ui_patch(patch, base=self.settings or PipelineSettings())
            elif self.settings is None:
                current = settings_from_dict(patch)
            else:
                current = settings_from_dict(patch, base=self.settings)
            current = apply_identity_bake_to_settings(current)
            self.settings = current
            processor = self.processor
        if reload_processor:
            # Soft force: attach if shared UI processor still compatible; reload only if topology changed.
            self.bootstrap(force=True)
        elif processor is not None:
            sync_processor_for_run(processor, current)
            with self._lock:
                self.processor = processor
            job_manager.set_processor(processor, current)
        return current

    def sync_from_shared(self) -> None:
        """Point API/jobs at the process-wide shared processor (e.g. after UI Init)."""
        processor, settings, _holders = shared_get()
        if processor is None or settings is None:
            return
        shared_attach(processor, settings, "api")
        with self._lock:
            self.processor = processor
            self.settings = settings
            self.status = "ready"
        job_manager.set_processor(processor, settings)

    def prepare_processor_for_job(self) -> tuple[PipelineSettings | None, str | None]:
        """Sync settings → processor before inference job (same as desktop Run)."""
        shared_proc, _shared_settings, holders = shared_get()
        with self._lock:
            settings = self.settings
            processor = self.processor
        # When sharing the UI GPU load, prefer the live settings object already on
        # the processor (UI Run / attach) so infer_batch_size etc. match the desktop.
        if (
            processor is not None
            and shared_proc is not None
            and processor is shared_proc
            and "ui" in holders
        ):
            live = getattr(processor, "settings", None)
            if live is not None:
                settings = live
                with self._lock:
                    self.settings = settings
        if settings is None:
            return None, "Settings not loaded"
        processor, err = runtime_prepare_job(settings, processor, on_log=self._log)
        if err or processor is None:
            return None, err or "Processor not loaded"
        with self._lock:
            self.processor = processor
        job_manager.set_processor(processor, settings)
        return settings, None

    def build_trt_engines(self) -> None:
        """Build missing TensorRT engines (same as desktop «Собрать TensorRT engines»)."""
        with self._lock:
            if self.settings is None:
                self.status = "error"
                self._log("Settings not configured")
                return
            settings = self.settings
            self.status = "building_engines"

        self._log("TensorRT: ручная сборка engines…")
        build_all_engines(settings, log=self._log)
        with self._lock:
            settings = self.settings
            if settings is None:
                self.status = "error"
                return
        from app.core.reid_engine import resolve_reid_backend
        from app.core.sam_memory_tracker import needs_osnet_embed

        reid_backend = resolve_reid_backend(
            getattr(settings, "reid_backend", None), settings.reid_model
        )
        need_reid_trt = bool(needs_osnet_embed(settings)) and reid_backend == "osnet"
        central = _trt_central(settings)
        strategy = str(getattr(settings, "tensorrt_engine_strategy", "central") or "central")
        self.engines_ready = engines_ready(
            detect_pt=Path(settings.detect_model),
            cross_pt=Path(settings.cross_check_model) if settings.cross_check_model else None,
            reid_pth=Path(settings.reid_model),
            imgsz=int(settings.tensorrt_imgsz or 640),
            max_batch=int(settings.tensorrt_max_batch),
            fp16=bool(settings.tensorrt_fp16),
            need_cross=bool(settings.cross_check_enabled),
            need_reid=need_reid_trt,
            strategy=strategy,
            central_dir=central,
        )
        self._log("TensorRT: сборка завершена")
        if self.processor is not None:
            # Hard reload so newly built .engine files are loaded into GPU.
            self.bootstrap(force_reload=True)
        else:
            with self._lock:
                self.status = "ready"

    def bootstrap(self, *, force: bool = False, force_reload: bool = False) -> None:
        with self._bootstrap_lock:
            with self._lock:
                if self.settings is None:
                    self.status = "error"
                    self._log("Settings not configured")
                    return
                settings = self.settings
                processor = self.processor

            shared_proc, _shared_settings, holders = shared_get()
            # Prefer process-wide shared processor (UI Init). Never wipe a compatible load
            # unless force_reload (TRT engine rebuild). Soft force=True still attaches.
            if not force_reload:
                candidate = processor if processor is not None else shared_proc
                if candidate is not None and not processor_needs_rebuild(candidate, settings):
                    sync_processor_for_run(candidate, settings)
                    shared_attach(candidate, settings, "api")
                    job_manager.set_processor(candidate, settings)
                    with self._lock:
                        self.processor = candidate
                        self.settings = settings
                        self.status = "ready"
                    if "ui" in holders or (shared_proc is not None and candidate is shared_proc):
                        self._log("API attached to shared UI processor (no duplicate VRAM)")
                    else:
                        self._log(
                            "Bootstrap skipped — processor already loaded (same path as UI Init)."
                        )
                    return

            with self._lock:
                self.status = "starting"
                self.processor = None
                job_manager.clear_processor()

            def _set_status(value: str) -> None:
                with self._lock:
                    self.status = value

            self._log(
                "Bootstrap: check/build TensorRT for current detect weights, then load models..."
            )
            processor = ensure_processor(
                settings,
                holder="api",
                force=force,
                force_reload=force_reload,
                on_log=self._log,
                on_status=_set_status,
                warmup=True,
                auto_build_trt=True,
            )
            if processor is None:
                with self._lock:
                    self.status = "gpu_missing" if not torch.cuda.is_available() else "error"
                    self.processor = None
                return

            with self._lock:
                self.processor = processor
                self.status = "ready"
            job_manager.set_processor(processor, settings)

    def shutdown(self) -> None:
        with self._lock:
            self.processor = None
            self.status = "stopped"
            job_manager.clear_processor()
        _, _, holders_before = shared_get()
        shared_detach("api", dispose_if_last=True)
        _, _, holders_after = shared_get()
        if "ui" in holders_after:
            self._log("API stopped — UI processor kept loaded (shared, no duplicate VRAM)")
        elif holders_before:
            self._log("API processor released")
        else:
            release_gpu_memory()
            self._log("API processor released")

    def health_paths(self) -> dict[str, str]:
        with self._lock:
            settings = self.settings
        if settings is None:
            return {}
        imgsz = int(settings.tensorrt_imgsz or 640)
        max_batch = int(settings.tensorrt_max_batch)
        fp16 = bool(settings.tensorrt_fp16)
        det_pt = Path(settings.detect_model)
        cross_pt = Path(settings.cross_check_model) if settings.cross_check_model else None
        reid_pth = Path(settings.reid_model)
        central = _trt_central(settings)
        strategy = str(getattr(settings, "tensorrt_engine_strategy", "central") or "central")
        paths = {
            "output_dir": str(settings.output_dir),
            "work_dir": str(settings.work_dir),
            "upload_dir": str(settings.resolve_upload_dir()),
            "models_dir": str(settings.models_dir),
            "trt_dir": str(central),
            "detect_weights": str(det_pt),
            "cross_weights": str(cross_pt) if cross_pt else "",
            "reid_weights": str(reid_pth),
        }
        det_eng = resolve_yolo_engine(
            det_pt,
            imgsz=imgsz,
            max_batch=max_batch,
            fp16=fp16,
            strategy=strategy,
            central_dir=central,
        )
        paths["detect_engine"] = str(det_eng)
        paths["detect_engine_exists"] = str(det_eng.exists())
        if cross_pt is not None:
            cross_eng = resolve_yolo_engine(
                cross_pt,
                imgsz=imgsz,
                max_batch=max_batch,
                fp16=fp16,
                strategy=strategy,
                central_dir=central,
            )
            paths["cross_engine"] = str(cross_eng)
            paths["cross_engine_exists"] = str(cross_eng.exists())
        reid_eng = resolve_reid_engine(
            reid_pth, fp16=fp16, strategy=strategy, central_dir=central
        )
        paths["reid_engine"] = str(reid_eng)
        paths["reid_engine_exists"] = str(reid_eng.exists())
        return paths


server_state = ServerState()
