"""Smart host-RAM budget: fit windowed decode + pipeline under a cap without slowing GPU."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.frame_pipeline import estimate_video_ram_gb

if TYPE_CHECKING:
    from app.config.settings import PipelineSettings


@dataclass(frozen=True, slots=True)
class RamBudgetPlan:
    """Resolved RAM plan for one video run."""

    budget_gb: float
    max_window_ram_gb: float
    windows_in_ram: int
    max_preload_ram_gb: float
    estimated_peak_gb: float
    baseline_rss_gb: float
    pipeline_gb: float
    models_gb: float
    reserve_gb: float
    spill_gb: float
    safety_gb: float
    job_batch: int
    gpu_depth: int
    finalizer_depth: int

    def summary_lines(self) -> list[str]:
        src = (
            f"RSS {self.baseline_rss_gb:.1f} GB"
            if self.baseline_rss_gb > 0
            else f"models ~{self.models_gb:.1f} GB"
        )
        return [
            f"Smart RAM: budget {self.budget_gb:.1f} GB → peak ~{self.estimated_peak_gb:.1f} GB",
            (
                f"  {src} | окно {self.max_window_ram_gb:.2f} GB × {self.windows_in_ram} "
                f"(+prefetch ≈2× decode) | pipeline {self.pipeline_gb:.2f} GB "
                f"(job {self.job_batch}, GPU q={self.gpu_depth})"
            ),
            (
                f"  spill {self.spill_gb:.1f} GB | safety {self.safety_gb:.1f} GB "
                f"| preload cap {self.max_preload_ram_gb:.1f} GB"
            ),
        ]


def estimate_models_host_ram_gb(settings: PipelineSettings) -> float:
    """Fallback when RSS unknown — завышенно, чтобы не раздувать окно."""
    override = float(getattr(settings, "ram_budget_models_gb", 0.0) or 0.0)
    if override > 0:
        return override
    base = 1.5
    if bool(getattr(settings, "use_tensorrt", False)):
        base = 1.2
    if bool(getattr(settings, "use_seg", True)):
        base += 1.0
    if bool(getattr(settings, "use_reid", True)):
        base += 0.7
    if bool(getattr(settings, "cross_check_enabled", False)):
        base += 0.5
    return base


def effective_gpu_pipeline_depth(settings: PipelineSettings, *, use_reid: bool) -> int:
    """Match video_processor._drive_gpu_pipeline depth for ReID serial overlap."""
    dev = str(getattr(settings, "inference_device", "cuda") or "cuda").casefold()
    reid_overlap = bool(
        use_reid
        and bool(getattr(settings, "reid_gpu_overlap", False))
        and dev != "cpu"
    )
    serial_cuda = use_reid and not reid_overlap
    if serial_cuda or reid_overlap:
        return 2
    return max(1, int(getattr(settings, "gpu_queue_depth", 4)))


def estimate_pipeline_host_ram_gb(
    width: int,
    height: int,
    job_batch: int,
    *,
    gpu_depth: int,
    finalizer_depth: int,
) -> float:
    """BGR frames in GPU/finalizer queues (upper bound)."""
    job = max(1, int(job_batch))
    depth = max(1, int(gpu_depth))
    fin = max(1, int(finalizer_depth))
    frame_slots = job * (depth + fin + 1)
    return estimate_video_ram_gb(width, height, frame_slots)


def resolve_smart_ram_plan(
    settings: PipelineSettings,
    *,
    width: int,
    height: int,
    job_batch: int,
    use_reid: bool,
    baseline_rss_gb: float | None = None,
) -> RamBudgetPlan | None:
    """
    Окно decode = budget − текущий RSS − pipeline − запас.
    RSS снимается после загрузки моделей — единственный надёжный способ уложиться в 10 GB.
    """
    if not bool(getattr(settings, "smart_ram_budget", True)):
        return None
    budget = float(getattr(settings, "max_process_ram_gb", 0.0) or 0.0)
    if budget <= 0:
        return None

    job = max(1, int(job_batch))
    gpu_depth = effective_gpu_pipeline_depth(settings, use_reid=use_reid)
    finalizer_depth = max(2, gpu_depth)
    pipeline_gb = estimate_pipeline_host_ram_gb(
        width,
        height,
        job,
        gpu_depth=gpu_depth,
        finalizer_depth=finalizer_depth,
    )
    models_gb = estimate_models_host_ram_gb(settings)
    spill_gb = max(0.15, float(getattr(settings, "ram_budget_spill_gb", 0.25)))
    safety_gb = max(0.3, float(getattr(settings, "ram_budget_safety_margin_gb", 0.5)))
    reserve_gb = 0.0
    baseline = float(baseline_rss_gb or 0.0)

    min_window_gb = estimate_video_ram_gb(width, height, job)
    window_ceiling = float(getattr(settings, "max_window_ram_gb", 4.0) or 0.0)

    if baseline > 0.4:
        # Prefetcher кладёт следующее окно в очередь, пока GPU жуёт текущее → до 2× decode BGR.
        prefetch_slots = 2
        used_gb = baseline + pipeline_gb + spill_gb + safety_gb
        window_pool_total = budget - used_gb
        window_pool_gb = max(min_window_gb, window_pool_total / prefetch_slots)
        models_gb = baseline
    else:
        reserve_gb = max(0.5, float(getattr(settings, "ram_budget_system_reserve_gb", 0.5)))
        used_gb = models_gb + pipeline_gb + reserve_gb + spill_gb + safety_gb
        window_pool_gb = budget - used_gb

    if window_pool_gb < min_window_gb:
        window_pool_gb = min_window_gb

    windows_in_ram = 1
    per_window_gb = window_pool_gb
    if window_ceiling > 0:
        per_window_gb = min(per_window_gb, window_ceiling)
    per_window_gb = max(min_window_gb, per_window_gb)

    if baseline > 0.4:
        estimated_peak = baseline + pipeline_gb + spill_gb + per_window_gb * 2
    else:
        estimated_peak = models_gb + pipeline_gb + reserve_gb + spill_gb + safety_gb + per_window_gb
    estimated_peak = min(estimated_peak + safety_gb * 0.5, budget + safety_gb)

    max_preload = max(
        min_window_gb,
        min(
            budget - (baseline if baseline > 0.4 else models_gb) - safety_gb,
            window_ceiling if window_ceiling > 0 else budget,
        ),
    )

    return RamBudgetPlan(
        budget_gb=budget,
        max_window_ram_gb=per_window_gb,
        windows_in_ram=1,
        max_preload_ram_gb=max_preload,
        estimated_peak_gb=min(estimated_peak, budget + 0.5),
        baseline_rss_gb=baseline,
        pipeline_gb=pipeline_gb,
        models_gb=models_gb,
        reserve_gb=reserve_gb,
        spill_gb=spill_gb,
        safety_gb=safety_gb,
        job_batch=job,
        gpu_depth=gpu_depth,
        finalizer_depth=finalizer_depth,
    )
