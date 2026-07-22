"""Application defaults and paths (Docker / env-driven)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("YOLO_DRT_MODELS_DIR", "/data/models"))
YOLO_DIR = MODELS_DIR / "YOLO"
YOLO_SEG_DIR = YOLO_DIR / "seg"
YOLO_HELMET_DIR = YOLO_DIR / "Helmet"
RD_DIR = MODELS_DIR / "RD"
TRT_DIR = MODELS_DIR / "TRT"
OUTPUT_DIR = Path(os.environ.get("YOLO_DRT_OUTPUT_DIR", "/data/output"))

DEFAULT_DETECT_POSE = YOLO_DIR / "yolo26x-pose.pt"
DEFAULT_HELMET_MODEL = YOLO_HELMET_DIR / "helmet-26m.pt"

OSNET_FILENAME = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)


@dataclass(slots=True)
class PipelineSettings:
    detect_model: Path = field(default_factory=lambda: DEFAULT_DETECT_POSE)
    seg_model: Path = field(default_factory=lambda: YOLO_SEG_DIR / "yolo26x-seg.pt")
    reid_model: Path = field(default_factory=lambda: RD_DIR / OSNET_FILENAME)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)

    detect_conf: float = 0.20
    seg_conf: float = 0.20
    use_seg: bool = False
    # Classic OSNet ReidTracker / optional SAM long re-entry. Off by default —
    # SAM identity does not need OSNet for F2F (Pass 2 can still load OSNet).
    use_reid: bool = False
    # SAM-style masklet identity (default). Bypasses OSNet gallery F2F.
    use_sam_identity: bool = True
    sam_identity_backend: str = "memory"  # memory | mock | ultralytics_sam2
    sam_model: Path | None = None
    sam_match_iou: float = 0.30
    # If True and use_reid: OSNet only for long-lost re-entry (not frame-to-frame).
    sam_osnet_reentry: bool = False
    sam_osnet_reentry_thresh: float = 0.70
    sam_osnet_reentry_min_miss: int = 30
    # Offline Pass 2: re-merge F2F tracklets after long occlusions (full-video only).
    use_offline_tracklet_link: bool = True
    tracklet_link_max_gap_frames: int = 300
    tracklet_link_min_sim: float = 0.60
    tracklet_link_use_reid: bool = True
    tracklet_link_samples_per_tracklet: int = 5
    tracklet_link_spatial_weight: float = 0.15
    match_iou_min: float = 0.45
    seg_fallback_iou_min: float = 0.25

    appearance_thresh: float = 0.52
    track_buffer: int = 300
    recovery_thresh: float = 0.42
    reid_gallery_size: int = 10
    reid_min_match_score: float = 0.30
    reid_debug_log: bool = False
    w_iou: float = 0.3
    w_app: float = 0.7

    overlay_alpha: float = 0.45
    draw_boxes: bool = True
    draw_masks: bool = False
    draw_centers: bool = False
    draw_pose: bool = True
    pose_kpt_conf: float = 0.25
    preview_every_n: int = 5

    use_amp: bool = True
    inference_device: str = "cuda"
    default_prompt: str = "person"

    infer_imgsz: int = 640
    seg_stride: int = 1
    frame_stride: int = 1
    realtime_mode: bool = True

    parallel_post: bool = True
    post_workers: int = 6
    decode_prefetch: int = 4

    gpu_pipeline: bool = True
    gpu_full_batch: bool = False
    infer_batch_size: int = 64
    max_infer_batch_size: int = 200
    max_job_batch_size: int = 200
    reid_embed_chunk: int = 0
    gpu_queue_depth: int = 4
    use_batch_detect: bool = True
    reid_batch_across_frames: bool = True
    # ReID + YOLO track на одном GPU: overlap обычно медленнее (конкуренция SM, queue>1).
    reid_gpu_overlap: bool = False
    gpu_mask_resize: bool = True

    preload_video: bool = True
    max_preload_ram_gb: float = 12.0
    max_window_ram_gb: float = 4.0
    smart_ram_budget: bool = True
    max_process_ram_gb: float = 10.0
    ram_budget_system_reserve_gb: float = 1.0
    ram_budget_models_gb: float = 0.0
    ram_budget_spill_gb: float = 0.25
    ram_budget_safety_margin_gb: float = 0.5
    frame_source_mode: str = "auto"
    window_frames: int = 0
    windows_in_ram: int = 1
    parallel_models: bool = False
    encode_mode: str = "manual"
    async_encode: bool = True
    encode_preset: str = "fast"
    encode_crf: int = 23
    encode_codec: str = "auto"
    encode_workers: int = 0

    use_tensorrt: bool = True
    tensorrt_imgsz: int = 640
    tensorrt_max_batch: int = 32
    tensorrt_fp16: bool = True
    tensorrt_workspace_gb: float = 2.0
    tensorrt_autocast_fast: bool = True
    tensorrt_engine_strategy: str = "central"
    tensorrt_rebuild_policy: str = "missing_only"
    tensorrt_manifest_dir: Path = field(default_factory=lambda: MODELS_DIR)

    cross_check_enabled: bool = True
    cross_check_model: Path | None = field(default_factory=lambda: DEFAULT_HELMET_MODEL)
    cross_check_object_prompt: str = "helmet"
    cross_check_conf: float = 0.20
    cross_check_min_intersection_px: float = 1.0
    cross_check_min_iou: float = 0.0
    cross_check_warning_text: str = "NO HELMET"
    cross_check_draw_head_box: bool = True
    cross_check_draw_boxes: bool = True

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        YOLO_DIR.mkdir(parents=True, exist_ok=True)
        YOLO_SEG_DIR.mkdir(parents=True, exist_ok=True)
        RD_DIR.mkdir(parents=True, exist_ok=True)
        TRT_DIR.mkdir(parents=True, exist_ok=True)
        YOLO_HELMET_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = PipelineSettings()
