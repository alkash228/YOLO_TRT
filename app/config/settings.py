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
    use_reid: bool = True
    match_iou_min: float = 0.45
    seg_fallback_iou_min: float = 0.25

    appearance_thresh: float = 0.52
    track_buffer: int = 150
    recovery_thresh: float = 0.42
    reid_gallery_size: int = 10
    w_iou: float = 0.3
    w_app: float = 0.7

    overlay_alpha: float = 0.45
    draw_boxes: bool = True
    draw_masks: bool = True
    draw_centers: bool = True
    draw_pose: bool = True
    pose_kpt_conf: float = 0.25
    preview_every_n: int = 5

    use_amp: bool = True
    inference_device: str = "cuda"
    default_prompt: str = "person"

    infer_imgsz: int = 640
    seg_stride: int = 1
    frame_stride: int = 1
    realtime_mode: bool = False

    parallel_post: bool = True
    post_workers: int = 6
    decode_prefetch: int = 4

    gpu_pipeline: bool = True
    gpu_full_batch: bool = True
    infer_batch_size: int = 0
    max_infer_batch_size: int = 32
    max_job_batch_size: int = 64
    reid_embed_chunk: int = 0
    gpu_queue_depth: int = 4
    use_batch_detect: bool = True
    reid_batch_across_frames: bool = True
    gpu_mask_resize: bool = True

    preload_video: bool = True
    max_preload_ram_gb: float = 8.0
    parallel_models: bool = False
    encode_mode: str = "manual"
    async_encode: bool = True

    use_tensorrt: bool = True
    tensorrt_imgsz: int = 640
    tensorrt_max_batch: int = 16
    tensorrt_fp16: bool = True
    tensorrt_workspace_gb: float = 4.0
    tensorrt_autocast_fast: bool = False
    # "central" -> models/TRT; "colocate" -> .engine рядом с .pt/.pth
    tensorrt_engine_strategy: str = "central"
    # "missing_only" | "always"
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
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "input").mkdir(parents=True, exist_ok=True)
        YOLO_DIR.mkdir(parents=True, exist_ok=True)
        YOLO_SEG_DIR.mkdir(parents=True, exist_ok=True)
        RD_DIR.mkdir(parents=True, exist_ok=True)
        TRT_DIR.mkdir(parents=True, exist_ok=True)
        YOLO_HELMET_DIR.mkdir(parents=True, exist_ok=True)
        self.tensorrt_manifest_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = PipelineSettings()
