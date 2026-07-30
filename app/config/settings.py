"""Application defaults and paths."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_DIR = MODELS_DIR / "YOLO"
YOLO_SEG_DIR = YOLO_DIR / "seg"
YOLO_HELMET_DIR = YOLO_DIR / "Helmet"
RD_DIR = MODELS_DIR / "RD"
TRT_DIR = MODELS_DIR / "TRT"
OUTPUT_DIR = PROJECT_ROOT / "output"
# Local work tree (same env names as Docker API).
WORK_DIR = Path(os.environ.get("YOLO_DRT_WORK_DIR", str(PROJECT_ROOT / "work")))


def _default_upload_dir() -> Path:
    """Host uploads: prefer system TEMP (usually C:) so decode matches UI Desktop I/O.

    Project disk (D:\\...\\work\\uploads) was ~2× slower than Desktop path on the
    same machine. Override with YOLO_DRT_UPLOAD_DIR if needed.
    """
    raw = os.environ.get("YOLO_DRT_UPLOAD_DIR", "").strip()
    if raw:
        return Path(raw)
    if os.name == "nt":
        return Path(tempfile.gettempdir()) / "yolo_drt_uploads"
    return WORK_DIR / "uploads"


UPLOAD_DIR = _default_upload_dir()

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
    # Uploads / temp inputs (host: TEMP on Windows; Docker: named volume).
    work_dir: Path = field(default_factory=lambda: WORK_DIR)
    upload_dir: Path = field(default_factory=_default_upload_dir)

    detect_conf: float = 0.20
    seg_conf: float = 0.20
    use_seg: bool = False
    # Classic OSNet ReidTracker / optional SAM long re-entry. Off by default — SAM identity
    # does not need OSNet for F2F (see needs_osnet_embed). Set True + sam_osnet_reentry for re-entry.
    use_reid: bool = False
    # SAM-style masklet identity (default). Bypasses OSNet gallery F2F.
    # Backend: sam_identity_backend="memory"|"mock"|"ultralytics_sam2"
    # Weights (optional): sam_model → ../sam3/cache/sam3.pt or models/SAM/sam2*.pt
    # Sibling Meta SAM3: ../sam3 (pip: git+https://github.com/facebookresearch/sam3.git)
    # Legacy OSNet path: use_sam_identity=False, use_reid=True
    use_sam_identity: bool = True
    sam_identity_backend: str = "memory"  # memory | mock | ultralytics_sam2
    sam_model: Path | None = None
    sam_match_iou: float = 0.30
    # If True and use_reid: OSNet only for long-lost re-entry (not frame-to-frame).
    sam_osnet_reentry: bool = False
    sam_osnet_reentry_thresh: float = 0.70
    sam_osnet_reentry_min_miss: int = 30
    # Offline Pass 2: re-merge F2F tracklets after long occlusions (full-video only).
    # Does not change live SamMemoryTracker / ReidTracker matching.
    # Enabled by default: Pass 2 keeps long-gap identity merging without slowing
    # the per-frame live tracker.
    use_offline_tracklet_link: bool = True
    tracklet_link_max_gap_frames: int = 300
    tracklet_link_min_sim: float = 0.60
    # Load/use OSNet only for this pass when embeddings were not stored in Pass 1.
    tracklet_link_use_reid: bool = True
    tracklet_link_samples_per_tracklet: int = 5
    tracklet_link_spatial_weight: float = 0.15
    match_iou_min: float = 0.45
    seg_fallback_iou_min: float = 0.25

    # Active-track Hungarian / motion continuity appearance floor (cosine).
    appearance_thresh: float = 0.55
    track_buffer: int = 300
    # Lost-track re-ID by appearance alone (same-PPE lookalikes need this high).
    recovery_thresh: float = 0.50
    # Center distance / avg bbox diagonal; above + IoU≈0 → spatial_gap (pan to other person).
    lost_spatial_sep: float = 1.25
    # Required cosine when spatial_gap (refuse soft 0.50–0.60 same-vest match).
    lost_spatial_strict_app: float = 0.72
    # After this many miss frames, raise appearance floor for recover-by-appearance.
    lost_stale_frames: int = 45
    lost_stale_app: float = 0.65
    # No active tracks + spatial_gap: prefer new ID unless very high sim.
    lost_reacquire_app: float = 0.70
    # Sole subject missed ≥ N frames (centered pan/cut): raise app floor.
    # N≫1 so brief flicker/occlusion still uses recovery_thresh / IoU.
    sole_gap_frames: int = 10
    centered_reacquire_app: float = 0.74
    # Break motion_id→object_id only on strong app collapse (see ReidTracker).
    motion_app_break: float = 0.40
    motion_app_drop: float = 0.20
    motion_app_break_miss: int = 3
    # False by default: per-frame motion_bind lines stall CPU finalizer → GPU idle (API worse than UI).
    reid_debug_log: bool = False
    reid_gallery_size: int = 10
    # EMA gallery: ema = alpha * new + (1-alpha) * ema (replaces plain mean of last N).
    reid_gallery_ema_alpha: float = 0.35
    # Skip gallery update when cosine(new, ema) is below this (matching still uses new emb).
    # Higher than recovery soft-floor to limit gradual gallery poisoning after a bad match.
    reid_gallery_update_min_sim: float = 0.45
    # Prefer COCO shoulders+hips torso crop for ReID when keypoints are confident.
    reid_torso_crop: bool = True
    # Expand torso box (fraction of torso w/h). Larger → more context, less "vest blob".
    reid_torso_pad: float = 0.20
    # If torso area < this × person bbox area, fall back to full-body crop.
    reid_torso_min_area_ratio: float = 0.12
    # Combined score floor for Hungarian edges (w_iou*iou + w_app*app).
    reid_min_match_score: float = 0.35
    # IoU-only continuity / recovery (sitting, brief ReID dip).
    iou_recovery_thresh: float = 0.55
    # Reject IoU recovery when appearance is this far below the gallery (ID-steal guard).
    iou_recovery_min_app: float = 0.28
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
    # "cuda" | "cpu" — устройство для PyTorch/Ultralytics (TRT только CUDA).
    inference_device: str = "cuda"
    default_prompt: str = "person"
    # 0 / None = всё видео (как «Max seconds» пустое в desktop UI).
    max_duration_seconds: float | None = None

    # YOLO letterbox (640 default; 512 в realtime). 0 = дефолт Ultralytics.
    infer_imgsz: int = 640
    # Seg каждые N кадров (2 = ~2× быстрее seg; между — bbox-маски).
    seg_stride: int = 1
    # Inference: 1 = все кадры, 2 = каждый 2-й, 0 = авто (realtime → 2).
    frame_stride: int = 0
    # Цель ~1×: imgsz=512, seg_stride=2; infer_batch явный или → tensorrt_max_batch.
    realtime_mode: bool = True

    parallel_post: bool = True
    post_workers: int = 6
    # Очередь готовых батчей (decode → GPU): обработали — слот освободился, читается следующий.
    decode_prefetch: int = 4

    # GPU pipeline (steps 2–6): dedicated CUDA worker, batched infer, motion tracker
    gpu_pipeline: bool = True
    # Полная загрузка GPU одним job на весь ролик — снято (YouTube-style: только батчи).
    gpu_full_batch: bool = False
    # Кадров/job. 0 = авто (realtime→max(64, tensorrt_max_batch)). Явное ≠ silent 200.
    infer_batch_size: int = 64
    # Размер одного YOLO forward (seg/cross); не размер job.
    max_infer_batch_size: int = 200
    # Лимит кадров в одном job при ReID (detect=track по кадрам); 0 = без лимита.
    max_job_batch_size: int = 200
    # 0 = один embed_batch на все кропы job (макс. загрузка GPU).
    reid_embed_chunk: int = 0
    gpu_queue_depth: int = 4
    # Ignored when use_reid=True (BoT-SORT track required for stable motion_id).
    use_batch_detect: bool = True
    reid_batch_across_frames: bool = True
    # ReID + YOLO track на одном GPU: overlap обычно медленнее (конкуренция SM, queue>1).
    # Имеет смысл только без track (predict_batch); при use_reid track принудителен → serial.
    reid_gpu_overlap: bool = False
    gpu_mask_resize: bool = True

    preload_video: bool = False
    # auto: preload если весь ролик влезает сюда; иначе windowed.
    max_preload_ram_gb: float = 12.0
    # Умный лимит host-RAM на прогон: окно + pipeline + модели ≤ cap (не трогает stride/imgsz).
    smart_ram_budget: bool = True
    max_process_ram_gb: float = 10.0
    ram_budget_system_reserve_gb: float = 1.0
    ram_budget_models_gb: float = 0.0  # 0 = авто по включённым моделям
    ram_budget_spill_gb: float = 0.25
    ram_budget_safety_margin_gb: float = 0.5
    # RAM на одно окно в windowed — потолок decode; Smart RAM только уменьшает, не увеличивает.
    max_window_ram_gb: float = 4.0
    # YouTube-style default: windowed batches from disk → GPU. auto → windowed.
    # Explicit preload|stream still honored.
    frame_source_mode: str = "windowed"  # auto | preload | stream | windowed
    # 0 = авто: кратно job×stride, влезает в max_window_ram_gb; >0 = подсказка (snap)
    window_frames: int = 0
    # 1 = только текущее окно; 2 = + decode-ahead (быстрее, ×2 RAM на окна)
    windows_in_ram: int = 1
    # Legacy: two threads on one GPU — disabled when gpu_pipeline=True
    parallel_models: bool = False
    # "parallel" — MP4 во время inference; "deferred" — в конце прогона; "manual" — по кнопке
    encode_mode: str = "manual"
    async_encode: bool = True
    # Сборка MP4: preset libx264 или p1–p7 для NVENC
    encode_preset: str = "fast"
    encode_crf: int = 23
    # auto → NVENC если доступен, иначе libx264
    encode_codec: str = "auto"
    # 0 = post_workers
    encode_workers: int = 0

    # TensorRT (.engine в models/TRT/)
    use_tensorrt: bool = False
    tensorrt_imgsz: int = 640
    # Batch engine; при сборке auto-cap по VRAM (yolo26x@640 batch=200 → OOM на 12GB).
    tensorrt_max_batch: int = 32
    tensorrt_fp16: bool = True
    tensorrt_workspace_gb: float = 2.0
    tensorrt_autocast_fast: bool = False

    cross_check_enabled: bool = False
    cross_check_model: Path | None = field(default_factory=lambda: DEFAULT_HELMET_MODEL)
    cross_check_object_prompt: str = "helmet"
    cross_check_conf: float = 0.35
    cross_check_min_intersection_px: float = 20.0
    cross_check_min_iou: float = 0.03
    cross_check_helmet_min_conf: float = 0.30
    cross_check_min_violation_streak: int = 2
    cross_check_verdict_history: int = 5
    cross_check_warning_text: str = "NO HELMET"
    cross_check_draw_head_box: bool = True
    cross_check_draw_boxes: bool = True

    def resolve_upload_dir(self) -> Path:
        if self.upload_dir is not None:
            return Path(self.upload_dir)
        return Path(self.work_dir) / "uploads"

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self.resolve_upload_dir().mkdir(parents=True, exist_ok=True)
        YOLO_DIR.mkdir(parents=True, exist_ok=True)
        YOLO_SEG_DIR.mkdir(parents=True, exist_ok=True)
        RD_DIR.mkdir(parents=True, exist_ok=True)
        TRT_DIR.mkdir(parents=True, exist_ok=True)
        YOLO_HELMET_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = PipelineSettings()
