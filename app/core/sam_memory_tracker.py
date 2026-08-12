"""
SAM-style identity: masklet / memory-bank association for stable object_id.

Architecture (hybrid YOLO + SAM identity)
----------------------------------------
1. YOLO person detect (+ optional seg) is the *speed* path for boxes/track IDs.
   Helmet cross-check is a separate per-frame accessory detector — not identity.
2. Frame-to-frame identity uses masklet memory (IoU + motion_id), NOT OSNet gallery
   thresholds. This mirrors SAM2/SAM3 video masklet association. Person-only.
3. OSNet/SOLIDER is optional and only for person long re-entry / Pass2
   (`sam_osnet_reentry=True` / tracklet link) — never applied to helmet boxes.
4. Real SAM2/SAM3 neural memory (`backend=ultralytics_sam2` / parent `sam3` weights)
   is a drop-in upgrade behind the same update() API; default `memory` works offline.
5. Default: `use_sam_identity=True` → SamMemoryTracker; `False` keeps OSNet ReidTracker.

Defaults / enable
-----------------
  PipelineSettings()  # use_sam_identity=True, use_reid=False, sam_osnet_reentry=False
  settings.sam_identity_backend = "memory"   # or "mock" / "ultralytics_sam2"
  # Optional weights (Ultralytics SAM2 or sibling repo):
  #   settings.sam_model = Path(r"D:\\Projects\\SAM3_construction\\sam3\\cache\\sam3.pt")
  # Sibling Meta SAM3 Video: pip install git+https://github.com/facebookresearch/sam3.git
  #   (see ../sam3/app/core/SAM31_core/sam31_engine.py) — full neural propagate is TODO.

Windows / deps
--------------
- Ultralytics 8.4+ already ships SAM2VideoPredictor / SAM3VideoPredictor in this venv.
- Meta `sam3` package is installed under ../sam3/.venv, not YOLO_DRT by default.
- Full SAM3.pt (~3.3GB) lives at ../sam3/cache/sam3.pt — do not auto-download here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import lap
import numpy as np

from app.core.detect_engine import DetectItem
from app.core.fusion import box_iou
from app.core.identity_gallery import IdentityGallery
from app.core.reid_tracker import TrackerUpdateResult


@dataclass
class MaskletState:
    """One identity slot in the temporal memory bank (SAM masklet analogue)."""

    object_id: int
    label: str
    last_bbox: np.ndarray
    last_motion_id: int = -1
    miss_count: int = 0
    active: bool = True
    # Optional appearance for long re-entry only (OSNet), never for F2F matching.
    ema_embedding: np.ndarray | None = None


class IdentityTracker(Protocol):
    total_id_switches: int
    total_reid_recoveries: int

    def reset(self) -> None: ...

    def update(
        self,
        detections: list[DetectItem],
        embeddings: np.ndarray,
        *,
        frame_idx: int = -1,
    ) -> TrackerUpdateResult: ...


def default_sam_model_candidates() -> list[Path]:
    """Nearby checkpoints (sibling sam3 repo / Ultralytics SAM2)."""
    root = Path(__file__).resolve().parents[2]
    parent = root.parent
    return [
        parent / "sam3" / "cache" / "sam3.pt",
        parent / "sam3" / "DOCKER_SAM" / "cache" / "sam3.pt",
        root / "models" / "SAM" / "sam2.1_b.pt",
        root / "models" / "SAM" / "sam2_b.pt",
        root / "models" / "SAM" / "sam3.pt",
    ]


def resolve_sam_model_path(explicit: Path | None = None) -> Path | None:
    if explicit is not None and Path(explicit).exists():
        return Path(explicit)
    for cand in default_sam_model_candidates():
        if cand.exists():
            return cand
    return None


def uses_sam_identity(settings: Any) -> bool:
    return bool(getattr(settings, "use_sam_identity", False))


def uses_stable_identity(settings: Any) -> bool:
    """True when pipeline should assign stable object_id (SAM or OSNet ReID)."""
    return uses_sam_identity(settings) or bool(getattr(settings, "use_reid", False))


def needs_osnet_embed(settings: Any) -> bool:
    """
    ReID crops/embed when classic ReID is on, or SAM asks for long re-entry.
    F2F SAM identity does not need embeddings for IoU/masklet matching.
    """
    if bool(getattr(settings, "use_reid", False)):
        return True
    if uses_sam_identity(settings):
        # Re-entry must not also require use_reid — that flag means classic ReidTracker.
        return bool(getattr(settings, "sam_osnet_reentry", False))
    return False


class SamMemoryTracker:
    """
    Masklet memory tracker: IoU + motion_id continuity → object_id.

    Same update() contract as ReidTracker so VideoProcessor can swap trackers.
    Embeddings are ignored unless allow_osnet_reentry=True and miss_count is high.
    """

    def __init__(
        self,
        *,
        match_iou: float = 0.30,
        track_buffer: int = 150,
        debug_log: bool = False,
        allow_osnet_reentry: bool = False,
        osnet_reentry_thresh: float = 0.70,
        osnet_reentry_min_miss: int = 30,
        backend: str = "memory",
        identity_gallery: IdentityGallery | None = None,
    ) -> None:
        self.match_iou = float(match_iou)
        self.track_buffer = int(track_buffer)
        self.debug_log = bool(debug_log)
        self.allow_osnet_reentry = bool(allow_osnet_reentry)
        self.osnet_reentry_thresh = float(osnet_reentry_thresh)
        self.osnet_reentry_min_miss = int(osnet_reentry_min_miss)
        self.backend = str(backend or "memory")
        # CPU gallery survives full video; short-term _lost is only for IoU recover.
        self.gallery = identity_gallery
        self._masklets: dict[int, MaskletState] = {}
        self._lost: dict[int, MaskletState] = {}
        self._motion_to_oid: dict[int, int] = {}
        self._next_id = 1
        self.total_id_switches = 0
        self.total_reid_recoveries = 0
        self._backend_note = ""
        if self.backend == "ultralytics_sam2":
            self._backend_note = (
                "ultralytics_sam2 requested: neural memory not wired in this slice; "
                "using IoU/masklet memory. Point settings.sam_model at weights and "
                "see module docstring for full SAM2/SAM3 TODO."
            )

    def set_identity_gallery(self, gallery: IdentityGallery | None) -> None:
        self.gallery = gallery

    def reset(self) -> None:
        self._masklets.clear()
        self._lost.clear()
        self._motion_to_oid.clear()
        self._next_id = 1
        self.total_id_switches = 0
        self.total_reid_recoveries = 0
        if self.gallery is not None:
            self.gallery.clear()

    def _mint(self, det: DetectItem) -> int:
        oid = self._next_id
        self._next_id += 1
        self._masklets[oid] = MaskletState(
            object_id=oid,
            label=det.label,
            last_bbox=det.xyxy.astype(np.float32, copy=True),
            last_motion_id=int(det.motion_id),
            miss_count=0,
            active=True,
        )
        if det.motion_id >= 0:
            self._motion_to_oid[int(det.motion_id)] = oid
        return oid

    def _bind(self, oid: int, det: DetectItem, emb: np.ndarray | None, *, frame_idx: int = -1) -> None:
        tr = self._masklets.get(oid) or self._lost.pop(oid, None)
        if tr is None:
            return
        tr.last_bbox = det.xyxy.astype(np.float32, copy=True)
        tr.last_motion_id = int(det.motion_id)
        tr.miss_count = 0
        tr.active = True
        tr.label = det.label
        if (
            self.allow_osnet_reentry
            and emb is not None
            and float(np.linalg.norm(emb)) > 1e-3
        ):
            vec = emb.astype(np.float32, copy=False).reshape(-1)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            if tr.ema_embedding is None:
                tr.ema_embedding = vec
            else:
                mixed = 0.35 * vec + 0.65 * tr.ema_embedding
                tr.ema_embedding = mixed / (np.linalg.norm(mixed) + 1e-8)
            if self.gallery is not None:
                self.gallery.upsert(oid, vec, frame_idx=frame_idx)
        self._masklets[oid] = tr
        if det.motion_id >= 0:
            self._motion_to_oid[int(det.motion_id)] = oid

    def _ensure_slot(self, oid: int, det: DetectItem) -> None:
        """Re-materialize a gallery identity that left short-term masklet memory."""
        if oid in self._masklets or oid in self._lost:
            return
        self._lost[oid] = MaskletState(
            object_id=oid,
            label=det.label,
            last_bbox=det.xyxy.astype(np.float32, copy=True),
            last_motion_id=int(det.motion_id),
            miss_count=max(self.osnet_reentry_min_miss, self.track_buffer + 1),
            active=False,
            ema_embedding=None if self.gallery is None else self.gallery.get(oid),
        )
        self._next_id = max(self._next_id, oid + 1)

    def _gallery_query(
        self,
        emb: np.ndarray,
        *,
        exclude: set[int],
        frame_idx: int,
    ) -> tuple[int, float] | None:
        if self.gallery is None:
            return None
        if float(np.linalg.norm(emb)) < 1e-3:
            return None
        hit = self.gallery.query(
            emb,
            exclude=exclude,
            min_sim=self.osnet_reentry_thresh,
        )
        if hit is None:
            return None
        return int(hit.object_id), float(hit.similarity)

    def _increment_miss_all(self) -> None:
        """Bump miss on unmatched masklets; demote past track_buffer into _lost.

        ``track_buffer <= 0`` = full-video memory: demote after a short grace so
        IoU/re-entry can use ``_lost``, but **never purge** lost IDs by TTL.
        """
        full_video = self.track_buffer <= 0
        # Demote active→lost so Phase-2/3 can recover; keep a small grace even in
        # full-video mode (does not drop identity — only moves pool).
        demote_after = (
            max(30, int(self.osnet_reentry_min_miss))
            if full_video
            else int(self.track_buffer)
        )
        for oid in list(self._masklets.keys()):
            tr = self._masklets[oid]
            if tr.active:
                continue
            tr.miss_count += 1
            if tr.miss_count > demote_after:
                mid = tr.last_motion_id
                if mid >= 0 and self._motion_to_oid.get(mid) == oid:
                    del self._motion_to_oid[mid]
                self._lost[oid] = self._masklets.pop(oid)
        for oid, tr in list(self._lost.items()):
            if oid in self._masklets:
                continue
            tr.miss_count += 1
            if full_video:
                # Full-video: identity lives in _lost + CPU gallery for the whole run.
                continue
            # Finite buffer: keep lost IDs long enough for leave/return.
            # With re-entry: 10× track_buffer (900 → 9000 frames ≈ 5 min @ 30 fps).
            lost_ttl = max(self.track_buffer * 2, self.track_buffer * 6)
            if self.allow_osnet_reentry:
                lost_ttl = max(lost_ttl, self.track_buffer * 10)
            if tr.miss_count > lost_ttl:
                mid = tr.last_motion_id
                if mid >= 0 and self._motion_to_oid.get(mid) == oid:
                    del self._motion_to_oid[mid]
                del self._lost[oid]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def update(
        self,
        detections: list[DetectItem],
        embeddings: np.ndarray,
        *,
        frame_idx: int = -1,
    ) -> TrackerUpdateResult:
        debug_lines: list[str] = []
        frame_recoveries = 0
        frame_id_switches = 0

        if self._backend_note and self.debug_log and frame_idx <= 0:
            debug_lines.append(f"sam_backend_note: {self._backend_note}")

        # Cap only for finite buffer; full-video (track_buffer<=0) keeps every lost ID.
        if self.track_buffer > 0 and len(self._lost) > 64:
            keep = sorted(self._lost.items(), key=lambda kv: kv[1].miss_count)[:64]
            self._lost = dict(keep)

        for tr in self._masklets.values():
            tr.active = False

        n = len(detections)
        if n == 0:
            self._increment_miss_all()
            return TrackerUpdateResult([], 0, 0, debug_lines)

        if embeddings is None or embeddings.shape[0] != n:
            embs = np.zeros((n, 1), dtype=np.float32)
        else:
            embs = embeddings

        assigned_det: set[int] = set()
        assigned_oid: set[int] = set()
        det_to_oid: dict[int, int] = {}

        # Phase 1: motion_id → masklet (BoT-SORT continuity hint).
        for di, det in enumerate(detections):
            if det.motion_id < 0:
                continue
            oid = self._motion_to_oid.get(int(det.motion_id))
            if oid is None:
                continue
            if oid not in self._masklets and oid not in self._lost:
                continue
            if oid in assigned_oid:
                continue
            tr = self._masklets.get(oid) or self._lost.get(oid)
            if tr is None:
                continue
            iou = box_iou(det.xyxy, tr.last_bbox) if tr.last_bbox is not None else 0.0
            # Soft gate: motion_id alone is enough when IoU is not terrible
            # (occlusion / brief miss); refuse only on clear spatial hijack.
            if iou < 0.05 and tr.miss_count == 0:
                continue
            det_to_oid[di] = oid
            assigned_det.add(di)
            assigned_oid.add(oid)
            if self.debug_log:
                debug_lines.append(
                    f"f{frame_idx} det{di} motion_bind oid={oid} mid={det.motion_id} iou={iou:.2f}"
                )

        # Phase 2: IoU Hungarian on remaining dets ↔ active+lost masklets.
        pool: list[tuple[int, MaskletState]] = []
        for oid, tr in self._masklets.items():
            if oid not in assigned_oid:
                pool.append((oid, tr))
        for oid, tr in self._lost.items():
            if oid not in assigned_oid:
                pool.append((oid, tr))

        free_dets = [di for di in range(n) if di not in assigned_det]
        if free_dets and pool:
            cost = np.ones((len(free_dets), len(pool)), dtype=np.float64)
            for i, di in enumerate(free_dets):
                det = detections[di]
                for j, (_oid, tr) in enumerate(pool):
                    iou = box_iou(det.xyxy, tr.last_bbox)
                    if det.label.casefold() != tr.label.casefold():
                        iou *= 0.5
                    cost[i, j] = 1.0 - iou
            _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=1.0 - self.match_iou)
            for i, di in enumerate(free_dets):
                j = int(x[i]) if i < len(x) else -1
                if j < 0 or j >= len(pool):
                    continue
                iou = 1.0 - float(cost[i, j])
                if iou < self.match_iou:
                    continue
                oid = pool[j][0]
                if oid in assigned_oid:
                    continue
                was_lost = oid in self._lost
                det_to_oid[di] = oid
                assigned_det.add(di)
                assigned_oid.add(oid)
                if was_lost:
                    frame_recoveries += 1
                    self.total_reid_recoveries += 1
                if self.debug_log:
                    tag = "iou_recover" if was_lost else "iou_match"
                    debug_lines.append(
                        f"f{frame_idx} det{di} {tag} oid={oid} iou={iou:.2f}"
                    )

        # Phase 3 (optional): long-lost appearance re-entry via CPU gallery / lost EMA.
        # Gallery lives for the full video on host RAM/SQLite — not VRAM.
        if self.allow_osnet_reentry:
            for di in range(n):
                if di in assigned_det:
                    continue
                emb = embs[di]
                if float(np.linalg.norm(emb)) < 1e-3:
                    continue
                emb_n = emb.astype(np.float32, copy=False).reshape(-1)
                emb_n = emb_n / (np.linalg.norm(emb_n) + 1e-8)

                best_oid, best_sim = -1, -1.0
                # Prefer short-term lost EMA when available.
                for oid, tr in list(self._lost.items()):
                    if oid in assigned_oid:
                        continue
                    if tr.miss_count < self.osnet_reentry_min_miss:
                        continue
                    if tr.ema_embedding is None:
                        continue
                    sim = self._cosine(emb_n, tr.ema_embedding)
                    if sim > best_sim:
                        best_sim, best_oid = sim, oid

                # Full-video CPU gallery (covers IDs already purged from _lost).
                ghit = self._gallery_query(
                    emb_n, exclude=assigned_oid | set(self._masklets.keys()), frame_idx=frame_idx
                )
                if ghit is not None:
                    g_oid, g_sim = ghit
                    if g_sim > best_sim:
                        best_sim, best_oid = g_sim, g_oid

                if best_oid >= 0 and best_sim >= self.osnet_reentry_thresh:
                    self._ensure_slot(best_oid, detections[di])
                    det_to_oid[di] = best_oid
                    assigned_det.add(di)
                    assigned_oid.add(best_oid)
                    frame_recoveries += 1
                    self.total_reid_recoveries += 1
                    if self.debug_log:
                        debug_lines.append(
                            f"f{frame_idx} det{di} gallery_reentry oid={best_oid} sim={best_sim:.2f}"
                        )

        stable_ids: list[int] = []
        for di, det in enumerate(detections):
            if di in det_to_oid:
                oid = det_to_oid[di]
                self._bind(oid, det, embs[di], frame_idx=frame_idx)
                stable_ids.append(oid)
            else:
                # Last chance: query gallery before minting a brand-new ID.
                reused = None
                if self.allow_osnet_reentry:
                    reused = self._gallery_query(
                        embs[di],
                        exclude=assigned_oid | set(self._masklets.keys()),
                        frame_idx=frame_idx,
                    )
                if reused is not None and reused[1] >= self.osnet_reentry_thresh:
                    oid, sim = reused
                    self._ensure_slot(oid, det)
                    self._bind(oid, det, embs[di], frame_idx=frame_idx)
                    assigned_oid.add(oid)
                    frame_recoveries += 1
                    self.total_reid_recoveries += 1
                    stable_ids.append(oid)
                    if self.debug_log:
                        debug_lines.append(
                            f"f{frame_idx} det{di} gallery_mint_reuse oid={oid} sim={sim:.2f}"
                        )
                else:
                    oid = self._mint(det)
                    if self.allow_osnet_reentry:
                        self._bind(oid, det, embs[di], frame_idx=frame_idx)
                    stable_ids.append(oid)
                    if self.debug_log:
                        debug_lines.append(f"f{frame_idx} det{di} new_masklet oid={oid}")

        self._increment_miss_all()

        return TrackerUpdateResult(
            stable_ids, frame_id_switches, frame_recoveries, debug_lines
        )


class MockSamMemoryTracker(SamMemoryTracker):
    """
    Deterministic mock for smoke tests: same API, no weights.
    Prefers motion_id as object_id when present; else sequential masklets.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("backend", "mock")
        kwargs.setdefault("debug_log", True)
        super().__init__(**kwargs)

    def update(
        self,
        detections: list[DetectItem],
        embeddings: np.ndarray,
        *,
        frame_idx: int = -1,
    ) -> TrackerUpdateResult:
        # Reuse memory association; mock only guarantees a stable API + backend tag.
        result = super().update(detections, embeddings, frame_idx=frame_idx)
        result.debug_lines.insert(0, f"mock_sam_identity backend={self.backend}")
        return result


def build_identity_tracker(
    settings: Any,
    *,
    identity_gallery: IdentityGallery | None = None,
) -> IdentityTracker:
    """
    Factory: SAM masklet tracker when use_sam_identity, else classic ReidTracker.
    """
    if uses_sam_identity(settings):
        backend = str(getattr(settings, "sam_identity_backend", "memory") or "memory")
        gallery = identity_gallery
        if gallery is None and bool(getattr(settings, "identity_gallery_enabled", True)):
            spill = None
            if bool(getattr(settings, "identity_gallery_spill", True)):
                work = getattr(settings, "work_dir", None)
                if work is not None:
                    spill = Path(work) / "identity_gallery.sqlite"
            gallery = IdentityGallery(
                min_sim=float(
                    getattr(
                        settings,
                        "identity_gallery_min_sim",
                        getattr(settings, "sam_osnet_reentry_thresh", 0.65),
                    )
                ),
                ema_alpha=float(getattr(settings, "identity_gallery_ema_alpha", 0.35)),
                update_min_sim=float(
                    getattr(settings, "reid_gallery_update_min_sim", 0.45)
                ),
                spill_path=spill,
            )
        common = dict(
            match_iou=float(getattr(settings, "sam_match_iou", 0.30)),
            track_buffer=int(getattr(settings, "track_buffer", 150)),
            debug_log=bool(getattr(settings, "reid_debug_log", False)),
            allow_osnet_reentry=bool(getattr(settings, "sam_osnet_reentry", False)),
            osnet_reentry_thresh=float(getattr(settings, "sam_osnet_reentry_thresh", 0.70)),
            osnet_reentry_min_miss=int(getattr(settings, "sam_osnet_reentry_min_miss", 30)),
            backend=backend,
            identity_gallery=gallery,
        )
        if backend == "mock":
            return MockSamMemoryTracker(**common)
        if backend == "ultralytics_sam2":
            # Probe weights once; still use masklet memory until neural wire-up lands.
            _ = resolve_sam_model_path(getattr(settings, "sam_model", None))
            return SamMemoryTracker(**common)
        return SamMemoryTracker(**common)

    from app.core.reid_tracker import ReidTracker

    return ReidTracker(
        appearance_thresh=settings.appearance_thresh,
        track_buffer=settings.track_buffer,
        gallery_size=settings.reid_gallery_size,
        w_iou=settings.w_iou,
        w_app=settings.w_app,
        recovery_thresh=settings.recovery_thresh,
        min_match_score=float(getattr(settings, "reid_min_match_score", 0.35)),
        iou_recovery_thresh=float(getattr(settings, "iou_recovery_thresh", 0.55)),
        iou_recovery_min_app=float(getattr(settings, "iou_recovery_min_app", 0.28)),
        debug_log=bool(getattr(settings, "reid_debug_log", False)),
        gallery_ema_alpha=float(getattr(settings, "reid_gallery_ema_alpha", 0.35)),
        gallery_update_min_sim=float(
            getattr(settings, "reid_gallery_update_min_sim", 0.45)
        ),
        lost_spatial_sep=float(getattr(settings, "lost_spatial_sep", 1.25)),
        lost_spatial_strict_app=float(
            getattr(settings, "lost_spatial_strict_app", 0.72)
        ),
        lost_stale_frames=int(getattr(settings, "lost_stale_frames", 45)),
        lost_stale_app=float(getattr(settings, "lost_stale_app", 0.65)),
        lost_reacquire_app=float(getattr(settings, "lost_reacquire_app", 0.70)),
        sole_gap_frames=int(getattr(settings, "sole_gap_frames", 10)),
        centered_reacquire_app=float(
            getattr(settings, "centered_reacquire_app", 0.74)
        ),
        motion_app_break=float(getattr(settings, "motion_app_break", 0.40)),
        motion_app_drop=float(getattr(settings, "motion_app_drop", 0.20)),
        motion_app_break_miss=int(getattr(settings, "motion_app_break_miss", 3)),
    )
