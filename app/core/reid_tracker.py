"""Stable ID assignment via ReID gallery + motion continuity."""
from __future__ import annotations

from dataclasses import dataclass, field

import lap
import numpy as np

from app.core.detect_engine import DetectItem
from app.core.fusion import box_iou


@dataclass
class TrackState:
    stable_id: int
    label: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    last_bbox: np.ndarray | None = None
    last_motion_id: int = -1
    miss_count: int = 0
    active: bool = True

    def mean_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        arr = np.stack(self.embeddings, axis=0)
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-6:
            return None
        return (mean / norm).astype(np.float32, copy=False)

    def push_embedding(self, emb: np.ndarray, max_size: int) -> None:
        self.embeddings.append(emb.astype(np.float32, copy=False))
        if len(self.embeddings) > max_size:
            self.embeddings = self.embeddings[-max_size:]


@dataclass(slots=True)
class TrackerUpdateResult:
    stable_ids: list[int]
    id_switches: int
    reid_recoveries: int


class ReidTracker:
    def __init__(
        self,
        appearance_thresh: float = 0.52,
        track_buffer: int = 150,
        gallery_size: int = 10,
        w_iou: float = 0.3,
        w_app: float = 0.7,
        recovery_thresh: float = 0.42,
        min_match_score: float = 0.30,
    ) -> None:
        self.appearance_thresh = float(appearance_thresh)
        self.track_buffer = int(track_buffer)
        self.gallery_size = int(gallery_size)
        self.w_iou = float(w_iou)
        self.w_app = float(w_app)
        self.recovery_thresh = float(recovery_thresh)
        self.min_match_score = float(min_match_score)
        self._tracks: dict[int, TrackState] = {}
        self._lost: dict[int, TrackState] = {}
        self._motion_to_stable: dict[int, int] = {}
        self._next_id = 1
        self.total_id_switches = 0
        self.total_reid_recoveries = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._lost.clear()
        self._motion_to_stable.clear()
        self._next_id = 1
        self.total_id_switches = 0
        self.total_reid_recoveries = 0

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _appearance(self, emb: np.ndarray, track: TrackState) -> float:
        track_emb = track.mean_embedding()
        if track_emb is None:
            return 0.0
        return self._cosine(emb, track_emb)

    def _score_pair(
        self,
        det: DetectItem,
        emb: np.ndarray,
        track: TrackState,
    ) -> float:
        iou = 0.0
        if track.last_bbox is not None:
            iou = box_iou(det.xyxy, track.last_bbox)
        track_emb = track.mean_embedding()
        app = self._cosine(emb, track_emb) if track_emb is not None else 0.0
        if track.label.casefold() != det.label.casefold():
            app *= 0.5
        return self.w_iou * iou + self.w_app * app

    def _min_app_for_kind(self, kind: str) -> float:
        if kind == "lost":
            return self.recovery_thresh
        return max(0.35, self.appearance_thresh - 0.12)

    def _assign_track(
        self,
        tid: int,
        tr: TrackState,
        di: int,
        det: DetectItem,
        emb: np.ndarray,
        kind: str,
        det_to_stable: dict[int, int],
        assigned_det: set[int],
        assigned_tid: set[int],
    ) -> bool:
        if di in assigned_det or tid in assigned_tid:
            return False
        if kind == "lost":
            self._lost.pop(tid, None)
            self.total_reid_recoveries += 1
        tr.active = True
        tr.miss_count = 0
        tr.last_bbox = det.xyxy.copy()
        tr.last_motion_id = det.motion_id
        tr.label = det.label
        tr.push_embedding(emb, self.gallery_size)
        self._tracks[tid] = tr
        if det.motion_id >= 0:
            self._motion_to_stable[det.motion_id] = tid
        assigned_det.add(di)
        assigned_tid.add(tid)
        det_to_stable[di] = tid
        return True

    def update(
        self,
        detections: list[DetectItem],
        embeddings: np.ndarray,
    ) -> TrackerUpdateResult:
        frame_recoveries = 0

        for tid in list(self._tracks.keys()):
            self._tracks[tid].active = False

        n = len(detections)
        if n == 0:
            self._increment_miss_all()
            return TrackerUpdateResult([], 0, 0)

        if embeddings.shape[0] != n:
            raise ValueError("embeddings count must match detections")

        pool: list[tuple[int, TrackState, str]] = []
        for tid, tr in self._tracks.items():
            pool.append((tid, tr, "active"))
        for tid, tr in self._lost.items():
            pool.append((tid, tr, "lost"))

        assigned_det: set[int] = set()
        assigned_tid: set[int] = set()
        det_to_stable: dict[int, int] = {}

        # Phase 1: BoT-SORT motion_id continuity (when IDs are stable within a segment).
        for di, det in enumerate(detections):
            if det.motion_id < 0:
                continue
            tid_hint = self._motion_to_stable.get(det.motion_id)
            if tid_hint is None:
                for tid, tr, kind in pool:
                    if tr.last_motion_id == det.motion_id:
                        tid_hint = tid
                        break
            if tid_hint is None or tid_hint in assigned_tid:
                continue
            tr = self._tracks.get(tid_hint) or self._lost.get(tid_hint)
            if tr is None:
                continue
            kind = "active" if tid_hint in self._tracks else "lost"
            app = self._appearance(embeddings[di], tr)
            if app >= self._min_app_for_kind(kind):
                was_lost = kind == "lost"
                if self._assign_track(
                    tid_hint, tr, di, det, embeddings[di], kind, det_to_stable, assigned_det, assigned_tid
                ):
                    if was_lost:
                        frame_recoveries += 1

        # Phase 2: Hungarian assignment for remaining detections vs tracks.
        rem_dets = [di for di in range(n) if di not in assigned_det]
        rem_pool = [(tid, tr, kind) for tid, tr, kind in pool if tid not in assigned_tid]

        if rem_dets and rem_pool:
            n_d = len(rem_dets)
            n_t = len(rem_pool)
            cost = np.full((n_d, n_t), 1e6, dtype=np.float64)
            for rdi, di in enumerate(rem_dets):
                det = detections[di]
                emb = embeddings[di]
                for rtj, (tid, tr, kind) in enumerate(rem_pool):
                    app = self._appearance(emb, tr)
                    if app < self._min_app_for_kind(kind):
                        continue
                    sc = self._score_pair(det, emb, tr)
                    if sc >= self.min_match_score:
                        cost[rdi, rtj] = 1.0 - sc

            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=1.0 - self.min_match_score)
            for rdi, rtj in enumerate(x):
                if rtj < 0 or rtj >= n_t:
                    continue
                if cost[rdi, rtj] >= 1e5:
                    continue
                di = rem_dets[rdi]
                tid, tr, kind = rem_pool[rtj]
                was_lost = kind == "lost"
                if self._assign_track(
                    tid, tr, di, detections[di], embeddings[di], kind, det_to_stable, assigned_det, assigned_tid
                ):
                    if was_lost:
                        frame_recoveries += 1

        # Phase 3: new stable IDs only when no lost track matches at recovery threshold.
        stable_ids: list[int] = []
        for di, det in enumerate(detections):
            if di in det_to_stable:
                stable_ids.append(det_to_stable[di])
                continue

            emb = embeddings[di]
            best_tid: int | None = None
            best_app = -1.0
            best_tr: TrackState | None = None
            for tid, tr in self._lost.items():
                if tid in assigned_tid:
                    continue
                app = self._appearance(emb, tr)
                if app > best_app:
                    best_app = app
                    best_tid = tid
                    best_tr = tr

            if best_tid is not None and best_tr is not None and best_app >= self.recovery_thresh:
                if self._assign_track(
                    best_tid,
                    best_tr,
                    di,
                    det,
                    emb,
                    "lost",
                    det_to_stable,
                    assigned_det,
                    assigned_tid,
                ):
                    frame_recoveries += 1
                    stable_ids.append(best_tid)
                    continue

            tid = self._next_id
            self._next_id += 1
            tr = TrackState(
                stable_id=tid,
                label=det.label,
                last_bbox=det.xyxy.copy(),
                last_motion_id=det.motion_id,
                active=True,
            )
            tr.push_embedding(emb, self.gallery_size)
            self._tracks[tid] = tr
            if det.motion_id >= 0:
                self._motion_to_stable[det.motion_id] = tid
            stable_ids.append(tid)

        for tid, tr in list(self._tracks.items()):
            if not tr.active:
                tr.miss_count += 1
                if tr.miss_count > self.track_buffer:
                    self._tracks.pop(tid, None)
                else:
                    self._lost[tid] = tr
                    self._tracks.pop(tid, None)

        for tid, tr in list(self._lost.items()):
            tr.miss_count += 1
            if tr.miss_count > self.track_buffer:
                self._lost.pop(tid, None)

        self._prune_motion_map()
        return TrackerUpdateResult(stable_ids, 0, frame_recoveries)

    def _prune_motion_map(self) -> None:
        live: set[int] = set()
        for tr in self._tracks.values():
            if tr.last_motion_id >= 0:
                live.add(tr.last_motion_id)
        for tr in self._lost.values():
            if tr.last_motion_id >= 0:
                live.add(tr.last_motion_id)
        for mid in list(self._motion_to_stable):
            if mid not in live:
                self._motion_to_stable.pop(mid, None)

    def _increment_miss_all(self) -> None:
        for tid, tr in list(self._tracks.items()):
            tr.miss_count += 1
            tr.active = False
            if tr.miss_count > self.track_buffer:
                self._tracks.pop(tid, None)
            else:
                self._lost[tid] = tr
                self._tracks.pop(tid, None)
        for tid, tr in list(self._lost.items()):
            tr.miss_count += 1
            if tr.miss_count > self.track_buffer:
                self._lost.pop(tid, None)
