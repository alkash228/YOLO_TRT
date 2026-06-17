"""Lightweight IoU motion IDs for batched YOLO predict (replaces per-frame model.track)."""
from __future__ import annotations

import numpy as np

from app.core.detect_engine import DetectItem
from app.core.fusion import box_iou


class MotionTracker:
    """Greedy IoU association — assigns motion_id in temporal order."""

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30) -> None:
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self._tracks: dict[int, dict[str, object]] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def assign(self, detections: list[DetectItem]) -> list[DetectItem]:
        if not detections:
            self._age_tracks(set())
            return []

        track_ids = list(self._tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))
        matches: list[tuple[int, int, float]] = []

        for ti, tid in enumerate(track_ids):
            tbbox = self._tracks[tid]["bbox"]
            for di in list(unmatched_dets):
                iou = box_iou(np.asarray(tbbox, dtype=np.float32), detections[di].xyxy)
                if iou >= self.iou_thresh:
                    matches.append((iou, ti, di))
        matches.sort(key=lambda x: x[0], reverse=True)

        det_to_track: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _iou, ti, di in matches:
            if di in det_to_track or ti in used_tracks:
                continue
            tid = track_ids[ti]
            det_to_track[di] = tid
            used_tracks.add(ti)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)

        out: list[DetectItem] = []
        matched_track_ids: set[int] = set()
        for di, det in enumerate(detections):
            if di in det_to_track:
                tid = int(det_to_track[di])
            else:
                tid = self._next_id
                self._next_id += 1
            matched_track_ids.add(tid)
            self._tracks[tid] = {"bbox": det.xyxy.copy(), "missed": 0}
            out.append(
                DetectItem(
                    xyxy=det.xyxy.copy(),
                    cls_id=det.cls_id,
                    label=det.label,
                    conf=det.conf,
                    motion_id=tid,
                    keypoints=det.keypoints.copy() if det.keypoints is not None else None,
                )
            )

        self._age_tracks(matched_track_ids)
        return out

    def _age_tracks(self, active_ids: set[int]) -> None:
        stale: list[int] = []
        for tid, state in self._tracks.items():
            if tid in active_ids:
                continue
            missed = int(state.get("missed", 0)) + 1
            if missed > self.max_age:
                stale.append(tid)
            else:
                state["missed"] = missed
        for tid in stale:
            self._tracks.pop(tid, None)
