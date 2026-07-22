"""Offline tracklet linking: merge non-overlapping F2F IDs after long occlusions.

Pass 1 (SamMemoryTracker / ReidTracker) produces short tracklets.
Pass 2 (this module) greedily merges tracklets that never co-occur in time,
preferring high OSNet cosine similarity and short temporal/spatial gaps.

Does not affect live frame-to-frame matching (no soft OSNet pan merges).
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


@dataclass(slots=True)
class Tracklet:
    object_id: int
    frame_indices: list[int] = field(default_factory=list)
    bboxes_xyxy: list[np.ndarray] = field(default_factory=list)
    embedding: np.ndarray | None = None

    @property
    def start_frame(self) -> int:
        return int(self.frame_indices[0]) if self.frame_indices else -1

    @property
    def end_frame(self) -> int:
        return int(self.frame_indices[-1]) if self.frame_indices else -1

    @property
    def n_frames(self) -> int:
        return len(self.frame_indices)


@dataclass(slots=True)
class LinkResult:
    id_map: dict[int, int]
    merges: list[tuple[int, int, float]]
    n_tracklets_in: int
    n_ids_out: int


def _xywh_to_xyxy(bbox: list | tuple | np.ndarray) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox[:4]]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _bbox_center(xyxy: np.ndarray) -> tuple[float, float]:
    return (
        0.5 * (float(xyxy[0]) + float(xyxy[2])),
        0.5 * (float(xyxy[1]) + float(xyxy[3])),
    )


def _bbox_diag(xyxy: np.ndarray) -> float:
    w = max(1.0, float(xyxy[2] - xyxy[0]))
    h = max(1.0, float(xyxy[3] - xyxy[1]))
    return float(np.hypot(w, h))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def temporal_overlap(a: Tracklet, b: Tracklet) -> bool:
    """True if both tracklets appear on any shared frame index set (interval overlap)."""
    if not a.frame_indices or not b.frame_indices:
        return False
    return not (a.end_frame < b.start_frame or b.end_frame < a.start_frame)


def temporal_gap(a: Tracklet, b: Tracklet) -> int:
    """Frames between end of earlier and start of later; 0 if abutting/overlapping."""
    if a.end_frame < b.start_frame:
        return int(b.start_frame - a.end_frame - 1)
    if b.end_frame < a.start_frame:
        return int(a.start_frame - b.end_frame - 1)
    return 0


def _normalize_embedding(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    if n < 1e-8:
        return None
    return (arr / n).astype(np.float32, copy=False)


def build_tracklets_from_frames(
    frames: list[dict[str, Any]],
    *,
    embedding_key: str = "embedding",
) -> list[Tracklet]:
    """Build one tracklet per object_id from result-JSON style frames[]."""
    by_id: dict[int, Tracklet] = {}
    emb_acc: dict[int, list[np.ndarray]] = defaultdict(list)
    for fr in frames:
        fi = int(fr.get("frame", 0))
        for inst in fr.get("instances") or []:
            oid = int(inst.get("object_id", 0))
            if oid <= 0:
                continue
            bbox = inst.get("bbox_xywh") or [0, 0, 0, 0]
            xyxy = _xywh_to_xyxy(bbox)
            tr = by_id.get(oid)
            if tr is None:
                tr = Tracklet(object_id=oid)
                by_id[oid] = tr
            tr.frame_indices.append(fi)
            tr.bboxes_xyxy.append(xyxy)
            emb = inst.get(embedding_key)
            if emb is not None:
                arr = _normalize_embedding(np.asarray(emb, dtype=np.float32))
                if arr is not None:
                    emb_acc[oid].append(arr)
    for oid, tr in by_id.items():
        vecs = emb_acc.get(oid) or []
        if vecs:
            mean = np.mean(np.stack(vecs, axis=0), axis=0)
            tr.embedding = _normalize_embedding(mean)
    return sorted(by_id.values(), key=lambda t: t.start_frame)


def build_tracklets_from_packets(packets: list[Any]) -> list[Tracklet]:
    """Build tracklets from FramePacket list (compact or full)."""
    by_id: dict[int, Tracklet] = {}
    for packet in sorted(packets, key=lambda p: int(p.frame_idx)):
        n = int(getattr(packet, "n_inst", 0) or 0)
        if n <= 0:
            continue
        fi = int(packet.frame_idx)
        ids = getattr(packet, "stable_ids", None)
        meta = getattr(packet, "instance_meta", None) or []
        stack = getattr(packet, "stack", None)
        for i in range(n):
            if ids is None or i >= len(ids):
                continue
            oid = int(ids[i])
            if oid <= 0:
                continue
            xyxy: np.ndarray | None = None
            if i < len(meta) and meta[i].get("bbox_xywh"):
                xyxy = _xywh_to_xyxy(meta[i]["bbox_xywh"])
            elif (
                stack is not None
                and getattr(stack, "size", 0) > 0
                and i < int(stack.shape[0])
            ):
                m = (stack[i] > 127).astype(np.uint8)
                if m.any():
                    x, y, bw, bh = cv2.boundingRect(m)
                    xyxy = np.array(
                        [float(x), float(y), float(x + bw), float(y + bh)],
                        dtype=np.float32,
                    )
            if xyxy is None:
                xyxy = np.zeros(4, dtype=np.float32)
            tr = by_id.get(oid)
            if tr is None:
                tr = Tracklet(object_id=oid)
                by_id[oid] = tr
            tr.frame_indices.append(fi)
            tr.bboxes_xyxy.append(xyxy)
    return sorted(by_id.values(), key=lambda t: t.start_frame)


def pair_link_score(
    earlier: Tracklet,
    later: Tracklet,
    *,
    max_gap_frames: int,
    min_sim: float,
    spatial_weight: float = 0.15,
    gap_weight: float = 0.10,
    require_embedding: bool = True,
) -> float | None:
    """
    Score for merging later into earlier's identity.
    Returns None if pair is ineligible.
    """
    if earlier.object_id == later.object_id:
        return None
    if temporal_overlap(earlier, later):
        return None
    # Orient so earlier ends before later starts
    a, b = earlier, later
    if a.end_frame > b.end_frame:
        a, b = later, earlier
    if a.end_frame >= b.start_frame:
        return None
    gap = temporal_gap(a, b)
    if gap > int(max_gap_frames):
        return None

    app = 0.0
    has_emb = a.embedding is not None and b.embedding is not None
    if has_emb:
        app = cosine_similarity(a.embedding, b.embedding)
        if app < float(min_sim):
            return None
    elif require_embedding:
        return None
    else:
        # Spatial-only fallback: require small gap and close centers
        if gap > max(5, int(max_gap_frames) // 20):
            return None
        app = float(min_sim)

    spatial_pen = 0.0
    if a.bboxes_xyxy and b.bboxes_xyxy:
        c0 = _bbox_center(a.bboxes_xyxy[-1])
        c1 = _bbox_center(b.bboxes_xyxy[0])
        dist = float(np.hypot(c0[0] - c1[0], c0[1] - c1[1]))
        diag = 0.5 * (_bbox_diag(a.bboxes_xyxy[-1]) + _bbox_diag(b.bboxes_xyxy[0]))
        spatial_pen = min(1.0, dist / max(1.0, diag)) * float(spatial_weight)

    gap_pen = (gap / max(1.0, float(max_gap_frames))) * float(gap_weight)
    return float(app - gap_pen - spatial_pen)


class _UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {i: i for i in ids}
        self.members: dict[int, set[int]] = {i: {i} for i in ids}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra > rb:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members[rb]
        del self.members[rb]


def _clusters_temporally_ok(
    tracklets_by_id: dict[int, Tracklet],
    ids_a: set[int],
    ids_b: set[int],
) -> bool:
    for ia in ids_a:
        ta = tracklets_by_id[ia]
        for ib in ids_b:
            if temporal_overlap(ta, tracklets_by_id[ib]):
                return False
    return True


def link_tracklets(
    tracklets: list[Tracklet],
    *,
    max_gap_frames: int = 300,
    min_sim: float = 0.60,
    spatial_weight: float = 0.15,
    gap_weight: float = 0.10,
    require_embedding: bool = True,
) -> LinkResult:
    """
    Greedy hierarchical merge: highest-scoring eligible pairs first.
    Final ID = min(original object_id) in each connected component.
    """
    if not tracklets:
        return LinkResult(id_map={}, merges=[], n_tracklets_in=0, n_ids_out=0)

    by_id = {t.object_id: t for t in tracklets}
    ids = list(by_id.keys())
    uf = _UnionFind(ids)

    candidates: list[tuple[float, int, int]] = []
    for i in range(len(tracklets)):
        for j in range(i + 1, len(tracklets)):
            a, b = tracklets[i], tracklets[j]
            score = pair_link_score(
                a,
                b,
                max_gap_frames=max_gap_frames,
                min_sim=min_sim,
                spatial_weight=spatial_weight,
                gap_weight=gap_weight,
                require_embedding=require_embedding,
            )
            if score is None:
                continue
            candidates.append((score, a.object_id, b.object_id))
    candidates.sort(key=lambda x: x[0], reverse=True)

    merges: list[tuple[int, int, float]] = []
    for score, ia, ib in candidates:
        ra, rb = uf.find(ia), uf.find(ib)
        if ra == rb:
            continue
        if not _clusters_temporally_ok(by_id, uf.members[ra], uf.members[rb]):
            continue
        uf.union(ia, ib)
        merges.append((ia, ib, float(score)))

    id_map: dict[int, int] = {}
    for oid in ids:
        root = uf.find(oid)
        final = min(uf.members[root])
        id_map[oid] = int(final)

    n_out = len({id_map[i] for i in ids})
    return LinkResult(
        id_map=id_map,
        merges=merges,
        n_tracklets_in=len(ids),
        n_ids_out=n_out,
    )


def remap_object_ids_in_frames(
    frames: list[dict[str, Any]],
    id_map: dict[int, int],
) -> int:
    """In-place remap instance object_id. Returns number of instance remaps."""
    n = 0
    for fr in frames:
        for inst in fr.get("instances") or []:
            oid = int(inst.get("object_id", 0))
            new_id = id_map.get(oid, oid)
            if new_id != oid:
                inst["object_id"] = int(new_id)
                n += 1
    return n


def remap_object_ids_in_packets(packets: list[Any], id_map: dict[int, int]) -> int:
    """In-place remap FramePacket.stable_ids. Returns number of instance remaps."""
    n = 0
    for packet in packets:
        ids = getattr(packet, "stable_ids", None)
        if ids is None or getattr(packet, "n_inst", 0) <= 0:
            continue
        arr = np.asarray(ids, dtype=np.int64)
        for i in range(arr.shape[0]):
            oid = int(arr[i])
            new_id = int(id_map.get(oid, oid))
            if new_id != oid:
                arr[i] = new_id
                n += 1
        packet.stable_ids = arr
    return n


def sample_indices(n: int, k: int) -> list[int]:
    """Evenly spaced indices in [0, n)."""
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def enrich_tracklets_from_video(
    tracklets: list[Tracklet],
    video_path: str | Path,
    embed_fn: Callable[[list[np.ndarray]], np.ndarray],
    *,
    samples_per_tracklet: int = 5,
    pad: float = 0.05,
    on_log: Callable[[str], None] | None = None,
) -> int:
    """
    Fill missing tracklet embeddings by sampling crops from the source video.
    embed_fn(crops_bgr) -> (N, D) float32 embeddings.
    Returns number of tracklets that received an embedding.
    """
    need = [t for t in tracklets if t.embedding is None and t.n_frames > 0]
    if not need:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if on_log:
            on_log(f"Tracklet link: cannot open video for ReID crops: {video_path}")
        return 0

    # Gather (tracklet_idx, frame_idx, bbox) samples, then read frames in order.
    jobs: list[tuple[int, int, np.ndarray]] = []
    for ti, tr in enumerate(need):
        idxs = sample_indices(tr.n_frames, int(samples_per_tracklet))
        for si in idxs:
            jobs.append((ti, int(tr.frame_indices[si]), tr.bboxes_xyxy[si]))
    jobs.sort(key=lambda j: j[1])

    crops_by_ti: dict[int, list[np.ndarray]] = defaultdict(list)
    frame_cache_idx = -1
    frame_bgr: np.ndarray | None = None
    for ti, fi, xyxy in jobs:
        if fi != frame_cache_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame_bgr = cap.read()
            frame_cache_idx = fi if ok else -1
            if not ok:
                frame_bgr = None
        if frame_bgr is None:
            continue
        from app.core.reid_engine import ReidEngine

        crop = ReidEngine.crop_from_bbox(frame_bgr, xyxy, pad=pad)
        if crop is None or crop.size == 0:
            continue
        crops_by_ti[ti].append(crop)
    cap.release()

    filled = 0
    for ti, crops in crops_by_ti.items():
        if not crops:
            continue
        try:
            embs = embed_fn(crops)
        except Exception as exc:
            if on_log:
                on_log(f"Tracklet link: embed failed: {exc}")
            continue
        if embs is None or len(embs) == 0:
            continue
        mean = np.mean(np.asarray(embs, dtype=np.float32), axis=0)
        need[ti].embedding = _normalize_embedding(mean)
        if need[ti].embedding is not None:
            filled += 1
    if on_log:
        on_log(
            f"Tracklet link: computed embeddings for {filled}/{len(need)} tracklets "
            f"({samples_per_tracklet} samples/tracklet)"
        )
    return filled


def rewrite_spill_chunks_with_packets(
    manifest: dict[str, Any],
    *,
    run_dir: Path,
    packets: list[Any],
) -> None:
    """Rewrite each spill chunk pickle using remapped packets (same chunk bounds)."""
    run_dir = Path(run_dir)
    by_frame = {int(p.frame_idx): p for p in packets}
    for entry in manifest.get("chunks") or []:
        rel = str(entry.get("path") or "")
        chunk_path = run_dir / rel
        if not chunk_path.is_file():
            continue
        with chunk_path.open("rb") as f:
            data = pickle.load(f)
        old_packets = data.get("packets") or []
        new_packets = []
        for p in old_packets:
            fi = int(p.frame_idx)
            new_packets.append(by_frame.get(fi, p))
        data["packets"] = new_packets
        with chunk_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_offline_link(
    *,
    tracklets: list[Tracklet],
    max_gap_frames: int = 300,
    min_sim: float = 0.60,
    spatial_weight: float = 0.15,
    require_embedding: bool = True,
) -> LinkResult:
    """Convenience wrapper around link_tracklets."""
    return link_tracklets(
        tracklets,
        max_gap_frames=max_gap_frames,
        min_sim=min_sim,
        spatial_weight=spatial_weight,
        require_embedding=require_embedding,
    )
