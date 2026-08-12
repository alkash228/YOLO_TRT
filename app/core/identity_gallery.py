"""CPU-only persistent identity gallery (job-lifetime "DB" for person ReID).

Stores L2-normalized float32 embeddings in host RAM and optionally SQLite.
Never allocates CUDA tensors — VRAM is only touched by the ReID model when
embedding the *current* crop batch.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)


def _l2_normalize(vec: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    if n < 1e-8:
        return None
    return (arr / n).astype(np.float32, copy=False)


@dataclass(slots=True)
class GalleryHit:
    object_id: int
    similarity: float


class IdentityGallery:
    """
    Full-video identity bank queried on leave/return.

    - Hot path: numpy matrix on CPU (tiny RAM: N×D float32).
    - Cold path: optional SQLite spill (survives mid-job inspect / crash recovery).
    """

    def __init__(
        self,
        *,
        dim: int | None = None,
        min_sim: float = 0.65,
        ema_alpha: float = 0.35,
        update_min_sim: float = 0.45,
        spill_path: Path | str | None = None,
        spill_every: int = 25,
    ) -> None:
        self.min_sim = float(min_sim)
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.update_min_sim = float(update_min_sim)
        self.spill_every = max(1, int(spill_every))
        self._dim = int(dim) if dim is not None else 0
        self._ids: list[int] = []
        self._id_to_row: dict[int, int] = {}
        self._matrix: np.ndarray | None = None  # (N, D) float32, CPU
        self._last_frame: dict[int, int] = {}
        self._n_updates: dict[int, int] = {}
        self._upserts_since_spill = 0
        self._spill_path = Path(spill_path) if spill_path else None
        self._db: sqlite3.Connection | None = None
        if self._spill_path is not None:
            self._open_spill()

    @property
    def dim(self) -> int:
        return int(self._dim)

    @property
    def size(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids.clear()
        self._id_to_row.clear()
        self._matrix = None
        self._last_frame.clear()
        self._n_updates.clear()
        self._upserts_since_spill = 0
        if self._db is not None:
            self._db.execute("DELETE FROM identities")
            self._db.commit()

    def close(self) -> None:
        if self._db is not None:
            try:
                self._flush_spill(force=True)
                self._db.close()
            except Exception as exc:
                _log.warning("IdentityGallery spill close failed: %s", exc)
            self._db = None

    def upsert(
        self,
        object_id: int,
        embedding: np.ndarray,
        *,
        frame_idx: int = -1,
        force: bool = False,
    ) -> bool:
        """Insert or EMA-update an identity. Returns True if gallery changed."""
        vec = _l2_normalize(embedding)
        if vec is None:
            return False
        oid = int(object_id)
        if oid <= 0:
            return False

        if self._dim <= 0:
            self._dim = int(vec.size)
        if int(vec.size) != self._dim:
            _log.warning(
                "IdentityGallery dim mismatch oid=%s got=%s want=%s — skip",
                oid,
                vec.size,
                self._dim,
            )
            return False

        row = self._id_to_row.get(oid)
        if row is None:
            self._append_row(oid, vec, frame_idx=frame_idx)
            self._touch_spill()
            return True

        assert self._matrix is not None
        old = self._matrix[row]
        sim = float(np.dot(old, vec))
        if not force and sim < self.update_min_sim:
            # Avoid poisoning a strong identity with a bad crop.
            self._last_frame[oid] = int(frame_idx)
            return False
        a = self.ema_alpha
        mixed = _l2_normalize((1.0 - a) * old + a * vec)
        if mixed is None:
            return False
        self._matrix[row] = mixed
        self._last_frame[oid] = int(frame_idx)
        self._n_updates[oid] = int(self._n_updates.get(oid, 0)) + 1
        self._touch_spill()
        return True

    def query(
        self,
        embedding: np.ndarray,
        *,
        exclude: set[int] | None = None,
        min_sim: float | None = None,
    ) -> GalleryHit | None:
        """Best cosine match on CPU. Does not touch GPU/VRAM."""
        if self._matrix is None or not self._ids:
            return None
        vec = _l2_normalize(embedding)
        if vec is None or int(vec.size) != self._dim:
            return None
        floor = float(self.min_sim if min_sim is None else min_sim)
        # (N,) scores via single CPU GEMV
        scores = self._matrix @ vec
        ex = exclude or set()
        best_i, best_s = -1, -1.0
        for i, oid in enumerate(self._ids):
            if oid in ex:
                continue
            s = float(scores[i])
            if s > best_s:
                best_s, best_i = s, i
        if best_i < 0 or best_s < floor:
            return None
        return GalleryHit(object_id=int(self._ids[best_i]), similarity=float(best_s))

    def get(self, object_id: int) -> np.ndarray | None:
        row = self._id_to_row.get(int(object_id))
        if row is None or self._matrix is None:
            return None
        return self._matrix[row].copy()

    def _append_row(self, oid: int, vec: np.ndarray, *, frame_idx: int) -> None:
        if self._matrix is None:
            self._matrix = vec.reshape(1, -1).copy()
        else:
            self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)])
        self._id_to_row[oid] = len(self._ids)
        self._ids.append(oid)
        self._last_frame[oid] = int(frame_idx)
        self._n_updates[oid] = 1

    def _touch_spill(self) -> None:
        self._upserts_since_spill += 1
        if self._upserts_since_spill >= self.spill_every:
            self._flush_spill(force=False)

    def _open_spill(self) -> None:
        assert self._spill_path is not None
        self._spill_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._spill_path), check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS identities (
                object_id INTEGER PRIMARY KEY,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                last_frame INTEGER NOT NULL,
                n_updates INTEGER NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._db.commit()
        self._load_spill()

    def _load_spill(self) -> None:
        if self._db is None:
            return
        rows = self._db.execute(
            "SELECT object_id, dim, embedding, last_frame, n_updates FROM identities"
        ).fetchall()
        for oid, dim, blob, last_frame, n_updates in rows:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            if self._dim <= 0:
                self._dim = int(dim)
            if int(vec.size) != self._dim:
                continue
            if int(oid) in self._id_to_row:
                continue
            self._append_row(int(oid), vec, frame_idx=int(last_frame))
            self._n_updates[int(oid)] = int(n_updates)
        if rows:
            _log.info("IdentityGallery loaded %d identities from %s", len(rows), self._spill_path)

    def _flush_spill(self, *, force: bool) -> None:
        if self._db is None or self._matrix is None:
            return
        if not force and self._upserts_since_spill <= 0:
            return
        now = time.time()
        for oid, row in self._id_to_row.items():
            blob = memoryview(self._matrix[row].astype(np.float32, copy=False))
            self._db.execute(
                """
                INSERT INTO identities(object_id, dim, embedding, last_frame, n_updates, updated_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(object_id) DO UPDATE SET
                    embedding=excluded.embedding,
                    last_frame=excluded.last_frame,
                    n_updates=excluded.n_updates,
                    updated_at=excluded.updated_at
                """,
                (
                    int(oid),
                    int(self._dim),
                    sqlite3.Binary(blob.tobytes()),
                    int(self._last_frame.get(oid, -1)),
                    int(self._n_updates.get(oid, 0)),
                    now,
                ),
            )
        self._db.commit()
        self._upserts_since_spill = 0
