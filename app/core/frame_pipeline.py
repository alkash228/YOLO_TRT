"""Video preload, parallel model inference, async encode."""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import imageio
import numpy as np

from app.core.cross_check import CrossCheckDetection, draw_cross_check_detections, draw_cross_check_overlay
from app.core.exporter import overlay_instance_masks_on_frame, pose_overlay_colors
from app.core.mask_json import mask_u8_to_rle_dict, rle_list_to_stack_u8
from app.core.pose_utils import draw_pose_on_frame

PACKET_STUB_BGR = np.empty((1, 1, 3), dtype=np.uint8)


@dataclass(slots=True)
class FramePacket:
    frame_idx: int
    frame_bgr: np.ndarray
    stack: np.ndarray
    stable_ids: np.ndarray
    scores: np.ndarray
    labels: list[str]
    n_inst: int
    detect_ms: float
    seg_ms: float
    reid_ms: float
    infer_ms: float
    reid_recoveries: int
    cross_ms: float = 0.0
    keypoints_list: list[np.ndarray | None] | None = None
    cross_check_verdicts: list | None = None
    cross_check_accessories: list[CrossCheckDetection] | None = None
    # Компактный кэш (manual encode): маски в RLE, frame_bgr — заглушка.
    masks_rle: list[dict] | None = None
    mask_hw: tuple[int, int] | None = None
    instance_meta: list[dict[str, object]] | None = None


def compact_packet_for_cache(packet: FramePacket) -> FramePacket:
    """Убрать полноразмерные пиксели из RAM-кэша; MP4/JSON подгружают кадр с диска.

    Idempotent: повторный compact не должен затирать уже готовые masks_rle
    (иначе JSON/instances становятся пустыми после double-compact в GPU path).
    """
    # Already compacted — keep RLE/meta, only ensure BGR stub.
    if packet.masks_rle is not None and packet.n_inst > 0:
        if packet.frame_bgr is PACKET_STUB_BGR or (
            isinstance(packet.frame_bgr, np.ndarray) and packet.frame_bgr.size <= 3
        ):
            return packet
        return FramePacket(
            frame_idx=packet.frame_idx,
            frame_bgr=PACKET_STUB_BGR,
            stack=np.zeros((0, 1, 1), dtype=np.uint8),
            stable_ids=packet.stable_ids,
            scores=packet.scores,
            labels=list(packet.labels),
            n_inst=packet.n_inst,
            detect_ms=packet.detect_ms,
            seg_ms=packet.seg_ms,
            reid_ms=packet.reid_ms,
            infer_ms=packet.infer_ms,
            reid_recoveries=packet.reid_recoveries,
            cross_ms=packet.cross_ms,
            keypoints_list=list(packet.keypoints_list or []),
            cross_check_verdicts=list(packet.cross_check_verdicts or []),
            cross_check_accessories=list(packet.cross_check_accessories or []),
            masks_rle=list(packet.masks_rle),
            mask_hw=packet.mask_hw,
            instance_meta=list(packet.instance_meta or []) if packet.instance_meta else None,
        )

    masks_rle: list[dict] | None = None
    mask_hw: tuple[int, int] | None = None
    instance_meta: list[dict[str, object]] | None = None
    if packet.n_inst > 0 and packet.stack.size > 0:
        mask_hw = (int(packet.stack.shape[1]), int(packet.stack.shape[2]))
        masks_rle = []
        meta: list[dict[str, object]] = []
        for i in range(packet.n_inst):
            m = (packet.stack[i] > 127).astype(np.uint8)
            masks_rle.append(mask_u8_to_rle_dict(packet.stack[i].astype(np.uint8, copy=False)))
            if not m.any():
                meta.append({"bbox_xywh": [0, 0, 0, 0], "area_px": 0})
            else:
                x, y, bw, bh = cv2.boundingRect(m)
                meta.append(
                    {
                        "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
                        "area_px": int(m.sum()),
                        "center_xy": [int(x + bw // 2), int(y + bh // 2)],
                    }
                )
        instance_meta = meta
        stack = np.zeros((0, 1, 1), dtype=np.uint8)
    else:
        stack = np.zeros((0, 1, 1), dtype=np.uint8)
    return FramePacket(
        frame_idx=packet.frame_idx,
        frame_bgr=PACKET_STUB_BGR,
        stack=stack,
        stable_ids=packet.stable_ids,
        scores=packet.scores,
        labels=list(packet.labels),
        n_inst=packet.n_inst,
        detect_ms=packet.detect_ms,
        seg_ms=packet.seg_ms,
        reid_ms=packet.reid_ms,
        infer_ms=packet.infer_ms,
        reid_recoveries=packet.reid_recoveries,
        cross_ms=packet.cross_ms,
        keypoints_list=list(packet.keypoints_list or []),
        cross_check_verdicts=list(packet.cross_check_verdicts or []),
        cross_check_accessories=list(packet.cross_check_accessories or []),
        masks_rle=masks_rle,
        mask_hw=mask_hw,
        instance_meta=instance_meta,
    )


def _scale_keypoints_list(
    keypoints_list: list[np.ndarray | None] | None,
    sx: float,
    sy: float,
) -> list[np.ndarray | None]:
    if not keypoints_list:
        return []
    out: list[np.ndarray | None] = []
    for kpts in keypoints_list:
        if kpts is None or not isinstance(kpts, np.ndarray) or kpts.size == 0:
            out.append(kpts)
            continue
        arr = np.array(kpts, dtype=np.float32, copy=True)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            arr[:, 0] *= float(sx)
            arr[:, 1] *= float(sy)
        out.append(arr)
    return out


def _scale_cross_check_geometry(
    verdicts: list | None,
    accessories: list | None,
    sx: float,
    sy: float,
) -> tuple[list, list]:
    from dataclasses import replace

    scaled_verdicts: list = []
    for v in verdicts or []:
        head = getattr(v, "head_xyxy", None)
        if head is None:
            scaled_verdicts.append(v)
            continue
        h = np.asarray(head, dtype=np.float32).copy().reshape(-1)
        if h.size >= 4:
            h[0] *= float(sx)
            h[1] *= float(sy)
            h[2] *= float(sx)
            h[3] *= float(sy)
            scaled_verdicts.append(replace(v, head_xyxy=h[:4]))
        else:
            scaled_verdicts.append(v)

    scaled_acc: list = []
    for det in accessories or []:
        xyxy = getattr(det, "xyxy", None)
        if xyxy is None:
            scaled_acc.append(det)
            continue
        box = np.asarray(xyxy, dtype=np.float32).copy().reshape(-1)
        if box.size >= 4:
            box[0] *= float(sx)
            box[1] *= float(sy)
            box[2] *= float(sx)
            box[3] *= float(sy)
            scaled_acc.append(replace(det, xyxy=box[:4]))
        else:
            scaled_acc.append(det)
    return scaled_verdicts, scaled_acc


def _scale_instance_meta(
    instance_meta: list[dict[str, object]] | None,
    sx: float,
    sy: float,
) -> list[dict[str, object]] | None:
    if not instance_meta:
        return instance_meta
    out: list[dict[str, object]] = []
    for m in instance_meta:
        if not isinstance(m, dict):
            out.append(m)  # type: ignore[arg-type]
            continue
        row = dict(m)
        bb = row.get("bbox_xywh")
        if bb is not None and len(bb) >= 4:
            x, y, w, h = [float(v) for v in bb[:4]]
            row["bbox_xywh"] = [
                int(round(x * sx)),
                int(round(y * sy)),
                max(1, int(round(w * sx))),
                max(1, int(round(h * sy))),
            ]
        cxy = row.get("center_xy")
        if cxy is not None and len(cxy) >= 2:
            row["center_xy"] = [
                int(round(float(cxy[0]) * sx)),
                int(round(float(cxy[1]) * sy)),
            ]
        out.append(row)
    return out


def materialize_packet_for_render(
    packet: FramePacket,
    frame_bgr: np.ndarray | None = None,
    *,
    height: int = 0,
    width: int = 0,
) -> FramePacket:
    """Развернуть RLE-маски и подставить frame_bgr перед overlay/encode."""
    fb = frame_bgr if frame_bgr is not None else packet.frame_bgr
    src_h = src_w = 0
    if packet.mask_hw:
        src_h, src_w = int(packet.mask_hw[0]), int(packet.mask_hw[1])

    if packet.masks_rle and packet.mask_hw:
        mh, mw = packet.mask_hw
        stack = rle_list_to_stack_u8(packet.masks_rle, mh, mw)
        src_h, src_w = int(mh), int(mw)
    elif packet.stack is not None and packet.stack.ndim == 3 and packet.stack.shape[0] > 0:
        stack = packet.stack
        src_h, src_w = int(stack.shape[1]), int(stack.shape[2])
    else:
        h = height or (packet.mask_hw[0] if packet.mask_hw else 1)
        w = width or (packet.mask_hw[1] if packet.mask_hw else 1)
        stack = np.zeros((0, h, w), dtype=np.uint8)

    kpts = list(packet.keypoints_list or [])
    verdicts = list(packet.cross_check_verdicts or [])
    accessories = list(packet.cross_check_accessories or [])
    instance_meta = list(packet.instance_meta or []) if packet.instance_meta else None

    # Masks / pose / head boxes live in inference coords; source frame may differ.
    if (
        fb is not None
        and getattr(fb, "ndim", 0) == 3
        and fb.shape[0] > 2
        and fb.shape[1] > 2
        and src_h > 0
        and src_w > 0
    ):
        fh, fw = int(fb.shape[0]), int(fb.shape[1])
        if (src_h, src_w) != (fh, fw):
            sx = float(fw) / float(src_w)
            sy = float(fh) / float(src_h)
            if stack.ndim == 3 and stack.shape[0] > 0:
                stack = np.stack(
                    [
                        cv2.resize(stack[i], (fw, fh), interpolation=cv2.INTER_NEAREST)
                        for i in range(int(stack.shape[0]))
                    ],
                    axis=0,
                )
            kpts = _scale_keypoints_list(kpts, sx, sy)
            verdicts, accessories = _scale_cross_check_geometry(verdicts, accessories, sx, sy)
            instance_meta = _scale_instance_meta(instance_meta, sx, sy)

    return FramePacket(
        frame_idx=packet.frame_idx,
        frame_bgr=fb,
        stack=stack,
        stable_ids=packet.stable_ids,
        scores=packet.scores,
        labels=list(packet.labels),
        n_inst=packet.n_inst,
        detect_ms=packet.detect_ms,
        seg_ms=packet.seg_ms,
        reid_ms=packet.reid_ms,
        infer_ms=packet.infer_ms,
        reid_recoveries=packet.reid_recoveries,
        cross_ms=packet.cross_ms,
        keypoints_list=kpts,
        cross_check_verdicts=verdicts,
        cross_check_accessories=accessories,
        masks_rle=None,
        mask_hw=None,
        instance_meta=instance_meta,
    )


@dataclass(slots=True)
class PostResult:
    frame_idx: int
    inst_rows: list[dict[str, object]]
    rgb: np.ndarray
    composed_bgr: np.ndarray
    post_ms: float


def estimate_video_ram_gb(width: int, height: int, frame_count: int) -> float:
    return (frame_count * height * width * 3) / (1024**3)


def load_all_frames(
    cap: cv2.VideoCapture,
    max_frames: int,
    *,
    max_ram_gb: float = 8.0,
    width: int = 0,
    height: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
    progress_every: int = 16,
) -> list[np.ndarray]:
    """Decode entire video (or up to max_frames) into RAM."""
    frames: list[np.ndarray] = []
    limit = max_frames if max_frames > 0 else 10**9
    total_hint = limit if max_frames > 0 else 0
    every = max(1, int(progress_every))
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if on_progress is not None and (
            len(frames) % every == 0 or (total_hint > 0 and len(frames) >= total_hint)
        ):
            on_progress(len(frames), total_hint if total_hint > 0 else len(frames))
        if width > 0 and height > 0 and len(frames) % 100 == 0:
            used = estimate_video_ram_gb(width, height, len(frames))
            if used > max_ram_gb:
                raise MemoryError(
                    f"Video preload exceeds max_preload_ram_gb={max_ram_gb:.1f} "
                    f"(estimated {used:.2f} GB after {len(frames)} frames)"
                )
    if on_progress is not None and frames:
        on_progress(len(frames), total_hint if total_hint > 0 else len(frames))
    return frames


class ModelParallelRunner:
    """Run detect and optionally seg concurrently on the same frame."""

    def __init__(self, detect_engine, seg_engine=None) -> None:
        self._detect = detect_engine
        self._seg = seg_engine
        workers = 2 if seg_engine is not None else 1
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="yolo-drt-models")

    def run(self, frame: np.ndarray) -> tuple[list, list, float, float, float]:
        t0 = time.perf_counter()
        f_det = self._pool.submit(self._detect.track, frame)
        f_seg = self._pool.submit(self._seg.predict, frame) if self._seg is not None else None
        t_det0 = time.perf_counter()
        detections_raw = f_det.result()
        detect_ms = (time.perf_counter() - t_det0) * 1000.0
        if f_seg is not None:
            t_seg0 = time.perf_counter()
            segments_raw = f_seg.result()
            seg_ms = (time.perf_counter() - t_seg0) * 1000.0
        else:
            segments_raw = []
            seg_ms = 0.0
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return detections_raw, segments_raw, detect_ms, seg_ms, wall_ms

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


class AsyncVideoWriter:
    """Background x264/NVENC encode — render loop never blocks on ffmpeg."""

    def __init__(
        self,
        path: str,
        fps: float,
        *,
        codec: str = "libx264",
        ffmpeg_params: list[str] | None = None,
        ffmpeg_exe: str | None = None,
        queue_size: int = 64,
    ) -> None:
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(8, queue_size))
        if ffmpeg_exe:
            import os

            os.environ["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg_exe)
        self._writer = imageio.get_writer(
            path,
            fps=fps,
            codec=codec,
            ffmpeg_params=ffmpeg_params or ["-crf", "23", "-preset", "fast"],
        )
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._loop, name="yolo-drt-encode", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._writer.append_data(item)
        except Exception as exc:
            self._error = exc

    def submit(self, rgb: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put(np.ascontiguousarray(rgb))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=600.0)
        self._writer.close()
        if self._error is not None:
            raise self._error


class OrderedPostExecutor:
    """Overlay + RLE in thread pool; results collected in frame order."""

    def __init__(self, workers: int, process_fn: Callable[[FramePacket], PostResult]) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="yolo-drt-post")
        self._process_fn = process_fn
        self._futures: dict[int, Future[PostResult]] = {}
        self._next_out = 0

    def submit(self, packet: FramePacket, *, frame_source: Sequence[np.ndarray] | None = None) -> None:
        src = list(frame_source) if frame_source is not None else None
        self._futures[packet.frame_idx] = self._executor.submit(
            self._process_fn,
            packet,
            src,
        )

    def drain(self) -> list[PostResult]:
        out: list[PostResult] = []
        while self._next_out in self._futures and self._futures[self._next_out].done():
            out.append(self._futures.pop(self._next_out).result())
            self._next_out += 1
        return out

    def finish(self) -> list[PostResult]:
        out: list[PostResult] = []
        while self._next_out in self._futures:
            out.append(self._futures.pop(self._next_out).result())
            self._next_out += 1
        return out

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def post_process_frame(
    packet: FramePacket,
    *,
    serialize_fn: Callable[..., list[dict[str, object]]],
    prompt_id_lookup: dict[str, int],
    overlay_alpha: float,
    draw_boxes: bool,
    draw_masks: bool,
    draw_centers: bool,
    draw_pose: bool = True,
    pose_kpt_conf: float = 0.25,
    cross_check_enabled: bool = False,
    cross_check_draw_head_box: bool = True,
    cross_check_draw_boxes: bool = True,
    pose_point_radius: int = 4,
    pose_line_thickness: int = 2,
) -> PostResult:
    t0 = time.perf_counter()
    kpts = packet.keypoints_list or []
    verdicts = packet.cross_check_verdicts or []
    inst_rows = serialize_fn(
        packet.stack if packet.n_inst else None,
        packet.stable_ids if packet.n_inst else None,
        packet.scores if packet.n_inst else None,
        packet.labels if packet.n_inst else None,
        prompt_id_lookup,
        keypoints_list=kpts,
        pose_kpt_conf=pose_kpt_conf,
        cross_check_verdicts=verdicts,
    )
    composed = overlay_instance_masks_on_frame(
        packet.frame_bgr,
        packet.stack,
        alpha=overlay_alpha,
        object_ids=packet.stable_ids if packet.n_inst else None,
        scores=packet.scores if packet.n_inst else None,
        prompt_labels=packet.labels if packet.n_inst else None,
        draw_boxes=draw_boxes,
        draw_masks=draw_masks,
        draw_centers=draw_centers,
    )
    if draw_pose and kpts:
        colors = pose_overlay_colors(max(packet.n_inst, len(kpts)))
        composed = draw_pose_on_frame(
            composed,
            kpts,
            colors,
            kpt_conf=pose_kpt_conf,
            draw_skeleton=True,
            point_radius=int(max(3, pose_point_radius)),
            line_thickness=int(max(2, pose_line_thickness)),
        )
    accessories = packet.cross_check_accessories or []
    if cross_check_enabled and cross_check_draw_boxes and accessories:
        composed = draw_cross_check_detections(composed, accessories)
    if cross_check_enabled and verdicts:
        composed = draw_cross_check_overlay(
            composed,
            verdicts,
            draw_head_box=cross_check_draw_head_box,
        )
    rgb = np.ascontiguousarray(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB))
    post_ms = (time.perf_counter() - t0) * 1000.0
    return PostResult(
        frame_idx=packet.frame_idx,
        inst_rows=inst_rows,
        rgb=rgb,
        composed_bgr=composed,
        post_ms=post_ms,
    )


def make_infer_packet(
    frame_idx: int,
    frame_bgr: np.ndarray,
    stack: np.ndarray,
    stable_ids: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    n_inst: int,
    detect_ms: float,
    seg_ms: float,
    reid_ms: float,
    infer_ms: float,
    reid_recoveries: int,
    cross_ms: float = 0.0,
    keypoints_list: list[np.ndarray | None] | None = None,
    cross_check_verdicts: list | None = None,
    cross_check_accessories: list[CrossCheckDetection] | None = None,
) -> FramePacket:
    return FramePacket(
        frame_idx=frame_idx,
        frame_bgr=frame_bgr,
        stack=stack,
        stable_ids=stable_ids,
        scores=scores,
        labels=labels,
        n_inst=n_inst,
        detect_ms=detect_ms,
        seg_ms=seg_ms,
        reid_ms=reid_ms,
        cross_ms=cross_ms,
        infer_ms=infer_ms,
        reid_recoveries=reid_recoveries,
        keypoints_list=keypoints_list or [],
        cross_check_verdicts=cross_check_verdicts or [],
        cross_check_accessories=cross_check_accessories or [],
    )
