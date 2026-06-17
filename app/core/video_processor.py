"""Offline video pipeline: YOLO26 detect/seg + OSNet ReID."""
from __future__ import annotations

import json
import queue as queue_module
import subprocess
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

from app.config.settings import PipelineSettings
from app.core.detect_engine import DetectEngine, DetectItem
from app.core.frame_pipeline import (
    AsyncVideoWriter,
    FramePacket,
    ModelParallelRunner,
    OrderedPostExecutor,
    PostResult,
    compact_packet_for_cache,
    estimate_video_ram_gb,
    load_all_frames,
    make_infer_packet,
    materialize_packet_for_render,
    post_process_frame,
)
from app.core.cross_check import CrossCheckDetection, CrossCheckVerdict, evaluate_cross_check_batch
from app.core.fusion import inherit_motion_ids, match_detections_to_segments, merge_seg_fallback_detections
from app.core.async_finalize import AsyncBatchFinalizer
from app.core.batch_prepare import prepare_batch_frames
from app.core.batch_utils import (
    cap_job_batch_long_video,
    effective_speed_tuning,
    frame_stride_summary,
    resolve_batch_prefetch_depth,
    resolve_gpu_queue_depth,
    resolve_infer_batch_size,
    resolve_reid_embed_chunk,
    resolve_use_streaming,
    source_frame_indices,
)
from app.core.gpu_monitor import GpuMonitor
from app.core.gpu_pipeline import (
    FrameBatchJob,
    GpuInferWorker,
    make_async_batch_jobs,
)
from app.core.motion_tracker import MotionTracker
from app.core.run_registry import (
    append_run_record,
    build_run_record,
    generate_run_charts,
    read_metrics_jsonl,
)
from app.core.gpu_cleanup import release_gpu_memory
from app.core.mask_json import mask_u8_to_rle_dict
from app.core.memory_stats import build_memory_report
from app.core.video_encode import (
    expand_packets_to_timeline,
    packets_path_for_run,
    save_run_packets,
)
from app.core.pose_utils import keypoints_to_json
from app.core.prompt_utils import label_match, prompt_terms
from app.core.reid_engine import ReidEngine
from app.core.reid_tracker import ReidTracker
from app.core.schema import (
    ProcessVideoResult,
    RunStats,
    VideoProgress,
    build_result_payload,
    label_slug,
    prompt_id_lookup_from_prompt,
    run_stats_to_dict,
    stats_summary_from_counters,
    video_data_json_dumps,
)
from app.core.seg_engine import SegEngine


class VideoProcessor:
    def __init__(
        self,
        detect_engine: DetectEngine,
        seg_engine: SegEngine | None,
        reid_engine: ReidEngine | None,
        settings: PipelineSettings,
        cross_check_engine: DetectEngine | None = None,
    ) -> None:
        self.detect_engine = detect_engine
        self.seg_engine = seg_engine
        self.reid_engine = reid_engine
        self.cross_check_engine = cross_check_engine
        self.settings = settings
        self.cross_check_violations_total = 0
        self._job_batch_size = 16
        self.motion_tracker = MotionTracker()
        self.tracker = ReidTracker(
            appearance_thresh=settings.appearance_thresh,
            track_buffer=settings.track_buffer,
            gallery_size=settings.reid_gallery_size,
            w_iou=settings.w_iou,
            w_app=settings.w_app,
            recovery_thresh=settings.recovery_thresh,
        )

    @staticmethod
    def _gpu_mem_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.memory_allocated() / (1024 * 1024))

    @staticmethod
    def _resolve_prompt_id(
        label: str,
        prompt_id_lookup: dict[str, int],
        fallback_lookup: dict[str, int],
        next_fallback_id: list[int],
    ) -> int | None:
        norm = label.strip().casefold()
        if not norm:
            return None
        if norm in prompt_id_lookup:
            return prompt_id_lookup[norm]
        for key, pid in prompt_id_lookup.items():
            if key in norm or norm in key:
                return pid
        if norm in fallback_lookup:
            return fallback_lookup[norm]
        pid = next_fallback_id[0]
        fallback_lookup[norm] = pid
        next_fallback_id[0] += 1
        return pid

    @classmethod
    def serialize_frame_instances(
        cls,
        instance_stack: np.ndarray | None,
        instance_object_ids: np.ndarray | None,
        instance_scores: np.ndarray | None,
        prompt_labels: list[str] | None,
        prompt_id_lookup: dict[str, int] | None = None,
        pose_kpt_conf: float = 0.25,
        keypoints_list: list[np.ndarray | None] | None = None,
        cross_check_verdicts: list[CrossCheckVerdict] | None = None,
    ) -> list[dict[str, object]]:
        if (
            instance_stack is None
            or not isinstance(instance_stack, np.ndarray)
            or instance_stack.ndim != 3
            or instance_stack.shape[0] == 0
        ):
            return []
        out: list[dict[str, object]] = []
        n = int(instance_stack.shape[0])
        pls = prompt_labels or []
        fallback_lookup: dict[str, int] = {}
        next_fallback_id = [max(prompt_id_lookup.values(), default=0) + 1] if prompt_id_lookup else [1]
        for i in range(n):
            m = (instance_stack[i] > 127).astype(np.uint8)
            if not m.any():
                continue
            x, y, bw, bh = cv2.boundingRect(m)
            oid = (
                int(instance_object_ids[i])
                if instance_object_ids is not None and i < len(instance_object_ids)
                else i + 1
            )
            score: float | None = None
            if instance_scores is not None and i < len(instance_scores):
                score = float(instance_scores[i])
            label = str(pls[i]) if i < len(pls) else ""
            mask_enc = mask_u8_to_rle_dict(instance_stack[i].astype(np.uint8, copy=False))
            row: dict[str, object] = {
                "object_id": oid,
                "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
                "area_px": int(m.sum()),
                "center_xy": [int(x + bw // 2), int(y + bh // 2)],
                "score": score,
                "prompt_label": label,
                "mask": mask_enc,
            }
            pid = cls._resolve_prompt_id(
                label=label,
                prompt_id_lookup=prompt_id_lookup or {},
                fallback_lookup=fallback_lookup,
                next_fallback_id=next_fallback_id,
            )
            if pid is not None:
                row["prompt_id"] = int(pid)
                slug = label_slug(label)
                if slug:
                    row[f"{slug}_id"] = int(pid)
            if keypoints_list is not None and i < len(keypoints_list):
                kpts = keypoints_list[i]
                if kpts is not None:
                    row["keypoints"] = keypoints_to_json(kpts, pose_kpt_conf)
            if cross_check_verdicts is not None and i < len(cross_check_verdicts):
                v = cross_check_verdicts[i]
                row["cross_check_ok"] = bool(v.ok)
                if not v.ok and v.warning:
                    row["warning"] = v.warning
                row["cross_check_intersection_px"] = round(v.best_intersection_px, 2)
                if v.head_xyxy is not None:
                    hx0, hy0, hx1, hy1 = [float(x) for x in v.head_xyxy.tolist()]
                    row["head_bbox_xyxy"] = [int(hx0), int(hy0), int(hx1), int(hy1)]
            out.append(row)
        return out

    def _make_post_fn(self, prompt_id_lookup: dict[str, int]):
        settings = self.settings

        def _run(
            packet: FramePacket,
            frame_source: list[np.ndarray] | None = None,
        ) -> PostResult:
            fb = packet.frame_bgr
            if frame_source is not None and 0 <= packet.frame_idx < len(frame_source):
                fb = frame_source[packet.frame_idx]
            work = materialize_packet_for_render(packet, fb)
            return post_process_frame(
                work,
                serialize_fn=self.serialize_frame_instances,
                prompt_id_lookup=prompt_id_lookup,
                overlay_alpha=settings.overlay_alpha,
                draw_boxes=settings.draw_boxes,
                draw_masks=settings.draw_masks,
                draw_centers=settings.draw_centers,
                draw_pose=settings.draw_pose,
                pose_kpt_conf=settings.pose_kpt_conf,
                cross_check_enabled=settings.cross_check_enabled,
                cross_check_draw_head_box=settings.cross_check_draw_head_box,
                cross_check_draw_boxes=settings.cross_check_draw_boxes,
            )

        return _run

    def _frames_payload_from_packets(
        self,
        packets: list[FramePacket],
        prompt_id_lookup: dict[str, int],
    ) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for packet in sorted(packets, key=lambda p: p.frame_idx):
            out.append(self._frame_dict_from_packet(packet, prompt_id_lookup))
        return out

    def _frame_dict_from_packet(
        self,
        packet: FramePacket,
        prompt_id_lookup: dict[str, int],
    ) -> dict[str, object]:
        if packet.masks_rle and packet.instance_meta and packet.n_inst > 0:
            inst_rows = self._instances_from_compact_packet(packet, prompt_id_lookup)
        else:
            kpts = packet.keypoints_list or []
            verdicts = packet.cross_check_verdicts or []
            inst_rows = self.serialize_frame_instances(
                packet.stack if packet.n_inst else None,
                packet.stable_ids if packet.n_inst else None,
                packet.scores if packet.n_inst else None,
                packet.labels if packet.n_inst else None,
                prompt_id_lookup,
                keypoints_list=kpts,
                pose_kpt_conf=self.settings.pose_kpt_conf,
                cross_check_verdicts=verdicts,
            )
        return {"frame": int(packet.frame_idx), "instances": inst_rows}

    def _instances_from_compact_packet(
        self,
        packet: FramePacket,
        prompt_id_lookup: dict[str, int],
    ) -> list[dict[str, object]]:
        masks_rle = packet.masks_rle or []
        meta = packet.instance_meta or []
        kpts = packet.keypoints_list or []
        verdicts = packet.cross_check_verdicts or []
        pls = packet.labels or []
        fallback_lookup: dict[str, int] = {}
        next_fallback_id = [max(prompt_id_lookup.values(), default=0) + 1] if prompt_id_lookup else [1]
        out: list[dict[str, object]] = []
        for i in range(packet.n_inst):
            if i >= len(masks_rle):
                break
            m = meta[i] if i < len(meta) else {}
            bbox = m.get("bbox_xywh", [0, 0, 0, 0])
            area = int(m.get("area_px", 0))
            if area <= 0:
                continue
            label = str(pls[i]) if i < len(pls) else ""
            oid = (
                int(packet.stable_ids[i])
                if packet.stable_ids is not None and i < len(packet.stable_ids)
                else i + 1
            )
            score: float | None = None
            if packet.scores is not None and i < len(packet.scores):
                score = float(packet.scores[i])
            center = m.get("center_xy")
            if not center and len(bbox) == 4:
                x, y, bw, bh = bbox
                center = [int(x + bw // 2), int(y + bh // 2)]
            row: dict[str, object] = {
                "object_id": oid,
                "bbox_xywh": list(bbox),
                "area_px": area,
                "center_xy": list(center) if center else [0, 0],
                "score": score,
                "prompt_label": label,
                "mask": masks_rle[i],
            }
            pid = self._resolve_prompt_id(
                label=label,
                prompt_id_lookup=prompt_id_lookup,
                fallback_lookup=fallback_lookup,
                next_fallback_id=next_fallback_id,
            )
            if pid is not None:
                row["prompt_id"] = int(pid)
                slug = label_slug(label)
                if slug:
                    row[f"{slug}_id"] = int(pid)
            if i < len(kpts) and kpts[i] is not None:
                row["keypoints"] = keypoints_to_json(kpts[i], self.settings.pose_kpt_conf)
            if i < len(verdicts):
                v = verdicts[i]
                row["cross_check_ok"] = bool(v.ok)
                if not v.ok and v.warning:
                    row["warning"] = v.warning
                row["cross_check_intersection_px"] = round(v.best_intersection_px, 2)
                if v.head_xyxy is not None:
                    hx0, hy0, hx1, hy1 = [float(x) for x in v.head_xyxy.tolist()]
                    row["head_bbox_xyxy"] = [int(hx0), int(hy0), int(hx1), int(hy1)]
            out.append(row)
        return out

    def _append_deferred_packet(
        self,
        packet: FramePacket,
        deferred_packets: list[FramePacket],
        *,
        compact: bool = False,
    ) -> None:
        deferred_packets.append(compact_packet_for_cache(packet) if compact else packet)

    def _pulse_infer_progress(
        self,
        *,
        current_frame: int,
        total: int,
        instances_peak: int,
        start_ts: float,
        on_progress: Callable[[VideoProgress], None] | None,
        gpu_monitor: GpuMonitor | None,
        phase: str = "inference",
    ) -> None:
        if on_progress is None:
            return
        out_frame = max(0, int(current_frame))
        elapsed = time.perf_counter() - start_ts
        proc_fps = out_frame / elapsed if elapsed > 0 and out_frame > 0 else 0.0
        eta = ((total - out_frame) / proc_fps) if proc_fps > 0 and total > 0 else 0.0
        gpu_util = 0.0
        if gpu_monitor is not None:
            latest = gpu_monitor.latest()
            if latest is not None:
                gpu_util = latest.gpu_util_pct
        on_progress(
            VideoProgress(
                current=out_frame,
                total=total if total > 0 else max(1, out_frame),
                fps=proc_fps,
                eta_seconds=eta,
                gpu_mem_mb=self._gpu_mem_mb(),
                elapsed_sec=elapsed,
                gpu_util_pct=gpu_util,
                instances_current=0,
                instances_peak=instances_peak,
                stats=None,
            )
        )

    def _report_infer_progress(
        self,
        *,
        packet: FramePacket,
        total: int,
        instances_peak: int,
        start_ts: float,
        infer_ms_by_frame: dict[int, float],
        metrics_path: Path,
        on_progress: Callable[[VideoProgress], None] | None,
        gpu_monitor: GpuMonitor | None,
        phase: str = "inference",
        infer_ms: float | None = None,
        write_metrics: bool = True,
    ) -> None:
        out_frame = int(packet.frame_idx) + 1
        elapsed = time.perf_counter() - start_ts
        proc_fps = out_frame / elapsed if elapsed > 0 else 0.0
        eta = ((total - out_frame) / proc_fps) if proc_fps > 0 and total > 0 else 0.0
        gpu_util = 0.0
        if gpu_monitor is not None:
            latest = gpu_monitor.latest()
            if latest is not None:
                gpu_util = latest.gpu_util_pct
        infer_ms_val = (
            float(infer_ms)
            if infer_ms is not None
            else float(infer_ms_by_frame.get(packet.frame_idx, packet.infer_ms))
        )
        stats = RunStats(
            frame=out_frame,
            total_frames=total if total > 0 else out_frame,
            fps=proc_fps,
            eta_sec=eta,
            gpu_mem_mb=self._gpu_mem_mb(),
            instances_current=packet.n_inst,
            instances_peak=instances_peak,
            id_switches=self.tracker.total_id_switches,
            reid_recoveries=self.tracker.total_reid_recoveries,
            detect_ms=packet.detect_ms,
            seg_ms=packet.seg_ms,
            reid_ms=packet.reid_ms,
            total_ms=infer_ms_val,
            elapsed_sec=elapsed,
            gpu_util_pct=gpu_util,
        )
        if write_metrics:
            with metrics_path.open("a", encoding="utf-8") as mf:
                row = run_stats_to_dict(stats)
                row["phase"] = phase
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
        if on_progress is not None:
            on_progress(
                VideoProgress(
                    current=out_frame,
                    total=total if total > 0 else out_frame,
                    fps=proc_fps,
                    eta_seconds=eta,
                    gpu_mem_mb=stats.gpu_mem_mb,
                    elapsed_sec=elapsed,
                    gpu_util_pct=gpu_util,
                    instances_current=packet.n_inst,
                    instances_peak=instances_peak,
                    stats=stats,
                )
            )

    def _emit_post_results(
        self,
        results: list[PostResult],
        *,
        video_sink,
        frames_payload: list[dict[str, object]],
        metrics_path: Path,
        total: int,
        frame_idx_ref: list[int],
        instances_peak: int,
        detect_ms_acc: list[float],
        seg_ms_acc: list[float],
        reid_ms_acc: list[float],
        total_ms_acc: list[float],
        infer_ms_by_frame: dict[int, float],
        packet_meta: dict[int, FramePacket],
        start_ts: float,
        on_progress: Callable[[VideoProgress], None] | None,
        on_preview: Callable[[np.ndarray], None] | None,
        on_debug_log: Callable[[str], None] | None,
        gpu_monitor: GpuMonitor | None = None,
    ) -> int:
        for result in results:
            packet = packet_meta.pop(result.frame_idx, None)
            if packet is None:
                continue

            if video_sink is not None:
                video_sink.submit(result.rgb)
            frames_payload.append({"frame": int(result.frame_idx), "instances": result.inst_rows})

            infer_ms = infer_ms_by_frame.pop(result.frame_idx, packet.infer_ms)
            total_ms = infer_ms + result.post_ms
            detect_ms_acc[0] += packet.detect_ms
            seg_ms_acc[0] += packet.seg_ms
            reid_ms_acc[0] += packet.reid_ms
            total_ms_acc[0] += total_ms

            out_frame = result.frame_idx + 1
            elapsed = time.perf_counter() - start_ts
            proc_fps = out_frame / elapsed if elapsed > 0 else 0.0
            eta = ((total - out_frame) / proc_fps) if proc_fps > 0 and total > 0 else 0.0
            gpu_util = 0.0
            if gpu_monitor is not None:
                latest = gpu_monitor.latest()
                if latest is not None:
                    gpu_util = latest.gpu_util_pct

            stats = RunStats(
                frame=out_frame,
                total_frames=total if total > 0 else out_frame,
                fps=proc_fps,
                eta_sec=eta,
                gpu_mem_mb=self._gpu_mem_mb(),
                instances_current=packet.n_inst,
                instances_peak=instances_peak,
                id_switches=self.tracker.total_id_switches,
                reid_recoveries=self.tracker.total_reid_recoveries,
                detect_ms=packet.detect_ms,
                seg_ms=packet.seg_ms,
                reid_ms=packet.reid_ms,
                total_ms=total_ms,
                elapsed_sec=elapsed,
                gpu_util_pct=gpu_util,
            )

            if on_progress:
                on_progress(
                    VideoProgress(
                        current=out_frame,
                        total=total if total > 0 else out_frame,
                        fps=proc_fps,
                        eta_seconds=eta,
                        gpu_mem_mb=stats.gpu_mem_mb,
                        elapsed_sec=elapsed,
                        gpu_util_pct=gpu_util,
                        instances_current=packet.n_inst,
                        instances_peak=instances_peak,
                        stats=stats,
                    )
                )

            if on_preview and result.frame_idx % max(1, self.settings.preview_every_n) == 0:
                on_preview(result.composed_bgr.copy())

            with metrics_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(run_stats_to_dict(stats), ensure_ascii=False) + "\n")

            if on_debug_log and packet.reid_recoveries > 0:
                on_debug_log(f"[f={result.frame_idx}] ReID recovery x{packet.reid_recoveries}")
            if on_debug_log and result.inst_rows:
                warns = [r.get("warning") for r in result.inst_rows if r.get("warning")]
                if warns:
                    on_debug_log(f"[f={result.frame_idx}] " + ", ".join(str(w) for w in warns))

            frame_idx_ref[0] = out_frame
        return instances_peak

    def _run_detect_seg(
        self,
        frame: np.ndarray,
        model_runner: ModelParallelRunner | None,
    ) -> tuple[list, list, float, float, float]:
        use_seg = self.settings.use_seg and self.seg_engine is not None
        if model_runner is not None:
            return model_runner.run(frame)
        t0 = time.perf_counter()
        detections_raw = self.detect_engine.track(frame)
        detect_ms = (time.perf_counter() - t0) * 1000.0
        if use_seg:
            t0 = time.perf_counter()
            segments_raw = self.seg_engine.predict(frame)
            seg_ms = (time.perf_counter() - t0) * 1000.0
        else:
            segments_raw = []
            seg_ms = 0.0
        wall_ms = detect_ms + seg_ms
        return detections_raw, segments_raw, detect_ms, seg_ms, wall_ms

    def _infer_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        terms: list[str],
        height: int,
        width: int,
        model_runner: ModelParallelRunner | None,
    ) -> tuple[FramePacket, int]:
        t_infer = time.perf_counter()

        detections_raw, segments_raw, detect_ms, seg_ms, model_wall_ms = self._run_detect_seg(
            frame, model_runner
        )

        detections = [d for d in detections_raw if label_match(d.label, terms)]
        segments = [s for s in segments_raw if label_match(s.label, terms)]
        if self.settings.use_seg and self.seg_engine is not None:
            detections = merge_seg_fallback_detections(
                detections,
                segments,
                match_iou_min=self.settings.seg_fallback_iou_min,
            )
        use_reid = self.settings.use_reid and self.reid_engine is not None
        if use_reid:
            detections = inherit_motion_ids(detections)

        crops: list[np.ndarray] = []
        valid_dets: list[DetectItem] = []
        for det in detections:
            crop = ReidEngine.crop_from_bbox(frame, det.xyxy)
            if crop is not None and crop.size > 0:
                crops.append(crop)
                valid_dets.append(det)

        t0 = time.perf_counter()
        if use_reid:
            embeddings = self.reid_engine.embed_batch(crops)
            reid_ms = (time.perf_counter() - t0) * 1000.0
            track_result = self.tracker.update(valid_dets, embeddings)
            stable_ids = (
                np.array(track_result.stable_ids, dtype=np.int64)
                if track_result.stable_ids
                else np.zeros((0,), dtype=np.int64)
            )
            reid_recoveries = track_result.reid_recoveries
        else:
            reid_ms = 0.0
            stable_ids = np.array(
                [
                    int(d.motion_id) if d.motion_id >= 0 else i + 1
                    for i, d in enumerate(valid_dets)
                ],
                dtype=np.int64,
            )
            reid_recoveries = 0

        masks = match_detections_to_segments(
            valid_dets,
            segments,
            match_iou_min=self.settings.match_iou_min,
            frame_h=height,
            frame_w=width,
        )

        n_inst = len(valid_dets)
        if n_inst > 0:
            stack = np.stack(masks, axis=0).astype(np.uint8)
            scores = np.array([d.conf for d in valid_dets], dtype=np.float32)
            labels = [d.label for d in valid_dets]
            keypoints_list = [d.keypoints for d in valid_dets]
        else:
            stack = np.zeros((0, height, width), dtype=np.uint8)
            scores = np.zeros((0,), dtype=np.float32)
            labels = []
            keypoints_list = []

        if model_runner is not None:
            infer_ms = model_wall_ms + reid_ms
        else:
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

        cross_verdicts: list[CrossCheckVerdict] = []
        cross_accessories: list[CrossCheckDetection] = []
        cross_ms = 0.0
        if (
            self.settings.cross_check_enabled
            and self.cross_check_engine is not None
            and valid_dets
        ):
            obj_terms = prompt_terms(self.settings.cross_check_object_prompt)
            t_cross = time.perf_counter()
            aux_raw = self.cross_check_engine.predict(frame)
            cross_ms = (time.perf_counter() - t_cross) * 1000.0
            accessories = [d for d in aux_raw if label_match(d.label, obj_terms)]
            cross_accessories = [
                CrossCheckDetection(d.label, float(d.conf), d.xyxy.copy()) for d in accessories
            ]
            cross_verdicts = evaluate_cross_check_batch(
                valid_dets,
                accessories,
                kpt_conf=self.settings.pose_kpt_conf,
                min_intersection_px=self.settings.cross_check_min_intersection_px,
                min_iou=self.settings.cross_check_min_iou,
                frame_w=width,
                frame_h=height,
                warn_text=self.settings.cross_check_warning_text,
            )
            self.cross_check_violations_total += sum(1 for v in cross_verdicts if not v.ok)

        packet = make_infer_packet(
            frame_idx=frame_idx,
            frame_bgr=frame,
            stack=stack,
            stable_ids=stable_ids,
            scores=scores,
            labels=labels,
            n_inst=n_inst,
            detect_ms=detect_ms,
            seg_ms=seg_ms,
            reid_ms=reid_ms,
            infer_ms=infer_ms + cross_ms,
            reid_recoveries=reid_recoveries,
            cross_ms=cross_ms,
            keypoints_list=keypoints_list,
            cross_check_verdicts=cross_verdicts,
            cross_check_accessories=cross_accessories,
        )
        return packet, n_inst

    def _use_gpu_pipeline(self) -> bool:
        dev = str(getattr(self.settings, "inference_device", "cuda") or "cuda").casefold()
        return bool(
            self.settings.gpu_pipeline
            and dev != "cpu"
            and torch.cuda.is_available()
        )

    def _overlay_snapshot(self) -> dict[str, object]:
        s = self.settings
        return {
            "overlay_alpha": float(s.overlay_alpha),
            "draw_boxes": bool(s.draw_boxes),
            "draw_masks": bool(s.draw_masks),
            "draw_centers": bool(s.draw_centers),
            "draw_pose": bool(s.draw_pose),
            "pose_kpt_conf": float(s.pose_kpt_conf),
            "cross_check_enabled": bool(s.cross_check_enabled),
            "cross_check_draw_head_box": bool(s.cross_check_draw_head_box),
            "cross_check_draw_boxes": bool(s.cross_check_draw_boxes),
        }

    def _make_batch_job_iter(
        self,
        *,
        all_frames: list[np.ndarray] | None,
        cap_stream: cv2.VideoCapture | None,
        n_frames_total: int,
        process_indices: list[int],
        frame_stride: int,
        should_stop: Callable[[], bool] | None,
        should_pause: Callable[[], bool] | None,
        on_debug_log: Callable[[str], None] | None,
    ) -> Iterator[FrameBatchJob]:
        from app.core.gpu_pipeline import chunk_frame_jobs

        bs = self._job_batch_size
        n_jobs = (len(process_indices) + bs - 1) // bs if process_indices else 0
        if all_frames is not None:
            indices = process_indices if process_indices else list(range(len(all_frames)))
            selected = [all_frames[i] for i in indices]
            jobs = chunk_frame_jobs(
                selected,
                batch_size=bs,
                frame_indices=indices,
            )
            if on_debug_log:
                on_debug_log(
                    f"Batch jobs: {bs} fr/job, {len(jobs)} jobs, preload RAM, stride={frame_stride}"
                )
            return iter(jobs)

        depth = resolve_batch_prefetch_depth(self.settings)
        feeder = make_async_batch_jobs(
            all_frames=None,
            cap=cap_stream,
            n_frames_total=n_frames_total,
            process_indices=process_indices,
            batch_size=bs,
            queue_depth=depth,
            frame_stride=frame_stride,
            should_stop=should_stop,
            should_pause=should_pause,
        )
        if on_debug_log:
            on_debug_log(
                f"Async decode loader: queue={depth}, {bs} fr/job, ~{n_jobs} jobs, "
                f"streaming, stride={frame_stride}"
            )
        return feeder

    def _cpu_finalize_batch(
        self,
        batch,
        frames_bgr: list[np.ndarray],
        terms: list[str],
        obj_terms: list[str],
        height: int,
        width: int,
    ) -> list[tuple[FramePacket, int]]:
        n = len(batch.frame_indices)
        if n == 0:
            return []

        per_det_ms = batch.detect_ms / max(1, n)
        per_seg_ms = batch.seg_ms / max(1, n)
        per_cross_ms = batch.cross_ms / max(1, n)

        use_reid = self.settings.use_reid and self.reid_engine is not None
        use_motion_tracker = self.settings.gpu_full_batch and not use_reid

        emb_by_slot: dict[tuple[int, int], np.ndarray] = {}
        reid_ms_total = 0.0
        frame_valid: list[list[DetectItem]] = []
        frame_segments: list[list] = []
        frame_cross_raw: list[list[DetectItem]] = []

        if batch.prepared_frames is not None:
            for pf in batch.prepared_frames:
                frame_valid.append(pf.valid_dets)
                frame_segments.append(pf.segments)
                frame_cross_raw.append(pf.cross_raw)
            if (
                use_reid
                and batch.reid_embeddings is not None
                and batch.reid_crop_map is not None
            ):
                reid_ms_total = float(batch.reid_ms)
                n_emb = int(batch.reid_embeddings.shape[0])
                crop_map = batch.reid_crop_map
                if n_emb == len(crop_map):
                    for slot, key in enumerate(crop_map):
                        emb_by_slot[key] = batch.reid_embeddings[slot]
                else:
                    crops_retry: list[np.ndarray] = []
                    keys_retry: list[tuple[int, int]] = []
                    for fi, pf in enumerate(batch.prepared_frames):
                        frame = frames_bgr[fi]
                        for di, det in enumerate(pf.valid_dets):
                            crop = ReidEngine.crop_from_bbox(frame, det.xyxy)
                            if crop is not None and crop.size > 0:
                                keys_retry.append((fi, di))
                                crops_retry.append(crop)
                    t0 = time.perf_counter()
                    retry_emb = self.reid_engine.embed_batch(crops_retry)
                    reid_ms_total += (time.perf_counter() - t0) * 1000.0
                    for slot, key in enumerate(keys_retry):
                        if slot < retry_emb.shape[0]:
                            emb_by_slot[key] = retry_emb[slot]
        else:
            prepared, all_crops, crop_map = prepare_batch_frames(
                detections=batch.detections,
                segments=batch.segments,
                cross_detections=batch.cross_detections,
                frames_bgr=frames_bgr,
                terms=terms,
                settings=self.settings,
                motion_tracker=self.motion_tracker,
                use_motion_tracker=use_motion_tracker,
            )
            for pf in prepared:
                frame_valid.append(pf.valid_dets)
                frame_segments.append(pf.segments)
                frame_cross_raw.append(pf.cross_raw)
            if use_reid and all_crops:
                t0 = time.perf_counter()
                trt_cap = (
                    self.reid_engine.trt_max_batch
                    if self.reid_engine is not None and getattr(self.reid_engine, "using_tensorrt", False)
                    else 0
                )
                chunk = resolve_reid_embed_chunk(
                    self.settings.reid_embed_chunk,
                    len(all_crops),
                    trt_max_batch=trt_cap,
                )
                if chunk <= 0:
                    embeddings = self.reid_engine.embed_batch(all_crops)
                else:
                    emb_parts: list[np.ndarray] = []
                    for i in range(0, len(all_crops), chunk):
                        emb_parts.append(self.reid_engine.embed_batch(all_crops[i : i + chunk]))
                    embeddings = (
                        np.concatenate(emb_parts, axis=0)
                        if emb_parts
                        else np.zeros((0, 512), dtype=np.float32)
                    )
                reid_ms_total = (time.perf_counter() - t0) * 1000.0
                if embeddings.shape[0] != len(crop_map):
                    raise RuntimeError(
                        f"ReID embed count mismatch: got {embeddings.shape[0]} for {len(crop_map)} crops"
                    )
                for slot, key in enumerate(crop_map):
                    emb_by_slot[key] = embeddings[slot]

        per_reid_ms = reid_ms_total / max(1, n) if self.settings.reid_batch_across_frames else 0.0
        out: list[tuple[FramePacket, int]] = []

        for fi in range(n):
            frame_idx = int(batch.frame_indices[fi])
            valid_dets = frame_valid[fi]
            segments = frame_segments[fi]
            frame = frames_bgr[fi]

            reid_ms = 0.0
            if use_reid:
                if valid_dets:
                    if self.settings.reid_batch_across_frames:
                        embs = np.stack(
                            [emb_by_slot[(fi, di)] for di in range(len(valid_dets))],
                            axis=0,
                        )
                        reid_ms = per_reid_ms
                    else:
                        crops = [ReidEngine.crop_from_bbox(frame, d.xyxy) for d in valid_dets]
                        crops = [c for c in crops if c is not None and c.size > 0]
                        t0 = time.perf_counter()
                        embs = self.reid_engine.embed_batch(crops)
                        reid_ms = (time.perf_counter() - t0) * 1000.0
                else:
                    embs = np.zeros((0, 512), dtype=np.float32)
                track_result = self.tracker.update(valid_dets, embs)
                stable_ids = (
                    np.array(track_result.stable_ids, dtype=np.int64)
                    if track_result.stable_ids
                    else np.zeros((0,), dtype=np.int64)
                )
                reid_recoveries = track_result.reid_recoveries
            else:
                stable_ids = np.array(
                    [
                        int(d.motion_id) if d.motion_id >= 0 else i + 1
                        for i, d in enumerate(valid_dets)
                    ],
                    dtype=np.int64,
                )
                reid_recoveries = 0

            masks = match_detections_to_segments(
                valid_dets,
                segments,
                match_iou_min=self.settings.match_iou_min,
                frame_h=height,
                frame_w=width,
            )
            n_inst = len(valid_dets)
            if n_inst > 0:
                stack = np.stack(masks, axis=0).astype(np.uint8)
                scores = np.array([d.conf for d in valid_dets], dtype=np.float32)
                labels = [d.label for d in valid_dets]
                keypoints_list = [d.keypoints for d in valid_dets]
            else:
                stack = np.zeros((0, height, width), dtype=np.uint8)
                scores = np.zeros((0,), dtype=np.float32)
                labels = []
                keypoints_list = []

            cross_verdicts: list[CrossCheckVerdict] = []
            cross_accessories: list[CrossCheckDetection] = []
            if (
                self.settings.cross_check_enabled
                and self.cross_check_engine is not None
                and valid_dets
            ):
                accessories = [d for d in frame_cross_raw[fi] if label_match(d.label, obj_terms)]
                cross_accessories = [
                    CrossCheckDetection(d.label, float(d.conf), d.xyxy.copy()) for d in accessories
                ]
                cross_verdicts = evaluate_cross_check_batch(
                    valid_dets,
                    accessories,
                    kpt_conf=self.settings.pose_kpt_conf,
                    min_intersection_px=self.settings.cross_check_min_intersection_px,
                    min_iou=self.settings.cross_check_min_iou,
                    frame_w=width,
                    frame_h=height,
                    warn_text=self.settings.cross_check_warning_text,
                )
                self.cross_check_violations_total += sum(1 for v in cross_verdicts if not v.ok)

            infer_ms = per_det_ms + per_seg_ms + per_cross_ms + reid_ms
            packet = make_infer_packet(
                frame_idx=frame_idx,
                frame_bgr=frame,
                stack=stack,
                stable_ids=stable_ids,
                scores=scores,
                labels=labels,
                n_inst=n_inst,
                detect_ms=per_det_ms,
                seg_ms=per_seg_ms,
                reid_ms=reid_ms,
                cross_ms=per_cross_ms,
                infer_ms=infer_ms,
                reid_recoveries=reid_recoveries,
                keypoints_list=keypoints_list,
                cross_check_verdicts=cross_verdicts,
                cross_check_accessories=cross_accessories,
            )
            out.append((packet, n_inst))
        return out

    def _drive_gpu_pipeline(
        self,
        *,
        all_frames: list[np.ndarray] | None,
        cap_stream: cv2.VideoCapture | None,
        n_frames_total: int,
        terms: list[str],
        obj_terms: list[str],
        height: int,
        width: int,
        deferred_encode: bool,
        deferred_packets: list[FramePacket],
        prompt_id_lookup: dict[str, int],
        post_exec: OrderedPostExecutor | None,
        drain_post: Callable[[], None],
        emit_kwargs: dict,
        frame_idx_ref: list[int],
        instances_peak: int,
        infer_ms_by_frame: dict[int, float],
        packet_meta: dict[int, FramePacket],
        should_stop: Callable[[], bool] | None,
        should_pause: Callable[[], bool] | None,
        on_debug_log: Callable[[str], None] | None,
        batch_jobs: Iterator[FrameBatchJob],
        n_process: int,
    ) -> tuple[int, int]:
        seg_eng = self.seg_engine if self.settings.use_seg else None
        use_reid = self.settings.use_reid and self.reid_engine is not None
        eff_bs = self._job_batch_size
        n_jobs = (n_process + eff_bs - 1) // eff_bs if n_process > 0 else 0
        # ReID + YOLO track на одном GPU: параллельные CUDA-потоки → deadlock (TRT/Ultralytics).
        serial_cuda = use_reid
        depth = 1 if serial_cuda else resolve_gpu_queue_depth(
            self.settings.gpu_queue_depth,
            n_jobs=n_jobs,
            job_batch_size=eff_bs,
            max_job_batch=self.settings.max_job_batch_size,
        )
        worker = GpuInferWorker(
            self.detect_engine,
            seg_eng,
            self.cross_check_engine,
            self.settings,
        )
        worker.start()
        detect_mode = "predict_batch" if not worker._detect_use_track else "track"
        if on_debug_log:
            req = self.settings.infer_batch_size
            req_txt = "авто→все кадры" if int(req) <= 0 and self.settings.gpu_full_batch else str(req)
            cap_note = ""
            if (
                self.settings.use_reid
                and int(self.settings.max_job_batch_size) > 0
                and int(req) > eff_bs
            ):
                cap_note = f" | job cap ReID {req}→{eff_bs}"
            queue_note = ""
            if depth != int(self.settings.gpu_queue_depth):
                queue_note = f" (запрос {self.settings.gpu_queue_depth})"
            serial_note = " | CUDA serial (ReID)" if serial_cuda else ""
            on_debug_log(
                f"GPU job: {eff_bs} кадр/батч × {n_jobs} job | inference {n_process} кадр | "
                f"detect={detect_mode} | YOLO chunk={self.settings.max_infer_batch_size} | "
                f"queue={depth}{queue_note} | запрос batch={req_txt}{cap_note}{serial_note}"
            )
        job_iter = batch_jobs
        pending_jobs: deque[FrameBatchJob] = deque()
        in_flight = 0

        def _close_batch_feeder() -> None:
            close_fn = getattr(job_iter, "close", None)
            if callable(close_fn):
                close_fn()

        def _prime() -> None:
            nonlocal in_flight
            while in_flight < depth:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                worker.submit(job)
                pending_jobs.append(job)
                in_flight += 1

        def _finalize_job(result, frames_bgr: list[np.ndarray]):
            packets = self._cpu_finalize_batch(
                result,
                frames_bgr,
                terms,
                obj_terms,
                height,
                width,
            )
            if deferred_encode:
                return [
                    (compact_packet_for_cache(packet), n_inst)
                    for packet, n_inst in packets
                ]
            return packets

        def _emit_packets(packets: list[tuple[FramePacket, int]]) -> None:
            nonlocal instances_peak
            if not packets:
                return
            last_packet: FramePacket | None = None
            for packet, n_inst in packets:
                instances_peak = max(instances_peak, n_inst)
                infer_ms_by_frame[packet.frame_idx] = packet.infer_ms
                if not deferred_encode:
                    packet_meta[packet.frame_idx] = packet
                frame_idx_ref[0] = max(frame_idx_ref[0], packet.frame_idx + 1)

                if deferred_encode:
                    self._append_deferred_packet(packet, deferred_packets)
                    last_packet = packet
                elif post_exec is not None:
                    post_exec.submit(packet, frame_source=all_frames)
                else:
                    work = packet
                    if all_frames is not None:
                        work = FramePacket(
                            frame_idx=packet.frame_idx,
                            frame_bgr=all_frames[packet.frame_idx],
                            stack=packet.stack,
                            stable_ids=packet.stable_ids,
                            scores=packet.scores,
                            labels=packet.labels,
                            n_inst=packet.n_inst,
                            detect_ms=packet.detect_ms,
                            seg_ms=packet.seg_ms,
                            reid_ms=packet.reid_ms,
                            cross_ms=packet.cross_ms,
                            infer_ms=packet.infer_ms,
                            reid_recoveries=packet.reid_recoveries,
                            keypoints_list=list(packet.keypoints_list or []),
                            cross_check_verdicts=list(packet.cross_check_verdicts or []),
                            cross_check_accessories=[
                                CrossCheckDetection(d.label, d.conf, d.xyxy.copy())
                                for d in (packet.cross_check_accessories or [])
                            ],
                        )
                    result_post = post_process_frame(
                        work,
                        serialize_fn=self.serialize_frame_instances,
                        prompt_id_lookup=prompt_id_lookup,
                        overlay_alpha=self.settings.overlay_alpha,
                        draw_boxes=self.settings.draw_boxes,
                        draw_masks=self.settings.draw_masks,
                        draw_centers=self.settings.draw_centers,
                        draw_pose=self.settings.draw_pose,
                        pose_kpt_conf=self.settings.pose_kpt_conf,
                        cross_check_enabled=self.settings.cross_check_enabled,
                        cross_check_draw_head_box=self.settings.cross_check_draw_head_box,
                        cross_check_draw_boxes=self.settings.cross_check_draw_boxes,
                    )
                    self._emit_post_results(
                        [result_post],
                        **{**emit_kwargs, "instances_peak": instances_peak},
                    )

            if deferred_encode and last_packet is not None:
                self._report_infer_progress(
                    packet=last_packet,
                    total=emit_kwargs["total"],
                    instances_peak=instances_peak,
                    start_ts=emit_kwargs["start_ts"],
                    infer_ms_by_frame=infer_ms_by_frame,
                    metrics_path=emit_kwargs["metrics_path"],
                    on_progress=emit_kwargs.get("on_progress"),
                    gpu_monitor=emit_kwargs.get("gpu_monitor"),
                    phase="inference",
                    write_metrics=False,
                )

        def _drain_reid_and_finalize() -> None:
            if finalizer is None:
                return
            for batch_packets in finalizer.drain():
                _emit_packets(batch_packets)

        finalizer: AsyncBatchFinalizer | None = None
        if not serial_cuda:
            finalizer = AsyncBatchFinalizer(_finalize_job, should_stop=should_stop)
            finalizer.start()

        self._pulse_infer_progress(
            current_frame=0,
            total=emit_kwargs["total"],
            instances_peak=instances_peak,
            start_ts=emit_kwargs["start_ts"],
            on_progress=emit_kwargs.get("on_progress"),
            gpu_monitor=emit_kwargs.get("gpu_monitor"),
            phase="warmup",
        )

        try:
            _prime()

            while in_flight > 0:
                if should_stop and should_stop():
                    break
                _drain_reid_and_finalize()
                try:
                    result = worker.get_result(timeout=0.05)
                except queue_module.Empty:
                    self._pulse_infer_progress(
                        current_frame=frame_idx_ref[0],
                        total=emit_kwargs["total"],
                        instances_peak=instances_peak,
                        start_ts=emit_kwargs["start_ts"],
                        on_progress=emit_kwargs.get("on_progress"),
                        gpu_monitor=emit_kwargs.get("gpu_monitor"),
                        phase="gpu",
                    )
                    continue

                job = pending_jobs.popleft()
                frames_bgr = job.frames_bgr
                in_flight -= 1
                job.frames_bgr = []

                if serial_cuda:
                    packets = _finalize_job(result, frames_bgr)
                    _emit_packets(packets)
                    try:
                        next_job = next(job_iter)
                        worker.submit(next_job)
                        pending_jobs.append(next_job)
                        in_flight += 1
                    except StopIteration:
                        pass
                else:
                    try:
                        next_job = next(job_iter)
                        worker.submit(next_job)
                        pending_jobs.append(next_job)
                        in_flight += 1
                    except StopIteration:
                        pass
                    if finalizer is not None:
                        finalizer.submit(result, frames_bgr)

                _drain_reid_and_finalize()
                drain_post()
                emit_kwargs["instances_peak"] = instances_peak

            worker.close()
            if finalizer is not None:
                for batch_packets in finalizer.finish():
                    _emit_packets(batch_packets)
            drain_post()
            emit_kwargs["instances_peak"] = instances_peak
        finally:
            _close_batch_feeder()

        return instances_peak, frame_idx_ref[0]

    def process_video(
        self,
        input_path: str,
        output_dir: str | Path,
        prompt: str,
        *,
        max_duration_seconds: float | None = None,
        on_progress: Callable[[VideoProgress], None] | None = None,
        on_preview: Callable[[np.ndarray], None] | None = None,
        on_debug_log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> ProcessVideoResult:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        self.detect_engine.reset_session()
        self.tracker.reset()
        self.motion_tracker.reset()
        self.cross_check_violations_total = 0

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Invalid video dimensions")

        if max_duration_seconds is not None and max_duration_seconds > 0:
            limit = int(max_duration_seconds * fps)
            if total > 0:
                total = min(total, limit)
            else:
                total = limit

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        out_dir = Path(output_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{run_id}_result.json"
        video_path = out_dir / f"{run_id}_annotated.mp4"
        report_path = out_dir / f"{run_id}_report.txt"
        metrics_path = out_dir / f"{run_id}_metrics.jsonl"
        gpu_samples_path = out_dir / f"{run_id}_gpu_samples.json"
        summary_path = out_dir / f"{run_id}_run_summary.json"

        gpu_monitor = GpuMonitor(interval_sec=0.5)
        gpu_monitor.start()
        if on_debug_log:
            if gpu_monitor.available:
                on_debug_log("GPU monitor: NVML active")
            else:
                on_debug_log("GPU monitor: NVML unavailable (install nvidia-ml-py)")

        terms = prompt_terms(prompt)
        prompt_id_lookup = prompt_id_lookup_from_prompt(prompt)
        frames_payload: list[dict[str, object]] = []

        deferred_encode = self.settings.encode_mode.strip().casefold() in ("deferred", "manual")
        manual_encode = self.settings.encode_mode.strip().casefold() == "manual"
        encode_parallel = self.settings.encode_mode.strip().casefold() == "parallel"
        use_async_writer = bool(
            self.settings.async_encode and encode_parallel and not deferred_encode
        )

        sync_writer = None
        async_writer: AsyncVideoWriter | None = None
        if encode_parallel and not use_async_writer:
            sync_writer = imageio.get_writer(
                str(video_path),
                fps=fps,
                codec="libx264",
                ffmpeg_params=["-crf", "18", "-preset", "medium"],
            )
        elif use_async_writer:
            async_writer = AsyncVideoWriter(str(video_path), fps)

        def video_sink():
            if async_writer is not None:
                return async_writer
            if sync_writer is not None:
                class _SyncSink:
                    def submit(self, rgb: np.ndarray) -> None:
                        sync_writer.append_data(np.ascontiguousarray(rgb))

                return _SyncSink()
            return None

        sink = video_sink()

        frame_idx = 0
        instances_peak = 0
        detect_ms_acc = [0.0]
        seg_ms_acc = [0.0]
        reid_ms_acc = [0.0]
        total_ms_acc = [0.0]
        start_ts = time.perf_counter()
        frame_idx_ref = [0]
        infer_ms_by_frame: dict[int, float] = {}
        packet_meta: dict[int, FramePacket] = {}

        tune = effective_speed_tuning(self.settings)
        frame_stride = int(tune["frame_stride"])
        requested_batch = int(tune["infer_batch"])
        source_frame_count = total
        process_indices = source_frame_indices(source_frame_count, frame_stride)
        n_process = len(process_indices)
        self._job_batch_size = resolve_infer_batch_size(
            requested_batch,
            n_process,
            gpu_full_batch=bool(tune["gpu_full_batch"]),
            max_cap=int(tune["max_infer_batch"]),
            use_reid=self.settings.use_reid,
            max_job_batch=int(tune["max_job_batch"]),
        )
        use_streaming = resolve_use_streaming(
            self.settings,
            source_frame_count=source_frame_count,
            width=width,
            height=height,
        )
        capped_batch = cap_job_batch_long_video(
            self._job_batch_size,
            n_process,
            self.settings,
            streaming=use_streaming,
            source_frame_count=source_frame_count,
        )
        if capped_batch != self._job_batch_size and on_debug_log:
            on_debug_log(
                f"Job batch {self._job_batch_size}→{capped_batch} "
                f"(длинное видео, max {self.settings.max_job_batch_size or 64})"
            )
        self._job_batch_size = capped_batch
        if on_debug_log and use_streaming and source_frame_count > 0:
            on_debug_log(
                f"Streaming decode: очередь батчей={resolve_batch_prefetch_depth(self.settings)}, "
                f"job={self._job_batch_size} кадр"
            )
        elif on_debug_log and use_streaming and source_frame_count <= 0:
            on_debug_log(
                f"Streaming decode (длина ролика неизвестна): "
                f"queue={resolve_batch_prefetch_depth(self.settings)}"
            )
        if on_debug_log and self.settings.realtime_mode:
            on_debug_log(
                f"Realtime ~1×: imgsz={tune['imgsz'] or 'auto'} | seg_stride={tune['seg_stride']} | "
                f"frame_stride={frame_stride} | job={self._job_batch_size} | YOLO chunk={tune['max_infer_batch']}"
            )
        elif on_debug_log and frame_stride > 1:
            on_debug_log(
                f"Frame stride={frame_stride}: inference {n_process}/{source_frame_count} кадров "
                f"(MP4 при сборке — hold-forward)"
            )
        elif (
            on_debug_log
            and self.settings.use_reid
            and int(self.settings.infer_batch_size) > 0
            and self._job_batch_size < int(self.settings.infer_batch_size)
        ):
            on_debug_log(
                f"ReID: job batch {self.settings.infer_batch_size}→{self._job_batch_size} "
                f"(max_job_batch_size={self.settings.max_job_batch_size})"
            )

        all_frames: list[np.ndarray] | None = None
        cap_stream: cv2.VideoCapture | None = None
        if not use_streaming:
            try:
                if on_debug_log:
                    on_debug_log("Preloading video frames into RAM…")
                t_pre = time.perf_counter()
                all_frames = load_all_frames(
                    cap,
                    total,
                    max_ram_gb=self.settings.max_preload_ram_gb,
                    width=width,
                    height=height,
                )
                source_frame_count = len(all_frames)
                process_indices = source_frame_indices(source_frame_count, frame_stride)
                n_process = len(process_indices)
                if on_debug_log:
                    elapsed_pre = time.perf_counter() - t_pre
                    on_debug_log(f"Preloaded {source_frame_count} frames in {elapsed_pre:.2f}s")
            except MemoryError as exc:
                if on_debug_log:
                    on_debug_log(f"Preload skipped: {exc}")
                all_frames = None
                use_streaming = True
        elif on_debug_log and not use_streaming:
            on_debug_log(f"Preload mode: {source_frame_count} кадр в RAM")
        cap.release()

        model_runner: ModelParallelRunner | None = None
        if not self._use_gpu_pipeline() and self.settings.parallel_models:
            seg_eng = self.seg_engine if self.settings.use_seg else None
            model_runner = ModelParallelRunner(self.detect_engine, seg_eng)

        obj_terms = prompt_terms(self.settings.cross_check_object_prompt)
        use_gpu_pipe = self._use_gpu_pipeline()

        post_exec: OrderedPostExecutor | None = None
        deferred_packets: list[FramePacket] = []
        if not deferred_encode and self.settings.parallel_post:
            post_exec = OrderedPostExecutor(
                workers=self.settings.post_workers,
                process_fn=self._make_post_fn(prompt_id_lookup),
            )

        emit_kwargs = dict(
            video_sink=sink,
            frames_payload=frames_payload,
            metrics_path=metrics_path,
            total=n_process,
            frame_idx_ref=frame_idx_ref,
            instances_peak=instances_peak,
            detect_ms_acc=detect_ms_acc,
            seg_ms_acc=seg_ms_acc,
            reid_ms_acc=reid_ms_acc,
            total_ms_acc=total_ms_acc,
            infer_ms_by_frame=infer_ms_by_frame,
            packet_meta=packet_meta,
            start_ts=start_ts,
            on_progress=on_progress,
            on_preview=on_preview,
            on_debug_log=on_debug_log,
            gpu_monitor=gpu_monitor,
        )

        def drain_post() -> None:
            if post_exec is None:
                return
            emit_kwargs["instances_peak"] = instances_peak
            flushed = post_exec.drain()
            if flushed:
                self._emit_post_results(flushed, **emit_kwargs)

        infer_start_ts = time.perf_counter()
        try:
            n_frames_total = source_frame_count

            if all_frames is None:
                cap_stream = cv2.VideoCapture(input_path)
                if not cap_stream.isOpened():
                    raise RuntimeError(f"Cannot reopen video: {input_path}")

            batch_jobs = self._make_batch_job_iter(
                all_frames=all_frames,
                cap_stream=cap_stream,
                n_frames_total=n_frames_total,
                process_indices=process_indices,
                frame_stride=frame_stride,
                should_stop=should_stop,
                should_pause=should_pause,
                on_debug_log=on_debug_log,
            )

            self._pulse_infer_progress(
                current_frame=0,
                total=n_process,
                instances_peak=instances_peak,
                start_ts=start_ts,
                on_progress=on_progress,
                gpu_monitor=gpu_monitor,
                phase="start",
            )

            if use_gpu_pipe:
                instances_peak, frame_idx = self._drive_gpu_pipeline(
                    all_frames=all_frames,
                    cap_stream=cap_stream,
                    n_frames_total=n_frames_total,
                    terms=terms,
                    obj_terms=obj_terms,
                    height=height,
                    width=width,
                    deferred_encode=deferred_encode,
                    deferred_packets=deferred_packets,
                    prompt_id_lookup=prompt_id_lookup,
                    post_exec=post_exec,
                    drain_post=drain_post,
                    emit_kwargs=emit_kwargs,
                    frame_idx_ref=frame_idx_ref,
                    instances_peak=instances_peak,
                    infer_ms_by_frame=infer_ms_by_frame,
                    packet_meta=packet_meta,
                    should_stop=should_stop,
                    should_pause=should_pause,
                    on_debug_log=on_debug_log,
                    batch_jobs=batch_jobs,
                    n_process=n_process,
                )
            else:
                processed_count = 0
                stream_read_idx = 0

                def _run_one_frame(frame_idx: int, frame: np.ndarray) -> None:
                    nonlocal instances_peak, processed_count
                    packet, n_inst = self._infer_frame(
                        frame_idx,
                        frame,
                        terms,
                        height,
                        width,
                        model_runner,
                    )
                    instances_peak = max(instances_peak, n_inst)
                    infer_ms_by_frame[frame_idx] = packet.infer_ms
                    if not deferred_encode:
                        packet_meta[frame_idx] = packet

                    if deferred_encode:
                        progress_total = n_process if n_process > 0 else frame_idx + 1
                        self._report_infer_progress(
                            packet=packet,
                            total=progress_total,
                            instances_peak=instances_peak,
                            start_ts=start_ts,
                            infer_ms_by_frame=infer_ms_by_frame,
                            metrics_path=metrics_path,
                            on_progress=on_progress,
                            gpu_monitor=gpu_monitor,
                            phase="inference",
                            write_metrics=False,
                        )
                        self._append_deferred_packet(packet, deferred_packets, compact=True)
                    elif post_exec is not None:
                        post_exec.submit(packet, frame_source=all_frames)
                        drain_post()
                    else:
                        work = materialize_packet_for_render(
                            packet,
                            all_frames[packet.frame_idx] if all_frames is not None else packet.frame_bgr,
                        )
                        result = post_process_frame(
                            work,
                            serialize_fn=self.serialize_frame_instances,
                            prompt_id_lookup=prompt_id_lookup,
                            overlay_alpha=self.settings.overlay_alpha,
                            draw_boxes=self.settings.draw_boxes,
                            draw_masks=self.settings.draw_masks,
                            draw_centers=self.settings.draw_centers,
                            draw_pose=self.settings.draw_pose,
                            pose_kpt_conf=self.settings.pose_kpt_conf,
                            cross_check_enabled=self.settings.cross_check_enabled,
                            cross_check_draw_head_box=self.settings.cross_check_draw_head_box,
                            cross_check_draw_boxes=self.settings.cross_check_draw_boxes,
                        )
                        self._emit_post_results([result], **{**emit_kwargs, "instances_peak": instances_peak})

                    emit_kwargs["instances_peak"] = instances_peak
                    processed_count += 1

                if process_indices:
                    for frame_idx in process_indices:
                        if should_stop and should_stop():
                            break
                        while should_pause and should_pause():
                            time.sleep(0.05)
                            if should_stop and should_stop():
                                break

                        if all_frames is not None:
                            frame = all_frames[frame_idx]
                        else:
                            while stream_read_idx < frame_idx:
                                ok, _ = cap_stream.read()
                                if not ok:
                                    break
                                stream_read_idx += 1
                            if stream_read_idx != frame_idx:
                                break
                            ok, frame = cap_stream.read()
                            if not ok:
                                break
                            stream_read_idx += 1

                        _run_one_frame(frame_idx, frame)
                elif cap_stream is not None:
                    while True:
                        if should_stop and should_stop():
                            break
                        while should_pause and should_pause():
                            time.sleep(0.05)
                            if should_stop and should_stop():
                                break
                        ok, frame = cap_stream.read()
                        if not ok:
                            break
                        fi = stream_read_idx
                        stream_read_idx += 1
                        if fi % frame_stride != 0:
                            continue
                        _run_one_frame(fi, frame)

                frame_idx = processed_count

            if deferred_encode:
                if manual_encode:
                    if on_debug_log:
                        on_debug_log(
                            f"Inference done ({frame_idx} frames). "
                            f"JSON + кэш кадров (MP4 только по кнопке)…"
                        )
                    pkl_path = packets_path_for_run(out_dir, run_id)
                    save_run_packets(
                        pkl_path,
                        run_id=run_id,
                        packets=deferred_packets,
                        fps=fps,
                        input_path=str(input_path),
                        prompt=prompt,
                        overlay=self._overlay_snapshot(),
                        width=width,
                        height=height,
                        frame_stride=frame_stride,
                        source_frame_count=source_frame_count,
                    )
                    if on_debug_log:
                        on_debug_log(
                            f"Packets: {pkl_path.name} ({len(deferred_packets)} keyframes, "
                            f"stride={frame_stride}, source={source_frame_count})"
                        )
                else:
                    if on_debug_log:
                        on_debug_log(f"Inference done ({frame_idx} frames). Building video + JSON…")
                    timeline = expand_packets_to_timeline(
                        deferred_packets,
                        source_frame_count=source_frame_count,
                        frame_stride=frame_stride,
                        frame_source=all_frames,
                    )
                    post_exec = OrderedPostExecutor(
                        workers=self.settings.post_workers,
                        process_fn=self._make_post_fn(prompt_id_lookup),
                    )
                    for packet in timeline:
                        post_exec.submit(packet, frame_source=all_frames)
                    remaining = post_exec.finish()
                    if sink is None:
                        sync_writer = imageio.get_writer(
                            str(video_path),
                            fps=fps,
                            codec="libx264",
                            ffmpeg_params=["-crf", "18", "-preset", "medium"],
                        )

                        class _DeferredSink:
                            def submit(self, rgb: np.ndarray) -> None:
                                sync_writer.append_data(np.ascontiguousarray(rgb))

                        emit_kwargs["video_sink"] = _DeferredSink()
                    emit_kwargs["instances_peak"] = instances_peak
                    self._emit_post_results(remaining, **emit_kwargs)
                    post_exec.shutdown()
            elif post_exec is not None:
                emit_kwargs["instances_peak"] = instances_peak
                remaining = post_exec.finish()
                self._emit_post_results(remaining, **emit_kwargs)
        finally:
            gpu_monitor.stop()
            if model_runner is not None:
                model_runner.shutdown()
            if post_exec is not None and not deferred_encode:
                post_exec.shutdown()
            if cap_stream is not None:
                cap_stream.release()
            if async_writer is not None:
                async_writer.close()
            elif sync_writer is not None:
                sync_writer.close()
            freed = release_gpu_memory()
            if on_debug_log and freed.get("freed_mb", 0) > 0.1:
                on_debug_log(
                    f"GPU cache cleared: {freed['before_mb']:.0f}→{freed['after_mb']:.0f} MB "
                    f"(−{freed['freed_mb']:.0f} MB)"
                )

        elapsed_total = time.perf_counter() - start_ts
        elapsed_infer = time.perf_counter() - infer_start_ts
        if source_frame_count <= 0:
            source_frame_count = max(
                int(frame_idx_ref[0]),
                int(frame_idx),
                len(deferred_packets),
            )
        if n_process <= 0 and source_frame_count > 0:
            n_process = len(deferred_packets) or int(frame_idx)
        n_frames = max(1, frame_idx)
        stride_info = frame_stride_summary(source_frame_count, frame_stride, frame_idx)
        video_duration_sec = source_frame_count / fps if fps > 0 else 0.0
        if on_debug_log and video_duration_sec > 0:
            speed_ratio = elapsed_infer / video_duration_sec
            on_debug_log(
                f"Speed: inference {elapsed_infer:.1f}s / video {video_duration_sec:.1f}s = {speed_ratio:.2f}× "
                f"(total wall {elapsed_total:.1f}s"
                f"{f', preload {elapsed_total - elapsed_infer:.1f}s' if elapsed_total - elapsed_infer > 0.3 else ''}) "
                f"({'~1× OK' if speed_ratio <= 1.15 else 'цель ~1× — включи Realtime или yolo26s/m'})"
            )
        stats_summary = stats_summary_from_counters(
            id_switches=self.tracker.total_id_switches,
            reid_recoveries=self.tracker.total_reid_recoveries,
            instances_peak=instances_peak,
            avg_detect_ms=detect_ms_acc[0] / n_frames,
            avg_seg_ms=seg_ms_acc[0] / n_frames,
            avg_reid_ms=reid_ms_acc[0] / n_frames,
            avg_total_ms=total_ms_acc[0] / n_frames,
        )
        stats_summary.update(stride_info)

        if manual_encode and deferred_packets and not frames_payload:
            if on_debug_log:
                on_debug_log(f"Building result JSON ({len(deferred_packets)} frames)…")
            frames_payload = self._frames_payload_from_packets(
                deferred_packets,
                prompt_id_lookup,
            )

        payload = build_result_payload(
            schema="yolo_drt_video_data_v1",
            input_path=str(input_path),
            prompt=prompt,
            fps=fps,
            width=width,
            height=height,
            frames_written=frame_idx,
            elapsed_sec=elapsed_total,
            frames=frames_payload,
            models={
                "detect": Path(self.detect_engine.model_path).name,
                "seg": (
                    Path(self.seg_engine.model_path).name
                    if self.seg_engine is not None
                    else "disabled"
                ),
                "reid": (
                    Path(self.reid_engine.model_path).name
                    if self.reid_engine is not None
                    else "disabled"
                ),
                "cross_check": (
                    Path(self.cross_check_engine.model_path).name
                    if self.cross_check_engine is not None
                    else "disabled"
                ),
            },
            draw_boxes=self.settings.draw_boxes,
            draw_masks=self.settings.draw_masks,
            draw_centers=self.settings.draw_centers,
            draw_pose=self.settings.draw_pose,
            run_id=run_id,
            stats_summary=stats_summary,
        )
        json_path.write_text(video_data_json_dumps(payload), encoding="utf-8")

        memory_info = build_memory_report(
            width=width,
            height=height,
            effective_batch_size=self._job_batch_size,
            frames_total=source_frame_count,
            preloaded=all_frames is not None,
        )
        pkl_path = packets_path_for_run(out_dir, run_id)
        has_video = video_path.exists()
        artifacts = {
            "json": str(json_path.name),
            "video": str(video_path.name) if has_video else None,
            "packets": str(pkl_path.name) if pkl_path.exists() else None,
        }

        report_lines = [
            f"YOLO_DRT run {run_id}",
            f"Input: {input_path}",
            (
                f"Кадры: {stride_info['source_frame_count']} всего → "
                f"inference {stride_info['processed_frame_count']} "
                f"(stride={stride_info['frame_stride']}, "
                f"пропущено {stride_info['frames_skipped']}, "
                f"−{stride_info['reduction_pct']:.1f}%)"
            ),
            f"Elapsed: {elapsed_total:.2f}s",
            (
                f"Streaming: {'да' if all_frames is None else 'нет'} | "
                f"batch queue={resolve_batch_prefetch_depth(self.settings)} | "
                f"job={self._job_batch_size} fr"
            ),
            f"Device: {getattr(self.settings, 'inference_device', 'cuda')}",
            f"Memory: 1 frame = {memory_info['frame_human']} ({memory_info['frame_bytes']} B) | "
            f"1 batch ({memory_info['effective_batch_size']} fr) = {memory_info['batch_human']}",
        ]
        if memory_info.get("preloaded"):
            report_lines.append(
                f"Preload RAM (frames only): {memory_info['preload_human']} "
                f"({memory_info['preload_mb']} MB)"
            )
        report_lines.extend(
            [
                f"Models: {payload['models']}",
                f"Stats: {stats_summary}",
                f"Cross-check violations (frames×person): {self.cross_check_violations_total}",
                f"GPU pipeline: {self.settings.gpu_pipeline} batch={self.settings.infer_batch_size}",
                f"Encode: {self.settings.encode_mode}",
                f"JSON: {json_path}",
                f"Video: {video_path if has_video else '(не собран — кнопка «Собрать видео»)'}",
            ]
        )
        if pkl_path.exists():
            report_lines.append(f"Packets: {pkl_path}")
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        gpu_samples = gpu_monitor.samples_to_dicts()
        if gpu_samples:
            gpu_samples_path.write_text(
                json.dumps(gpu_samples, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        gpu_stats = gpu_monitor.summary()
        metrics_rows = read_metrics_jsonl(metrics_path)
        chart_paths = generate_run_charts(
            out_dir,
            run_id,
            gpu_samples=gpu_samples or None,
            metrics_rows=metrics_rows or None,
            stats_summary=stats_summary,
        )
        pipeline_info = {
            "gpu_pipeline": self.settings.gpu_pipeline,
            "gpu_full_batch": self.settings.gpu_full_batch,
            "infer_batch_size": self.settings.infer_batch_size,
            "max_infer_batch_size": self.settings.max_infer_batch_size,
            "effective_batch_size": self._job_batch_size,
            "detect_mode": (
                "track"
                if self.settings.use_reid
                else ("predict_batch" if self.settings.gpu_full_batch else "track_or_batch")
            ),
            "gpu_queue_depth": self.settings.gpu_queue_depth,
            "batch_prefetch_depth": resolve_batch_prefetch_depth(self.settings),
            "streaming_decode": all_frames is None,
            "use_batch_detect": self.settings.use_batch_detect,
            "use_seg": self.settings.use_seg,
            "use_reid": self.settings.use_reid,
            "cross_check_enabled": self.settings.cross_check_enabled,
            "encode_mode": self.settings.encode_mode,
            "parallel_post": self.settings.parallel_post,
            "inference_device": getattr(self.settings, "inference_device", "cuda"),
            "frame_stride": frame_stride,
            "source_frame_count": source_frame_count,
            "processed_frame_count": stride_info["processed_frame_count"],
        }
        record = build_run_record(
            run_id=run_id,
            out_dir=out_dir,
            input_path=str(input_path),
            prompt=prompt,
            frames=frame_idx,
            source_frames=source_frame_count,
            elapsed_sec=elapsed_total,
            width=width,
            height=height,
            video_fps=fps,
            models=payload["models"],
            stats_summary=stats_summary,
            pipeline=pipeline_info,
            gpu_stats=gpu_stats,
            chart_paths=chart_paths,
            memory=memory_info,
            artifacts=artifacts,
        )
        summary_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        append_run_record(Path(output_dir), record)

        if has_video:
            self._mux_audio_if_possible(input_path, video_path)
        fps_proc = frame_idx / elapsed_total if elapsed_total > 0 else 0.0
        return ProcessVideoResult(
            out_dir=str(out_dir.resolve()),
            run_id=run_id,
            elapsed_sec=elapsed_total,
            fps_processed=fps_proc,
            frames=frame_idx,
            record=record,
            packets_path=str(pkl_path.resolve()) if pkl_path.exists() else None,
            video_path=str(video_path.resolve()) if has_video else None,
            has_video=has_video,
        )

    @staticmethod
    def _mux_audio_if_possible(src_video: str, video_only: Path) -> None:
        if not video_only.exists():
            return
        tmp = video_only.with_name(video_only.stem + "_audio.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_only),
            "-i",
            str(src_video),
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-shortest",
            str(tmp),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            tmp.replace(video_only)
        except (FileNotFoundError, subprocess.CalledProcessError):
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
