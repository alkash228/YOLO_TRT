"""Offline video pipeline: YOLO26 detect/seg + OSNet ReID."""
from __future__ import annotations

import gc
import json
import queue as queue_module
import shutil
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
from app.core.cross_check import (
    CrossCheckDetection,
    CrossCheckVerdict,
    evaluate_cross_check_batch,
    smooth_helmet_verdicts,
)
from app.core.fusion import dedupe_detections, inherit_motion_ids, match_detections_to_segments, merge_seg_fallback_detections
from app.core.async_finalize import AsyncBatchFinalizer
from app.core.batch_prepare import prepare_batch_frames
from app.core.batch_utils import (
    cap_job_batch_long_video,
    effective_speed_tuning,
    frame_stride_summary,
    resolve_batch_prefetch_depth,
    resolve_frame_source_ex,
    resolve_gpu_queue_depth,
    resolve_infer_batch_size,
    resolve_reid_embed_chunk,
    resolve_window_frames,
    source_frame_indices,
)
from app.core.gpu_monitor import GpuMonitor
from app.core.process_resource_monitor import ProcessResourceMonitor
from app.core.gpu_pipeline import (
    FrameBatchJob,
    GpuInferWorker,
    ReidEmbedWorker,
    chunk_frame_jobs,
    enrich_gpu_batch_reid,
    make_async_batch_jobs,
)
from app.core.motion_tracker import MotionTracker
from app.core.run_registry import (
    append_run_record,
    build_run_record,
    generate_run_charts,
    read_metrics_jsonl,
)
from app.core.mask_json import mask_u8_to_rle_dict
from app.core.memory_stats import build_memory_report
from app.core.video_encode import (
    build_encode_writer_args,
    encode_manifest_to_video,
    encode_packets_to_video,
    packets_path_for_run,
    save_run_packets,
)
from app.core.packet_spill import PacketSpillWriter
from app.core.pose_utils import keypoints_to_json
from app.core.prompt_utils import label_match, prompt_terms
from app.core.reid_engine import ReidEngine
from app.core.sam_memory_tracker import (
    build_identity_tracker,
    needs_osnet_embed,
)
from app.core.tracklet_linker import (
    build_tracklets_from_frames,
    build_tracklets_from_packets,
    enrich_tracklets_from_video,
    link_tracklets,
    remap_object_ids_in_frames,
    remap_object_ids_in_packets,
    rewrite_spill_chunks_with_packets,
)
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
    write_result_json,
)
from app.core.process_memory import current_process_rss_gb
from app.core.ram_budget import RamBudgetPlan, resolve_smart_ram_plan
from app.core.seg_engine import SegEngine
from app.core.window_frame_loader import VideoWindow, WindowPrefetcher, WindowedBatchJobs


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
        self._on_debug_log: Callable[[str], None] | None = None
        self._on_progress_tick: Callable[[int, int, str], None] | None = None
        self._progress_slot: Any = None
        # Per-run stage timers (decode / GPU worker wall / CPU finalize).
        self._stage_timing: dict[str, float] = {
            "decode_sec": 0.0,
            "gpu_infer_sec": 0.0,
            "cpu_finalize_sec": 0.0,
        }
        # SAM masklet identity OR classic OSNet ReidTracker (see use_sam_identity).
        self.tracker = build_identity_tracker(settings)
        self._helmet_verdict_history: dict[int, list[bool]] = {}

    def _apply_cross_check(
        self,
        *,
        valid_dets: list,
        width: int,
        height: int,
        stable_ids: np.ndarray,
        frame=None,
        accessories: list | None = None,
    ) -> tuple[list[CrossCheckVerdict], list[CrossCheckDetection], float]:
        cross_verdicts: list[CrossCheckVerdict] = []
        cross_accessories: list[CrossCheckDetection] = []
        cross_ms = 0.0
        if (
            not self.settings.cross_check_enabled
            or self.cross_check_engine is None
            or not valid_dets
        ):
            return cross_verdicts, cross_accessories, cross_ms
        if accessories is None:
            if frame is None:
                return cross_verdicts, cross_accessories, cross_ms
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
            helmet_min_conf=float(getattr(self.settings, "cross_check_helmet_min_conf", 0.0) or 0.0),
        )
        streak = int(getattr(self.settings, "cross_check_min_violation_streak", 2) or 2)
        if streak > 1 and len(stable_ids) == len(cross_verdicts):
            cross_verdicts = smooth_helmet_verdicts(
                stable_ids,
                cross_verdicts,
                self._helmet_verdict_history,
                min_violation_streak=streak,
                history_len=int(getattr(self.settings, "cross_check_verdict_history", 5) or 5),
            )
        self.cross_check_violations_total += sum(1 for v in cross_verdicts if not v.ok)
        return cross_verdicts, cross_accessories, cross_ms

    @staticmethod
    def _gpu_mem_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.memory_allocated() / (1024 * 1024))

    @staticmethod
    def _persist_source_in_run(input_path: str, out_dir: Path, run_id: str) -> str:
        """Copy source MP4 into run folder so deferred/manual encode survives job staging cleanup."""
        src = Path(input_path)
        if not src.is_file():
            return str(input_path)
        dest = out_dir / f"{run_id}_source{src.suffix or '.mp4'}"
        try:
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dest)
        except OSError:
            return str(src.resolve())
        return str(dest.resolve())

    @staticmethod
    def _resolve_prompt_id(
        label: str,
        prompt_id_lookup: dict[str, int],
        fallback_lookup: dict[str, int],
        next_fallback_id: list[int],
    ) -> int | None:
        from app.core.instance_serialize import resolve_prompt_id

        return resolve_prompt_id(label, prompt_id_lookup, fallback_lookup, next_fallback_id)

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
        from app.core.instance_serialize import serialize_frame_instances as _serialize

        return _serialize(
            instance_stack,
            instance_object_ids,
            instance_scores,
            prompt_labels,
            prompt_id_lookup=prompt_id_lookup,
            pose_kpt_conf=pose_kpt_conf,
            keypoints_list=keypoints_list,
            cross_check_verdicts=cross_check_verdicts,
        )

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

    def _apply_offline_tracklet_link(
        self,
        *,
        deferred_packets: list[FramePacket],
        frames_payload: list[dict[str, object]],
        spill_manifest_path: Path | None,
        out_dir: Path,
        input_path: str,
        on_debug_log: Callable[[str], None] | None = None,
    ) -> dict[str, object] | None:
        """
        Pass 2: merge non-overlapping tracklets after F2F tracking.
        Remaps packet stable_ids and/or frames_payload object_id in place.
        """
        if not bool(getattr(self.settings, "use_offline_tracklet_link", False)):
            return None

        packets: list[FramePacket] | None = None
        manifest: dict | None = None
        if spill_manifest_path is not None and spill_manifest_path.is_file():
            from app.core.packet_spill import load_all_spilled_packets, load_packets_manifest

            manifest = load_packets_manifest(spill_manifest_path)
            packets = load_all_spilled_packets(manifest, run_dir=out_dir)
            if deferred_packets:
                by_fi = {int(p.frame_idx): p for p in packets}
                for p in deferred_packets:
                    by_fi[int(p.frame_idx)] = p
                packets = sorted(by_fi.values(), key=lambda p: int(p.frame_idx))
        elif deferred_packets:
            packets = list(deferred_packets)

        if packets:
            tracklets = build_tracklets_from_packets(packets)
        elif frames_payload:
            tracklets = build_tracklets_from_frames(frames_payload)
        else:
            return None

        if len(tracklets) < 2:
            if on_debug_log:
                on_debug_log(
                    f"Tracklet link: skip ({len(tracklets)} tracklet(s), need ≥2)"
                )
            return None

        max_gap = int(getattr(self.settings, "tracklet_link_max_gap_frames", 300))
        min_sim = float(getattr(self.settings, "tracklet_link_min_sim", 0.60))
        spatial_w = float(getattr(self.settings, "tracklet_link_spatial_weight", 0.15))
        use_reid = bool(getattr(self.settings, "tracklet_link_use_reid", True))
        samples = int(getattr(self.settings, "tracklet_link_samples_per_tracklet", 5))

        missing_emb = sum(1 for t in tracklets if t.embedding is None)
        temp_engine: ReidEngine | None = None
        embed_engine = self.reid_engine
        if use_reid and missing_emb > 0:
            if embed_engine is None:
                reid_path = Path(self.settings.reid_model)
                if reid_path.is_file():
                    try:
                        if on_debug_log:
                            on_debug_log(
                                "Tracklet link: loading OSNet for offline embeddings…"
                            )
                        temp_engine = ReidEngine(
                            reid_path,
                            device=getattr(self.settings, "inference_device", "cuda"),
                            use_amp=bool(self.settings.use_amp),
                            use_tensorrt=bool(self.settings.use_tensorrt),
                            tensorrt_fp16=bool(self.settings.tensorrt_fp16),
                        )
                        embed_engine = temp_engine
                    except Exception as exc:
                        if on_debug_log:
                            on_debug_log(f"Tracklet link: OSNet load failed: {exc}")
                elif on_debug_log:
                    on_debug_log(
                        f"Tracklet link: ReID weights missing ({reid_path.name}); "
                        "appearance merge skipped"
                    )
            if embed_engine is not None:
                enrich_tracklets_from_video(
                    tracklets,
                    input_path,
                    embed_fn=embed_engine.embed_batch,
                    samples_per_tracklet=max(1, samples),
                    on_log=on_debug_log,
                )
        if temp_engine is not None:
            # Do NOT empty_cache here — kills allocator reuse and slows the next API job.
            del temp_engine

        has_any_emb = any(t.embedding is not None for t in tracklets)
        require_emb = bool(use_reid and has_any_emb)
        if use_reid and not has_any_emb and on_debug_log:
            on_debug_log(
                "Tracklet link: no embeddings available — "
                "skipping (enable tracklet_link_use_reid + OSNet weights)"
            )
            return None

        result = link_tracklets(
            tracklets,
            max_gap_frames=max_gap,
            min_sim=min_sim,
            spatial_weight=spatial_w,
            require_embedding=require_emb,
        )
        changed = sum(1 for o, n in result.id_map.items() if o != n)
        if changed <= 0:
            if on_debug_log:
                on_debug_log(
                    f"Tracklet link: no merges "
                    f"({result.n_tracklets_in} tracklets, gap≤{max_gap}, sim≥{min_sim})"
                )
            return {
                "n_tracklets_in": result.n_tracklets_in,
                "n_ids_out": result.n_ids_out,
                "merges": 0,
            }

        n_pkt = 0
        n_fr = 0
        if packets is not None:
            n_pkt = remap_object_ids_in_packets(packets, result.id_map)
            if deferred_packets and packets is not deferred_packets:
                by_fi = {int(p.frame_idx): p for p in packets}
                for i, p in enumerate(deferred_packets):
                    remapped = by_fi.get(int(p.frame_idx))
                    if remapped is not None:
                        deferred_packets[i] = remapped
            if manifest is not None and spill_manifest_path is not None:
                rewrite_spill_chunks_with_packets(
                    manifest, run_dir=out_dir, packets=packets
                )
        if frames_payload:
            n_fr = remap_object_ids_in_frames(frames_payload, result.id_map)

        info: dict[str, object] = {
            "n_tracklets_in": result.n_tracklets_in,
            "n_ids_out": result.n_ids_out,
            "merges": len(result.merges),
            "instances_remapped": n_pkt + n_fr,
            "id_map": {str(k): v for k, v in result.id_map.items() if k != v},
        }
        if on_debug_log:
            on_debug_log(
                f"Tracklet link: {result.n_tracklets_in}→{result.n_ids_out} IDs "
                f"({len(result.merges)} merges, remapped {n_pkt + n_fr} instances)"
            )
        return info

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

    def _sync_progress_resources(
        self,
        gpu_monitor: GpuMonitor | None,
        process_res_mon: ProcessResourceMonitor | None,
    ) -> None:
        slot = getattr(self, "_progress_slot", None)
        if slot is None:
            return
        proc = process_res_mon.latest() if process_res_mon is not None else None
        gpu = gpu_monitor.latest() if gpu_monitor is not None else None
        slot.set_resources(
            process_rss_mb=proc.process_rss_mb if proc is not None else 0.0,
            cuda_allocated_mb=proc.cuda_allocated_mb if proc is not None else 0.0,
            cuda_reserved_mb=proc.cuda_reserved_mb if proc is not None else 0.0,
            gpu_device_used_mb=float(gpu.mem_used_mb) if gpu is not None else 0.0,
            gpu_util_pct=float(gpu.gpu_util_pct) if gpu is not None else 0.0,
        )

    def _write_progress_hook(self, current: int, total: int, phase: str) -> None:
        """Throttled progress slot (API) or legacy tick callback."""
        out_frame = max(0, int(current))
        total_out = max(0, int(total)) if int(total) > 0 else max(1, out_frame)
        slot = getattr(self, "_progress_slot", None)
        if slot is not None:
            slot.maybe_update(out_frame, total_out, phase)
            self._sync_progress_resources(
                getattr(self, "_gpu_monitor", None),
                getattr(self, "_process_res_mon", None),
            )
            return
        tick = getattr(self, "_on_progress_tick", None)
        if tick is not None:
            tick(out_frame, total_out, phase)

    def _pulse_infer_progress(
        self,
        *,
        current_frame: int,
        total: int,
        instances_peak: int,
        start_ts: float,
        on_progress: Callable[[VideoProgress], None] | None,
        gpu_monitor: GpuMonitor | None,
        process_res_mon: ProcessResourceMonitor | None = None,
        phase: str = "inference",
        fps_start_ts: float | None = None,
        min_interval_sec: float = 0.0,
    ) -> None:
        slot = getattr(self, "_progress_slot", None)
        tick = getattr(self, "_on_progress_tick", None)
        if on_progress is None and tick is None and slot is None:
            return
        now = time.perf_counter()
        # Throttle wait-loop spam only; batch-complete pulses pass min_interval_sec=0.
        if min_interval_sec > 0:
            last = float(getattr(self, "_last_progress_pulse_ts", 0.0) or 0.0)
            if now - last < min_interval_sec:
                return
        self._last_progress_pulse_ts = now
        out_frame = max(0, int(current_frame))
        total_out = total if total > 0 else max(1, out_frame)
        self._write_progress_hook(out_frame, total_out, phase)
        if on_progress is None:
            return
        elapsed = now - start_ts
        fps_anchor = float(fps_start_ts) if fps_start_ts is not None else start_ts
        fps_elapsed = max(1e-6, now - fps_anchor)
        proc_fps = out_frame / fps_elapsed if out_frame > 0 else 0.0
        eta = ((total - out_frame) / proc_fps) if proc_fps > 0 and total > 0 else 0.0
        gpu_util = 0.0
        gpu_mem = 0.0
        if gpu_monitor is not None:
            latest = gpu_monitor.latest()
            if latest is not None:
                gpu_util = latest.gpu_util_pct
        if process_res_mon is not None:
            proc = process_res_mon.latest()
            if proc is not None:
                gpu_mem = float(proc.cuda_allocated_mb)
        elif gpu_mem <= 0.0:
            gpu_mem = self._gpu_mem_mb()
        on_progress(
            VideoProgress(
                current=out_frame,
                total=total_out,
                fps=proc_fps,
                eta_seconds=eta,
                gpu_mem_mb=gpu_mem,
                elapsed_sec=elapsed,
                gpu_util_pct=gpu_util,
                instances_current=0,
                instances_peak=instances_peak,
                stats=None,
                phase=phase,
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
        process_res_mon: ProcessResourceMonitor | None = None,
        phase: str = "inference",
        infer_ms: float | None = None,
        write_metrics: bool = True,
        current_frame: int | None = None,
    ) -> None:
        out_frame = (
            max(1, int(current_frame))
            if current_frame is not None
            else int(packet.frame_idx) + 1
        )
        total_out = total if total > 0 else out_frame
        self._write_progress_hook(out_frame, total_out, phase)
        # Silent API path: skip RunStats / cuda.memory_allocated / VideoProgress.
        if not write_metrics and on_progress is None:
            return
        elapsed = time.perf_counter() - start_ts
        proc_fps = out_frame / elapsed if elapsed > 0 else 0.0
        eta = ((total - out_frame) / proc_fps) if proc_fps > 0 and total > 0 else 0.0
        gpu_util = 0.0
        if gpu_monitor is not None:
            latest = gpu_monitor.latest()
            if latest is not None:
                gpu_util = latest.gpu_util_pct
        gpu_mem = 0.0
        if process_res_mon is not None:
            proc = process_res_mon.latest()
            if proc is not None:
                gpu_mem = float(proc.cuda_allocated_mb)
        if gpu_mem <= 0.0:
            gpu_mem = self._gpu_mem_mb()
        infer_ms_val = (
            float(infer_ms)
            if infer_ms is not None
            else float(infer_ms_by_frame.get(packet.frame_idx, packet.infer_ms))
        )
        stats = RunStats(
            frame=out_frame,
            total_frames=total_out,
            fps=proc_fps,
            eta_sec=eta,
            gpu_mem_mb=gpu_mem,
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
                    total=total_out,
                    fps=proc_fps,
                    eta_seconds=eta,
                    gpu_mem_mb=stats.gpu_mem_mb,
                    elapsed_sec=elapsed,
                    gpu_util_pct=gpu_util,
                    instances_current=packet.n_inst,
                    instances_peak=instances_peak,
                    stats=stats,
                    phase=phase,
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
                        phase="post",
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
        use_sam = bool(getattr(self.settings, "use_sam_identity", False))
        want_identity = use_sam or (
            self.settings.use_reid and self.reid_engine is not None
        )
        need_osnet = needs_osnet_embed(self.settings) and self.reid_engine is not None
        if want_identity:
            detections = inherit_motion_ids(detections)
            detections = dedupe_detections(detections, iou_min=0.55)

        use_torso = bool(getattr(self.settings, "reid_torso_crop", True))
        kpt_conf = float(getattr(self.settings, "pose_kpt_conf", 0.25))
        torso_pad = float(getattr(self.settings, "reid_torso_pad", 0.20))
        torso_min_area = float(getattr(self.settings, "reid_torso_min_area_ratio", 0.12))
        crops: list[np.ndarray | None] = []
        valid_dets: list[DetectItem] = []
        for det in detections:
            valid_dets.append(det)
            if need_osnet:
                crops.append(
                    ReidEngine.crop_for_reid(
                        frame,
                        det.xyxy,
                        det.keypoints,
                        kpt_conf=kpt_conf,
                        use_torso=use_torso,
                        torso_pad=torso_pad,
                        torso_min_area_ratio=torso_min_area,
                    )
                )
            else:
                crops.append(None)
        crop_list = [c for c in crops if c is not None and c.size > 0]

        t0 = time.perf_counter()
        if want_identity:
            if need_osnet and crop_list:
                raw_embs = self.reid_engine.embed_batch(crop_list)
                emb_dim = int(raw_embs.shape[1])
                embs = np.zeros((len(valid_dets), emb_dim), dtype=np.float32)
                ci = 0
                for di, crop in enumerate(crops):
                    if crop is not None and crop.size > 0:
                        embs[di] = raw_embs[ci]
                        ci += 1
                embeddings = embs
            else:
                embeddings = np.zeros((len(valid_dets), 512), dtype=np.float32)
            reid_ms = (time.perf_counter() - t0) * 1000.0
            track_result = self.tracker.update(valid_dets, embeddings, frame_idx=frame_idx)
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

        cross_verdicts, cross_accessories, cross_ms = self._apply_cross_check(
            valid_dets=valid_dets,
            frame=frame,
            width=width,
            height=height,
            stable_ids=stable_ids,
        )

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
            # TRT track/predict сами pad'ят до engine batch; pad хвоста job→bs
            # только копирует last frame (и раньше светил «28→64») без пользы.
            jobs = chunk_frame_jobs(
                selected,
                batch_size=bs,
                frame_indices=indices,
                pad_last_job=False,
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
        *,
        on_debug_log: Callable[[str], None] | None = None,
    ) -> list[tuple[FramePacket, int]]:
        n = len(batch.frame_indices)
        if n == 0:
            return []

        per_det_ms = batch.detect_ms / max(1, n)
        per_seg_ms = batch.seg_ms / max(1, n)
        per_cross_ms = batch.cross_ms / max(1, n)

        use_sam = bool(getattr(self.settings, "use_sam_identity", False))
        want_identity = use_sam or (
            self.settings.use_reid and self.reid_engine is not None
        )
        need_osnet = needs_osnet_embed(self.settings) and self.reid_engine is not None
        # BoT-SORT track when stable identity is needed (SAM or OSNet).
        use_motion_tracker = self.settings.gpu_full_batch and not want_identity

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
                need_osnet
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
                    # GPU ReID уже на main; finalizer не трогает CUDA (deadlock с enrich).
                    n_use = min(n_emb, len(crop_map))
                    for slot in range(n_use):
                        emb_by_slot[crop_map[slot]] = batch.reid_embeddings[slot]
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
            if need_osnet and all_crops:
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
            if want_identity:
                if valid_dets:
                    if need_osnet and self.settings.reid_batch_across_frames:
                        emb_dim = int(next(iter(emb_by_slot.values())).shape[0]) if emb_by_slot else 512
                        embs = np.stack(
                            [
                                emb_by_slot.get(
                                    (fi, di),
                                    np.zeros(emb_dim, dtype=np.float32),
                                )
                                for di in range(len(valid_dets))
                            ],
                            axis=0,
                        )
                        reid_ms = per_reid_ms
                    elif need_osnet:
                        use_torso = bool(getattr(self.settings, "reid_torso_crop", True))
                        kpt_conf = float(getattr(self.settings, "pose_kpt_conf", 0.25))
                        torso_pad = float(getattr(self.settings, "reid_torso_pad", 0.20))
                        torso_min_area = float(
                            getattr(self.settings, "reid_torso_min_area_ratio", 0.12)
                        )
                        crops = [
                            ReidEngine.crop_for_reid(
                                frame,
                                d.xyxy,
                                d.keypoints,
                                kpt_conf=kpt_conf,
                                use_torso=use_torso,
                                torso_pad=torso_pad,
                                torso_min_area_ratio=torso_min_area,
                            )
                            for d in valid_dets
                        ]
                        crop_list = [c for c in crops if c is not None and c.size > 0]
                        t0 = time.perf_counter()
                        if crop_list:
                            raw_embs = self.reid_engine.embed_batch(crop_list)
                            emb_dim = int(raw_embs.shape[1])
                            embs = np.zeros((len(valid_dets), emb_dim), dtype=np.float32)
                            ci = 0
                            for di, c in enumerate(crops):
                                if c is not None and c.size > 0:
                                    embs[di] = raw_embs[ci]
                                    ci += 1
                        else:
                            embs = np.zeros((len(valid_dets), 512), dtype=np.float32)
                        reid_ms = (time.perf_counter() - t0) * 1000.0
                    else:
                        embs = np.zeros((len(valid_dets), 512), dtype=np.float32)
                else:
                    embs = np.zeros((0, 512), dtype=np.float32)
                track_result = self.tracker.update(valid_dets, embs, frame_idx=frame_idx)
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

            cross_verdicts, cross_accessories, _ = self._apply_cross_check(
                valid_dets=valid_dets,
                width=width,
                height=height,
                stable_ids=stable_ids,
                accessories=[d for d in frame_cross_raw[fi] if label_match(d.label, obj_terms)],
            )

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
        infer_done_ref: list[int],
        should_stop: Callable[[], bool] | None,
        should_pause: Callable[[], bool] | None,
        on_debug_log: Callable[[str], None] | None,
        batch_jobs: Iterator[FrameBatchJob],
        n_process: int,
        spill_flush_hooks: list[Callable[[], None]] | None = None,
    ) -> tuple[int, int]:
        seg_eng = self.seg_engine if self.settings.use_seg else None
        use_sam = bool(getattr(self.settings, "use_sam_identity", False))
        need_osnet = needs_osnet_embed(self.settings) and self.reid_engine is not None
        want_identity = use_sam or (
            self.settings.use_reid and self.reid_engine is not None
        )
        eff_bs = self._job_batch_size
        n_jobs = (n_process + eff_bs - 1) // eff_bs if n_process > 0 else 0
        worker = GpuInferWorker(
            self.detect_engine,
            seg_eng,
            self.cross_check_engine,
            self.settings,
            on_batch_log=on_debug_log,
        )
        worker.start()
        detect_mode = "predict_batch" if not worker._detect_use_track else "track"
        dev = str(getattr(self.settings, "inference_device", "cuda") or "cuda").casefold()
        # Track (обязателен при stable identity) уже грузит GPU; второй cuda-stream не ускоряет.
        reid_overlap = bool(
            need_osnet
            and bool(getattr(self.settings, "reid_gpu_overlap", False))
            and dev != "cpu"
            and torch.cuda.is_available()
            and not worker._detect_use_track
        )
        # OSNet serial pipeline only when embeds are required; SAM-only skips it.
        serial_cuda = need_osnet and not reid_overlap
        # ReID serial: YOLO job N+1 на GPU, пока CPU tracker обрабатывает job N (без YOLO||ReID).
        pipeline_cpu = serial_cuda or (want_identity and not need_osnet)
        # Preload держит кадры в RAM — глубина очереди = ссылки на job, не копии кадров.
        # Раньше SAM/ReID жёстко ставили queue=2 → GPU worker часто простаивал между job.
        requested_depth = resolve_gpu_queue_depth(
            self.settings.gpu_queue_depth,
            n_jobs=n_jobs,
            job_batch_size=eff_bs,
            max_job_batch=self.settings.max_job_batch_size,
        )
        if reid_overlap:
            # Два CUDA-потребителя (detect + OSNet stream) — не раздувать in-flight.
            depth = min(2, requested_depth)
        elif serial_cuda:
            # YOLO||CPU finalize: минимум 2, иначе нет overlap.
            depth = max(2, min(requested_depth, 3))
        elif pipeline_cpu:
            # SAM masklet без live OSNet: CPU tracker лёгкий — можно глубже (до settings).
            depth = max(2, requested_depth)
        else:
            depth = requested_depth
        reid_worker: ReidEmbedWorker | None = None
        if reid_overlap:
            assert self.reid_engine is not None
            reid_worker = ReidEmbedWorker(
                self.reid_engine,
                self.settings,
                terms,
                queue_depth=max(2, int(self.settings.gpu_queue_depth)),
            )
            reid_worker.start()
        if on_debug_log:
            req = self.settings.infer_batch_size
            req_txt = "авто→все кадры" if int(req) <= 0 and self.settings.gpu_full_batch else str(req)
            cap_note = ""
            if (
                need_osnet
                and int(self.settings.max_job_batch_size) > 0
                and int(req) > eff_bs
            ):
                cap_note = f" | job cap ReID {req}→{eff_bs}"
            queue_note = ""
            if depth != int(self.settings.gpu_queue_depth):
                queue_note = f" (запрос {self.settings.gpu_queue_depth})"
            if use_sam:
                mode_note = " | SAM masklet identity"
            elif reid_overlap:
                mode_note = " | ReID GPU overlap (cuda stream)"
            elif serial_cuda:
                mode_note = " | ReID pipeline (YOLO||CPU tracker)"
                if need_osnet and bool(getattr(self.settings, "reid_gpu_overlap", False)):
                    mode_note += " [gpu overlap off]"
            else:
                mode_note = ""
            trt_b = int(getattr(self.detect_engine, "trt_max_batch", 0) or 0)
            trt_note = ""
            if trt_b > 0 and eff_bs > trt_b:
                trt_note = (
                    f" | TRT detect b{trt_b} (job {eff_bs}→{eff_bs // trt_b}× launches/job; "
                    f"rebuild engine b{min(eff_bs, int(self.settings.tensorrt_max_batch))} to fill GPU)"
                )
            on_debug_log(
                f"GPU job: {eff_bs} кадр/батч × {n_jobs} job | inference {n_process} кадр | "
                f"detect={detect_mode} | YOLO chunk={self.settings.max_infer_batch_size} | "
                f"queue={depth}{queue_note} | запрос batch={req_txt}{cap_note}{mode_note}{trt_note}"
            )
        job_iter = batch_jobs
        pending_jobs: deque[FrameBatchJob] = deque()
        in_flight = 0
        # GPU-completed frames (ahead of CPU finalize) — progress must not wait for emit.
        gpu_done_ref = [0]

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
            t0 = time.perf_counter()
            n_fr = len(getattr(result, "frame_indices", []) or [])
            if on_debug_log and n_fr < self._job_batch_size:
                on_debug_log(f"Finalizer: tracker {n_fr} fr…")
            packets = self._cpu_finalize_batch(
                result,
                frames_bgr,
                terms,
                obj_terms,
                height,
                width,
                on_debug_log=on_debug_log,
            )
            self._stage_timing["cpu_finalize_sec"] = float(
                self._stage_timing.get("cpu_finalize_sec", 0.0)
            ) + max(0.0, time.perf_counter() - t0)
            if on_debug_log and n_fr < self._job_batch_size:
                ms = (time.perf_counter() - t0) * 1000.0
                on_debug_log(f"Finalizer: {len(packets)} pkts in {ms:.0f}ms")
            if deferred_encode:
                # Compact once here; _emit_packets must NOT compact again
                # (second compact used to wipe masks_rle → empty JSON instances).
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
                infer_done_ref[0] += 1

                if deferred_encode:
                    # Already compacted in _finalize_job when deferred_encode.
                    self._append_deferred_packet(packet, deferred_packets, compact=False)
                    last_packet = packet
                    # Manual/deferred never hits _emit_post_results — still accumulate ms.
                    emit_kwargs["detect_ms_acc"][0] += float(packet.detect_ms)
                    emit_kwargs["seg_ms_acc"][0] += float(packet.seg_ms)
                    emit_kwargs["reid_ms_acc"][0] += float(packet.reid_ms)
                    emit_kwargs["total_ms_acc"][0] += float(
                        infer_ms_by_frame.get(packet.frame_idx, packet.infer_ms)
                    )
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
                    process_res_mon=emit_kwargs.get("process_res_mon"),
                    phase="inference",
                    write_metrics=False,
                    current_frame=infer_done_ref[0],
                )
            _try_pending_window_spill()

        def _drain_pipeline() -> None:
            if reid_worker is not None:
                for enriched, frames in reid_worker.poll():
                    if finalizer is not None:
                        finalizer.submit(enriched, frames)
            if finalizer is not None:
                for batch_packets in finalizer.drain():
                    _emit_packets(batch_packets)

        def _submit_finalize(result, frames_bgr: list) -> None:
            if finalizer is None:
                packets = _finalize_job(result, frames_bgr)
                _emit_packets(packets)
                return
            # Иначе RAM: 100+ job × 64 кадра в очереди finalizer → swap/«зависание».
            max_pending = max(2, depth)
            while finalizer.pending_depth() >= max_pending:
                for batch_packets in finalizer.drain():
                    _emit_packets(batch_packets)
                if finalizer.pending_depth() >= max_pending:
                    time.sleep(0.005)
            finalizer.submit(result, frames_bgr)

        finalizer: AsyncBatchFinalizer | None = None
        if pipeline_cpu or reid_overlap or not serial_cuda:
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

        def _flush_before_window_spill() -> None:
            if reid_worker is not None:
                for enriched, frames in reid_worker.poll():
                    if finalizer is not None:
                        finalizer.submit(enriched, frames)
            if finalizer is not None:
                for batch_packets in finalizer.flush_pending():
                    _emit_packets(batch_packets)
            _drain_pipeline()

        if spill_flush_hooks is not None:
            spill_flush_hooks.clear()
            spill_flush_hooks.append(_flush_before_window_spill)

        def _window_pipeline_idle() -> bool:
            if in_flight > 0:
                return False
            if finalizer is not None and finalizer.is_busy():
                return False
            return True

        def _window_spill_ready() -> bool:
            info = getattr(job_iter, "pending_spill_info", None)
            if info is None:
                return False
            lo, hi = int(info.start_frame), int(info.end_frame)
            got = sum(1 for p in deferred_packets if lo <= int(p.frame_idx) < hi)
            return got >= int(info.infer_count)

        def _try_pending_window_spill() -> None:
            try_finish = getattr(job_iter, "try_finish_pending_window", None)
            if not callable(try_finish) or not getattr(job_iter, "has_pending_spill", False):
                return
            if not _window_spill_ready():
                return
            for hook in spill_flush_hooks or []:
                hook()
            if not _window_spill_ready():
                return
            try_finish(ready=True)

        def _drain_all_pending_windows() -> None:
            force_finish = getattr(job_iter, "force_finish_pending_window", None)
            if not callable(force_finish):
                return
            deadline = time.monotonic() + 180.0
            while getattr(job_iter, "has_pending_spill", False):
                for hook in spill_flush_hooks or []:
                    hook()
                _drain_pipeline()
                if _window_spill_ready():
                    force_finish()
                    continue
                if _window_pipeline_idle():
                    info = getattr(job_iter, "pending_spill_info", None)
                    if info is not None:
                        lo, hi = int(info.start_frame), int(info.end_frame)
                        got = sum(1 for p in deferred_packets if lo <= int(p.frame_idx) < hi)
                        if on_debug_log:
                            on_debug_log(
                                f"Window spill flush: {got}/{info.infer_count} pkts ready, force…"
                            )
                    force_finish()
                    continue
                if on_debug_log and time.monotonic() > deadline:
                    on_debug_log("Window spill: timeout 180s, force finish")
                    force_finish()
                    break
                time.sleep(0.01)

        stall_log_ts = 0.0

        try:
            _prime()

            while in_flight > 0:
                if should_stop and should_stop():
                    break
                _drain_pipeline()
                try:
                    result = worker.get_result(timeout=0.05)
                except queue_module.Empty:
                    now = time.monotonic()
                    if on_debug_log and now - stall_log_ts >= 15.0:
                        stall_log_ts = now
                        fin_q = finalizer.pending_depth() if finalizer is not None else 0
                        fin_busy = (
                            finalizer.is_busy()
                            if finalizer is not None
                            else False
                        )
                        job_note = ""
                        if pending_jobs:
                            pj = pending_jobs[0]
                            if pj.frame_indices:
                                pn = pj.infer_count
                                idx_end = (
                                    pj.frame_indices[pn - 1]
                                    if pn > 0
                                    else pj.frame_indices[0]
                                )
                                job_note = (
                                    f" job=[{pj.frame_indices[0]}..{idx_end}]"
                                    f" n={pn}"
                                )
                        on_debug_log(
                            f"Waiting GPU: in_flight={in_flight} "
                            f"pending_jobs={len(pending_jobs)} finalizer={fin_q}"
                            f"{'+work' if fin_busy else ''} "
                            f"infer_done={infer_done_ref[0]}/{n_process}{job_note}"
                        )
                    if not pending_jobs and in_flight > 0:
                        if on_debug_log:
                            on_debug_log(
                                f"GPU queue reset: in_flight {in_flight}→0 "
                                f"(нет pending_jobs, infer_done={infer_done_ref[0]})"
                            )
                        in_flight = 0
                        break
                    self._pulse_infer_progress(
                        current_frame=max(infer_done_ref[0], gpu_done_ref[0]),
                        total=emit_kwargs["total"],
                        instances_peak=instances_peak,
                        start_ts=emit_kwargs["start_ts"],
                        on_progress=emit_kwargs.get("on_progress"),
                        gpu_monitor=emit_kwargs.get("gpu_monitor"),
                    process_res_mon=emit_kwargs.get("process_res_mon"),
                        phase="gpu",
                        fps_start_ts=emit_kwargs.get("infer_start_ts"),
                        min_interval_sec=0.25,
                    )
                    continue

                # Do NOT torch.cuda.synchronize() here — that serializes the GPU
                # pipeline vs UI and kills overlap with CPU finalize / next submit.

                job = pending_jobs.popleft()
                frames_bgr = job.frames_bgr
                in_flight -= 1
                real_n = job.infer_count
                frames_infer = frames_bgr[:real_n]
                job.frames_bgr = []
                gpu_done_ref[0] += real_n
                self._pulse_infer_progress(
                    current_frame=max(gpu_done_ref[0], infer_done_ref[0]),
                    total=emit_kwargs["total"],
                    instances_peak=instances_peak,
                    start_ts=emit_kwargs["start_ts"],
                    on_progress=emit_kwargs.get("on_progress"),
                    gpu_monitor=emit_kwargs.get("gpu_monitor"),
                    process_res_mon=emit_kwargs.get("process_res_mon"),
                    phase="inference",
                    fps_start_ts=emit_kwargs.get("infer_start_ts"),
                    min_interval_sec=0.25,
                )

                is_tail = real_n < self._job_batch_size or len(frames_bgr) > real_n

                def _post_log(msg: str) -> None:
                    if on_debug_log and is_tail:
                        on_debug_log(msg)

                if on_debug_log and is_tail:
                    idx_lo = job.frame_indices[0] if job.frame_indices else -1
                    idx_hi = job.frame_indices[real_n - 1] if real_n > 0 else idx_lo
                    pad_note = (
                        f" (padded→{len(frames_bgr)})"
                        if len(frames_bgr) > real_n
                        else ""
                    )
                    on_debug_log(
                        f"CPU post: {real_n} fr [{idx_lo}..{idx_hi}]{pad_note}…"
                    )

                if serial_cuda:
                    assert self.reid_engine is not None
                    enriched = enrich_gpu_batch_reid(
                        result,
                        frames_infer,
                        reid_engine=self.reid_engine,
                        settings=self.settings,
                        terms=terms,
                        on_log=_post_log,
                    )
                    _post_log("CPU post: submit finalizer…")
                    if finalizer is not None:
                        _submit_finalize(enriched, frames_infer)
                    else:
                        packets = _finalize_job(enriched, frames_infer)
                        _emit_packets(packets)
                    _post_log("CPU post: done")
                    try:
                        next_job = next(job_iter)
                        worker.submit(next_job)
                        pending_jobs.append(next_job)
                        in_flight += 1
                    except StopIteration:
                        if not pending_jobs:
                            in_flight = 0
                elif reid_overlap:
                    assert reid_worker is not None
                    reid_worker.submit(result, frames_infer)
                    try:
                        next_job = next(job_iter)
                        worker.submit(next_job)
                        pending_jobs.append(next_job)
                        in_flight += 1
                    except StopIteration:
                        if not pending_jobs:
                            in_flight = 0
                else:
                    try:
                        next_job = next(job_iter)
                        worker.submit(next_job)
                        pending_jobs.append(next_job)
                        in_flight += 1
                    except StopIteration:
                        if not pending_jobs:
                            in_flight = 0
                    if finalizer is not None:
                        finalizer.submit(result, frames_infer)

                _drain_pipeline()
                _try_pending_window_spill()
                drain_post()
                emit_kwargs["instances_peak"] = instances_peak

            _drain_all_pending_windows()

            if on_debug_log:
                on_debug_log("GPU jobs done, draining finalizer…")
            worker.close()
            _drain_pipeline()
            if reid_worker is not None:
                for enriched, frames in reid_worker.finish():
                    if finalizer is not None:
                        finalizer.submit(enriched, frames)
            if finalizer is not None:
                if on_debug_log:
                    on_debug_log("Finalizer: draining remaining jobs…")
                for batch_packets in finalizer.finish():
                    _emit_packets(batch_packets)
            _drain_all_pending_windows()
            drain_post()
            emit_kwargs["instances_peak"] = instances_peak
            self._stage_timing["gpu_infer_sec"] = float(
                self._stage_timing.get("gpu_infer_sec", 0.0)
            ) + float(getattr(worker, "gpu_wall_sec", 0.0) or 0.0)
        finally:
            if spill_flush_hooks is not None:
                spill_flush_hooks.clear()
            _close_batch_feeder()

        return instances_peak, frame_idx_ref[0]

    def _drive_windowed_pipeline(
        self,
        *,
        input_path: str,
        source_frame_count: int,
        frame_stride: int,
        terms: list[str],
        obj_terms: list[str],
        height: int,
        width: int,
        deferred_encode: bool,
        deferred_packets: list[FramePacket],
        spill_writer: PacketSpillWriter,
        prompt_id_lookup: dict[str, int],
        post_exec: OrderedPostExecutor | None,
        drain_post: Callable[[], None],
        emit_kwargs: dict,
        frame_idx_ref: list[int],
        instances_peak: int,
        infer_ms_by_frame: dict[int, float],
        packet_meta: dict[int, FramePacket],
        infer_done_ref: list[int],
        should_stop: Callable[[], bool] | None,
        should_pause: Callable[[], bool] | None,
        on_debug_log: Callable[[str], None] | None,
        n_process: int,
        use_gpu_pipe: bool,
        model_runner: ModelParallelRunner | None,
        manual_encode: bool,
        frames_payload: list[dict[str, object]],
        infer_per_window: int,
        windows_in_ram: int | None = None,
    ) -> tuple[int, int]:
        infer_per_window = max(1, int(infer_per_window))
        stride = max(1, int(frame_stride))
        win_ram = (
            int(windows_in_ram)
            if windows_in_ram is not None
            else int(getattr(self.settings, "windows_in_ram", 1))
        )
        windows_in_ram = max(1, win_ram)
        ahead = max(0, windows_in_ram - 1)
        source_span = infer_per_window * stride
        n_windows = (
            (source_frame_count + source_span - 1) // source_span
            if source_frame_count > 0
            else "?"
        )

        if on_debug_log:
            win_mb = estimate_video_ram_gb(width, height, infer_per_window) * 1024
            jobs_pw = infer_per_window // max(1, self._job_batch_size)
            on_debug_log(
                f"Windowed preload: infer {infer_per_window} fr/окно "
                f"({jobs_pw}×job {self._job_batch_size}, src span {source_span}), "
                f"RAM ~{win_mb:.0f} MB, ahead={ahead}, ~{n_windows} окон"
            )

        prefetcher = WindowPrefetcher(
            input_path,
            infer_per_window=infer_per_window,
            frame_stride=stride,
            total_frames=source_frame_count,
            windows_ahead=ahead,
            should_stop=should_stop,
            should_pause=should_pause,
        )
        prefetcher.start()

        def _log_window(num: int, window: VideoWindow) -> None:
            if not on_debug_log:
                return
            win_process = window.frame_indices
            n_jobs = (len(win_process) + self._job_batch_size - 1) // self._job_batch_size
            tail = len(win_process) % self._job_batch_size
            tail_note = f", tail {tail}" if tail else ""
            on_debug_log(
                f"Window {num}: src [{window.start_frame}..{window.end_frame}), "
                f"infer {len(win_process)} = {n_jobs}×job {self._job_batch_size}{tail_note}"
            )

        def _finish_window(num: int, window: VideoWindow) -> None:
            lo, hi = int(window.start_frame), int(window.end_frame)
            window_packets = [
                p for p in deferred_packets if lo <= int(p.frame_idx) < hi
            ]
            if window_packets:
                keep = [
                    p for p in deferred_packets if not (lo <= int(p.frame_idx) < hi)
                ]
                deferred_packets.clear()
                deferred_packets.extend(keep)
            if deferred_encode and window_packets:
                if on_debug_log:
                    on_debug_log(
                        f"Window {num} spill: {len(window_packets)} pkts → disk…"
                    )
                spill_writer.write_chunk(
                    window_packets,
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                )

        spill_flush_hooks: list[Callable[[], None]] = []
        if use_gpu_pipe:
            batch_feeder = WindowedBatchJobs(
                prefetcher,
                batch_size=self._job_batch_size,
                on_window_start=_log_window,
                on_window_end=_finish_window,
            )
            try:
                instances_peak, _ = self._drive_gpu_pipeline(
                    all_frames=None,
                    cap_stream=None,
                    n_frames_total=source_frame_count,
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
                    infer_done_ref=infer_done_ref,
                    should_stop=should_stop,
                    should_pause=should_pause,
                    on_debug_log=on_debug_log,
                    batch_jobs=batch_feeder,
                    n_process=n_process,
                    spill_flush_hooks=spill_flush_hooks,
                )
            finally:
                batch_feeder.close()
                if on_debug_log:
                    on_debug_log("GPU pipeline idle.")
            self._stage_timing["decode_sec"] = float(
                self._stage_timing.get("decode_sec", 0.0)
            ) + float(getattr(prefetcher, "decode_sec", 0.0) or 0.0)
        else:
            window_num = 0
            try:
                while True:
                    if should_stop and should_stop():
                        break
                    window = prefetcher.next_window()
                    if window is None or not window.frames_bgr:
                        break

                    win_process = list(window.frame_indices)
                    if not win_process:
                        window_num += 1
                        continue

                    selected = window.frames_bgr
                    _log_window(window_num, window)

                    for frame_idx, frame in zip(win_process, selected, strict=False):
                        if should_stop and should_stop():
                            break
                        while should_pause and should_pause():
                            time.sleep(0.05)
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
                        frame_idx_ref[0] = max(frame_idx_ref[0], frame_idx + 1)
                        infer_done_ref[0] += 1
                        if deferred_encode:
                            progress_total = n_process if n_process > 0 else infer_done_ref[0]
                            self._report_infer_progress(
                                packet=packet,
                                total=progress_total,
                                instances_peak=instances_peak,
                                start_ts=emit_kwargs["start_ts"],
                                infer_ms_by_frame=infer_ms_by_frame,
                                metrics_path=emit_kwargs["metrics_path"],
                                on_progress=emit_kwargs.get("on_progress"),
                                gpu_monitor=emit_kwargs.get("gpu_monitor"),
                    process_res_mon=emit_kwargs.get("process_res_mon"),
                                phase="inference",
                                write_metrics=False,
                                current_frame=infer_done_ref[0],
                            )
                            self._append_deferred_packet(
                                packet, deferred_packets, compact=True
                            )
                        elif post_exec is not None:
                            post_exec.submit(packet, frame_source=None)
                            drain_post()
                        emit_kwargs["instances_peak"] = instances_peak

                    _finish_window(window_num, window)
                    del window.frames_bgr[:]
                    del window
                    gc.collect()
                    window_num += 1
            finally:
                prefetcher.close()
            self._stage_timing["decode_sec"] = float(
                self._stage_timing.get("decode_sec", 0.0)
            ) + float(getattr(prefetcher, "decode_sec", 0.0) or 0.0)

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
        on_progress_tick: Callable[[int, int, str], None] | None = None,
        progress_slot: Any = None,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> ProcessVideoResult:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        # Soft reset: clear BoT-SORT IDs only. Do NOT tear down TRT predictor
        # (hard reset after API warmup reloads .engine and ~2× slows Pass1).
        self.detect_engine.reset_session()
        self.tracker.reset()
        self.motion_tracker.reset()
        self.cross_check_violations_total = 0
        self._helmet_verdict_history.clear()
        self._on_debug_log = on_debug_log
        # Lightweight (current, total, phase) — no VideoProgress; used by API jobs.
        self._on_progress_tick = on_progress_tick
        self._progress_slot = progress_slot
        self._stage_timing = {
            "decode_sec": 0.0,
            "gpu_infer_sec": 0.0,
            "cpu_finalize_sec": 0.0,
        }
        if on_debug_log and getattr(self.settings, "reid_debug_log", False):
            on_debug_log(
                "ReID debug: ON (new_id, id_switch, iou_recover, motion_iou, dup)"
            )

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
        input_path = self._persist_source_in_run(input_path, out_dir, run_id)
        if on_debug_log:
            on_debug_log(f"Source persisted for encode: {Path(input_path).name}")
        json_path = out_dir / f"{run_id}_result.json"
        video_path = out_dir / f"{run_id}_annotated.mp4"
        report_path = out_dir / f"{run_id}_report.txt"
        metrics_path = out_dir / f"{run_id}_metrics.jsonl"
        gpu_samples_path = out_dir / f"{run_id}_gpu_samples.json"
        summary_path = out_dir / f"{run_id}_run_summary.json"

        gpu_monitor = GpuMonitor(interval_sec=0.5)
        gpu_monitor.start()
        process_res_mon = ProcessResourceMonitor(interval_sec=0.5)
        process_res_mon.start()
        self._gpu_monitor = gpu_monitor
        self._process_res_mon = process_res_mon
        if on_debug_log:
            if gpu_monitor.available:
                on_debug_log("GPU monitor: NVML (util + whole-GPU VRAM for reference)")
            else:
                on_debug_log("GPU monitor: NVML unavailable (install nvidia-ml-py)")
            on_debug_log("Process monitor: this PID RSS + torch.cuda allocated/reserved")

        terms = prompt_terms(prompt)
        prompt_id_lookup = prompt_id_lookup_from_prompt(prompt)
        frames_payload: list[dict[str, object]] = []

        deferred_encode = self.settings.encode_mode.strip().casefold() in ("deferred", "manual")
        manual_encode = self.settings.encode_mode.strip().casefold() == "manual"
        # Keep a stable copy in run_dir for manual/deferred MP4 build (API staging dir is deleted).
        manifest_input_path = str(input_path)
        if deferred_encode or manual_encode:
            manifest_input_path = self._archive_run_source(input_path, out_dir, run_id)
        encode_parallel = self.settings.encode_mode.strip().casefold() == "parallel"
        use_async_writer = bool(
            self.settings.async_encode and encode_parallel and not deferred_encode
        )

        sync_writer = None
        async_writer: AsyncVideoWriter | None = None
        if encode_parallel and not use_async_writer:
            enc_codec, enc_params, enc_exe = build_encode_writer_args(
                codec=getattr(self.settings, "encode_codec", "auto"),
                preset=getattr(self.settings, "encode_preset", "fast"),
                crf=int(getattr(self.settings, "encode_crf", 23)),
            )
            if enc_exe:
                import os

                os.environ["IMAGEIO_FFMPEG_EXE"] = str(enc_exe)
            sync_writer = imageio.get_writer(
                str(video_path),
                fps=fps,
                codec=enc_codec,
                ffmpeg_params=enc_params,
            )
        elif use_async_writer:
            enc_codec, enc_params, enc_exe = build_encode_writer_args(
                codec=getattr(self.settings, "encode_codec", "auto"),
                preset=getattr(self.settings, "encode_preset", "fast"),
                crf=int(getattr(self.settings, "encode_crf", 23)),
            )
            async_writer = AsyncVideoWriter(
                str(video_path),
                fps,
                codec=enc_codec,
                ffmpeg_params=enc_params,
                ffmpeg_exe=enc_exe,
            )

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
        infer_done_ref = [0]
        elapsed_preload_sec = 0.0
        elapsed_pass2_sec = 0.0
        elapsed_finalize_sec = 0.0
        finalize_start_ts: float | None = None
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
        capped_batch = cap_job_batch_long_video(
            self._job_batch_size,
            n_process,
            self.settings,
            streaming=False,
            source_frame_count=source_frame_count,
        )
        if capped_batch != self._job_batch_size and on_debug_log:
            on_debug_log(
                f"Job batch {self._job_batch_size}→{capped_batch} "
                f"(длинное видео, max {self.settings.max_job_batch_size or 200})"
            )
        self._job_batch_size = capped_batch
        # Do NOT clamp job batch to settings.tensorrt_max_batch (engine *build* profile,
        # often 32). UI infer_batch_size (e.g. 64) is the job size; YOLO/TRT already
        # chunks forwards to the loaded engine's max batch (e.g. b8).

        baseline_rss = current_process_rss_gb(collect_gc=True)
        ram_plan: RamBudgetPlan | None = resolve_smart_ram_plan(
            self.settings,
            width=width,
            height=height,
            job_batch=self._job_batch_size,
            use_reid=self.settings.use_reid,
            baseline_rss_gb=baseline_rss,
        )
        preload_cap: float | None = None
        window_ram_gb: float | None = None
        windows_in_ram_eff: int | None = None
        if ram_plan is not None:
            preload_cap = ram_plan.max_preload_ram_gb
            window_ram_gb = ram_plan.max_window_ram_gb
            windows_in_ram_eff = ram_plan.windows_in_ram
            if on_debug_log:
                for line in ram_plan.summary_lines():
                    on_debug_log(line)

        frame_source_requested = str(
            getattr(self.settings, "frame_source_mode", "auto") or "auto"
        ).casefold().strip()
        frame_source_mode, frame_source_reason = resolve_frame_source_ex(
            self.settings,
            source_frame_count=source_frame_count,
            width=width,
            height=height,
            max_preload_ram_gb=preload_cap,
        )
        use_windowed = frame_source_mode == "windowed"
        use_streaming = frame_source_mode == "stream"
        if on_debug_log:
            on_debug_log(
                f"Frame source resolve: requested={frame_source_requested!r} → "
                f"{frame_source_mode} ({frame_source_reason})"
            )
            if (
                frame_source_requested == "preload"
                and frame_source_mode != "preload"
            ):
                on_debug_log(
                    "WARNING: explicit preload was demoted before decode — "
                    "this should not happen (bug)"
                )
        effective_window_frames = int(self.settings.window_frames)
        infer_per_window = 0
        window_align_info: dict[str, int | float | bool] = {}
        if use_windowed:
            effective_window_frames, window_align_info = resolve_window_frames(
                self.settings,
                job_batch=self._job_batch_size,
                frame_stride=frame_stride,
                width=width,
                height=height,
                max_window_ram_gb=window_ram_gb,
                windows_in_ram=windows_in_ram_eff,
            )
            infer_per_window = int(window_align_info.get("infer_per_window", 0))
            if ram_plan is not None and on_debug_log and width > 0 and height > 0:
                win_mb = estimate_video_ram_gb(width, height, infer_per_window) * 1024
                total_win_mb = win_mb * max(1, windows_in_ram_eff or 1)
                on_debug_log(
                    f"Smart RAM window: infer {infer_per_window} fr "
                    f"({win_mb:.0f} MB×{windows_in_ram_eff or 1} ≈ {total_win_mb:.0f} MB decode)"
                )
        if on_debug_log and use_windowed and source_frame_count > 0:
            win_mb = estimate_video_ram_gb(width, height, infer_per_window) * 1024
            hint = int(window_align_info.get("hint", 0))
            src_span = int(window_align_info.get("source_span", effective_window_frames))
            align_note = ""
            if hint > 0 and src_span != hint:
                align_note = f" (hint src {hint}→span {src_span})"
            elif hint <= 0:
                align_note = " (auto)"
            jobs_pw = int(window_align_info.get("jobs_per_window", 0))
            ahead_log = max(
                0,
                int(windows_in_ram_eff if windows_in_ram_eff is not None else self.settings.windows_in_ram)
                - 1,
            )
            on_debug_log(
                f"Windowed decode: infer {infer_per_window} fr/окно{align_note} "
                f"= {jobs_pw}×job {self._job_batch_size}, src span {src_span}, "
                f"RAM ~{win_mb:.0f} MB, ahead={ahead_log}"
            )
        elif on_debug_log and use_streaming and source_frame_count > 0:
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
        packet_spill: PacketSpillWriter | None = None
        if use_windowed:
            packet_spill = PacketSpillWriter(out_dir, run_id)
            if on_debug_log:
                on_debug_log(
                    f"Frame source: windowed ({frame_source_mode}), "
                    f"не streaming — GPU batch из RAM-окна"
                )
        elif not use_streaming:
            try:
                if on_debug_log:
                    on_debug_log("Preloading video frames into RAM…")
                preload_total = max(1, int(total) if total > 0 else source_frame_count)
                self._pulse_infer_progress(
                    current_frame=0,
                    total=preload_total,
                    instances_peak=instances_peak,
                    start_ts=start_ts,
                    on_progress=on_progress,
                    gpu_monitor=gpu_monitor,
                    process_res_mon=process_res_mon,
                    phase="preload",
                )
                t_pre = time.perf_counter()

                def _preload_progress(loaded: int, total_src: int) -> None:
                    self._pulse_infer_progress(
                        current_frame=loaded,
                        total=max(1, total_src if total_src > 0 else preload_total),
                        instances_peak=instances_peak,
                        start_ts=start_ts,
                        on_progress=on_progress,
                        gpu_monitor=gpu_monitor,
                    process_res_mon=process_res_mon,
                        phase="preload",
                    )

                all_frames = load_all_frames(
                    cap,
                    total,
                    max_ram_gb=self.settings.max_preload_ram_gb,
                    width=width,
                    height=height,
                    on_progress=_preload_progress if on_progress is not None else None,
                    progress_every=max(16, frame_stride * 8),
                )
                source_frame_count = len(all_frames)
                process_indices = source_frame_indices(source_frame_count, frame_stride)
                n_process = len(process_indices)
                elapsed_preload_sec = time.perf_counter() - t_pre
                if on_debug_log:
                    on_debug_log(
                        f"Preloaded {source_frame_count} frames in {elapsed_preload_sec:.2f}s"
                    )
                # Clear preload total so clients don't treat 439/439 as infer done.
                self._pulse_infer_progress(
                    current_frame=0,
                    total=max(1, n_process),
                    instances_peak=instances_peak,
                    start_ts=start_ts,
                    on_progress=on_progress,
                    gpu_monitor=gpu_monitor,
                    process_res_mon=process_res_mon,
                    phase="start",
                )
            except MemoryError as exc:
                if on_debug_log:
                    on_debug_log(
                        "WARNING: OOM during explicit/auto preload — falling back to "
                        f"stream decode (requested={frame_source_requested!r}): {exc}"
                    )
                all_frames = None
                use_streaming = True
                frame_source_mode = "stream"
                elapsed_preload_sec = 0.0
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
            infer_done_ref=infer_done_ref,
            start_ts=start_ts,
            on_progress=on_progress,
            on_preview=on_preview,
            on_debug_log=on_debug_log,
            gpu_monitor=gpu_monitor,
            process_res_mon=process_res_mon,
        )

        def drain_post() -> None:
            if post_exec is None:
                return
            emit_kwargs["instances_peak"] = instances_peak
            flushed = post_exec.drain()
            if flushed:
                self._emit_post_results(flushed, **emit_kwargs)

        infer_start_ts = time.perf_counter()
        emit_kwargs["infer_start_ts"] = infer_start_ts
        infer_end_ts: float | None = None
        spill_manifest_path: Path | None = None
        try:
            n_frames_total = source_frame_count

            if use_windowed:
                self._pulse_infer_progress(
                    current_frame=0,
                    total=n_process,
                    instances_peak=instances_peak,
                    start_ts=start_ts,
                    on_progress=on_progress,
                    gpu_monitor=gpu_monitor,
                    process_res_mon=process_res_mon,
                    phase="start",
                    fps_start_ts=infer_start_ts,
                )
                assert packet_spill is not None
                instances_peak, frame_idx = self._drive_windowed_pipeline(
                    input_path=input_path,
                    source_frame_count=source_frame_count,
                    frame_stride=frame_stride,
                    terms=terms,
                    obj_terms=obj_terms,
                    height=height,
                    width=width,
                    deferred_encode=deferred_encode,
                    deferred_packets=deferred_packets,
                    spill_writer=packet_spill,
                    prompt_id_lookup=prompt_id_lookup,
                    post_exec=post_exec,
                    drain_post=drain_post,
                    emit_kwargs=emit_kwargs,
                    frame_idx_ref=frame_idx_ref,
                    instances_peak=instances_peak,
                    infer_ms_by_frame=infer_ms_by_frame,
                    packet_meta=packet_meta,
                    infer_done_ref=infer_done_ref,
                    should_stop=should_stop,
                    should_pause=should_pause,
                    on_debug_log=on_debug_log,
                    n_process=n_process,
                    use_gpu_pipe=use_gpu_pipe,
                    model_runner=model_runner,
                    manual_encode=manual_encode,
                    frames_payload=frames_payload,
                    infer_per_window=infer_per_window,
                    windows_in_ram=windows_in_ram_eff,
                )
            else:
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
                    process_res_mon=process_res_mon,
                    phase="start",
                    fps_start_ts=infer_start_ts,
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
                        infer_done_ref=infer_done_ref,
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
                        infer_done_ref[0] += 1
                        if not deferred_encode:
                            packet_meta[frame_idx] = packet

                        if deferred_encode:
                            progress_total = n_process if n_process > 0 else infer_done_ref[0]
                            self._report_infer_progress(
                                packet=packet,
                                total=progress_total,
                                instances_peak=instances_peak,
                                start_ts=start_ts,
                                infer_ms_by_frame=infer_ms_by_frame,
                                metrics_path=metrics_path,
                                on_progress=on_progress,
                                gpu_monitor=gpu_monitor,
                    process_res_mon=process_res_mon,
                                phase="inference",
                                write_metrics=False,
                                current_frame=infer_done_ref[0],
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
                if packet_spill is not None and deferred_packets:
                    tail_start = int(deferred_packets[0].frame_idx)
                    tail_end = int(deferred_packets[-1].frame_idx) + 1
                    if on_debug_log:
                        on_debug_log(
                            f"Tail spill: {len(deferred_packets)} pkts "
                            f"(frames {tail_start}..{tail_end})…"
                        )
                    packet_spill.write_chunk(
                        deferred_packets,
                        start_frame=tail_start,
                        end_frame=tail_end,
                    )
                    deferred_packets.clear()
                if packet_spill is not None and packet_spill.chunk_count > 0:
                    spill_manifest_path = packet_spill.finalize(
                        fps=fps,
                        input_path=manifest_input_path,
                        prompt=prompt,
                        overlay=self._overlay_snapshot(),
                        width=width,
                        height=height,
                        frame_stride=frame_stride,
                        source_frame_count=source_frame_count,
                        window_frames=int(window_align_info.get("source_span", effective_window_frames)),
                    )
                # End-of-YOLO timestamp before Pass 2 (OSNet link must not inflate infer FPS).
                infer_end_ts = time.perf_counter()
                t_pass2 = time.perf_counter()
                self._apply_offline_tracklet_link(
                    deferred_packets=deferred_packets,
                    frames_payload=frames_payload,
                    spill_manifest_path=spill_manifest_path,
                    out_dir=out_dir,
                    input_path=str(input_path),
                    on_debug_log=on_debug_log,
                )
                elapsed_pass2_sec += time.perf_counter() - t_pass2
                finalize_start_ts = time.perf_counter()
                if manual_encode:
                    if on_debug_log:
                        on_debug_log(
                            f"Inference done ({frame_idx} frames). "
                            f"JSON + кэш кадров (MP4 только по кнопке)…"
                        )
                    if spill_manifest_path is not None:
                        if on_debug_log:
                            on_debug_log(
                                f"Packets spill: {packet_spill.chunk_count} chunks, "
                                f"{packet_spill.total_packets} keyframes → "
                                f"{spill_manifest_path.name}"
                            )
                    elif deferred_packets:
                        pkl_path = packets_path_for_run(out_dir, run_id)
                        save_run_packets(
                            pkl_path,
                            run_id=run_id,
                            packets=deferred_packets,
                            fps=fps,
                            input_path=manifest_input_path,
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
                    from app.core.video_encode import encode_manifest_to_video, encode_packets_to_video

                    if spill_manifest_path is not None:
                        from app.core.packet_spill import load_packets_manifest

                        manifest = load_packets_manifest(spill_manifest_path)
                        encode_manifest_to_video(
                            manifest,
                            run_dir=out_dir,
                            video_path=video_path,
                        )
                    else:
                        encode_packets_to_video(
                            deferred_packets,
                            video_path=video_path,
                            fps=fps,
                            prompt=prompt,
                            overlay=self._overlay_snapshot(),
                            input_path=manifest_input_path,
                            width=width,
                            height=height,
                            frame_stride=frame_stride,
                            source_frame_count=source_frame_count,
                        )
                    if sink is None and video_path.exists():
                        if on_debug_log:
                            on_debug_log(f"Video encoded (streaming): {video_path.name}")
            elif post_exec is not None:
                emit_kwargs["instances_peak"] = instances_peak
                remaining = post_exec.finish()
                self._emit_post_results(remaining, **emit_kwargs)
        finally:
            proc_mon = getattr(self, "_process_res_mon", None)
            if proc_mon is not None:
                proc_mon.stop()
            self._process_res_mon = None
            self._gpu_monitor = None
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
            # Drop hot-path callbacks so a later API silent job cannot inherit UI ticks.
            self._on_progress_tick = None
            self._progress_slot = None
            self._on_debug_log = None
            # Skip empty_cache between jobs: synchronize+empty_cache regresses fps_infer
            # across API repeats (warmup→main→repeat). Models stay loaded.

        elapsed_total = time.perf_counter() - start_ts
        if infer_end_ts is None:
            infer_end_ts = time.perf_counter()
        elapsed_infer = max(0.0, infer_end_ts - infer_start_ts)
        if source_frame_count <= 0:
            source_frame_count = max(
                int(frame_idx_ref[0]),
                int(frame_idx),
                len(deferred_packets),
            )
        if n_process <= 0 and source_frame_count > 0:
            n_process = len(deferred_packets) or int(frame_idx)
        # Infer frames actually run (stride-aware). Do NOT use frame_idx (source span).
        n_frames = max(
            1,
            int(infer_done_ref[0]),
            len(deferred_packets),
            int(n_process) if int(n_process) > 0 else 0,
        )
        stride_info = frame_stride_summary(source_frame_count, frame_stride, n_frames)
        video_duration_sec = source_frame_count / fps if fps > 0 else 0.0
        if on_debug_log and video_duration_sec > 0:
            speed_ratio = elapsed_infer / video_duration_sec if video_duration_sec > 0 else 0.0
            on_debug_log(
                f"Speed: inference {elapsed_infer:.1f}s / video {video_duration_sec:.1f}s = {speed_ratio:.2f}× "
                f"(total wall {elapsed_total:.1f}s"
                f"{f', preload {elapsed_preload_sec:.1f}s' if elapsed_preload_sec > 0.05 else ''}"
                f"{f', Pass2 {elapsed_pass2_sec:.1f}s' if elapsed_pass2_sec > 0.05 else ''}) "
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
        stats_summary["elapsed_preload_sec"] = round(elapsed_preload_sec, 3)
        stats_summary["elapsed_infer_sec"] = round(elapsed_infer, 3)
        stats_summary["elapsed_pass2_sec"] = round(elapsed_pass2_sec, 3)
        stats_summary["fps_infer"] = round(
            (n_frames / elapsed_infer) if elapsed_infer > 0 else 0.0, 3
        )

        # Parallel / streaming encode: remap JSON IDs (MP4 may keep Pass-1 labels).
        if (
            not deferred_encode
            and bool(getattr(self.settings, "use_offline_tracklet_link", False))
            and frames_payload
        ):
            t_pass2 = time.perf_counter()
            self._apply_offline_tracklet_link(
                deferred_packets=deferred_packets,
                frames_payload=frames_payload,
                spill_manifest_path=spill_manifest_path,
                out_dir=out_dir,
                input_path=str(input_path),
                on_debug_log=on_debug_log,
            )
            elapsed_pass2_sec += time.perf_counter() - t_pass2
            stats_summary["elapsed_pass2_sec"] = round(elapsed_pass2_sec, 3)

        if manual_encode and deferred_packets and not frames_payload:
            if on_debug_log:
                on_debug_log(f"Building result JSON ({len(deferred_packets)} frames)…")
            frames_payload = self._frames_payload_from_packets(
                deferred_packets,
                prompt_id_lookup,
            )

        if finalize_start_ts is None:
            finalize_start_ts = time.perf_counter()

        payload = build_result_payload(
            schema="yolo_drt_video_data_v1",
            input_path=manifest_input_path,
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
        if spill_manifest_path is not None and not frames_payload:
            from app.core.packet_spill import load_packets_manifest

            manifest = load_packets_manifest(spill_manifest_path)
            total_pkts = int(manifest.get("total_packets") or 0)
            payload["packets_manifest"] = spill_manifest_path.name
            payload["frames"] = []
            payload["frames_in_spill"] = True
            payload["total_packets"] = total_pkts
            if on_debug_log:
                on_debug_log(
                    f"Writing result JSON (metadata, {total_pkts} frames in spill chunks)…"
                )
            write_result_json(json_path, payload)
            if on_debug_log:
                on_debug_log(f"Result JSON saved: {json_path.name}")
        else:
            if on_debug_log and frames_payload:
                on_debug_log(f"Writing result JSON ({len(frames_payload)} frames)…")
            write_result_json(json_path, payload)
            if on_debug_log and frames_payload:
                on_debug_log(f"Result JSON saved: {json_path.name}")

        memory_info = build_memory_report(
            width=width,
            height=height,
            effective_batch_size=self._job_batch_size,
            frames_total=source_frame_count,
            preloaded=all_frames is not None,
        )
        pkl_path = packets_path_for_run(out_dir, run_id)
        from app.core.packet_spill import find_packets_manifest, manifest_path_for_run

        spill_manifest = find_packets_manifest(out_dir, run_id)
        has_video = video_path.exists()
        artifacts = {
            "json": str(json_path.name),
            "video": str(video_path.name) if has_video else None,
            "packets": (
                str(spill_manifest.name)
                if spill_manifest is not None
                else (str(pkl_path.name) if pkl_path.exists() else None)
            ),
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
                f"Frame source: {frame_source_mode} | "
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
        elif use_windowed:
            win_mb = estimate_video_ram_gb(width, height, infer_per_window) * 1024
            src_span = int(window_align_info.get("source_span", effective_window_frames))
            report_lines.append(
                f"Windowed RAM (infer {infer_per_window} fr, src span {src_span}): "
                f"{win_mb:.0f} MB, ahead={max(0, int(self.settings.windows_in_ram) - 1)}"
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
        elif spill_manifest is not None:
            report_lines.append(f"Packets manifest: {spill_manifest}")

        gpu_samples = gpu_monitor.samples_to_dicts()
        if gpu_samples:
            gpu_samples_path.write_text(
                json.dumps(gpu_samples, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        gpu_stats = gpu_monitor.summary()
        process_memory_stats = process_res_mon.summary()
        proc_samples_path = out_dir / f"{run_id}_process_memory_samples.json"
        proc_samples = process_res_mon.samples_to_dicts()
        if proc_samples:
            proc_samples_path.write_text(
                json.dumps(proc_samples, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if process_memory_stats:
            stats_summary["process_memory"] = process_memory_stats
            report_lines.append(
                "Process RAM (this app): "
                f"peak {process_memory_stats['process_rss_peak_mb']:.0f} MB "
                f"(+{process_memory_stats['process_rss_delta_peak_mb']:.0f} vs job start)"
            )
            report_lines.append(
                "CUDA (this process, torch): "
                f"alloc peak {process_memory_stats['cuda_allocated_peak_mb']:.0f} MB, "
                f"reserved peak {process_memory_stats['cuda_reserved_peak_mb']:.0f} MB"
            )
        if gpu_stats:
            peak_dev = float(
                gpu_stats.get("peak_gpu_device_used_mb")
                or gpu_stats.get("peak_mem_used_mb")
                or 0.0
            )
            report_lines.append(
                f"GPU VRAM total on device (NVML, all processes): peak {peak_dev:.0f} MB"
            )
            if process_memory_stats:
                gpu_stats = {**gpu_stats, "process_memory": process_memory_stats}
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        if on_debug_log and process_memory_stats:
            on_debug_log(
                "Memory peaks (this process): "
                f"RAM {process_memory_stats['process_rss_peak_mb']:.0f} MB, "
                f"CUDA alloc {process_memory_stats['cuda_allocated_peak_mb']:.0f} MB "
                f"(device total NVML peak "
                f"{float((gpu_stats or {}).get('peak_gpu_device_used_mb') or 0):.0f} MB)"
            )
        metrics_rows = read_metrics_jsonl(metrics_path)
        elapsed_finalize_sec = max(
            0.0,
            time.perf_counter() - (finalize_start_ts or time.perf_counter()),
        )
        wall_now = time.perf_counter() - start_ts
        accounted = elapsed_preload_sec + elapsed_infer + elapsed_pass2_sec
        if elapsed_finalize_sec < 0.01:
            elapsed_finalize_sec = max(0.0, wall_now - accounted)
        stats_summary["elapsed_preload_sec"] = round(elapsed_preload_sec, 3)
        stats_summary["elapsed_infer_sec"] = round(elapsed_infer, 3)
        stats_summary["elapsed_pass2_sec"] = round(elapsed_pass2_sec, 3)
        stats_summary["elapsed_finalize_sec"] = round(elapsed_finalize_sec, 3)
        stats_summary["video_duration_sec"] = round(video_duration_sec, 3)
        # Recompute with n_frames (processed), not source span — same as early assign.
        stats_summary["fps_infer"] = round(
            (n_frames / elapsed_infer) if elapsed_infer > 0 else 0.0, 3
        )
        # Pure stage timers (may overlap; sum can exceed elapsed_infer).
        stats_summary["stage_decode_sec"] = round(
            float(self._stage_timing.get("decode_sec", 0.0)), 3
        )
        stats_summary["stage_gpu_infer_sec"] = round(
            float(self._stage_timing.get("gpu_infer_sec", 0.0)), 3
        )
        stats_summary["stage_cpu_finalize_sec"] = round(
            float(self._stage_timing.get("cpu_finalize_sec", 0.0)), 3
        )
        stats_summary["stage_pass2_sec"] = round(elapsed_pass2_sec, 3)
        if on_debug_log:
            on_debug_log(
                "Stages: "
                f"decode={stats_summary['stage_decode_sec']:.3f}s "
                f"gpu_wall={stats_summary['stage_gpu_infer_sec']:.3f}s "
                f"cpu_finalize={stats_summary['stage_cpu_finalize_sec']:.3f}s "
                f"pass2={stats_summary['stage_pass2_sec']:.3f}s "
                f"| infer_wall={elapsed_infer:.3f}s"
            )
        payload["stats_summary"] = stats_summary
        json_path.write_text(video_data_json_dumps(payload), encoding="utf-8")
        if on_debug_log:
            on_debug_log("Building run charts…")
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
                if (
                    self.settings.use_reid
                    or bool(getattr(self.settings, "use_sam_identity", False))
                )
                else ("predict_batch" if self.settings.gpu_full_batch else "track_or_batch")
            ),
            "gpu_queue_depth": self.settings.gpu_queue_depth,
            "batch_prefetch_depth": resolve_batch_prefetch_depth(self.settings),
            "frame_source_mode": frame_source_mode,
            "frame_source_requested": frame_source_requested,
            "frame_source_reason": frame_source_reason,
            "windowed_decode": use_windowed,
            "window_frames": int(window_align_info.get("source_span", effective_window_frames))
            if use_windowed
            else 0,
            "window_infer_frames": int(infer_per_window) if use_windowed else 0,
            "window_jobs_per_window": int(window_align_info.get("jobs_per_window", 0))
            if use_windowed
            else 0,
            "streaming_decode": all_frames is None and not use_windowed,
            "use_batch_detect": self.settings.use_batch_detect,
            "use_seg": self.settings.use_seg,
            "use_reid": self.settings.use_reid,
            "use_sam_identity": bool(getattr(self.settings, "use_sam_identity", False)),
            "sam_identity_backend": str(
                getattr(self.settings, "sam_identity_backend", "memory")
            ),
            "use_offline_tracklet_link": bool(
                getattr(self.settings, "use_offline_tracklet_link", False)
            ),
            "tracklet_link_max_gap_frames": int(
                getattr(self.settings, "tracklet_link_max_gap_frames", 300)
            ),
            "tracklet_link_min_sim": float(
                getattr(self.settings, "tracklet_link_min_sim", 0.60)
            ),
            "cross_check_enabled": self.settings.cross_check_enabled,
            "encode_mode": self.settings.encode_mode,
            "parallel_post": self.settings.parallel_post,
            "inference_device": getattr(self.settings, "inference_device", "cuda"),
            "frame_stride": frame_stride,
            "source_frame_count": source_frame_count,
            "processed_frame_count": stride_info["processed_frame_count"],
            "elapsed_preload_sec": round(elapsed_preload_sec, 3),
            "elapsed_infer_sec": round(elapsed_infer, 3),
            "elapsed_pass2_sec": round(elapsed_pass2_sec, 3),
            "elapsed_finalize_sec": round(elapsed_finalize_sec, 3),
            "video_duration_sec": round(video_duration_sec, 3),
            "stage_decode_sec": stats_summary["stage_decode_sec"],
            "stage_gpu_infer_sec": stats_summary["stage_gpu_infer_sec"],
            "stage_cpu_finalize_sec": stats_summary["stage_cpu_finalize_sec"],
            "stage_pass2_sec": stats_summary["stage_pass2_sec"],
        }
        if ram_plan is not None:
            pipeline_info["smart_ram_baseline_gb"] = round(ram_plan.baseline_rss_gb, 2)
            pipeline_info["smart_ram_budget_gb"] = ram_plan.budget_gb
            pipeline_info["smart_ram_peak_gb"] = round(ram_plan.estimated_peak_gb, 2)
            pipeline_info["smart_ram_window_gb"] = round(ram_plan.max_window_ram_gb, 2)
            pipeline_info["windows_in_ram_effective"] = ram_plan.windows_in_ram
        record = build_run_record(
            run_id=run_id,
            out_dir=out_dir,
            input_path=manifest_input_path,
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
        if on_debug_log:
            on_debug_log("Run complete.")

        if has_video:
            self._mux_audio_if_possible(manifest_input_path, video_path)
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
    def _archive_run_source(src_path: str, out_dir: Path, run_id: str) -> str:
        """Copy source video into run folder so deferred/manual encode survives API staging cleanup."""
        src = Path(src_path)
        if not src.is_file():
            return str(src_path)
        suffix = src.suffix if src.suffix else ".mp4"
        dest = out_dir / f"{run_id}_source{suffix}"
        try:
            if dest.is_file() and dest.stat().st_size == src.stat().st_size:
                return str(dest.resolve())
            shutil.copy2(src, dest)
            return str(dest.resolve())
        except OSError:
            return str(src_path)

    @staticmethod
    def _mux_audio_if_possible(src_video: str, video_only: Path) -> None:
        from app.core.ffmpeg_utils import mux_audio_if_possible

        mux_audio_if_possible(src_video, video_only)
