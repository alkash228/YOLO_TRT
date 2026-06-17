"""Persistent run index, CSV export, and research charts."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INDEX_VERSION = 1
INDEX_FILENAME = "runs_index.json"
CSV_FILENAME = "runs_history.csv"


def index_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / INDEX_FILENAME


def load_runs_index(output_dir: str | Path) -> dict[str, Any]:
    path = index_path(output_dir)
    if not path.exists():
        return {"version": INDEX_VERSION, "runs": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"version": INDEX_VERSION, "runs": []}


def save_runs_index(output_dir: str | Path, data: dict[str, Any]) -> Path:
    path = index_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def append_run_record(output_dir: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    data = load_runs_index(output_dir)
    runs: list[dict[str, Any]] = data["runs"]
    rid = str(record.get("run_id", ""))
    runs = [r for r in runs if str(r.get("run_id")) != rid]
    runs.insert(0, record)
    data["runs"] = runs[:500]
    save_runs_index(output_dir, data)
    export_runs_csv(output_dir, runs)
    return record


def export_runs_csv(output_dir: str | Path, runs: list[dict[str, Any]] | None = None) -> Path:
    if runs is None:
        runs = load_runs_index(output_dir).get("runs", [])
    path = Path(output_dir) / CSV_FILENAME
    fields = [
        "run_id",
        "timestamp",
        "input_video",
        "prompt",
        "frames",
        "elapsed_sec",
        "fps_processed",
        "resolution",
        "detect_model",
        "seg_model",
        "reid_model",
        "cross_check_model",
        "avg_detect_ms",
        "avg_seg_ms",
        "avg_reid_ms",
        "avg_total_ms",
        "avg_gpu_util_pct",
        "peak_gpu_util_pct",
        "instances_peak",
        "id_switches",
        "reid_recoveries",
        "gpu_pipeline",
        "infer_batch_size",
        "out_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            models = r.get("models") or {}
            stats = r.get("stats_summary") or {}
            gpu = r.get("gpu_stats") or {}
            pipe = r.get("pipeline") or {}
            w.writerow(
                {
                    "run_id": r.get("run_id"),
                    "timestamp": r.get("timestamp"),
                    "input_video": r.get("input_video"),
                    "prompt": r.get("prompt"),
                    "frames": r.get("frames"),
                    "elapsed_sec": r.get("elapsed_sec"),
                    "fps_processed": r.get("fps_processed"),
                    "resolution": r.get("resolution"),
                    "detect_model": models.get("detect"),
                    "seg_model": models.get("seg"),
                    "reid_model": models.get("reid"),
                    "cross_check_model": models.get("cross_check"),
                    "avg_detect_ms": stats.get("avg_detect_ms"),
                    "avg_seg_ms": stats.get("avg_seg_ms"),
                    "avg_reid_ms": stats.get("avg_reid_ms"),
                    "avg_total_ms": stats.get("avg_total_ms"),
                    "avg_gpu_util_pct": gpu.get("avg_gpu_util_pct"),
                    "peak_gpu_util_pct": gpu.get("peak_gpu_util_pct"),
                    "instances_peak": stats.get("instances_peak"),
                    "id_switches": stats.get("id_switches"),
                    "reid_recoveries": stats.get("reid_recoveries"),
                    "gpu_pipeline": pipe.get("gpu_pipeline"),
                    "infer_batch_size": pipe.get("infer_batch_size"),
                    "out_dir": r.get("out_dir"),
                }
            )
    return path


def build_run_record(
    *,
    run_id: str,
    out_dir: Path,
    input_path: str,
    prompt: str,
    frames: int,
    elapsed_sec: float,
    width: int,
    height: int,
    video_fps: float,
    models: dict[str, str],
    stats_summary: dict[str, Any],
    pipeline: dict[str, Any],
    gpu_stats: dict[str, Any] | None,
    chart_paths: dict[str, str],
    memory: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    source_frames: int | None = None,
) -> dict[str, Any]:
    fps_proc = frames / elapsed_sec if elapsed_sec > 0 else 0.0
    rec: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir.resolve()),
        "input_video": str(input_path),
        "prompt": prompt,
        "frames": int(frames),
        "source_frames": int(source_frames) if source_frames is not None else int(frames),
        "elapsed_sec": round(float(elapsed_sec), 3),
        "fps_processed": round(fps_proc, 3),
        "resolution": f"{width}x{height}",
        "video_fps": round(float(video_fps), 3),
        "models": dict(models),
        "stats_summary": dict(stats_summary),
        "pipeline": dict(pipeline),
        "gpu_stats": gpu_stats,
        "charts": dict(chart_paths),
    }
    if memory:
        rec["memory"] = dict(memory)
    if artifacts:
        rec["artifacts"] = dict(artifacts)
    return rec


def generate_run_charts(
    out_dir: Path,
    run_id: str,
    *,
    gpu_samples: list[dict[str, float]] | None,
    metrics_rows: list[dict[str, Any]] | None,
    stats_summary: dict[str, Any] | None,
) -> dict[str, str]:
    """Save PNG charts into run folder; return relative paths."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    style = {"figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d28", "axes.edgecolor": "#4a5068"}
    plt.rcParams.update(
        {
            **style,
            "text.color": "#e8ebf5",
            "axes.labelcolor": "#e8ebf5",
            "xtick.color": "#b8bdd0",
            "ytick.color": "#b8bdd0",
            "grid.color": "#2f3548",
        }
    )

    if gpu_samples:
        fig, ax1 = plt.subplots(figsize=(10, 3.5), dpi=120)
        ts = [s["t_sec"] for s in gpu_samples]
        util = [s["gpu_util_pct"] for s in gpu_samples]
        mem = [s["mem_used_mb"] for s in gpu_samples]
        ax1.plot(ts, util, color="#4c7cf3", linewidth=1.5, label="GPU %")
        ax1.set_ylabel("GPU util %")
        ax1.set_xlabel("Time (s)")
        ax1.grid(True, alpha=0.35)
        ax2 = ax1.twinx()
        ax2.plot(ts, mem, color="#f5a623", linewidth=1.2, alpha=0.85, label="VRAM MB")
        ax2.set_ylabel("VRAM MB")
        ax1.set_title(f"GPU load — {run_id}")
        fig.tight_layout()
        p = out_dir / f"{run_id}_chart_gpu.png"
        fig.savefig(p, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths["gpu"] = p.name

    if metrics_rows:
        frames = [int(r.get("frame", i + 1)) for i, r in enumerate(metrics_rows)]
        total_ms = [float(r.get("total_ms", 0)) for r in metrics_rows]
        detect_ms = [float(r.get("detect_ms", 0)) for r in metrics_rows]
        seg_ms = [float(r.get("seg_ms", 0)) for r in metrics_rows]
        reid_ms = [float(r.get("reid_ms", 0)) for r in metrics_rows]

        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)
        ax.plot(frames, total_ms, color="#e8ebf5", linewidth=1.0, alpha=0.9, label="total")
        ax.plot(frames, detect_ms, color="#4c7cf3", linewidth=0.8, alpha=0.8, label="detect")
        ax.plot(frames, seg_ms, color="#50c878", linewidth=0.8, alpha=0.8, label="seg")
        ax.plot(frames, reid_ms, color="#f5a623", linewidth=0.8, alpha=0.8, label="reid")
        ax.set_xlabel("Frame")
        ax.set_ylabel("ms")
        ax.set_title(f"Per-frame latency — {run_id}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        p = out_dir / f"{run_id}_chart_frame_ms.png"
        fig.savefig(p, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths["frame_ms"] = p.name

    if stats_summary:
        labels = []
        vals = []
        for key, label in (
            ("avg_detect_ms", "detect"),
            ("avg_seg_ms", "seg"),
            ("avg_reid_ms", "reid"),
        ):
            v = stats_summary.get(key)
            if v is not None and float(v) > 0.01:
                labels.append(label)
                vals.append(float(v))
        if labels:
            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
            colors = ["#4c7cf3", "#50c878", "#f5a623", "#c77dff"][: len(vals)]
            ax.bar(labels, vals, color=colors)
            ax.set_ylabel("avg ms / frame")
            ax.set_title(f"Stage breakdown — {run_id}")
            ax.grid(True, axis="y", alpha=0.35)
            fig.tight_layout()
            p = out_dir / f"{run_id}_chart_timing.png"
            fig.savefig(p, facecolor=fig.get_facecolor())
            plt.close(fig)
            paths["timing"] = p.name

    return paths


def read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
