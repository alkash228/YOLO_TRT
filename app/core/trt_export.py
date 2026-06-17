"""Build TensorRT engines for YOLO (Ultralytics) and OSNet ReID."""
from __future__ import annotations

import gc
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import PipelineSettings, TRT_DIR
from app.core.trt_paths import (
    TrtEngineRecord,
    instructions_path as trt_instructions_path,
    reid_engine_name,
    reid_onnx_name,
    resolve_reid_engine,
    resolve_reid_onnx,
    resolve_yolo_engine,
    save_manifest,
    yolo_engine_name,
)

LogFn = Callable[[str], None]


@dataclass(slots=True)
class TrtBuildResult:
    role: str
    source: Path
    engine: Path | None
    ok: bool
    message: str


def _log(fn: LogFn | None, msg: str) -> None:
    if fn is not None:
        fn(msg)


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1 << 20) if path.exists() else 0.0


def _tensorrt_version_major() -> int:
    import tensorrt as trt

    return int(trt.__version__.split(".", 1)[0])


def _supports_weak_fp16_flag() -> bool:
    import tensorrt as trt

    return hasattr(trt.BuilderFlag, "FP16")


def _release_cuda(log: LogFn | None = None) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        gc.collect()
        _log(log, "CUDA cache очищен")
    except Exception:
        pass


def _onnx_fixed_batch_hw(onnx_path: Path) -> tuple[int, int, int]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if len(dims) >= 4:
            b = dims[0].dim_value or 1
            h = dims[2].dim_value or 640
            w = dims[3].dim_value or 640
            return int(b), int(h), int(w)
    return 1, 640, 640


def _autocast_providers(onnx_path: Path, log: LogFn | None) -> list[str]:
    """ORT reference-run: yolo26x batch>=8 на CUDA часто OOM после YOLO export."""
    batch, h, w = _onnx_fixed_batch_hw(onnx_path)
    try:
        import onnxruntime as ort

        if batch >= 8 and max(h, w) >= 512:
            _log(
                log,
                f"ModelOpt reference-run: CPU (batch={batch}, {h}x{w}; CUDA часто OOM на 12GB)",
            )
            return ["cpu"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            _log(log, "ModelOpt reference-run: GPU (CUDA EP)")
            return ["cuda:0"]
    except Exception as exc:
        _log(log, f"ModelOpt: CUDA EP недоступен ({exc})")
    _log(log, "ModelOpt reference-run: CPU")
    return ["cpu"]


def _modelopt_to_fp16(
    onnx_path: Path,
    *,
    autocast_fast: bool,
    log: LogFn | None,
):
    import onnx
    import modelopt.onnx.autocast as autocast
    from modelopt.onnx.autocast.convert import convert_to_f16

    if autocast_fast:
        _log(log, "ModelOpt fast (без reference-run)")
        model = onnx.load(str(onnx_path), load_external_data=True)
        return convert_to_f16(model, low_precision_type="fp16", keep_io_types=True)

    attempts: list[tuple[str, list[str] | None]] = [
        ("accurate", _autocast_providers(onnx_path, log)),
        ("CPU", ["cpu"]),
        ("fast", None),
    ]
    seen: set[tuple[str, ...]] = set()
    last_exc: Exception | None = None
    for label, providers in attempts:
        key = tuple(providers or ())
        if key in seen:
            continue
        seen.add(key)
        try:
            if providers is None:
                _log(log, f"ModelOpt fast fallback ({last_exc})")
                model = onnx.load(str(onnx_path), load_external_data=True)
                return convert_to_f16(model, low_precision_type="fp16", keep_io_types=True)
            _log(log, f"ModelOpt AutoCast [{label}] ('Skipping node' — норма)")
            return autocast.convert_to_mixed_precision(
                onnx_path=str(onnx_path),
                low_precision_type="fp16",
                keep_io_types=True,
                providers=providers,
            )
        except Exception as exc:
            last_exc = exc
            _log(log, f"ModelOpt [{label}] ошибка: {str(exc)[:240]}")
            _release_cuda(log)
    raise RuntimeError(f"ModelOpt AutoCast failed: {last_exc}") from last_exc


def _prepare_onnx_for_build(
    onnx_path: Path,
    *,
    fp16: bool,
    autocast_fast: bool,
    log: LogFn | None,
) -> Path:
    """TRT 11+ убрал BuilderFlag.FP16 — FP16 через ModelOpt AutoCast в ONNX."""
    onnx_path = onnx_path.resolve()
    if not fp16:
        return onnx_path
    if _supports_weak_fp16_flag():
        return onnx_path

    fp16_path = onnx_path.with_name(f"{onnx_path.stem}_fp16{onnx_path.suffix}")
    if fp16_path.exists() and fp16_path.stat().st_mtime >= onnx_path.stat().st_mtime:
        _log(log, f"Кэш ModelOpt: {fp16_path.name}")
        return fp16_path

    try:
        import onnx  # noqa: F401 — проверка зависимости
        import modelopt.onnx.autocast  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT 11+ требует nvidia-modelopt[onnx] для FP16: "
            "pip install \"nvidia-modelopt[onnx]\""
        ) from exc

    _release_cuda(log)
    _log(log, f"ModelOpt -> {fp16_path.name}")
    converted = _modelopt_to_fp16(onnx_path, autocast_fast=autocast_fast, log=log)
    import onnx as onnx_mod

    onnx_mod.save(converted, str(fp16_path))
    _log(log, f"ModelOpt готов: {fp16_path.name} ({_size_mb(fp16_path):.1f} MB)")
    return fp16_path


def build_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    workspace_gb: float,
    autocast_fast: bool = False,
    profile_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] | None = None,
    log: LogFn | None = None,
) -> TrtBuildResult:
    try:
        import tensorrt as trt
    except ImportError:
        return TrtBuildResult("engine", onnx_path, None, False, "tensorrt Python package not installed")

    onnx_path = onnx_path.resolve()
    if not onnx_path.exists():
        return TrtBuildResult("engine", onnx_path, None, False, f"ONNX не найден: {onnx_path}")

    try:
        build_onnx = _prepare_onnx_for_build(
            onnx_path, fp16=fp16, autocast_fast=autocast_fast, log=log
        )
    except RuntimeError as exc:
        return TrtBuildResult("engine", onnx_path, None, False, str(exc))

    use_weak_fp16 = fp16 and _supports_weak_fp16_flag()
    prec = "FP16" if fp16 else "FP32"
    if fp16 and not use_weak_fp16:
        prec = "FP16 (ModelOpt strong-typed)"
    _log(log, f"TensorRT compile на GPU: {trt.__version__} {prec} -> {engine_path.name}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    ws = int(float(workspace_gb) * (1 << 30))
    if ws > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)

    flag = 0 if _tensorrt_version_major() >= 10 else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(build_onnx)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        return TrtBuildResult("engine", onnx_path, None, False, errs)

    if profile_shapes:
        profile = builder.create_optimization_profile()
        for name, (mn, opt, mx) in profile_shapes.items():
            profile.set_shape(name, mn, opt, mx)
        config.add_optimization_profile(profile)

    if use_weak_fp16 and getattr(builder, "platform_has_fast_fp16", True):
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return TrtBuildResult("engine", onnx_path, None, False, "build_serialized_network failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    _log(log, f"ENGINE OK: {engine_path.name} ({_size_mb(engine_path):.1f} MB)")
    return TrtBuildResult("engine", onnx_path, engine_path, True, "OK")


def find_trtexec() -> Path | None:
    found = shutil.which("trtexec")
    if found:
        return Path(found)
    candidates = [
        Path(r"C:\TensorRT\bin\trtexec.exe"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\bin\trtexec.exe"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def export_yolo_engine(
    pt_path: Path,
    out_engine: Path,
    *,
    imgsz: int,
    max_batch: int,
    fp16: bool,
    workspace_gb: float,
    autocast_fast: bool = False,
    log: LogFn | None = None,
) -> TrtBuildResult:
    role = "yolo"
    if not pt_path.exists():
        return TrtBuildResult(role, pt_path, None, False, f"Файл не найден: {pt_path}")
    out_engine.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO

        _log(log, f"YOLO шаг 1/2: ONNX export ({pt_path.name}, batch={max_batch})...")
        model = YOLO(str(pt_path))
        exported = model.export(
            format="onnx",
            imgsz=int(imgsz),
            dynamic=False,
            batch=int(max_batch),
            simplify=True,
            device=0,
            verbose=False,
        )
        onnx_path = Path(str(exported)).resolve()
        del model
        _release_cuda(log)
        _log(log, f"ONNX OK: {onnx_path.name} ({_size_mb(onnx_path):.1f} MB)")
        _log(log, f"YOLO шаг 2/2: TensorRT engine -> {out_engine.name}...")
        eng = build_engine_from_onnx(
            onnx_path,
            out_engine,
            fp16=fp16,
            workspace_gb=workspace_gb,
            autocast_fast=autocast_fast,
            log=log,
        )
        return TrtBuildResult(role, pt_path, eng.engine, eng.ok, eng.message)
    except Exception as exc:
        return TrtBuildResult(role, pt_path, None, False, str(exc))


def export_reid_onnx(
    pth_path: Path,
    onnx_path: Path,
    *,
    log: LogFn | None = None,
) -> TrtBuildResult:
    if not pth_path.exists():
        return TrtBuildResult("reid_onnx", pth_path, None, False, f"Файл не найден: {pth_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        from torchreid.utils import FeatureExtractor

        _log(log, f"ReID ONNX export -> {onnx_path.name}")
        ext = FeatureExtractor(
            model_name="osnet_ain_x1_0",
            model_path=str(pth_path),
            device="cpu",
            image_size=(256, 128),
        )
        model = ext.model.eval()
        for m in model.modules():
            if hasattr(m, "training"):
                m.training = False
        dummy = torch.randn(1, 3, 256, 128, dtype=torch.float32)
        export_kw: dict = {"dynamo": False}
        if hasattr(torch.onnx, "ExportOptions"):
            export_kw["export_params"] = True
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
            opset_version=18,
            do_constant_folding=True,
            **export_kw,
        )
        return TrtBuildResult("reid_onnx", pth_path, onnx_path, onnx_path.exists(), "OK")
    except Exception as exc:
        return TrtBuildResult("reid_onnx", pth_path, None, False, str(exc))


def build_engine_trtexec(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    workspace_gb: float,
    min_batch: int = 1,
    opt_batch: int = 16,
    max_batch: int = 32,
    autocast_fast: bool = False,
    log: LogFn | None = None,
) -> TrtBuildResult:
    trtexec = find_trtexec()
    if trtexec is None:
        return TrtBuildResult(
            "reid_engine",
            onnx_path,
            None,
            False,
            "trtexec не найден в PATH — см. TENSORRT_INSTRUCTIONS.md",
        )
    try:
        build_onnx = _prepare_onnx_for_build(
            onnx_path, fp16=fp16, autocast_fast=autocast_fast, log=log
        )
    except RuntimeError as exc:
        return TrtBuildResult("reid_engine", onnx_path, None, False, str(exc))

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(trtexec),
        f"--onnx={build_onnx}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{int(float(workspace_gb) * 1024)}",
        f"--minShapes=images:{min_batch}x3x256x128",
        f"--optShapes=images:{opt_batch}x3x256x128",
        f"--maxShapes=images:{max_batch}x3x256x128",
    ]
    if fp16 and _supports_weak_fp16_flag():
        cmd.append("--fp16")
    _log(log, "trtexec: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            return TrtBuildResult("reid_engine", onnx_path, None, False, tail)
        return TrtBuildResult("reid_engine", onnx_path, engine_path, engine_path.exists(), "OK")
    except Exception as exc:
        return TrtBuildResult("reid_engine", onnx_path, None, False, str(exc))


def build_engine_tensorrt_python(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    workspace_gb: float,
    max_batch: int,
    autocast_fast: bool = False,
    log: LogFn | None = None,
) -> TrtBuildResult:
    opt_batch = min(16, max_batch)
    profile = {
        "images": (
            (1, 3, 256, 128),
            (opt_batch, 3, 256, 128),
            (max_batch, 3, 256, 128),
        ),
    }
    r = build_engine_from_onnx(
        onnx_path,
        engine_path,
        fp16=fp16,
        workspace_gb=workspace_gb,
        autocast_fast=autocast_fast,
        profile_shapes=profile,
        log=log,
    )
    return TrtBuildResult("reid_engine", onnx_path, r.engine, r.ok, r.message)


def export_reid_engine(
    pth_path: Path,
    out_engine: Path,
    *,
    fp16: bool,
    workspace_gb: float,
    max_batch: int,
    autocast_fast: bool = False,
    log: LogFn | None = None,
) -> list[TrtBuildResult]:
    onnx_path = out_engine.parent / reid_onnx_name(pth_path.stem)
    results: list[TrtBuildResult] = []
    r_onnx = export_reid_onnx(pth_path, onnx_path, log=log)
    results.append(r_onnx)
    if not r_onnx.ok:
        return results

    r_eng = build_engine_tensorrt_python(
        onnx_path,
        out_engine,
        fp16=fp16,
        workspace_gb=workspace_gb,
        max_batch=max_batch,
        autocast_fast=autocast_fast,
        log=log,
    )
    if not r_eng.ok:
        r_eng2 = build_engine_trtexec(
            onnx_path,
            out_engine,
            fp16=fp16,
            workspace_gb=workspace_gb,
            max_batch=max_batch,
            autocast_fast=autocast_fast,
            log=log,
        )
        results.append(r_eng2)
    else:
        results.append(r_eng)
    return results


def build_all_engines(
    settings: PipelineSettings,
    *,
    log: LogFn | None = None,
) -> list[TrtBuildResult]:
    """Собрать .engine для detect, cross-check (helmet), ReID."""
    strategy = str(settings.tensorrt_engine_strategy)
    central_dir = Path(settings.models_dir) / "TRT"
    manifest_dir = Path(settings.tensorrt_manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rebuild_policy = str(settings.tensorrt_rebuild_policy).strip().casefold()
    missing_only = rebuild_policy in ("missing_only", "missing", "if_missing")

    imgsz = int(settings.tensorrt_imgsz or settings.infer_imgsz or 640)
    max_batch = max(1, int(settings.tensorrt_max_batch))
    fp16 = bool(settings.tensorrt_fp16)
    ws = float(settings.tensorrt_workspace_gb)
    autocast_fast = bool(settings.tensorrt_autocast_fast)
    results: list[TrtBuildResult] = []
    records: list[TrtEngineRecord] = []
    now = datetime.now(timezone.utc).isoformat()
    built_paths: list[Path] = []

    specs: list[tuple[str, Path]] = [("detect", Path(settings.detect_model))]
    if settings.cross_check_enabled and settings.cross_check_model is not None:
        specs.append(("cross_check", Path(settings.cross_check_model)))
    specs.append(("reid", Path(settings.reid_model)))

    _log(
        log,
        f"TensorRT сборка: {len(specs)} моделей, strategy={strategy}, "
        f"policy={rebuild_policy}, manifest={manifest_dir.resolve()}",
    )

    for role, src in specs:
        if not src.exists():
            msg = f"Пропуск {role}: нет файла {src}"
            _log(log, msg)
            results.append(TrtBuildResult(role, src, None, False, msg))
            continue

        if role == "reid":
            out = resolve_reid_engine(
                src,
                fp16=fp16,
                strategy=strategy,
                central_dir=central_dir,
            )
        else:
            out = resolve_yolo_engine(
                src,
                imgsz=imgsz,
                max_batch=max_batch,
                fp16=fp16,
                strategy=strategy,
                central_dir=central_dir,
            )

        if missing_only and out.exists():
            _log(log, f">>> [{role}] пропуск: engine уже есть ({out.name})")
            results.append(TrtBuildResult(role, src, out, True, "skipped: exists"))
            records.append(
                TrtEngineRecord(
                    role=role,
                    source=str(src.resolve()),
                    engine=str(out.resolve()),
                    imgsz=256 if role == "reid" else imgsz,
                    max_batch=max_batch,
                    fp16=fp16,
                    built_at=now,
                    notes="existing",
                )
            )
            built_paths.append(out)
            continue

        _log(log, f">>> [{role}] старт: {src.name} -> {out.parent}")

        if role == "reid":
            for step in export_reid_engine(
                src,
                out,
                fp16=fp16,
                workspace_gb=ws,
                max_batch=max_batch,
                autocast_fast=autocast_fast,
                log=log,
            ):
                results.append(
                    TrtBuildResult(role if step.role == "reid_engine" else step.role, src, step.engine, step.ok, step.message)
                )
            if out.exists():
                _log(log, f">>> [{role}] ГОТОВО: {out.name} ({_size_mb(out):.1f} MB)")
                records.append(
                    TrtEngineRecord(
                        role=role,
                        source=str(src.resolve()),
                        engine=str(out.resolve()),
                        imgsz=256,
                        max_batch=max_batch,
                        fp16=fp16,
                        built_at=now,
                        notes="OSNet 256x128",
                    )
                )
                built_paths.append(out)
            else:
                last = results[-1] if results else None
                _log(log, f">>> [{role}] ОШИБКА: {last.message[:200] if last else 'unknown'}")
            continue

        r = export_yolo_engine(
            src,
            out,
            imgsz=imgsz,
            max_batch=max_batch,
            fp16=fp16,
            workspace_gb=ws,
            autocast_fast=autocast_fast,
            log=log,
        )
        results.append(TrtBuildResult(role, src, out if r.ok else None, r.ok, r.message))
        if r.ok and out.exists():
            _log(log, f">>> [{role}] ГОТОВО: {out.name} ({_size_mb(out):.1f} MB)")
            built_paths.append(out)
        else:
            _log(log, f">>> [{role}] ОШИБКА: {r.message[:200]}")
        if r.ok and out.exists():
            records.append(
                TrtEngineRecord(
                    role=role,
                    source=str(src.resolve()),
                    engine=str(out.resolve()),
                    imgsz=imgsz,
                    max_batch=max_batch,
                    fp16=fp16,
                    built_at=now,
                    notes=src.name,
                )
            )

    if records:
        save_manifest(records, manifest_dir)
    write_instructions(settings, trt_dir=manifest_dir)
    ok_n = sum(1 for r in results if r.ok and r.role in ("detect", "cross_check", "reid", "reid_engine"))
    fail_n = sum(1 for r in results if not r.ok and r.role in ("detect", "cross_check", "reid", "reid_engine", "reid_onnx"))
    _log(log, f"=== СБОРКА ЗАВЕРШЕНА: OK={ok_n}, FAIL={fail_n} ===")
    for eng in built_paths:
        _log(log, f"  engine: {eng} ({_size_mb(eng):.1f} MB)")
    if not built_paths:
        _log(log, "  (файлов .engine не найдено)")
    _log(log, f"Manifest + инструкции: {manifest_dir}")
    return results


def write_instructions(settings: PipelineSettings, *, trt_dir: Path | None = None) -> Path:
    root = trt_dir or TRT_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = trt_instructions_path(root)
    imgsz = int(settings.tensorrt_imgsz or settings.infer_imgsz or 640)
    max_batch = int(settings.tensorrt_max_batch)
    fp16 = settings.tensorrt_fp16
    ws = settings.tensorrt_workspace_gb
    det = Path(settings.detect_model)
    cross = Path(settings.cross_check_model) if settings.cross_check_model else None
    reid = Path(settings.reid_model)
    det_eng = yolo_engine_name(det.stem, imgsz, max_batch, fp16)
    cross_eng = yolo_engine_name(cross.stem, imgsz, max_batch, fp16) if cross else "—"
    reid_eng = reid_engine_name(reid.stem, fp16)
    reid_onnx = reid_onnx_name(reid.stem)
    trtexec = find_trtexec()
    trtexec_line = str(trtexec) if trtexec else "trtexec  (добавь в PATH)"

    body = f"""# TensorRT — инструкции YOLO_DRT

Сгенерировано: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Модели (3 штуки)

| Роль | PyTorch | Назначение | TensorRT engine |
|------|---------|------------|-----------------|
| **detect** | `{det}` | Основная pose: person + keypoints + BoT-SORT | `{root / det_eng}` |
| **cross_check** | `{cross or "—"}` | helmet: пересечение голова ∩ каска | `{root / cross_eng if cross else "—"}` |
| **reid** | `{reid}` | OSNet stable ID | `{root / reid_eng}` |

Папка engines: `{root.resolve()}`

## Требования

- NVIDIA GPU + драйвер
- CUDA (совместим с PyTorch в .venv)
- TensorRT 10+ или 11 (`pip install tensorrt`). **TRT 11**: FP16 через ModelOpt AutoCast, не через `--fp16` / `BuilderFlag.FP16`
- `pip install "nvidia-modelopt[onnx]" onnxslim`
- `trtexec` в PATH: `{trtexec_line}
- Ultralytics (YOLO export)

## Быстрый способ (UI)

1. Выбери модели на вкладке Pipeline (detect = **yolo26x-pose.pt**, cross = **helmet-26m.pt**, ReID = OSNet .pth).
2. Нажми **«Собрать TensorRT engines»** — создаст `.engine` и этот файл.
3. Включи галочку **«Использовать TensorRT»**.
4. **Init engines** → **Run**.

## Ручная сборка YOLO (Ultralytics)

```powershell
cd {Path(settings.detect_model).parent.parent.parent if det.parent.name == "YOLO" else det.parent}
{sys.executable} -c "
from ultralytics import YOLO
from pathlib import Path
imgsz={imgsz}; batch={max_batch}; fp16={fp16}
for pt in [r'{det}', r'{cross or ""}']:
    if not pt: continue
    m=YOLO(pt)
    m.export(format='engine', imgsz=imgsz, half=fp16, batch=batch, workspace={int(ws * (1 << 30))}, device=0)
"
```

Переименуй выход в `{det_eng}` / `{cross_eng}` и положи в `{root}`.

## Ручная сборка ReID (ONNX → TensorRT)

```powershell
# 1) ONNX (или кнопка в UI)
{sys.executable} -c "from app.core.trt_export import export_reid_onnx; from pathlib import Path; export_reid_onnx(Path(r'{reid}'), Path(r'{root / reid_onnx}'))"

# 2) Engine
{trtexec_line} --onnx={root / reid_onnx} --saveEngine={root / reid_eng} --fp16 --memPoolSize=workspace:{int(ws * 1024)} ^
  --minShapes=images:1x3x256x128 --optShapes=images:{max_batch}x3x256x128 --maxShapes=images:{max_batch}x3x256x128
```

## Использование в коде

- `use_tensorrt=True` в настройках → `DetectEngine` / `ReidEngine` грузят `.engine` из `{root}`.
- Если engine отсутствует — fallback на `.pt` / `.pth` с предупреждением в лог.

## Параметры сборки (settings)

- `tensorrt_imgsz` = {imgsz}
- `tensorrt_max_batch` = {max_batch}
- `tensorrt_fp16` = {fp16}
- `tensorrt_workspace_gb` = {ws}

## Устранение проблем

- **OOM при export**: уменьши `tensorrt_max_batch` или `tensorrt_workspace_gb`.
- **trtexec not found**: установи TensorRT, добавь `bin` в PATH.
- **BuilderFlag.FP16 / --fp16**: в TensorRT 11 убраны — UI собирает через ModelOpt AutoCast автоматически.
- **ReID export fail**: `pip install onnx "nvidia-modelopt[onnx]"` и torchreid.
- После смены GPU или CUDA — пересобери engines.
"""
    path.write_text(body, encoding="utf-8")
    return path
