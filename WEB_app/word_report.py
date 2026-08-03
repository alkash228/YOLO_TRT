"""Word (.docx) incident report per violator stable_id."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.video_encode import _EncodeJob, _RenderContext, _render_encode_job, resolve_run_packets
from WEB_app.video_builder import (
    collect_presence_frame_indices,
    collect_violation_frame_indices,
    count_presence_by_stable_id,
    count_violations_by_stable_id,
    filter_person_single_id,
    find_packet_at_frame,
    load_run_metadata,
    resolve_overlay,
)

TEST_COMPANIES: tuple[str, ...] = (
    'ООО «Обнал»',
    'ООО «Рога и копыта»',
)

_MONTHS_RU = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


def format_incident_datetime_ru(value: datetime) -> str:
    return (
        f"{value.day} {_MONTHS_RU[value.month - 1]} {value.year} г., "
        f"{value.hour:02d}:{value.minute:02d}"
    )


def parse_incident_datetime(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        return datetime.now()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.now()


def middle_presence_frame(
    data: dict[str, Any], run_dir: Path, stable_id: int
) -> int | None:
    presence = collect_presence_frame_indices(data, run_dir, stable_id)
    if not presence:
        violations = collect_violation_frame_indices(data, run_dir, stable_id)
        if not violations:
            return None
        return violations[len(violations) // 2]
    return presence[len(presence) // 2]


def read_source_frame_bgr(input_path: str, frame_idx: int) -> np.ndarray | None:
    """Grab one source frame. Sequential decode — HEVC-safe (no POS_FRAMES seek)."""
    from app.core.window_frame_loader import read_frame_bgr_sequential

    return read_frame_bgr_sequential(str(input_path), int(frame_idx))


def render_person_frame_rgb(
    *,
    data: dict[str, Any],
    run_dir: Path,
    run_id: str,
    stable_id: int,
    meta: dict[str, Any],
    overlay: dict[str, Any],
    frame_idx: int,
) -> np.ndarray:
    input_path = str(meta.get("input_path") or "")
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    frame_bgr = read_source_frame_bgr(input_path, frame_idx) if input_path else None

    packet = find_packet_at_frame(data, run_dir, frame_idx)
    if packet is None:
        if frame_bgr is None:
            raise RuntimeError(f"Кадр {frame_idx} недоступен в исходном видео")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return rgb

    if frame_bgr is None and packet.frame_bgr is not None and packet.frame_bgr.size > 0:
        frame_bgr = packet.frame_bgr

    if frame_bgr is not None:
        height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])

    filtered = filter_person_single_id(packet, stable_id)
    if filtered.n_inst <= 0:
        if frame_bgr is None:
            raise RuntimeError(f"Человек ID {stable_id} не найден на кадре {frame_idx}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    prompt = str(meta.get("prompt") or "person").strip().casefold() or "person"
    ctx = _RenderContext(
        target_w=max(1, width),
        target_h=max(1, height),
        prompt_lookup={prompt: 0},
        overlay=overlay,
    )
    job = _EncodeJob(src_i=int(frame_idx), carry=filtered, frame_bgr=frame_bgr)
    _, rgb = _render_encode_job(job, ctx)
    return rgb


def build_word_report(
    run_dir: str | Path,
    run_id: str,
    stable_id: int,
    *,
    company: str,
    organization: str,
    incident_datetime: datetime,
    overlay_override: dict[str, Any] | None = None,
) -> Path:
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Cm, Pt
    except ImportError as exc:
        raise RuntimeError(
            "Установите python-docx: pip install python-docx"
        ) from exc

    run_path = Path(run_dir)
    meta = load_run_metadata(run_path, run_id)
    data, _ = resolve_run_packets(run_path, run_id=run_id)
    overlay = resolve_overlay(meta, overlay_override=overlay_override)

    frame_idx = middle_presence_frame(data, run_path, stable_id)
    if frame_idx is None:
        raise RuntimeError(f"Нет кадров с нарушителем ID {stable_id}")

    rgb = render_person_frame_rgb(
        data=data,
        run_dir=run_path,
        run_id=run_id,
        stable_id=stable_id,
        meta=meta,
        overlay=overlay,
        frame_idx=frame_idx,
    )

    vcounts = count_violations_by_stable_id(data, run_path)
    pcounts = count_presence_by_stable_id(data, run_path)
    violation_count = int(vcounts.get(int(stable_id), 0))
    presence_count = int(pcounts.get(int(stable_id), 0))

    reports_dir = run_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    image_path = reports_dir / f"{run_id}_id{stable_id}_mid.jpg"
    docx_path = reports_dir / f"{run_id}_akt_id{stable_id}.docx"

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(image_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError("Не удалось сохранить кадр для отчёта")

    org = (organization or company or "").strip() or "—"
    company_name = (company or org).strip() or "—"
    incident_text = format_incident_datetime_ru(incident_datetime)
    source_name = Path(str(meta.get("input_path") or "")).name or "—"

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("АКТ фиксации нарушения\nтребований охраны труда")
    run.bold = True
    run.font.size = Pt(16)

    doc.add_paragraph()

    def add_row(label: str, value: str) -> None:
        p = doc.add_paragraph()
        p.add_run(f"{label}: ").bold = True
        p.add_run(value)

    add_row("Организация", org)
    add_row("Объект / заказчик (тест)", company_name)
    add_row("Дата и время инцидента", incident_text)
    add_row("Вид нарушения", "Работа без защитной каски (NO HELMET)")
    add_row("Идентификатор нарушителя", f"№ {stable_id}")
    add_row("Кадров присутствия в видео", str(presence_count))
    add_row("Срабатываний без каски", str(violation_count))
    add_row("Исходное видео", source_name)

    doc.add_paragraph()
    cap = doc.add_paragraph("Фотофиксация нарушителя:")
    cap.runs[0].bold = True
    doc.add_picture(str(image_path), width=Cm(14))

    doc.save(str(docx_path))
    return docx_path
