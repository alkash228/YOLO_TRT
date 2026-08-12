"""Serialize instance masks/ids for encode overlay — no torch / YOLO imports."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.mask_json import mask_u8_to_rle_dict
from app.core.pose_utils import keypoints_to_json
from app.core.schema import label_slug


def resolve_prompt_id(
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


def serialize_frame_instances(
    instance_stack: np.ndarray | None,
    instance_object_ids: np.ndarray | None,
    instance_scores: np.ndarray | None,
    prompt_labels: list[str] | None,
    prompt_id_lookup: dict[str, int] | None = None,
    pose_kpt_conf: float = 0.25,
    keypoints_list: list[np.ndarray | None] | None = None,
    cross_check_verdicts: list[Any] | None = None,
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
        pid = resolve_prompt_id(
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
            # Legacy key: highlight box (person∩helmet when OK, person-top otherwise).
            if v.head_xyxy is not None:
                hx0, hy0, hx1, hy1 = [float(x) for x in v.head_xyxy.tolist()]
                row["head_bbox_xyxy"] = [int(hx0), int(hy0), int(hx1), int(hy1)]
        out.append(row)
    return out
