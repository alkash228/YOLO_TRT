"""Baked UI fast profile — shared by desktop API defaults and Docker."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROFILE_PATH = Path(__file__).resolve().parent / "ui_fast_profile.json"


def load_ui_fast_pipeline() -> dict[str, Any]:
    """Return pipeline dict from ui_fast_profile.json (no model paths)."""
    raw = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    pipe = raw.get("pipeline") if isinstance(raw, dict) else None
    if not isinstance(pipe, dict):
        return {}
    return dict(pipe)
