"""Prompt parsing and label matching."""
from __future__ import annotations


def parse_prompt_segments(prompt: str) -> list[str]:
    if not prompt or not str(prompt).strip():
        return []
    raw = str(prompt).replace("\n", " ").replace(";", ".")
    parts = [p.strip() for p in raw.split(".") if p.strip()]
    return parts if parts else [str(prompt).strip()]


def _norm_label(s: str) -> str:
    return s.strip().casefold().replace("_", "-")


def label_match(label: str, terms: list[str]) -> bool:
    """Exact class name match (case-insensitive). Prompt term must equal model label."""
    if not terms:
        return True
    lab = _norm_label(label)
    return any(lab == _norm_label(t) for t in terms)


def prompt_terms(prompt: str) -> list[str]:
    terms: list[str] = []
    for seg in parse_prompt_segments(prompt):
        t = seg.strip().casefold()
        if t:
            terms.append(t)
    return list(dict.fromkeys(terms))
