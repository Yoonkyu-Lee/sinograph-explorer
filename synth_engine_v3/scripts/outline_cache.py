"""Phase 13 — cache outline_stroke file loaders.

v2's `load_ehanja_outline` / `load_mmh_outline` walk the ENTIRE JSONL file and
SVG-path-flatten every matching char on every call. Profiling (Phase 12) shows
180 ms / 156 ms per call warm — 200× slower than the median-based sources
which already index the raw JSONL by char.

This module monkey-patches v2 `outline_stroke` so:
  - each JSONL is scanned ONCE → dict[char → raw record]
  - parsed `OutlineStrokeData` per (path, char) is memoized
  - callers still receive a `.copy()` so stroke_ops can mutate freely

v2 sources are untouched. Import this module *after* v2's `outline_stroke` is
available (importing `mask_adapter` guarantees it).

Added at Phase 13.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np

import outline_stroke as v2_outline  # v2 module, via sys.path from mask_adapter


# path (str) -> char -> raw record dict
_RAW_INDEX: dict[str, dict[str, dict]] = {}
_RAW_INDEX_LOCK = Lock()

# (path_str, char) -> OutlineStrokeData (parsed & transformed)
_PARSED: dict[tuple[str, str], v2_outline.OutlineStrokeData | None] = {}
_PARSED_LOCK = Lock()


def _build_raw_index(path_str: str) -> dict[str, dict]:
    """Scan a JSONL once, index records by char / character key."""
    idx: dict[str, dict] = {}
    with open(path_str, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # e-hanja / kanjivg use 'char'; MMH graphics.txt uses 'character'
            c = rec.get("char") or rec.get("character")
            if c:
                idx[c] = rec
    return idx


def _get_raw_index(path: Path) -> dict[str, dict]:
    key = str(path)
    idx = _RAW_INDEX.get(key)
    if idx is not None:
        return idx
    with _RAW_INDEX_LOCK:
        idx = _RAW_INDEX.get(key)
        if idx is None:
            idx = _build_raw_index(key)
            _RAW_INDEX[key] = idx
    return idx


def _parse_ehanja_record(rec: dict, char: str) -> v2_outline.OutlineStrokeData | None:
    sx, sy, tx, ty = v2_outline._parse_transform(rec.get("transform"))
    strokes: list = []
    for s in rec.get("strokes", []):
        raw = v2_outline._flatten_svg_path(s["d"])
        if raw.shape[0] < 3:
            continue
        xf = v2_outline._apply_transform(raw, sx, sy, tx, ty)
        strokes.append(v2_outline.OutlineStroke(
            order=int(s["order"]), kind=str(s["kind"]), polygon=xf,
        ))
    strokes.sort(key=lambda s: s.order)
    vb = tuple(rec["viewbox"]) if rec.get("viewbox") else (1024, 1152)
    return v2_outline.OutlineStrokeData(char=char, viewbox=vb, strokes=strokes)


def _parse_mmh_record(rec: dict, char: str) -> v2_outline.OutlineStrokeData | None:
    raw_strokes = rec.get("strokes", [])
    if not raw_strokes:
        return None
    strokes: list = []
    for i, d in enumerate(raw_strokes, 1):
        raw = v2_outline._flatten_svg_path(d)
        if raw.shape[0] < 3:
            continue
        xf = v2_outline._apply_transform(raw, 1.0, -1.0, 0.0, -v2_outline.MMH_Y_PIVOT)
        strokes.append(v2_outline.OutlineStroke(order=i, kind="normal", polygon=xf))
    return v2_outline.OutlineStrokeData(
        char=char, viewbox=(v2_outline.MMH_BOX, v2_outline.MMH_BOX), strokes=strokes,
    )


def _cached_load(path: Path, char: str, parser) -> v2_outline.OutlineStrokeData | None:
    key = (str(path), char)
    cached = _PARSED.get(key)
    if cached is not None:
        return cached.copy()
    if key in _PARSED:
        # previously parsed as None — char is not in the file
        return None
    with _PARSED_LOCK:
        cached = _PARSED.get(key)
        if cached is not None:
            return cached.copy()
        if key in _PARSED:
            return None
        idx = _get_raw_index(path)
        rec = idx.get(char)
        if rec is None:
            _PARSED[key] = None
            return None
        data = parser(rec, char)
        _PARSED[key] = data
    return data.copy() if data is not None else None


def cached_load_ehanja_outline(manifest_path: Path, char: str):
    return _cached_load(manifest_path, char, _parse_ehanja_record)


def cached_load_mmh_outline(graphics_path: Path, char: str):
    return _cached_load(graphics_path, char, _parse_mmh_record)


def install() -> None:
    """Replace v2 outline_stroke loaders with cached versions.

    Idempotent — safe to call multiple times (e.g. once per mp worker process,
    since module-level caches are per-process).
    """
    v2_outline.load_ehanja_outline = cached_load_ehanja_outline
    v2_outline.load_mmh_outline = cached_load_mmh_outline


# Install on import so any downstream caller that imports this module gets the
# fast path automatically. Workers that spawn fresh Python interpreters must
# also import this (mask_adapter handles that).
install()
