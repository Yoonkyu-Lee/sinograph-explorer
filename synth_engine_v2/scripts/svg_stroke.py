"""
SVG stroke-based glyph source with per-stroke variation.

NOTE (deprecated for new work): this module renders MakeMeAHanzi glyphs from
**stroke medians** (centerlines) as thick polylines. For higher-fidelity
rendering use `outline_stroke.py` which fills the true canonical outline
polygons. This file is retained for reference and to keep legacy configs
(`kind: svg_stroke`) working; no active development planned.

Original docstring:


Unlike FontSource which rasterizes a font into a single monolithic bitmap,
SvgStrokeSource represents a glyph as a list of individual strokes (centerline
polylines from MakeMeAHanzi). Each stroke is independently perturbable, so
the SAME character in the SAME "typeface family" produces genuinely different
glyph shapes per sample — width varies per stroke, endpoints jitter, strokes
can be dropped or rotated.

This is the "per-stroke variation" axis: inside the base_source block, but a
different source kind than FontSource.

MakeMeAHanzi coordinate system:
- coordinates live in a 0..1024 box
- y-axis is flipped: screen_y = 900 - mmh_y (standard transform matrix)

Data source: canonical_characters.jsonl from sinograph_canonical_v1.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from pipeline import CANVAS, PAD


MMH_BOX = 1024
MMH_Y_PIVOT = 900  # screen_y = MMH_Y_PIVOT - mmh_y


@dataclass
class StrokeData:
    """A glyph represented as stroke centerlines + per-stroke widths.

    Coordinate frame is math-up: higher `y` means character top. The rasterizer
    converts to screen-down using `box` (x/y extent) and `y_pivot` (the y value
    that maps to screen_y=0 top of canvas).

    Defaults match MakeMeAHanzi: 1024×1024 box, pivot 900. e-hanja medianized
    data overrides these (typically box=1152, y_pivot=1152).
    """
    medians: list[list[list[float]]]  # stroke -> list of [x, y]
    widths:  list[float]              # one width per stroke
    char:    str = ""
    box:     int = MMH_BOX            # x/y max extent in math-up coords
    y_pivot: int = MMH_Y_PIVOT        # screen_y = y_pivot - y

    def copy(self) -> "StrokeData":
        return StrokeData(
            medians=[[list(p) for p in m] for m in self.medians],
            widths=list(self.widths),
            char=self.char,
            box=self.box,
            y_pivot=self.y_pivot,
        )


# ---------- loader -----------------------------------------------------------


# Cache for MMH canonical JSONL: keyed on path → {char: raw_record}
_MMH_INDEX: dict[str, dict[str, dict]] = {}


def _get_mmh_index(canonical_jsonl: Path) -> dict[str, dict]:
    key = str(canonical_jsonl)
    idx = _MMH_INDEX.get(key)
    if idx is not None:
        return idx
    idx = {}
    with open(canonical_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            c = rec.get("character")
            if c:
                idx[c] = rec
    _MMH_INDEX[key] = idx
    return idx


def load_stroke_data(canonical_jsonl: Path, char: str,
                      default_width: float = 48.0) -> StrokeData | None:
    """Find the record for `char` in the canonical JSONL and return StrokeData.

    Returns None if the character isn't covered by MakeMeAHanzi.
    """
    rec = _get_mmh_index(canonical_jsonl).get(char)
    if rec is None:
        return None
    medians = rec.get("media", {}).get("stroke_medians", [])
    if not medians:
        return None
    return StrokeData(
        medians=[[list(p) for p in m] for m in medians],
        widths=[default_width] * len(medians),
        char=char,
    )


# ---------- stroke ops -------------------------------------------------------
# Each op: (StrokeData, rng, **params) -> StrokeData (new copy)

STROKE_OPS: dict[str, Callable[..., StrokeData]] = {}


def _register_stroke_op(name: str):
    def deco(fn):
        STROKE_OPS[name] = fn
        return fn
    return deco


def _polyline_length(pts) -> float:
    """Total path length of a polyline (list of [x, y])."""
    if len(pts) < 2:
        return 0.0
    arr = np.asarray(pts, dtype=float)
    diffs = np.diff(arr, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _resolve_std(absolute_std: float, ratio: float | None,
                  length: float,
                  std_min: float, std_max: float) -> float:
    """Pick jitter std: length * ratio (clipped) if ratio given, else absolute.

    The ratio mode lets a config express "jitter as a fraction of each stroke's
    own length" so dots barely move while long strokes wiggle naturally; and
    it is unit-free so the same value works across MMH (box 1024), e-hanja
    (1152), KanjiVG (109).
    """
    if ratio is None:
        return float(absolute_std)
    eff = length * float(ratio)
    if eff < std_min:
        eff = std_min
    if eff > std_max:
        eff = std_max
    return float(eff)


@_register_stroke_op("width_jitter")
def width_jitter(data, rng, *, mean=None, std=5.0) -> StrokeData:
    """Per-stroke width perturbation (Gaussian).

    mean: override base width for all strokes (None = keep current widths)
    std:  per-stroke standard deviation
    """
    base = [float(mean)] * len(data.widths) if mean is not None else data.widths
    new_widths = [max(2.0, w + float(rng.normal(0.0, std))) for w in base]
    return replace(data.copy(), widths=new_widths)


@_register_stroke_op("endpoint_jitter")
def endpoint_jitter(data, rng, *, std=6.0, std_ratio=None,
                      std_min=0.0, std_max=float("inf")) -> StrokeData:
    """Shift first and last waypoint of each stroke by Gaussian noise.

    Length-proportional mode: set `std_ratio` — effective std becomes
    `clip(stroke_length * std_ratio, std_min, std_max)`. Dots (short strokes)
    stay nearly put, long strokes wiggle more. Leave `std_ratio=None` to use
    absolute `std` across all strokes (legacy behaviour).
    """
    new = data.copy()
    for m in new.medians:
        if len(m) < 2:
            continue
        length = _polyline_length(m)
        eff = _resolve_std(std, std_ratio, length, std_min, std_max)
        for idx in (0, -1):
            m[idx][0] += float(rng.normal(0.0, eff))
            m[idx][1] += float(rng.normal(0.0, eff))
    return new


@_register_stroke_op("control_jitter")
def control_jitter(data, rng, *, std=4.0, std_ratio=None,
                     std_min=0.0, std_max=float("inf")) -> StrokeData:
    """Perturb every waypoint (not just endpoints) of every stroke.

    Same length-proportional semantics as `endpoint_jitter`.
    """
    new = data.copy()
    for m in new.medians:
        if len(m) < 1:
            continue
        length = _polyline_length(m)
        eff = _resolve_std(std, std_ratio, length, std_min, std_max)
        for p in m:
            p[0] += float(rng.normal(0.0, eff))
            p[1] += float(rng.normal(0.0, eff))
    return new


@_register_stroke_op("stroke_rotate")
def stroke_rotate(data, rng, *, angle_std=4.0) -> StrokeData:
    """Rotate each stroke independently about its own centroid."""
    new = data.copy()
    for m in new.medians:
        if not m:
            continue
        cx = sum(p[0] for p in m) / len(m)
        cy = sum(p[1] for p in m) / len(m)
        angle = float(rng.normal(0.0, angle_std))
        rad = math.radians(angle)
        cs, sn = math.cos(rad), math.sin(rad)
        for p in m:
            dx, dy = p[0] - cx, p[1] - cy
            p[0] = cx + dx * cs - dy * sn
            p[1] = cy + dx * sn + dy * cs
    return new


@_register_stroke_op("drop_stroke")
def drop_stroke(data, rng, *, prob_per_stroke=0.05) -> StrokeData:
    """Randomly omit strokes (simulates faded / missing ink)."""
    keep = [rng.random() > prob_per_stroke for _ in data.medians]
    return StrokeData(
        medians=[m for m, k in zip(data.medians, keep) if k],
        widths=[w for w, k in zip(data.widths, keep) if k],
        char=data.char,
    )


@_register_stroke_op("stroke_translate")
def stroke_translate(data, rng, *, std=3.0, std_ratio=None,
                       std_min=0.0, std_max=float("inf")) -> StrokeData:
    """Shift each stroke as a whole by a small random offset.

    Length-proportional mode available via `std_ratio` (same semantics as
    other ops): short strokes translate less, long strokes more. This keeps
    each stroke's relative displacement comparable.
    """
    new = data.copy()
    for m in new.medians:
        length = _polyline_length(m) if m else 0.0
        eff = _resolve_std(std, std_ratio, length, std_min, std_max)
        dx = float(rng.normal(0.0, eff))
        dy = float(rng.normal(0.0, eff))
        for p in m:
            p[0] += dx
            p[1] += dy
    return new


def apply_stroke_ops(data: StrokeData, ops: list[dict],
                      rng: np.random.Generator) -> StrokeData:
    """Run a list of op specs against the stroke data in order."""
    for raw in ops or []:
        spec = dict(raw)
        name = spec.pop("op")
        if name not in STROKE_OPS:
            raise ValueError(f"unknown stroke op: {name!r}. known: {sorted(STROKE_OPS)}")
        prob = spec.pop("prob", 1.0)
        if rng.random() > prob:
            continue
        data = STROKE_OPS[name](data, rng, **spec)
    return data


# ---------- rasterizer -------------------------------------------------------


def rasterize_strokes(data: StrokeData, canvas: int = CANVAS, pad: int = PAD) -> Image.Image:
    """Draw strokes as thick polylines onto an L-mode image.

    Coordinate source: `data.box` (x/y extent) and `data.y_pivot`. MakeMeAHanzi
    defaults match the original hardcoded values; e-hanja medianized data
    carries its own box/pivot via the loader.
    """
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    target = canvas - 2 * pad
    scale = target / data.box

    for median, width in zip(data.medians, data.widths):
        if len(median) < 2:
            continue
        pts = []
        for x, y in median:
            sx = x * scale + pad
            sy = (data.y_pivot - y) * scale + pad
            pts.append((sx, sy))
        w = max(1, int(round(width * scale)))
        # thick polyline via line segments
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=255, width=w)
        # round caps at every waypoint so joints/ends look smooth
        half = w / 2
        for p in pts:
            draw.ellipse([p[0] - half, p[1] - half, p[0] + half, p[1] + half], fill=255)
    return img


# ---------- source -----------------------------------------------------------


@dataclass
class SvgStrokeSource:
    """Source type: produces glyph mask from MakeMeAHanzi stroke data.

    One instance = one character's base stroke data + a list of stroke ops
    that will be applied per render call using the supplied rng.
    """
    base_data: StrokeData
    stroke_ops: list[dict] = field(default_factory=list)
    canvas: int = CANVAS
    pad: int = PAD
    # Canonical source-kind AND tag prefix. Distinguishes MMH / ehanja_median /
    # kanjivg_median (all three use this same class). Used by source-aware
    # augment gating via `ctx.source_kind`.
    source_tag: str = "svg_stroke"

    @property
    def kind(self) -> str:
        return self.source_tag

    def render_mask(self, char: str, rng: np.random.Generator) -> Image.Image | None:
        if char != self.base_data.char:
            return None
        varied = apply_stroke_ops(self.base_data, self.stroke_ops, rng)
        if not varied.medians:
            return None
        return rasterize_strokes(varied, canvas=self.canvas, pad=self.pad)

    def tag(self) -> str:
        return f"{self.source_tag}-{self.base_data.char}"


def discover_svg_stroke_sources(char: str, canonical_jsonl: Path,
                                  stroke_ops: list[dict] | None = None,
                                  base_width: float = 48.0,
                                  canvas: int = CANVAS, pad: int = PAD) -> list[SvgStrokeSource]:
    """Return a single-item list if the character is MakeMeAHanzi-covered, else empty."""
    data = load_stroke_data(canonical_jsonl, char, default_width=base_width)
    if data is None:
        return []
    return [SvgStrokeSource(
        base_data=data,
        stroke_ops=list(stroke_ops or []),
        canvas=canvas,
        pad=pad,
    )]


# ---------- e-hanja median adapter ------------------------------------------


DEFAULT_EHANJA_MEDIAN_JSONL = Path(
    "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/"
    "db_src/e-hanja_online/strokes_medianized.jsonl"
)

DEFAULT_KANJIVG_JSONL = Path(
    "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/"
    "db_src/KanjiVG/strokes_kanjivg.jsonl"
)


# Cache of median JSONL files indexed by char for O(1) lookup. Keyed on the
# string form of the path so e-hanja and KanjiVG (same schema, different file)
# don't collide. Essential for corpus runs where a single file is queried
# thousands of times.
_MEDIAN_JSONL_INDEX: dict[str, dict[str, dict]] = {}


def _get_median_index(jsonl_path: Path) -> dict[str, dict]:
    key = str(jsonl_path)
    idx = _MEDIAN_JSONL_INDEX.get(key)
    if idx is not None:
        return idx
    idx = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            c = rec.get("char")
            if c:
                idx[c] = rec
    _MEDIAN_JSONL_INDEX[key] = idx
    return idx


def load_ehanja_median_data(jsonl_path: Path, char: str,
                              width_scale: float = 1.0,
                              width_override: float | None = None) -> StrokeData | None:
    """Load e-hanja medianized strokes (output of `medianize_outlines.py`).

    The file format stores per-stroke median + width + viewbox + y_pivot.
    We hand the same median/width arrays to `StrokeData` but override `box`
    and `y_pivot` so the shared rasterizer maps e-hanja's frame correctly.

    Parameters
    ----------
    width_scale : float
        Multiplier applied to every per-stroke estimated width. e-hanja's
        canonical glyphs are typically thinner than MMH's default 48; set
        `width_scale=1.5` to visually match MMH thickness.
    width_override : float or None
        If set, ignore per-stroke estimates and use this single width for all
        strokes (matches the MMH default-width behaviour).
    """
    idx = _get_median_index(jsonl_path)
    rec = idx.get(char)
    if rec is None:
        return None
    strokes = rec.get("strokes", [])
    if not strokes:
        return None
    strokes = sorted(strokes, key=lambda s: s.get("order", 0))
    medians = [[list(p) for p in s["median"]] for s in strokes]
    if width_override is not None:
        widths = [float(width_override)] * len(strokes)
    else:
        widths = [float(s["width"]) * float(width_scale) for s in strokes]
    vb = rec.get("viewbox") or [1024, 1152]
    # For a non-square viewbox use the *larger* extent as the single scale
    # factor so the rendered character doesn't get squashed.
    box = max(int(vb[0]), int(vb[1]))
    y_pivot = int(rec.get("y_pivot", box))
    return StrokeData(
        medians=medians,
        widths=widths,
        char=char,
        box=box,
        y_pivot=y_pivot,
    )


def discover_ehanja_median_sources(char: str,
                                     jsonl_path: Path | None = None,
                                     stroke_ops: list[dict] | None = None,
                                     width_scale: float = 1.0,
                                     width_override: float | None = None,
                                     canvas: int = CANVAS, pad: int = PAD) -> list[SvgStrokeSource]:
    """Return a single-item list if the character is in the e-hanja medianized
    manifest, else empty. Returns `SvgStrokeSource` instances (same class as
    MMH) — the only difference is `base_data.box` / `y_pivot`."""
    path = jsonl_path or DEFAULT_EHANJA_MEDIAN_JSONL
    data = load_ehanja_median_data(path, char,
                                     width_scale=width_scale,
                                     width_override=width_override)
    if data is None:
        return []
    return [SvgStrokeSource(
        base_data=data,
        stroke_ops=list(stroke_ops or []),
        canvas=canvas,
        pad=pad,
        source_tag="ehanja_median",
    )]


# ---------- KanjiVG median adapter ------------------------------------------

# KanjiVG extract JSONL uses the same schema as e-hanja medianized data —
# char/cp/viewbox/y_pivot/strokes[].{order,kind,median,width}. So the loader
# is literally the same; only the default path differs.

load_kanjivg_median_data = load_ehanja_median_data


def discover_kanjivg_median_sources(char: str,
                                     jsonl_path: Path | None = None,
                                     stroke_ops: list[dict] | None = None,
                                     width_scale: float = 1.0,
                                     width_override: float | None = None,
                                     canvas: int = CANVAS, pad: int = PAD) -> list[SvgStrokeSource]:
    """Return a single-item list if the character is in the KanjiVG extract,
    else empty. KanjiVG native CSS stroke-width=3 on viewbox=109 is thin;
    bump width_scale to ~5 for MMH-like thickness."""
    path = jsonl_path or DEFAULT_KANJIVG_JSONL
    data = load_kanjivg_median_data(path, char,
                                      width_scale=width_scale,
                                      width_override=width_override)
    if data is None:
        return []
    return [SvgStrokeSource(
        base_data=data,
        stroke_ops=list(stroke_ops or []),
        canvas=canvas,
        pad=pad,
        source_tag="kanjivg_median",
    )]
