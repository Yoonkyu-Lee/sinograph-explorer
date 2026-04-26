"""
Unified outline-based stroke source.

Both MakeMeAHanzi and e-hanja ship per-stroke SVG outline `d` strings. This
module loads either format into a common `OutlineStrokeData` representation
and renders it via `PIL.ImageDraw.polygon`. Stroke ops (rotate, translate,
control jitter, endpoint jitter, drop, width jitter) work uniformly on the
polygon geometry regardless of source.

Source adapters
---------------
  load_ehanja_outline(manifest_path, char)
    Reads `db_src/e-hanja_online/strokes_animated.jsonl` (Phase 2 output).
    Applies the SVG's `<g transform="scale(1,-1) translate(tx,ty)">` at load
    time so downstream vertices are in viewBox coordinates (typically
    1024×1152).

  load_mmh_outline(graphics_path, char)
    Reads MakeMeAHanzi's `graphics.txt`. Uses the conventional MMH frame
    (1024×1024 box, y-flip pivot=900); paths are transformed at load time so
    vertices sit in the same viewBox space as e-hanja data (0..vb_w × 0..vb_h
    roughly, with some allowed overshoot per MMH's ascender space).

Why one source class instead of two
-----------------------------------
Rendering is identical once the polygons are in viewBox space. The only
source-specific bit is the loader. Keeping them separate (`EHanjaStrokeSource`
vs `MmhStrokeSource`) would duplicate the rasterizer and the ops registry
without benefit. We keep one `OutlineStrokeSource` and two discovery
functions.

Relation to `svg_stroke.py`
---------------------------
`svg_stroke.py` still exists as the legacy **median-based** rasterizer for
MakeMeAHanzi. It renders strokes as thick polylines using `widths`; this
module renders them as filled polygons using the outline `strokes`. The two
produce slightly different looks (median = rounded uniform, outline = true
canonical shape). Both are usable; outline is preferred for higher fidelity.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


# ---------- MakeMeAHanzi conventions ----------------------------------------


MMH_BOX = 1024         # MMH glyphs fit in a 0..1024 box
MMH_Y_PIVOT = 900      # screen_y = pivot - mmh_y  (matches svg_stroke.py)


# ---------- data model ------------------------------------------------------


@dataclass
class OutlineStroke:
    """One stroke represented as a closed polygon in viewBox coordinates."""
    order: int
    kind: str                   # "normal" or "radical"
    polygon: np.ndarray         # (N, 2) float64, in viewBox coords
    dilate_radius: float = 0.0   # width_jitter output, applied at render


@dataclass
class OutlineStrokeData:
    char: str
    viewbox: tuple[int, int]        # (w, h) in pixels
    strokes: list[OutlineStroke]

    def copy(self) -> "OutlineStrokeData":
        return OutlineStrokeData(
            char=self.char,
            viewbox=self.viewbox,
            strokes=[
                OutlineStroke(
                    order=s.order,
                    kind=s.kind,
                    polygon=s.polygon.copy(),
                    dilate_radius=s.dilate_radius,
                )
                for s in self.strokes
            ],
        )


# ---------- SVG path & transform helpers ------------------------------------


_TRANSFORM_RE = re.compile(
    r"scale\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s+"
    r"translate\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)"
)


def _parse_transform(s: str | None) -> tuple[float, float, float, float]:
    """Return (sx, sy, tx, ty). Identity if input doesn't match the shape
    `scale(sx,sy) translate(tx,ty)` used by e-hanja/MMH."""
    if not s:
        return (1.0, 1.0, 0.0, 0.0)
    m = _TRANSFORM_RE.match(s)
    if not m:
        return (1.0, 1.0, 0.0, 0.0)
    return tuple(float(x) for x in m.groups())  # type: ignore[return-value]


def _apply_transform(pts: np.ndarray,
                      sx: float, sy: float, tx: float, ty: float) -> np.ndarray:
    """Apply `scale(sx,sy) translate(tx,ty)` to (N, 2) points.

    SVG composes right-to-left on points: translate first, then scale.
    """
    out = np.empty_like(pts)
    out[:, 0] = sx * (pts[:, 0] + tx)
    out[:, 1] = sy * (pts[:, 1] + ty)
    return out


def _flatten_svg_path(d: str, samples_per_curve: int = 8) -> np.ndarray:
    """Parse an SVG path `d` string into a polygon vertex array.

    Lines contribute endpoints only; curves (quadratic/cubic Bezier, arcs) are
    sampled `samples_per_curve` times along t∈[0,1]. Duplicate sample points
    at segment joins are dropped.
    """
    from svgpathtools import parse_path, Line

    path = parse_path(d)
    pts: list[tuple[float, float]] = []
    for seg in path:
        if isinstance(seg, Line):
            if not pts:
                p0 = seg.start
                pts.append((p0.real, p0.imag))
            p1 = seg.end
            pts.append((p1.real, p1.imag))
        else:
            for i in range(samples_per_curve + 1):
                t = i / samples_per_curve
                p = seg.point(t)
                if pts and (abs(p.real - pts[-1][0]) < 1e-6
                            and abs(p.imag - pts[-1][1]) < 1e-6):
                    continue
                pts.append((p.real, p.imag))
    return np.asarray(pts, dtype=np.float64)


# ---------- source adapters (loaders) ---------------------------------------


def load_ehanja_outline(manifest_path: Path, char: str) -> OutlineStrokeData | None:
    """Look up `char` in the e-hanja strokes-animated manifest (Phase 2).

    Applies the outer-group SVG transform at load time so returned polygons
    live in the viewBox coord space (typically 1024×1152).
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("char") != char:
                continue
            sx, sy, tx, ty = _parse_transform(rec.get("transform"))
            strokes: list[OutlineStroke] = []
            for s in rec["strokes"]:
                raw = _flatten_svg_path(s["d"])
                if raw.shape[0] < 3:
                    continue
                xf = _apply_transform(raw, sx, sy, tx, ty)
                strokes.append(OutlineStroke(
                    order=int(s["order"]),
                    kind=str(s["kind"]),
                    polygon=xf,
                ))
            strokes.sort(key=lambda s: s.order)
            vb = tuple(rec["viewbox"]) if rec.get("viewbox") else (1024, 1152)
            return OutlineStrokeData(char=char, viewbox=vb, strokes=strokes)
    return None


def load_mmh_outline(graphics_path: Path, char: str) -> OutlineStrokeData | None:
    """Look up `char` in MakeMeAHanzi's `graphics.txt`.

    MMH's implicit frame is 0..1024 with y-flipped about pivot 900
    (matches `svg_stroke.py`). We model this as viewBox (1024, 1024) and
    apply `scale(1,-1) translate(0, -900)` at load time so output coords
    sit in a screen-space frame consistent with e-hanja outputs.
    """
    with graphics_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("character") != char:
                continue
            raw_strokes = rec.get("strokes", [])
            if not raw_strokes:
                return None
            # MMH has no kind/order metadata — all "normal", order = index+1
            strokes: list[OutlineStroke] = []
            for i, d in enumerate(raw_strokes, 1):
                raw = _flatten_svg_path(d)
                if raw.shape[0] < 3:
                    continue
                xf = _apply_transform(raw, 1.0, -1.0, 0.0, -MMH_Y_PIVOT)
                strokes.append(OutlineStroke(
                    order=i,
                    kind="normal",
                    polygon=xf,
                ))
            return OutlineStrokeData(
                char=char,
                viewbox=(MMH_BOX, MMH_BOX),
                strokes=strokes,
            )
    return None


# ---------- stroke ops ------------------------------------------------------


OUTLINE_OPS: dict[str, Callable[..., OutlineStrokeData]] = {}


def _register_op(name: str):
    def deco(fn):
        OUTLINE_OPS[name] = fn
        return fn
    return deco


def _polygon_extent(pts: np.ndarray) -> float:
    """Longer axis of polygon's axis-aligned bbox — proxy for 'stroke length'.

    Outlines are closed loops so polyline length doesn't map to stroke length.
    The bbox long axis is a decent cheap substitute (good for straight strokes,
    slightly underestimates curved ones, acceptable for jitter scaling).
    """
    if pts is None or len(pts) < 2:
        return 0.0
    ranges = pts.max(axis=0) - pts.min(axis=0)
    return float(ranges.max())


def _resolve_outline_std(absolute_std: float, ratio: float | None,
                          length: float,
                          std_min: float, std_max: float) -> float:
    if ratio is None:
        return float(absolute_std)
    eff = length * float(ratio)
    if eff < std_min:
        eff = std_min
    if eff > std_max:
        eff = std_max
    return float(eff)


@_register_op("stroke_rotate")
def stroke_rotate(data, rng, *, angle_std=4.0) -> OutlineStrokeData:
    new = data.copy()
    for s in new.strokes:
        pts = s.polygon
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        angle = math.radians(float(rng.normal(0.0, angle_std)))
        cs, sn = math.cos(angle), math.sin(angle)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        s.polygon = np.column_stack([cx + dx * cs - dy * sn,
                                      cy + dx * sn + dy * cs])
    return new


@_register_op("stroke_translate")
def stroke_translate(data, rng, *, std=3.0, std_ratio=None,
                       std_min=0.0, std_max=float("inf")) -> OutlineStrokeData:
    """Shift each stroke's polygon by a single (dx, dy).

    Length-proportional mode: set `std_ratio` — effective std becomes
    `clip(bbox_long_axis * std_ratio, std_min, std_max)`. Short strokes
    translate less.
    """
    new = data.copy()
    for s in new.strokes:
        length = _polygon_extent(s.polygon)
        eff = _resolve_outline_std(std, std_ratio, length, std_min, std_max)
        dx = float(rng.normal(0.0, eff))
        dy = float(rng.normal(0.0, eff))
        s.polygon = s.polygon + np.array([dx, dy])
    return new


@_register_op("control_jitter")
def control_jitter(data, rng, *, std=2.0, std_ratio=None,
                     std_min=0.0, std_max=float("inf")) -> OutlineStrokeData:
    """Per-vertex Gaussian noise on all outline vertices.

    Length-proportional mode: see `stroke_translate` for semantics.
    """
    new = data.copy()
    for s in new.strokes:
        length = _polygon_extent(s.polygon)
        eff = _resolve_outline_std(std, std_ratio, length, std_min, std_max)
        noise = rng.normal(0.0, eff, size=s.polygon.shape)
        s.polygon = s.polygon + noise
    return new


@_register_op("endpoint_jitter")
def endpoint_jitter(data, rng, *, std=6.0, n_per_end=3,
                      std_ratio=None,
                      std_min=0.0, std_max=float("inf")) -> OutlineStrokeData:
    """Heuristic endpoint jitter on outline polygons.

    There is no explicit "endpoint" on a closed outline; we approximate by
    finding the bbox long axis and jittering the `n_per_end` vertices at each
    extreme along that axis. Length-proportional mode: see other ops.
    """
    new = data.copy()
    for s in new.strokes:
        pts = s.polygon
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        axis = 0 if x_range >= y_range else 1
        length = max(x_range, y_range)
        eff = _resolve_outline_std(std, std_ratio, length, std_min, std_max)
        order_idx = np.argsort(pts[:, axis])
        low = order_idx[:n_per_end]
        high = order_idx[-n_per_end:]
        for idx in list(low) + list(high):
            pts[idx, 0] += float(rng.normal(0.0, eff))
            pts[idx, 1] += float(rng.normal(0.0, eff))
    return new


@_register_op("drop_stroke")
def drop_stroke(data, rng, *, prob_per_stroke=0.05) -> OutlineStrokeData:
    kept = [s for s in data.strokes if rng.random() > prob_per_stroke]
    return OutlineStrokeData(char=data.char, viewbox=data.viewbox, strokes=kept)


@_register_op("width_jitter")
def width_jitter(data, rng, *, radius_std=1.2, mean=0.0,
                  min_radius=-999.0, max_radius=999.0) -> OutlineStrokeData:
    """Set per-stroke morphological dilate/erode radius. Applied at render.

    Because thin strokes (1–2 px at render scale) can vanish under even a
    `MinFilter(5)` (radius −2), callers typically want to either bias toward
    dilate (`mean` > 0) or cap erode magnitude (`min_radius=-1`).

    Parameters
    ----------
    radius_std : float
        Gaussian std of sampled radius (in pixels at render scale).
    mean : float
        Distribution center. Positive = dilate-leaning.
    min_radius, max_radius : float
        Post-sample clamp; use to bound erosion/dilation.
    """
    new = data.copy()
    for s in new.strokes:
        r = float(rng.normal(mean, radius_std))
        r = max(min_radius, min(max_radius, r))
        s.dilate_radius = r
    return new


def apply_stroke_ops(data: OutlineStrokeData, ops: list[dict],
                      rng: np.random.Generator) -> OutlineStrokeData:
    for raw in ops or []:
        spec = dict(raw)
        name = spec.pop("op")
        if name not in OUTLINE_OPS:
            raise ValueError(
                f"unknown outline stroke op: {name!r}. "
                f"known: {sorted(OUTLINE_OPS)}"
            )
        prob = spec.pop("prob", 1.0)
        if rng.random() > prob:
            continue
        data = OUTLINE_OPS[name](data, rng, **spec)
    return data


# ---------- rasterizer ------------------------------------------------------


# Imported from pipeline so PAD adjustments stay single-source-of-truth.
from pipeline import CANVAS, PAD  # noqa: E402


def rasterize_outlines(data: OutlineStrokeData,
                        canvas: int = CANVAS, pad: int = PAD) -> Image.Image:
    """Render outline polygons into an L-mode glyph mask.

    Fits the viewBox into (canvas - 2*pad) preserving aspect; centers the
    glyph. If any stroke carries a non-negligible `dilate_radius`, each stroke
    is drawn separately and dilated/eroded before unioning. Otherwise a single
    pass polygon-fill is used (faster common case).
    """
    vb_w, vb_h = data.viewbox
    target = canvas - 2 * pad
    scale = min(target / vb_w, target / vb_h)
    off_x = (canvas - vb_w * scale) / 2
    off_y = (canvas - vb_h * scale) / 2

    has_width_jitter = any(abs(s.dilate_radius) > 0.5 for s in data.strokes)

    if not has_width_jitter:
        img = Image.new("L", (canvas, canvas), 0)
        draw = ImageDraw.Draw(img)
        for s in data.strokes:
            pts = s.polygon
            if pts.shape[0] < 3:
                continue
            screen = [(float(x * scale + off_x), float(y * scale + off_y))
                      for x, y in pts]
            draw.polygon(screen, fill=255)
        return img

    final = Image.new("L", (canvas, canvas), 0)
    for s in data.strokes:
        pts = s.polygon
        if pts.shape[0] < 3:
            continue
        screen = [(float(x * scale + off_x), float(y * scale + off_y))
                  for x, y in pts]
        layer = Image.new("L", (canvas, canvas), 0)
        ImageDraw.Draw(layer).polygon(screen, fill=255)
        r = int(round(s.dilate_radius))
        if r >= 1:
            layer = layer.filter(ImageFilter.MaxFilter(2 * r + 1))
        elif r <= -1:
            layer = layer.filter(ImageFilter.MinFilter(2 * (-r) + 1))
        final = ImageChops.lighter(final, layer)
    return final


# ---------- source class & discovery ----------------------------------------


@dataclass
class OutlineStrokeSource:
    """Glyph source backed by per-stroke outline polygons.

    `base_data` carries the character's strokes in viewBox coords. Per-render
    we clone, apply stroke ops, then rasterize.
    """
    base_data: OutlineStrokeData
    stroke_ops: list[dict] = field(default_factory=list)
    canvas: int = CANVAS
    pad: int = PAD
    source_tag: str = "outline"     # set by the discover function

    @property
    def kind(self) -> str:
        return self.source_tag

    def render_mask(self, char: str, rng: np.random.Generator) -> Image.Image | None:
        if char != self.base_data.char:
            return None
        varied = apply_stroke_ops(self.base_data, self.stroke_ops, rng)
        if not varied.strokes:
            return None
        return rasterize_outlines(varied, canvas=self.canvas, pad=self.pad)

    def tag(self) -> str:
        return f"{self.source_tag}-{self.base_data.char}"


DEFAULT_EHANJA_MANIFEST = Path(
    "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/"
    "db_src/e-hanja_online/strokes_animated.jsonl"
)

DEFAULT_MMH_GRAPHICS = Path(
    "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/"
    "db_src/MAKEMEAHANZI/graphics.txt"
)


def discover_ehanja_outline_sources(char: str,
                                      manifest_path: Path | None = None,
                                      stroke_ops: list[dict] | None = None,
                                      canvas: int = CANVAS,
                                      pad: int = PAD) -> list[OutlineStrokeSource]:
    path = manifest_path or DEFAULT_EHANJA_MANIFEST
    data = load_ehanja_outline(path, char)
    if data is None:
        return []
    return [OutlineStrokeSource(
        base_data=data,
        stroke_ops=list(stroke_ops or []),
        canvas=canvas,
        pad=pad,
        source_tag="ehanja_stroke",
    )]


def discover_mmh_outline_sources(char: str,
                                   graphics_path: Path | None = None,
                                   stroke_ops: list[dict] | None = None,
                                   canvas: int = CANVAS,
                                   pad: int = PAD) -> list[OutlineStrokeSource]:
    path = graphics_path or DEFAULT_MMH_GRAPHICS
    data = load_mmh_outline(path, char)
    if data is None:
        return []
    return [OutlineStrokeSource(
        base_data=data,
        stroke_ops=list(stroke_ops or []),
        canvas=canvas,
        pad=pad,
        source_tag="mmh_stroke",
    )]
