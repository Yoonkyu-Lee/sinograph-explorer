"""TrueType outline extraction for CUDA rasterizer (Phase OPT-2.1, doc/20).

For a given (font_path, face_index, char, canvas_size, pad), produce a
flat array of line-segment edges in pixel coordinates (top-left origin).
Bezier curves (quadratic + cubic) are recursively subdivided to a flatness
tolerance. Result is cached per (font, face, char_code, canvas, pad).

The glyph is sized + centered to match v2's `FontSource.render_mask`
binary-search policy so CUDA-rasterized output occupies the same region
as PIL output (parity for downstream pipeline).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import freetype
import numpy as np
from PIL import ImageFont

# Default flatness tolerance for Bezier subdivision (pixels).
DEFAULT_TOL = 0.5


@dataclass
class OutlineData:
    """Flat polygon edges for one glyph at one render size."""
    edges: np.ndarray            # (M, 4) float32 (x0, y0, x1, y1) px, top-left origin
    canvas_size: int
    glyph_size: int              # font pixel size used (post binary search)
    bbox_centered: tuple         # final glyph bbox in pixel coords (x0,y0,x1,y1)
    char: str
    font_tag: str

    @property
    def n_edges(self) -> int:
        return 0 if self.edges is None else int(self.edges.shape[0])


# ---------- Bezier flatten ----------

def _flatten_quadratic(p0, p1, p2, tol2, out):
    """Adaptive subdivision until perpendicular distance from p1 to line
    p0-p2 is below sqrt(tol2). out collects (x, y) endpoints (excluding p0)."""
    dx = p2[0] - p0[0]
    dy = p2[1] - p0[1]
    cross = (p1[0] - p0[0]) * dy - (p1[1] - p0[1]) * dx
    length_sq = dx * dx + dy * dy
    # squared perpendicular distance: cross^2 / length_sq
    if length_sq <= 1e-12 or cross * cross <= tol2 * length_sq:
        out.append(p2)
        return
    m01 = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
    m12 = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    m_mid = ((m01[0] + m12[0]) * 0.5, (m01[1] + m12[1]) * 0.5)
    _flatten_quadratic(p0, m01, m_mid, tol2, out)
    _flatten_quadratic(m_mid, m12, p2, tol2, out)


def _flatten_cubic(p0, p1, p2, p3, tol2, out):
    """Adaptive subdivision for cubic Beziers (CFF fonts)."""
    # Distance of p1, p2 from line p0-p3
    dx = p3[0] - p0[0]
    dy = p3[1] - p0[1]
    length_sq = dx * dx + dy * dy
    c1 = (p1[0] - p0[0]) * dy - (p1[1] - p0[1]) * dx
    c2 = (p2[0] - p0[0]) * dy - (p2[1] - p0[1]) * dx
    max_c = max(c1 * c1, c2 * c2)
    if length_sq <= 1e-12 or max_c <= tol2 * length_sq:
        out.append(p3)
        return
    m01 = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
    m12 = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    m23 = ((p2[0] + p3[0]) * 0.5, (p2[1] + p3[1]) * 0.5)
    m012 = ((m01[0] + m12[0]) * 0.5, (m01[1] + m12[1]) * 0.5)
    m123 = ((m12[0] + m23[0]) * 0.5, (m12[1] + m23[1]) * 0.5)
    m_mid = ((m012[0] + m123[0]) * 0.5, (m012[1] + m123[1]) * 0.5)
    _flatten_cubic(p0, m01, m012, m_mid, tol2, out)
    _flatten_cubic(m_mid, m123, m23, p3, tol2, out)


# ---------- outline → edges ----------

def _walk_outline(outline, tol: float) -> list[tuple[float, float, float, float]]:
    """Walk the outline via freetype.decompose callbacks; returns line edges
    in font-design coordinates (still 26.6 fixed-point divided by 64)."""
    tol2 = tol * tol
    state = {"current": None, "start": None, "edges": []}

    def _to_px(v):
        # freetype FT_Vector → float pixel coords
        return (v.x / 64.0, v.y / 64.0)

    def move_to(a, _ctx):
        state["current"] = _to_px(a)
        state["start"] = state["current"]
        return 0

    def line_to(a, _ctx):
        cur = state["current"]
        nxt = _to_px(a)
        state["edges"].append((cur[0], cur[1], nxt[0], nxt[1]))
        state["current"] = nxt
        return 0

    def conic_to(ctrl, end, _ctx):
        cur = state["current"]
        c = _to_px(ctrl)
        e = _to_px(end)
        flat = [cur]
        _flatten_quadratic(cur, c, e, tol * tol, flat)
        for i in range(len(flat) - 1):
            state["edges"].append((flat[i][0], flat[i][1], flat[i + 1][0], flat[i + 1][1]))
        state["current"] = e
        return 0

    def cubic_to(c1, c2, end, _ctx):
        cur = state["current"]
        a = _to_px(c1)
        b = _to_px(c2)
        e = _to_px(end)
        flat = [cur]
        _flatten_cubic(cur, a, b, e, tol * tol, flat)
        for i in range(len(flat) - 1):
            state["edges"].append((flat[i][0], flat[i][1], flat[i + 1][0], flat[i + 1][1]))
        state["current"] = e
        return 0

    outline.decompose(state,
                       move_to=move_to,
                       line_to=line_to,
                       conic_to=conic_to,
                       cubic_to=cubic_to)
    return state["edges"]


# ---------- size search + centering (uses PIL.ImageFont — mirrors v2 base_source) ----------

def _pil_search_size_and_offset(font_path: str, face_index: int, char: str,
                                  canvas_size: int, pad: int):
    """Reproduce v2 FontSource.render_mask's size+placement using PIL —
    PIL's bbox semantics (layout bbox with side-bearings) differ subtly
    from freetype's pure outline get_bbox. Using PIL here keeps the CUDA
    rasterizer pixel-aligned with the existing PIL code path.

    Returns (font_size_px, x_off_px, y_off_px) such that the freetype
    outline rendered at `font_size_px` and translated by (x_off, y_off)
    occupies the same canvas region as PIL would draw it.
    """
    target = canvas_size - 2 * pad
    lo, hi = 16, canvas_size
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(font_path, mid, index=face_index)
        except Exception:
            return 64, 0.0, 0.0
        bbox = font.getbbox(char)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= target and h <= target:
            best = (mid, bbox)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        font = ImageFont.truetype(font_path, 64, index=face_index)
        best = (64, font.getbbox(char))
    size, bbox = best
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # PIL's ImageDraw.text draws starting at (x_top_left, y_top_left). The
    # glyph's pixel content occupies (x + bbox[0], y + bbox[1]) → (x +
    # bbox[2], y + bbox[3]). So x = (canvas - w)//2 - bbox[0] places the
    # top-left of the glyph at (canvas - w)//2 from canvas left.
    x_anchor = (canvas_size - w) // 2 - bbox[0]
    y_anchor = (canvas_size - h) // 2 - bbox[1]
    # In PIL: glyph appears in top-left coord system at (x_anchor + bbox[0],
    # y_anchor + bbox[1]). The freetype outline is in baseline-relative coords
    # (y up). We'll transform raw freetype output to top-left using face metrics
    # below.
    return size, float(x_anchor), float(y_anchor), bbox


def _font_tag(font_path: Path, face_index: int) -> str:
    return f"{Path(font_path).stem}-{face_index}"


# ---------- public: cached outline extraction ----------

# The (font_path, face_index, char_code, canvas, pad) tuple is the cache key.
# A typical 76 k class corpus with ~10 fonts each → up to 760 k entries; LRU
# size 200 k keeps memory in check (each OutlineData ~ 1-3 KB) while still
# hitting hot fonts.
_CACHE_SIZE = 200_000


def _binary_search_freetype(face: freetype.Face, char: str,
                              target_px: int, max_size: int) -> int:
    """Largest freetype-no-hinting pixel size whose outline bbox fits target."""
    lo, hi = 16, max_size
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        face.set_pixel_sizes(0, mid)
        try:
            face.load_char(char,
                            freetype.FT_LOAD_NO_BITMAP | freetype.FT_LOAD_NO_HINTING)
        except Exception:
            return best
        cbox = face.glyph.outline.get_bbox()
        w = (cbox.xMax - cbox.xMin) / 64.0
        h = (cbox.yMax - cbox.yMin) / 64.0
        if w <= target_px and h <= target_px:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


@lru_cache(maxsize=_CACHE_SIZE)
def _get_outline_cached(font_path_str: str, face_index: int, char_code: int,
                         canvas_size: int, pad: int, tol: float) -> OutlineData:
    """Extract glyph outline at a freetype-driven size + center geometrically.

    NOTE: We deliberately use freetype's outline.get_bbox (no hinting) for
    sizing instead of PIL's hinted getbbox. Reason: the CUDA rasterizer
    consumes the SAME outlines at runtime, so freetype-self-consistent
    placement is what matters. Pixel parity with PIL is sacrificed (IoU vs
    PIL ~0.6-0.8 for complex glyphs due to hinting+side-bearing differences),
    but the rasterizer output is internally consistent and the augment chain
    is robust to small position offsets. See doc/20 §5.5.
    """
    char = chr(char_code)
    face = freetype.Face(font_path_str, face_index)
    target_px = canvas_size - 2 * pad
    size = _binary_search_freetype(face, char, target_px, max_size=canvas_size)
    face.set_pixel_sizes(0, size)
    face.load_char(char, freetype.FT_LOAD_NO_BITMAP | freetype.FT_LOAD_NO_HINTING)
    outline = face.glyph.outline
    cbox = outline.get_bbox()
    glyph_w = (cbox.xMax - cbox.xMin) / 64.0
    glyph_h = (cbox.yMax - cbox.yMin) / 64.0

    # geometrically center the outline bbox in the canvas
    target_x_min = (canvas_size - glyph_w) / 2.0
    target_y_min = (canvas_size - glyph_h) / 2.0
    cbox_xmin = cbox.xMin / 64.0
    cbox_ymax = cbox.yMax / 64.0

    raw_edges = _walk_outline(outline, tol)
    if not raw_edges:
        edges = np.zeros((0, 4), dtype=np.float32)
        bbox = (0.0, 0.0, 0.0, 0.0)
    else:
        arr = np.asarray(raw_edges, dtype=np.float32)
        # freetype (y-up) → top-left pixel coords (y-down).
        #   px = target_x_min + (fx - cbox.xMin)
        #   py = target_y_min + (cbox.yMax - fy)
        arr[:, 0] = arr[:, 0] - cbox_xmin + target_x_min
        arr[:, 2] = arr[:, 2] - cbox_xmin + target_x_min
        arr[:, 1] = cbox_ymax - arr[:, 1] + target_y_min
        arr[:, 3] = cbox_ymax - arr[:, 3] + target_y_min
        # Lab 8 prep: sort edges by y_min (= min(y0, y1)) so the v2 GPU
        # kernel can early-exit when the current chunk's leading edge
        # passes the tile bound. Stable sort keeps original within-y order.
        y_min = np.minimum(arr[:, 1], arr[:, 3])
        order = np.argsort(y_min, kind="stable")
        edges = np.ascontiguousarray(arr[order])
        bbox = (
            float(target_x_min),
            float(target_y_min),
            float(target_x_min + glyph_w),
            float(target_y_min + glyph_h),
        )

    return OutlineData(
        edges=edges,
        canvas_size=canvas_size,
        glyph_size=size,
        bbox_centered=bbox,
        char=char,
        font_tag=_font_tag(Path(font_path_str), face_index),
    )


def get_outline(font_path: Path | str, face_index: int, char: str,
                canvas_size: int, pad: int,
                tol: float = DEFAULT_TOL) -> OutlineData:
    """Extract glyph outline as a flat edge list (px coords, top-left origin).

    Cached per (font, face_idx, char_code, canvas, pad, tol). Typical hit rate
    in a corpus run is near-100 % since the same glyph is re-rendered many
    times across samples.
    """
    return _get_outline_cached(str(Path(font_path)), int(face_index),
                                int(ord(char)), int(canvas_size), int(pad),
                                float(tol))


def cache_info() -> dict:
    info = _get_outline_cached.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "size": info.currsize,
        "maxsize": info.maxsize,
    }


def cache_clear() -> None:
    _get_outline_cached.cache_clear()
