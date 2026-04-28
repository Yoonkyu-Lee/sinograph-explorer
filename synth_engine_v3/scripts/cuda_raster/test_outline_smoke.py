"""Smoke test outline_cache vs PIL — extract a glyph outline, rasterize via
CPU numpy point-in-polygon (even-odd rule), compare to PIL.ImageDraw text
mask via IoU. Verifies that:
    (a) freetype outline extraction works
    (b) Bezier flatten produces correct contours
    (c) coordinate / centering matches PIL closely enough for IoU > 0.95

This is a CPU-only test before the CUDA kernel arrives — it uses the same
edge data the kernel will consume but with a slow numpy reference impl.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))   # synth_engine_v3/scripts
sys.path.insert(0, str(HERE))          # this dir

from outline_cache import get_outline                              # noqa: E402


CANVAS = 384
PAD = 48


def render_pil_at_size(font_path: Path, face_index: int, char: str,
                         size: int, target_x_min: float, target_y_min: float) -> np.ndarray:
    """Render PIL reference whose **ink bbox** lands at (target_x_min,
    target_y_min). PIL's `font.getbbox` returns the advance / layout bbox,
    not the actual ink extent — probe once at a known origin to learn the
    ink-relative-to-origin offset, then place accordingly.
    """
    font = ImageFont.truetype(str(font_path), size, index=face_index)
    probe = Image.new("L", (CANVAS + 200, CANVAS + 200), 0)
    PROBE_ORIG = 100
    ImageDraw.Draw(probe).text((PROBE_ORIG, PROBE_ORIG), char, fill=255, font=font)
    probe_arr = np.asarray(probe)
    ys, xs = np.where(probe_arr > 127)
    if ys.size == 0:
        return np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    ink_offset_x = int(xs.min()) - PROBE_ORIG
    ink_offset_y = int(ys.min()) - PROBE_ORIG
    # Re-render onto CANVAS-sized canvas with origin chosen so ink lands at
    # (target_x_min, target_y_min)
    x_anchor = int(round(target_x_min)) - ink_offset_x
    y_anchor = int(round(target_y_min)) - ink_offset_y
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    ImageDraw.Draw(mask).text((x_anchor, y_anchor), char, fill=255, font=font)
    return np.asarray(mask)


def render_pil_v2_style(font_path: Path, face_index: int, char: str) -> np.ndarray:
    """Original v2 FontSource.render_mask path (PIL-driven binary search +
    PIL bbox centering) — used purely for visual comparison."""
    target = CANVAS - 2 * PAD
    lo, hi = 16, CANVAS
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(str(font_path), mid, index=face_index)
        except Exception:
            return np.zeros((CANVAS, CANVAS), dtype=np.uint8)
        bbox = font.getbbox(char)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= target and h <= target:
            best = (font, bbox)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        font = ImageFont.truetype(str(font_path), 64, index=face_index)
        best = (font, font.getbbox(char))
    font, bbox = best
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(mask)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (CANVAS - w) // 2 - bbox[0]
    y = (CANVAS - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    return np.asarray(mask)


def rasterize_numpy_pip(edges: np.ndarray, H: int, W: int) -> np.ndarray:
    """Reference rasterization via per-pixel point-in-polygon (even-odd).

    For each pixel center (x+0.5, y+0.5), count edges crossed by horizontal
    ray going right; odd count → inside.

    Slow O(H*W*E) but vectorized in numpy. Used only for parity testing.
    """
    if len(edges) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    # pixel centers
    xs = np.arange(W, dtype=np.float32) + 0.5
    ys = np.arange(H, dtype=np.float32) + 0.5
    # E edges
    x0 = edges[:, 0]; y0 = edges[:, 1]
    x1 = edges[:, 2]; y1 = edges[:, 3]
    # crossing test (vectorize per scanline to keep memory tractable)
    out = np.zeros((H, W), dtype=np.uint8)
    for yi, py in enumerate(ys):
        # which edges does horizontal y=py cross?
        # condition: (y0 <= py) != (y1 <= py)  — strict inequality on one side
        cross_y = (y0 <= py) != (y1 <= py)
        if not cross_y.any():
            continue
        ye_a = y0[cross_y]; ye_b = y1[cross_y]
        xe_a = x0[cross_y]; xe_b = x1[cross_y]
        # x intersection at y=py
        t = (py - ye_a) / (ye_b - ye_a + 1e-12)
        xi = xe_a + t * (xe_b - xe_a)               # (E_active,)
        # for each pixel x, count xi <= x
        cnt = (xi[None, :] < xs[:, None]).sum(axis=1)
        out[yi] = (cnt % 2 == 1).astype(np.uint8) * 255
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a > 127
    b_b = b > 127
    inter = (a_b & b_b).sum()
    union = (a_b | b_b).sum()
    return float(inter) / float(max(union, 1))


def _win_fonts() -> Path:
    if sys.platform.startswith("win"):
        return Path("C:/Windows/Fonts")
    if Path("/mnt/c/Windows/Fonts").is_dir():
        return Path("/mnt/c/Windows/Fonts")
    return Path("/usr/share/fonts/truetype")


def main() -> None:
    fdir = _win_fonts()
    cases = [
        (fdir / "batang.ttc", 0, "鑑"),
        (fdir / "batang.ttc", 0, "金"),
        (fdir / "batang.ttc", 0, "媤"),
        (fdir / "batang.ttc", 0, "一"),
        (fdir / "malgun.ttf", 0, "鑑"),
        (fdir / "simsun.ttc", 0, "金"),
    ]

    out_dir = Path(__file__).parent / "out_outline_smoke"
    out_dir.mkdir(exist_ok=True)

    for font_path, face_idx, char in cases:
        if not font_path.exists():
            print(f"  [skip] {font_path} not present")
            continue
        # outline → numpy raster (the "truth" for the CUDA path)
        od = get_outline(font_path, face_idx, char, CANVAS, PAD)
        np_mask = rasterize_numpy_pip(od.edges, CANVAS, CANVAS)
        # PIL reference at the SAME freetype-determined size + same centering
        pil_ref = render_pil_at_size(
            font_path, face_idx, char, od.glyph_size,
            od.bbox_centered[0], od.bbox_centered[1],
        )
        # PIL v2-style (different size policy) — visual comparison only
        pil_v2 = render_pil_v2_style(font_path, face_idx, char)

        score_self = iou(pil_ref, np_mask)              # outline vs PIL @ same size
        score_v2 = iou(pil_v2, np_mask)                  # outline vs v2 baseline
        tag = f"{font_path.stem}_{face_idx}_U+{ord(char):04X}"
        print(f"  {char}  {tag}  edges={od.n_edges:>4}  size={od.glyph_size:>3}  "
              f"IoU(self)={score_self:.4f}  IoU(v2)={score_v2:.4f}")
        # Save 4-up: [outline-numpy | PIL @ same size | abs diff | v2-style PIL]
        diff = np.abs(np_mask.astype(int) - pil_ref.astype(int)).astype(np.uint8)
        side = np.concatenate([np_mask, pil_ref, diff, pil_v2], axis=1)
        Image.fromarray(side).save(out_dir / f"{tag}.png", compress_level=1)
        # Detail metric: pixel agreement %, also count edge-band pixels
        agree = float((np_mask > 127) == (pil_ref > 127)).mean() if False else \
                 float(((np_mask > 127) == (pil_ref > 127)).sum() / np_mask.size)
        # Centroid distance — quick offset diagnostic
        def _centroid(m):
            ys, xs = np.where(m > 127)
            if ys.size == 0: return (0.0, 0.0)
            return (float(xs.mean()), float(ys.mean()))
        cn = _centroid(np_mask); cp = _centroid(pil_ref)
        dx = cn[0] - cp[0]; dy = cn[1] - cp[1]
        print(f"      pixel-agree={agree:.4f}  centroid_diff=({dx:+.2f}, {dy:+.2f})")

    print("\nLayout in out_outline_smoke/*: [outline_raster | PIL @ same size | v2-style PIL]")
    print("IoU(self) = parity vs PIL at outline_cache's freetype-driven size  → CUDA target")
    print("IoU(v2)   = parity vs v2 PIL path                                  → reference only")


if __name__ == "__main__":
    main()
