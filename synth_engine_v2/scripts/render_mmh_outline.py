"""
One-off: render a character using MakeMeAHanzi's `stroke_svg_paths` (filled
outline mode) instead of `stroke_medians` (centerline polyline mode).

This is the 'font-like' render: actual stroke shapes with natural tapering
at stroke ends, matching the underlying AR PL UMing-style glyphs.
Not integrated into the engine — just a visual reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import parse_path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CANONICAL_JSONL = Path(
    "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/"
    "sinograph_canonical_v1/out/canonical_characters.jsonl"
)

CANVAS = 1024
MMH_Y_PIVOT = 900  # standard MakeMeAHanzi transform


def load_paths(char: str) -> list[str]:
    with CANONICAL_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("character") == char:
                return rec.get("media", {}).get("stroke_svg_paths", []) or []
    return []


def sample_path(path_str: str, samples_per_segment: int = 30) -> list[tuple[float, float]]:
    """Walk each bezier segment, sample points at uniform parameter intervals,
    return MakeMeAHanzi-native coordinates (pre y-flip)."""
    path = parse_path(path_str)
    pts = []
    for seg in path:
        n = max(samples_per_segment, int(seg.length() / 6))
        for t in np.linspace(0, 1, n, endpoint=False):
            p = seg.point(t)
            pts.append((float(p.real), float(p.imag)))
    # close the polygon by adding the last endpoint
    if path:
        last = path[-1].point(1.0)
        pts.append((float(last.real), float(last.imag)))
    return pts


def render_outline(char: str, canvas: int = 256) -> Image.Image:
    paths = load_paths(char)
    if not paths:
        raise SystemExit(f"no stroke_svg_paths for {char!r}")

    # High-res render then downsample for smoother edges
    hi = canvas * 4
    img_hi = Image.new("L", (hi, hi), 0)
    draw = ImageDraw.Draw(img_hi)

    # MakeMeAHanzi coords live in 1024x1024 with y-flip via transform
    scale = hi / 1024.0
    pad = hi / 16  # small padding so the glyph doesn't touch edges
    usable = hi - 2 * pad
    s = usable / 1024.0

    for path_str in paths:
        mmh_pts = sample_path(path_str, samples_per_segment=20)
        if not mmh_pts:
            continue
        # y-flip + scale + center
        screen_pts = [
            (px * s + pad,
             (MMH_Y_PIVOT - py) * s + pad)
            for (px, py) in mmh_pts
        ]
        draw.polygon(screen_pts, fill=255)

    # downsample (antialiasing effect via LANCZOS)
    img = img_hi.resize((canvas, canvas), Image.LANCZOS)
    # Invert: glyph should be black on white
    arr = 255 - np.asarray(img)
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("char")
    ap.add_argument("--out", default=None)
    ap.add_argument("--canvas", type=int, default=256)
    args = ap.parse_args()

    char = args.char
    if len(char) != 1:
        raise SystemExit(f"single char expected, got {char!r}")
    cp_hex = f"{ord(char):X}"

    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "out" / f"U+{cp_hex}" / f"mmh_outline.{cp_hex}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    img = render_outline(char, canvas=args.canvas)
    img.save(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
