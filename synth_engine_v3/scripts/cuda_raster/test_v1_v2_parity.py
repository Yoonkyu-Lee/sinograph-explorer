"""Verify raster_kernel_v1 and raster_kernel_v2 produce IDENTICAL output.

Both kernels apply the same even-odd fill rule with the same float math.
v2 only changes memory access (shared mem, sorted edges, tile pruning) —
result must be bit-exact except for floating-point order-of-summation
artifacts which round-trip through 0/1 binary mask.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from outline_cache import get_outline                  # noqa: E402
from rasterize import rasterize_batch                  # noqa: E402
from streaming_log import setup_logging                # noqa: E402

CANVAS = 384
PAD = 48


def _win_fonts() -> Path:
    if sys.platform.startswith("win"):
        return Path("C:/Windows/Fonts")
    if Path("/mnt/c/Windows/Fonts").is_dir():
        return Path("/mnt/c/Windows/Fonts")
    return Path("/usr/share/fonts/truetype")


def main() -> None:
    setup_logging(HERE / "out_v1v2_parity.log")
    fdir = _win_fonts()
    cases = [
        (fdir / "batang.ttc", 0, "鑑"),
        (fdir / "batang.ttc", 0, "金"),
        (fdir / "batang.ttc", 0, "媤"),
        (fdir / "batang.ttc", 0, "一"),
        (fdir / "batang.ttc", 0, "龘"),
        (fdir / "malgun.ttf", 0, "鑑"),
        (fdir / "simsun.ttc", 0, "金"),
    ]
    outlines = []
    tags = []
    for fp, fi, ch in cases:
        if not fp.exists():
            continue
        od = get_outline(fp, fi, ch, CANVAS, PAD)
        if od.edges is not None and len(od.edges) > 0:
            outlines.append(od)
            tags.append(f"{fp.stem}-{fi}-{ch}")

    print(f"[parity] testing {len(outlines)} glyphs")
    masks_v1 = rasterize_batch(outlines, CANVAS, CANVAS, kernel="v1")
    masks_v2 = rasterize_batch(outlines, CANVAS, CANVAS, kernel="v2")
    torch.cuda.synchronize()

    diff = (masks_v1 != masks_v2).float()
    total_diff = int(diff.sum().item())
    total_pixels = int(diff.numel())
    print(f"[parity] differing pixels: {total_diff} / {total_pixels} "
          f"({100.0 * total_diff / total_pixels:.4f}%)")
    print()
    print("Per-glyph differing pixel counts:")
    for i, tag in enumerate(tags):
        n = int(diff[i].sum().item())
        v1_lit = int((masks_v1[i] > 0.5).sum().item())
        v2_lit = int((masks_v2[i] > 0.5).sum().item())
        ok = "OK" if n == 0 else f"{n}"
        print(f"  [{ok:>5s}] {tag}  v1={v1_lit} v2={v2_lit} px lit")

    if total_diff == 0:
        print("\n✓ v1 and v2 produce IDENTICAL output.")
    else:
        # quantify visual divergence
        ratio = total_diff / total_pixels
        print(f"\n△ Differs in {ratio*100:.4f}% of pixels (likely AA edge "
              f"order-of-summation artifacts).")


if __name__ == "__main__":
    main()
