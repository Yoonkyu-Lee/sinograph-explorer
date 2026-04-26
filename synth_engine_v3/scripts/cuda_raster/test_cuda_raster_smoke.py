"""End-to-end smoke for CUDA rasterizer (Phase OPT-2.2 verification).

Builds the extension (first run only — slow), rasterizes a small batch of
glyphs, compares to PIL reference at the same freetype-driven size with
ink-bbox alignment. Pass criterion: IoU ≥ 0.85 across the batch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from outline_cache import get_outline                    # noqa: E402
from rasterize import rasterize_batch                    # noqa: E402

CANVAS = 384
PAD = 48


def render_pil_aligned(font_path, face_index, char, size, target_x_min, target_y_min):
    font = ImageFont.truetype(str(font_path), size, index=face_index)
    probe = Image.new("L", (CANVAS + 200, CANVAS + 200), 0)
    ImageDraw.Draw(probe).text((100, 100), char, fill=255, font=font)
    arr = np.asarray(probe)
    ys, xs = np.where(arr > 127)
    if ys.size == 0:
        return np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    ox = int(xs.min()) - 100; oy = int(ys.min()) - 100
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    ImageDraw.Draw(mask).text(
        (int(round(target_x_min)) - ox, int(round(target_y_min)) - oy),
        char, fill=255, font=font,
    )
    return np.asarray(mask)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a > 127
    b_b = b > 127
    inter = (a_b & b_b).sum()
    union = (a_b | b_b).sum()
    return float(inter) / float(max(union, 1))


def main() -> None:
    cases = [
        (Path("C:/Windows/Fonts/batang.ttc"), 0, "鑑"),
        (Path("C:/Windows/Fonts/batang.ttc"), 0, "金"),
        (Path("C:/Windows/Fonts/batang.ttc"), 0, "媤"),
        (Path("C:/Windows/Fonts/batang.ttc"), 0, "一"),
        (Path("C:/Windows/Fonts/malgun.ttf"), 0, "鑑"),
        (Path("C:/Windows/Fonts/simsun.ttc"), 0, "金"),
        (Path("C:/Windows/Fonts/batang.ttc"), 0, "龘"),
        (Path("C:/Windows/Fonts/batang.ttc"), 0, "𤨒"),
    ]

    out_dir = HERE / "out_cuda_smoke"
    out_dir.mkdir(exist_ok=True)

    print("[cuda] extracting outlines…")
    outlines = []
    pil_refs = []
    tags = []
    for font_path, face_idx, char in cases:
        if not font_path.exists():
            continue
        od = get_outline(font_path, face_idx, char, CANVAS, PAD)
        if od.edges is None or len(od.edges) == 0:
            continue
        outlines.append(od)
        ref = render_pil_aligned(font_path, face_idx, char, od.glyph_size,
                                  od.bbox_centered[0], od.bbox_centered[1])
        pil_refs.append(ref)
        tags.append(f"{font_path.stem}_{face_idx}_U+{ord(char):04X}_{char}")

    print(f"[cuda] {len(outlines)} glyphs → rasterizing on GPU…")
    t0 = time.perf_counter()
    masks = rasterize_batch(outlines, CANVAS, CANVAS, device="cuda")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[cuda] rasterize_batch returned {tuple(masks.shape)} in {elapsed*1000:.1f} ms "
          f"(includes lazy build / first call)")

    masks_u8 = (masks.cpu().numpy()[:, 0] * 255).astype(np.uint8)

    print()
    print("Per-glyph IoU vs PIL reference:")
    bad = []
    for i, tag in enumerate(tags):
        score = iou(pil_refs[i], masks_u8[i])
        ok = "OK" if score >= 0.85 else "LOW"
        print(f"  [{ok}] {tag}  IoU={score:.4f}  edges={outlines[i].n_edges}")
        if score < 0.85:
            bad.append((tag, score))
        # save 3-up: cuda | pil | diff
        diff = np.abs(masks_u8[i].astype(int) - pil_refs[i].astype(int)).astype(np.uint8)
        side = np.concatenate([masks_u8[i], pil_refs[i], diff], axis=1)
        Image.fromarray(side).save(out_dir / f"{tag}.png", compress_level=1)

    print()
    print(f"Saved comparisons → {out_dir}")
    if bad:
        print(f"  WARNING: {len(bad)} glyphs below IoU 0.85 threshold")
    else:
        print("  All glyphs passed.")


if __name__ == "__main__":
    main()
