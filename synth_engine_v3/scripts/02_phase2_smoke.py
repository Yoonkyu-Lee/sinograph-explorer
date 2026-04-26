"""Phase 2 smoke — end-to-end style block on GPU for a small batch.

Pipeline for each sample:
  mask (from v2 font source)
  -> background.solid / gradient / noise (rotated by batch index)
  -> stroke_weight.dilate (small)
  -> shadow.drop
  -> fill.hsv_contrast
  -> outline.simple
  -> glow.outer (prob 0.3)
  => (N, 3, 384, 384) GPU canvas -> center crop 256 -> PNG
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from mask_adapter import get_font_sources_for, render_mask, masks_to_tensor
from pipeline_gpu import (
    CANVAS, GPUContext, finalize_center_crop, tensor_to_pil_batch, run_pipeline,
)
import style_gpu  # noqa: F401 — registers layers


CHARS_DEFAULT = ["鑑", "學", "学", "斈", "媤", "乶", "畓", "裡", "裏", "あ", "カ", "한"]


def build_batch(chars: list[str], rng: random.Random, device: str = "cuda"):
    masks, tags, kinds = [], [], []
    for ch in chars:
        srcs = get_font_sources_for(ch)
        if not srcs:
            masks.append(None); tags.append(None); kinds.append(""); continue
        src = rng.choice(srcs)
        masks.append(render_mask(ch, src))
        tags.append(src.tag() if masks[-1] is not None else None)
        kinds.append("font" if masks[-1] is not None else "")
    mask_t = masks_to_tensor(masks, device=device)
    return mask_t, tags, kinds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", nargs="+", default=CHARS_DEFAULT)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="synth_engine_v3/out/03_phase2_style_smoke")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    batch_chars = args.chars * args.reps
    N = len(batch_chars)
    print(f"batch N={N}")

    py_rng = random.Random(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    # 1) masks
    t0 = time.perf_counter()
    mask_t, tags, kinds = build_batch(batch_chars, py_rng, device=args.device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"mask build: {(t1-t0)*1000:.1f} ms")

    # 2) canvas init (white)
    canvas = torch.ones(N, 3, CANVAS, CANVAS, device=args.device)
    ctx = GPUContext(canvas=canvas, mask=mask_t, rng=gen, chars=batch_chars, source_kinds=kinds, device=args.device)

    # 3) spec — 3 varieties rotated by sample index
    spec = {
        "style": [
            # each background has prob 1/3; only one fires per sample
            {"layer": "background.solid", "color": (250, 248, 240), "prob": 0.34},
            {"layer": "background.gradient", "start": (255, 255, 230), "end": (180, 210, 240),
             "direction": "vertical", "prob": 0.5},
            {"layer": "background.noise", "scale": 0.8, "smooth": 1.5, "prob": 0.5},
            {"layer": "stroke_weight.dilate", "amount": [0, 1], "prob": 0.5},
            {"layer": "shadow.drop", "offset": [2, 8], "blur": [1.5, 4.0], "opacity": [0.3, 0.6],
             "color": (30, 30, 30)},
            {"layer": "fill.hsv_contrast", "saturation": [0.4, 0.9], "value": [0.15, 0.9],
             "min_contrast": 70},
            {"layer": "outline.simple", "color": (20, 20, 20), "width": [1, 2], "prob": 0.5},
            {"layer": "glow.outer", "color": (255, 200, 120), "radius": 4, "blur": 5.0,
             "strength": 0.5, "prob": 0.3},
        ],
    }

    t2 = time.perf_counter()
    ctx = run_pipeline(ctx, spec)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    print(f"style pipeline: {(t3-t2)*1000:.1f} ms ({N/(t3-t2):.1f} samples/s)")

    # 4) finalize + save
    final = finalize_center_crop(ctx.canvas)
    imgs = tensor_to_pil_batch(final)
    saved = 0
    for i, (img, ch, tag) in enumerate(zip(imgs, batch_chars, tags)):
        if tag is None:
            continue
        notation = f"U+{ord(ch):04X}"
        img.save(out / f"{i:03d}_{notation}_{tag}.png", compress_level=1)
        saved += 1
    print(f"saved {saved} PNGs to {out}")

    # sanity
    assert final.shape == (N, 3, 256, 256)
    assert final.min() >= 0 and final.max() <= 1
    print("phase2 OK")


if __name__ == "__main__":
    main()
