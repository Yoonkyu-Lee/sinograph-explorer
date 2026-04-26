"""Phase 3 smoke — style + augment end-to-end on GPU.

Reuses Phase 2 style block; appends a realistic augment block matching
`full_random_multi.yaml` shape (subset that is ported in Phase 3 MVP).
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
import style_gpu   # noqa: F401
import augment_gpu  # noqa: F401

CHARS_DEFAULT = ["鑑", "學", "学", "斈", "媤", "乶", "畓", "裡", "裏", "あ", "カ", "한"]


def build_batch(chars, rng, device="cuda"):
    masks, tags, kinds = [], [], []
    for ch in chars:
        srcs = get_font_sources_for(ch)
        if not srcs:
            masks.append(None); tags.append(None); kinds.append(""); continue
        src = rng.choice(srcs)
        masks.append(render_mask(ch, src))
        tags.append(src.tag() if masks[-1] is not None else None)
        kinds.append("font" if masks[-1] is not None else "")
    return masks_to_tensor(masks, device=device), tags, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", nargs="+", default=CHARS_DEFAULT)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="synth_engine_v3/out/04_phase3_augment_smoke")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    batch_chars = args.chars * args.reps
    N = len(batch_chars)
    print(f"batch N={N}")

    py_rng = random.Random(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    t0 = time.perf_counter()
    mask_t, tags, kinds = build_batch(batch_chars, py_rng, args.device)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    print(f"mask build:       {(t1-t0)*1000:7.1f} ms")

    canvas = torch.ones(N, 3, CANVAS, CANVAS, device=args.device)
    ctx = GPUContext(canvas=canvas, mask=mask_t, rng=gen,
                     chars=batch_chars, source_kinds=kinds, device=args.device)

    spec = {
        "style": [
            {"layer": "background.solid", "color": (250, 248, 240), "prob": 0.4},
            {"layer": "background.gradient", "start": (255, 255, 230), "end": (180, 210, 240),
             "direction": "vertical", "prob": 0.4},
            {"layer": "background.noise", "scale": 0.7, "smooth": 1.5, "prob": 0.3},
            {"layer": "stroke_weight.dilate", "amount": [0, 1], "prob": 0.5},
            {"layer": "shadow.drop", "offset": [2, 8], "blur": [1.5, 4.0], "opacity": [0.3, 0.6],
             "color": (30, 30, 30), "prob": 0.5},
            {"layer": "fill.hsv_contrast", "saturation": [0.4, 0.9], "value": [0.15, 0.9],
             "min_contrast": 70},
            {"layer": "outline.simple", "color": (20, 20, 20), "width": [1, 2], "prob": 0.3},
            {"layer": "glow.outer", "color": (255, 200, 120), "radius": 4, "blur": 5.0,
             "strength": 0.5, "prob": 0.2},
        ],
        "augment": [
            {"op": "rotate", "angle": [-12, 12], "prob": 0.7},
            {"op": "perspective", "strength": [0.04, 0.15], "prob": 0.4},
            {"op": "scale_translate", "scale": [0.85, 1.05], "translate": [-0.04, 0.04],
             "prob": 0.4},
            {"op": "color_jitter", "brightness": [0.8, 1.2], "contrast": [0.8, 1.2],
             "saturation": [0.7, 1.3]},
            {"op": "gamma", "gamma": [0.85, 1.15]},
            {"op": "gaussian_blur", "sigma": [0.0, 1.5], "prob": 0.5},
            {"op": "gaussian_noise", "std": [0, 8], "prob": 0.5},
            {"op": "shadow_gradient", "strength": [0.1, 0.3], "direction": [0, 360], "prob": 0.3},
            {"op": "vignette", "strength": [0.1, 0.4], "prob": 0.3},
            {"op": "chromatic_aberration", "shift": [0, 2], "prob": 0.2},
            {"op": "lens_distort", "k": [-0.08, 0.08], "prob": 0.2},
            {"op": "elastic", "alpha": [3, 8], "sigma": [5, 7], "prob": 0.25},
            {"op": "jpeg", "quality": [40, 95], "prob": 0.4},
            {"op": "low_light", "brightness": [0.35, 0.65], "noise_std": [4, 12], "prob": 0.1},
        ],
    }

    # warmup (kornia graph compile, etc.)
    _ = run_pipeline(ctx, spec); torch.cuda.synchronize()

    # rebuild canvas for clean measurement (ctx mutated by warmup)
    ctx = GPUContext(canvas=torch.ones(N, 3, CANVAS, CANVAS, device=args.device),
                     mask=mask_t.clone(), rng=gen,
                     chars=batch_chars, source_kinds=kinds, device=args.device)
    t2 = time.perf_counter()
    ctx = run_pipeline(ctx, spec)
    torch.cuda.synchronize(); t3 = time.perf_counter()
    print(f"style+augment:    {(t3-t2)*1000:7.1f} ms   ({N/(t3-t2):7.1f} samples/s)")

    final = finalize_center_crop(ctx.canvas)
    t4 = time.perf_counter()
    imgs = tensor_to_pil_batch(final)
    saved = 0
    for i, (img, ch, tag) in enumerate(zip(imgs, batch_chars, tags)):
        if tag is None: continue
        img.save(out / f"{i:03d}_U+{ord(ch):04X}_{tag}.png", compress_level=1)
        saved += 1
    t5 = time.perf_counter()
    print(f"finalize+save:    {(t5-t4)*1000:7.1f} ms   ({N/(t5-t4):7.1f} samples/s)")
    print(f"end-to-end (ex-mask): {(t5-t2)*1000:7.1f} ms  ({N/(t5-t2):7.1f} samples/s)")
    print(f"saved {saved} PNGs to {out}")
    assert final.shape == (N, 3, 256, 256)
    print("phase3 OK")


if __name__ == "__main__":
    main()
