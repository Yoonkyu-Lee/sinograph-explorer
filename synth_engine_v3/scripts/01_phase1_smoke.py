"""Phase 1 smoke — verify v2 mask import + tensor stack + GPU upload.

Picks a handful of characters, renders per-sample font masks on CPU, stacks to
a (N,1,H,W) GPU tensor, and saves both the raw PIL masks and the round-tripped
tensor as PNGs so a human can spot corruption. Also times mask raster + upload.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from PIL import Image

from mask_adapter import batch_render, CANVAS


CHARS_DEFAULT = ["鑑", "學", "学", "斈", "媤", "乶", "畓", "裡", "裏", "あ", "カ", "한"]


def tensor_to_pil(t: torch.Tensor) -> list[Image.Image]:
    """(N,1,H,W) float [0,1] -> list of PIL L-mode images."""
    arr = (t.squeeze(1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(arr[i], mode="L") for i in range(arr.shape[0])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", nargs="+", default=CHARS_DEFAULT)
    ap.add_argument("--reps", type=int, default=3, help="repeat each char to form a larger batch")
    ap.add_argument("--out", default="synth_engine_v3/out/02_phase1_mask_adapter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-batches", type=int, default=5, help="how many batches to time")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    batch_chars = args.chars * args.reps
    print(f"batch size = {len(batch_chars)}; chars = {' '.join(args.chars)}")
    rng = random.Random(args.seed)

    # warmup (font scan cache)
    _ = batch_render(batch_chars, rng=random.Random(args.seed), device="cuda")

    t_raster = 0.0
    t_upload = 0.0
    n = 0
    final_t, final_tags = None, None
    for b in range(args.bench_batches):
        rng_b = random.Random(args.seed + b)
        t0 = time.perf_counter()
        # render all masks CPU-side, then stack
        from mask_adapter import get_font_sources_for, render_mask, masks_to_tensor
        masks, tags = [], []
        for ch in batch_chars:
            srcs = get_font_sources_for(ch)
            if not srcs:
                masks.append(None); tags.append(None); continue
            src = rng_b.choice(srcs)
            masks.append(render_mask(ch, src))
            tags.append(src.tag() if masks[-1] is not None else None)
        t1 = time.perf_counter()
        t_raster += t1 - t0
        mask_t = masks_to_tensor(masks, device="cuda")
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_upload += t2 - t1
        n += len(batch_chars)
        final_t, final_tags = mask_t, tags

    print(f"rasterized {n} masks over {args.bench_batches} batches")
    print(f"  CPU raster:   {t_raster*1000/args.bench_batches:7.2f} ms/batch   ({n/t_raster:7.1f} masks/s, serial)")
    print(f"  upload+stack: {t_upload*1000/args.bench_batches:7.2f} ms/batch   ({n/t_upload:7.1f} masks/s)")
    print(f"final tensor: shape={tuple(final_t.shape)} dtype={final_t.dtype} device={final_t.device} "
          f"min={final_t.min().item():.3f} max={final_t.max().item():.3f} mean={final_t.mean().item():.3f}")

    # save a few for visual check
    pil_masks = tensor_to_pil(final_t)
    saved = 0
    for i, (img, ch, tag) in enumerate(zip(pil_masks, batch_chars, final_tags)):
        if tag is None:
            continue
        notation = f"U+{ord(ch):04X}"
        fname = f"{i:02d}_{notation}_{tag}.png"
        img.save(out / fname)
        saved += 1
    print(f"saved {saved} mask PNGs to {out}")

    # assertion — catch silent failures
    assert final_t.shape[1:] == (1, CANVAS, CANVAS), f"unexpected mask shape {tuple(final_t.shape)}"
    assert final_t.device.type == "cuda"
    assert final_t.max().item() > 0.1, "all-black batch — font scan likely failed"
    print("phase1 OK")


if __name__ == "__main__":
    main()
