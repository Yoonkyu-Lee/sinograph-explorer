"""Phase 4 — verify source-aware per-sample gating.

Same seed, same masks, split batch:
  first half tagged source_kind="font"
  second half tagged source_kind="svg_stroke"

Apply elastic with skip_if_kinds=["svg_stroke"] and only elastic changes.

Expect: second-half canvas == canvas-before-elastic; first-half differs.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

from mask_adapter import get_font_sources_for, render_mask, masks_to_tensor
from pipeline_gpu import CANVAS, GPUContext, run_block
import style_gpu  # noqa
import augment_gpu  # noqa


CHARS = ["鑑", "學", "学", "斈", "媤", "乶", "畓", "裡", "裏", "あ", "カ", "한"]


def build_masks(chars, rng, device):
    masks = []
    for ch in chars:
        srcs = get_font_sources_for(ch)
        masks.append(render_mask(ch, rng.choice(srcs)) if srcs else None)
    return masks_to_tensor(masks, device=device)


def main():
    device = "cuda"
    seed = 123
    py_rng = random.Random(seed)
    mask_t = build_masks(CHARS * 2, py_rng, device)  # 24 masks
    N = mask_t.shape[0]
    half = N // 2

    # Tag first half as font, second half as svg_stroke.
    kinds = ["font"] * half + ["svg_stroke"] * (N - half)

    # Start with a uniform gray canvas so elastic warp is visible.
    base_canvas = torch.full((N, 3, CANVAS, CANVAS), 0.5, device=device)
    base_canvas = base_canvas * (1.0 - mask_t) + 0.1 * mask_t  # dark glyph

    # --- Run A: identical pipeline, but elastic is gated by skip_if_kinds.
    gen_A = torch.Generator(device=device).manual_seed(seed)
    ctx_A = GPUContext(canvas=base_canvas.clone(), mask=mask_t.clone(),
                       rng=gen_A, chars=CHARS * 2, source_kinds=kinds, device=device)
    specs = [{"op": "augment.elastic", "alpha": [12, 20], "sigma": [5, 7],
              "skip_if_kinds": ["svg_stroke"]}]
    ctx_A = run_block(ctx_A, specs)

    # --- Compare:
    #     svg_stroke half must equal base_canvas (elastic skipped)
    #     font half must differ from base_canvas (elastic applied)
    diff = (ctx_A.canvas - base_canvas).abs().amax(dim=(1, 2, 3))  # (N,)
    font_max = diff[:half].max().item()
    svg_max = diff[half:].max().item()
    print(f"per-sample max diff after gated elastic:")
    print(f"  font  (should be >>0, elastic applied):   max={font_max:.4f}, mean={diff[:half].mean().item():.4f}")
    print(f"  svg_stroke (should be ~0, elastic skip):  max={svg_max:.6f}, mean={diff[half:].mean().item():.6f}")
    assert svg_max < 1e-5, f"svg_stroke samples were MODIFIED (gate broken): max={svg_max}"
    assert font_max > 0.01, f"font samples were NOT modified (elastic no-op): max={font_max}"
    print("phase4 gating OK")

    # --- Also verify only_if_kinds inverse gate
    gen_B = torch.Generator(device=device).manual_seed(seed)
    ctx_B = GPUContext(canvas=base_canvas.clone(), mask=mask_t.clone(),
                       rng=gen_B, chars=CHARS * 2, source_kinds=kinds, device=device)
    specs2 = [{"op": "augment.elastic", "alpha": [12, 20], "sigma": [5, 7],
               "only_if_kinds": ["font"]}]
    ctx_B = run_block(ctx_B, specs2)
    diff2 = (ctx_B.canvas - base_canvas).abs().amax(dim=(1, 2, 3))
    font2 = diff2[:half].max().item()
    svg2 = diff2[half:].max().item()
    print(f"only_if_kinds=['font'] gate:")
    print(f"  font:       max={font2:.4f}")
    print(f"  svg_stroke: max={svg2:.6f}")
    assert svg2 < 1e-5
    assert font2 > 0.01
    print("phase4 only_if_kinds OK")


if __name__ == "__main__":
    main()
