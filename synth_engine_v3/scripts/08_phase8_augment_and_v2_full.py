"""Phase 8 smoke — verify the 5 newly-ported augment ops (motion_blur,
salt_pepper_noise, paper_texture, ink_bleed, binarize) + run v2's
`full_random_multi.yaml` end-to-end through v3 with ZERO layer/op stripping.

Success criteria:
  1. Every op in v2's full_random_multi.yaml resolves to a v3 REGISTRY entry.
  2. 40-sample mixed-kind batch renders to PNG without exception.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from mask_adapter import (
    CANVAS, batch_render_from_spec, get_sources_for_char, masks_to_tensor, render_mask,
)
from pipeline_gpu import GPUContext, finalize_center_crop, run_pipeline, tensor_to_pil_batch, REGISTRY
import style_gpu    # noqa: F401
import augment_gpu  # noqa: F401


CHARS = ["鑑", "學", "学", "斈", "裡", "裏", "媤", "乶", "畓", "裡"]
NEW_OPS = ["motion_blur", "salt_pepper_noise", "paper_texture", "ink_bleed", "binarize"]


def per_op_probe(out_dir: Path, seed: int) -> None:
    device = "cuda"
    rng_np = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    masks, kinds = [], []
    for ch in CHARS:
        srcs = get_sources_for_char(ch, {"kind": "font"})
        src = srcs[0] if srcs else None
        m = render_mask(ch, src, rng=rng_np) if src is not None else None
        masks.append(m); kinds.append("font" if m is not None else "")
    mask_t = masks_to_tensor(masks, device=device)
    N = mask_t.shape[0]

    # base: bg solid + fill.hsv_contrast → something non-trivial to degrade
    base_style = [
        {"layer": "background.solid", "color": (255, 255, 255)},
        {"layer": "fill.hsv_contrast", "saturation": [0.5, 0.9], "value": [0.1, 0.6],
         "min_contrast": 80},
    ]

    probe_specs = {
        "motion_blur": {"op": "motion_blur", "kernel": 7, "angle": 30},
        "salt_pepper_noise": {"op": "salt_pepper_noise", "amount": 0.03},
        "paper_texture": {"op": "paper_texture", "strength": 0.3},
        "ink_bleed": {"op": "ink_bleed", "radius": 1.5},
        "binarize": {"op": "binarize", "threshold": 128},
    }

    for op_name in NEW_OPS:
        sub = out_dir / op_name
        sub.mkdir(parents=True, exist_ok=True)
        canvas = torch.ones(N, 3, CANVAS, CANVAS, device=device)
        ctx = GPUContext(canvas=canvas, mask=mask_t.clone(), rng=gen,
                         chars=list(CHARS), source_kinds=kinds, device=device)
        spec = {"style": base_style, "augment": [probe_specs[op_name]]}
        ctx = run_pipeline(ctx, spec)
        final = finalize_center_crop(ctx.canvas)
        torch.cuda.synchronize()
        imgs = tensor_to_pil_batch(final)
        for j, ch in enumerate(CHARS):
            imgs[j].save(sub / f"{j:02d}_{ch}.png", compress_level=1)
        print(f"  [{op_name}] rendered {N} samples")


def v2_full_random_multi_full(out_dir: Path, seed: int) -> None:
    v2_cfg_path = Path(
        "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/"
        "synth_engine_v2/configs/full_random_multi.yaml"
    )
    with open(v2_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # verify full coverage — NO stripping expected now
    known = set(REGISTRY.keys())
    missing = []
    for s in cfg.get("style", []) or []:
        if s.get("layer") not in known:
            missing.append(s.get("layer"))
    for a in cfg.get("augment", []) or []:
        n = f"augment.{a.get('op')}"
        if n not in known:
            missing.append(n)
    if missing:
        print(f"  MISSING v3 registrations: {missing}")
    else:
        print("  all v2 full_random_multi style + augment entries have v3 handlers")

    device = "cuda"
    chars = CHARS * 4  # 40 samples
    rng_np = np.random.default_rng(seed)
    mask_t, tags, kinds = batch_render_from_spec(chars, cfg.get("base_source", {}), rng=rng_np)
    valid = [i for i, t in enumerate(tags) if t is not None]
    if not valid:
        print("  no valid masks")
        return
    mask_t_v = mask_t[valid].to(device)
    chars_v = [chars[i] for i in valid]; kinds_v = [kinds[i] for i in valid]
    gen = torch.Generator(device=device).manual_seed(seed)
    canvas = torch.ones(len(valid), 3, CANVAS, CANVAS, device=device)
    ctx = GPUContext(canvas=canvas, mask=mask_t_v, rng=gen,
                     chars=chars_v, source_kinds=kinds_v, device=device)
    ctx = run_pipeline(ctx, cfg)
    final = finalize_center_crop(ctx.canvas)
    torch.cuda.synchronize()

    sub = out_dir / "v2_full_random_multi_full"
    sub.mkdir(parents=True, exist_ok=True)
    imgs = tensor_to_pil_batch(final)
    kind_counts: dict[str, int] = {}
    for j, i in enumerate(valid):
        ch = chars[i]
        kind = kinds[i] or "?"
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        imgs[j].save(sub / f"{j:03d}_{ch}_{kind}.png", compress_level=1)
    print(f"  rendered {len(valid)}/{len(chars)} samples; kinds={kind_counts}")
    print(f"  output: {sub}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_engine_v3/out/08_phase8_augment_and_v2_full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    required = {f"augment.{o}" for o in NEW_OPS}
    missing = required - set(REGISTRY.keys())
    assert not missing, f"missing registrations: {missing}"
    all_augs = sorted(n for n in REGISTRY if n.startswith("augment."))
    print(f"registered augment ops ({len(all_augs)}): {all_augs}")
    print()
    print("=== per-op probe ===")
    per_op_probe(out, args.seed)
    print()
    print("=== v2 full_random_multi.yaml FULL (style + augment) ===")
    v2_full_random_multi_full(out, args.seed)
    print()
    print(f"outputs in: {out}")
    print("phase8 OK")


if __name__ == "__main__":
    main()
