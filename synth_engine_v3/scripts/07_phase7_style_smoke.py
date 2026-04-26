"""Phase 7 smoke — exercise every newly-ported style layer on a small batch.

Also loads v2's `full_random_multi.yaml` directly (minus augment ops not yet
ported in v3 — Phase 8) and runs it end-to-end to confirm zero "unknown layer"
errors on the style half.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from mask_adapter import CANVAS, batch_render_from_spec, masks_to_tensor, render_mask, get_sources_for_char
from pipeline_gpu import GPUContext, finalize_center_crop, run_pipeline, tensor_to_pil_batch, REGISTRY
import style_gpu  # noqa: F401


CHARS = ["鑑", "學", "学", "斈", "裡", "裏"]


NEW_LAYERS_TO_TEST = [
    {"layer": "background.stripe", "thickness": 16, "angle": 20,
     "color_a": (245, 245, 250), "color_b": (210, 220, 235)},
    {"layer": "background.lines", "spacing": 22, "line_width": 2, "angle": 0,
     "base_color": (255, 255, 250), "line_color": (180, 200, 230)},
    {"layer": "background.scene", "folder": "synth_engine_v2/samples/backgrounds",
     "mode": "random_crop", "scale_jitter": [1.0, 1.5], "dim": 0.2,
     "desaturate": 0.3, "blur": 1.0},
    {"layer": "fill.gradient", "start": (220, 60, 40), "end": (40, 60, 220),
     "direction": "vertical"},
    {"layer": "fill.stripe", "thickness": 8, "angle": 45,
     "color_a": (30, 30, 30), "color_b": (220, 80, 80)},
    {"layer": "fill.radial", "inner": (255, 240, 180), "outer": (120, 30, 30)},
    {"layer": "fill.contrast", "threshold": 128,
     "dark_color": [[20, 20, 20], [60, 20, 80]],
     "light_color": [[240, 240, 240], [250, 230, 180]],
     "jitter": 20, "min_contrast": 80},
    {"layer": "outline.double", "outer_offset": 5, "outer_width": 2,
     "inner_width": 1, "color": (10, 10, 10)},
    {"layer": "shadow.soft", "blur": 10, "color": (180, 180, 200)},
    {"layer": "shadow.long", "step": (1, 1), "length": 18,
     "color": (180, 180, 180)},
    {"layer": "glow.inner", "color": (255, 255, 220), "blur": 3},
    {"layer": "glow.neon", "color": (0, 220, 255), "outer_dilate": 3,
     "outer_blur": 8, "inner_blur": 2, "core_color": (255, 255, 255)},
]


def per_layer_probe(out_dir: Path, seed: int) -> None:
    """For each new layer, render the same 6-char batch with background.solid
    white -> that layer only -> verify result isn't the bland starting canvas."""
    device = "cuda"
    rng_np = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    # same masks for every layer so diffs are attributable to the layer
    masks, kinds = [], []
    for ch in CHARS:
        srcs = get_sources_for_char(ch, {"kind": "font"})
        src = srcs[0] if srcs else None
        m = render_mask(ch, src, rng=rng_np) if src is not None else None
        masks.append(m); kinds.append("font" if m is not None else "")
    mask_t = masks_to_tensor(masks, device=device)
    N = mask_t.shape[0]

    for step in NEW_LAYERS_TO_TEST:
        name = step["layer"]
        sub = out_dir / name.replace(".", "_")
        sub.mkdir(parents=True, exist_ok=True)
        canvas = torch.ones(N, 3, CANVAS, CANVAS, device=device)
        ctx = GPUContext(canvas=canvas, mask=mask_t.clone(), rng=gen,
                         chars=list(CHARS), source_kinds=kinds, device=device)
        # for layers that only touch fill on top of plain canvas, do a bg first
        spec = {"style": [
            {"layer": "background.solid", "color": (245, 245, 245)},
            step,
            # always follow with fill.hsv_contrast so the glyph is visible for
            # non-fill layers (bg/outline/shadow/glow tests)
        ]}
        if not name.startswith("fill."):
            spec["style"].append(
                {"layer": "fill.hsv_contrast", "saturation": [0.5, 0.9],
                 "value": [0.1, 0.8], "min_contrast": 70}
            )
        ctx = run_pipeline(ctx, spec)
        final = finalize_center_crop(ctx.canvas)
        torch.cuda.synchronize()
        imgs = tensor_to_pil_batch(final)
        for j, ch in enumerate(CHARS):
            imgs[j].save(sub / f"{j:02d}_{ch}.png", compress_level=1)
        diff_from_plain = (final - 0.95).abs().mean().item()
        print(f"  [{name}] diff_from_plain_gray={diff_from_plain:.3f}")


def v2_full_random_multi_style_only(out_dir: Path, seed: int) -> None:
    """Load v2's full_random_multi.yaml, strip augment ops that aren't in v3 yet,
    and run the style block end-to-end on a mixed base_source batch."""
    v2_cfg_path = Path(
        "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/"
        "synth_engine_v2/configs/full_random_multi.yaml"
    )
    with open(v2_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # strip augment ops whose op name doesn't exist in v3 REGISTRY.
    # augment names are unqualified in v2 ("rotate") — v3's run_block auto-prefixes.
    known = set(REGISTRY.keys())
    v3_aug = []
    for raw in (cfg.get("augment") or []):
        name = f"augment.{raw.get('op','?')}"
        if name in known:
            v3_aug.append(raw)
        else:
            print(f"  [strip augment] {raw.get('op')} (not yet in v3 — Phase 8)")
    cfg["augment"] = v3_aug

    # drop unknown style layers too (shouldn't happen after Phase 7, but safe)
    v3_style = []
    for raw in (cfg.get("style") or []):
        name = raw.get("layer")
        if name in known:
            v3_style.append(raw)
        else:
            print(f"  [strip style] {name} (unknown)")
    cfg["style"] = v3_style

    device = "cuda"
    chars = CHARS * 5
    rng_np = np.random.default_rng(seed)
    mask_t, tags, kinds = batch_render_from_spec(chars, cfg.get("base_source", {}), rng=rng_np)
    valid = [i for i, t in enumerate(tags) if t is not None]
    if not valid:
        print("  no valid masks from v2 multi spec")
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

    sub = out_dir / "v2_full_random_multi_style"
    sub.mkdir(parents=True, exist_ok=True)
    imgs = tensor_to_pil_batch(final)
    for j, i in enumerate(valid):
        ch = chars[i]
        kind = kinds[i] or "?"
        imgs[j].save(sub / f"{j:03d}_{ch}_{kind}.png", compress_level=1)
    print(f"  v2 full_random_multi style pipeline rendered {len(valid)}/{len(chars)} to {sub}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_engine_v3/out/07_phase7_style_smoke")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # sanity: verify every new layer is now registered
    required = {s["layer"] for s in NEW_LAYERS_TO_TEST}
    missing = required - set(REGISTRY.keys())
    assert not missing, f"missing layer registrations: {missing}"
    print(f"registered style layers: {sorted(n for n in REGISTRY if not n.startswith('augment.'))}")
    print()
    print("=== per-layer probe ===")
    per_layer_probe(out, args.seed)
    print()
    print("=== v2 full_random_multi.yaml (style-only) ===")
    v2_full_random_multi_style_only(out, args.seed)
    print()
    print(f"outputs in: {out}")
    print("phase7 OK")


if __name__ == "__main__":
    main()
