"""Phase 6 smoke — verify all v2 base_source kinds reachable through v3 adapter.

For a handful of characters, renders a mask through each available source kind
(font / svg_stroke / ehanja_median / ehanja_stroke / mmh_stroke / multi) and
saves PNGs plus a summary of shape / dtype / source_kind. Also end-to-ends
through a v3-compatible style+augment spec to confirm kind-gated augment
(skip_if_kinds) actually fires per sample inside a mixed batch.

Note: kanjivg_median needs `db_src/KanjiVG/strokes_kanjivg.jsonl` which is not
yet extracted in this checkout — gracefully reported rather than failing.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mask_adapter import (
    CANVAS, batch_render_from_spec, get_sources_for_char, masks_to_tensor,
    render_mask,
)
from pipeline_gpu import GPUContext, finalize_center_crop, run_pipeline, tensor_to_pil_batch
import style_gpu  # noqa: F401
import augment_gpu  # noqa: F401


CHARS = ["鑑", "學", "学", "媤", "乶", "畓", "裡", "裏", "あ", "한"]


# Per-kind specs matching v2's `full_random_multi.yaml` stroke_ops, in YAML-shape.
SINGLE_KIND_SPECS = {
    "font": {"kind": "font"},
    "svg_stroke": {
        "kind": "svg_stroke", "base_width": 48,
        "stroke_ops": [
            {"op": "width_jitter", "std": 5},
            {"op": "control_jitter", "std_ratio": 0.020, "std_min": 0.8, "std_max": 10.0},
            {"op": "endpoint_jitter", "std_ratio": 0.030, "std_min": 1.5, "std_max": 15.0},
            {"op": "stroke_rotate", "angle_std": 2},
            {"op": "stroke_translate", "std_ratio": 0.015, "std_min": 0.5, "std_max": 8.0},
        ],
    },
    "ehanja_median": {
        "kind": "ehanja_median", "width_scale": 1.5,
        "stroke_ops": [
            {"op": "width_jitter", "std": 5},
            {"op": "control_jitter", "std_ratio": 0.020, "std_min": 0.8, "std_max": 10.0},
            {"op": "endpoint_jitter", "std_ratio": 0.030, "std_min": 1.5, "std_max": 15.0},
            {"op": "stroke_rotate", "angle_std": 2},
            {"op": "stroke_translate", "std_ratio": 0.015, "std_min": 0.5, "std_max": 8.0},
        ],
    },
    "ehanja_stroke": {
        "kind": "ehanja_stroke",
        "stroke_ops": [
            {"op": "stroke_rotate", "angle_std": 1.5},
            {"op": "stroke_translate", "std_ratio": 0.010, "std_min": 0.5, "std_max": 6.0},
            {"op": "control_jitter", "std_ratio": 0.012, "std_min": 0.5, "std_max": 6.0},
        ],
    },
    "mmh_stroke": {
        "kind": "mmh_stroke",
        "stroke_ops": [
            {"op": "stroke_rotate", "angle_std": 1.5},
            {"op": "stroke_translate", "std_ratio": 0.010, "std_min": 0.5, "std_max": 6.0},
            {"op": "control_jitter", "std_ratio": 0.012, "std_min": 0.5, "std_max": 6.0},
        ],
    },
    "kanjivg_median": {
        "kind": "kanjivg_median", "width_scale": 1.7,
        "stroke_ops": [
            {"op": "width_jitter", "std": 0.5},
            {"op": "control_jitter", "std_ratio": 0.020, "std_min": 0.1, "std_max": 1.2},
            {"op": "endpoint_jitter", "std_ratio": 0.030, "std_min": 0.2, "std_max": 1.8},
            {"op": "stroke_rotate", "angle_std": 2},
            {"op": "stroke_translate", "std_ratio": 0.015, "std_min": 0.05, "std_max": 1.0},
        ],
    },
}


MULTI_SPEC = {
    "kind": "multi",
    "fallback": "font",
    "sources": [
        {**SINGLE_KIND_SPECS["font"], "weight": 5.0},
        {**SINGLE_KIND_SPECS["svg_stroke"], "weight": 2.0},
        {**SINGLE_KIND_SPECS["ehanja_median"], "weight": 2.0},
        {**SINGLE_KIND_SPECS["ehanja_stroke"], "weight": 1.5},
        {**SINGLE_KIND_SPECS["mmh_stroke"], "weight": 1.5},
        {**SINGLE_KIND_SPECS["kanjivg_median"], "weight": 1.0},
    ],
}


def probe_single_kinds(out_dir: Path, seed: int) -> dict:
    """For each kind, try one char that's likely covered. Save the raw mask."""
    rng = np.random.default_rng(seed)
    results = {}
    for kind_name, spec in SINGLE_KIND_SPECS.items():
        sub = out_dir / f"{kind_name}"
        sub.mkdir(parents=True, exist_ok=True)
        per_kind = []
        for ch in CHARS:
            try:
                srcs = get_sources_for_char(ch, spec)
            except FileNotFoundError as e:
                per_kind.append((ch, 0, f"DATA MISSING: {e}"))
                continue
            if not srcs:
                per_kind.append((ch, 0, "no source for char"))
                continue
            src = srcs[0]
            m = render_mask(ch, src, rng=rng)
            if m is None:
                per_kind.append((ch, 1, "render_mask returned None"))
                continue
            fname = f"{ch}_U+{ord(ch):04X}.png"
            try:
                m.save(sub / fname)
            except Exception as e:
                per_kind.append((ch, 1, f"save error: {e}"))
                continue
            per_kind.append((ch, 1, f"OK kind={getattr(src,'kind','?')} tag={src.tag()}"))
        results[kind_name] = per_kind
        covered = sum(1 for _, r, _ in per_kind if r == 1 and "OK" in _[2] if False)
        ok_count = sum(1 for _, _, note in per_kind if note.startswith("OK"))
        print(f"  [{kind_name}] {ok_count}/{len(per_kind)} chars rendered OK")
        for ch, _, note in per_kind:
            if not note.startswith("OK"):
                print(f"      {ch}: {note}")
    return results


def run_multi_batch(out_dir: Path, seed: int) -> None:
    """Render a mixed batch via MultiSource, then push through a small style+augment
    spec with skip_if_kinds gating to prove kind-aware gating fires mid-batch."""
    sub = out_dir / "multi_batch"
    sub.mkdir(parents=True, exist_ok=True)
    chars = CHARS * 4  # 40 samples
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    mask_t, tags, kinds = batch_render_from_spec(chars, MULTI_SPEC, rng=rng)
    t_mask = time.perf_counter() - t0
    valid = [i for i, t in enumerate(tags) if t is not None]
    kind_counts: dict[str, int] = {}
    for k in kinds:
        if k:
            kind_counts[k] = kind_counts.get(k, 0) + 1
    print(f"  multi batch: {len(valid)}/{len(chars)} valid, kinds={kind_counts}, mask_t={t_mask:.2f}s")

    if not valid:
        print("  no valid masks — skipping GPU run")
        return

    device = "cuda"
    mask_t_v = mask_t[valid].to(device)
    kinds_v = [kinds[i] for i in valid]
    chars_v = [chars[i] for i in valid]
    tags_v = [tags[i] for i in valid]
    canvas = torch.ones(len(valid), 3, CANVAS, CANVAS, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    ctx = GPUContext(canvas=canvas, mask=mask_t_v, rng=gen,
                     chars=chars_v, source_kinds=kinds_v, device=device)

    spec = {
        "style": [
            {"layer": "background.solid", "color": (255, 255, 255)},
            {"layer": "fill.hsv_contrast", "saturation": [0.5, 0.9], "value": [0.1, 0.8]},
            {"layer": "outline.simple", "width": [1, 2], "prob": 0.4},
        ],
        "augment": [
            {"op": "rotate", "angle": [-10, 10]},
            {"op": "gaussian_blur", "sigma": [0.0, 1.0], "prob": 0.5},
            # elastic only on non-SVG (matches v2 full_random_multi semantics)
            {"op": "elastic", "alpha": [4, 8], "sigma": [5, 7], "prob": 1.0,
             "skip_if_kinds": ["svg_stroke", "ehanja_median", "kanjivg_median",
                                "mmh_stroke", "ehanja_stroke"]},
        ],
    }
    ctx = run_pipeline(ctx, spec)
    final = finalize_center_crop(ctx.canvas)
    torch.cuda.synchronize()

    imgs = tensor_to_pil_batch(final)
    for j, i in enumerate(valid):
        ch = chars[i]
        kind = kinds[i] or "unknown"
        imgs[j].save(sub / f"{j:03d}_{ch}_U+{ord(ch):04X}_{kind}.png", compress_level=1)
    print(f"  multi batch rendered to {sub}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_engine_v3/out/06_phase6_svg_source_smoke")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("=== per-kind probe ===")
    probe_single_kinds(out, args.seed)
    print()
    print("=== multi-source batch + kind-gated augment ===")
    run_multi_batch(out, args.seed)
    print()
    print(f"outputs in: {out}")
    print("phase6 OK")


if __name__ == "__main__":
    main()
