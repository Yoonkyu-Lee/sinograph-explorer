"""generate_v3.py — single-character v3 generator driven by a YAML config.

v2 analog: synth_engine_v2/scripts/generate.py

Usage:
    python generate_v3.py 鑑 --config configs/full_random_v3.yaml --count 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from mask_adapter import get_sources_for_char, render_mask, masks_to_tensor
from pipeline_gpu import (
    CANVAS, GPUContext, finalize_center_crop, run_pipeline, tensor_to_pil_batch,
)
import style_gpu  # noqa: F401 — registers layers
import augment_gpu  # noqa: F401


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_masks_cpu(chars: list[str], base_source_spec: dict,
                      rng: np.random.Generator) -> tuple[list, list[str | None], list[str]]:
    """Render one mask per entry in `chars` using the v2 base_source spec
    (supports font / svg_stroke / ehanja_median / kanjivg_median /
    ehanja_stroke / mmh_stroke / multi with fallback)."""
    masks, tags, kinds = [], [], []
    for ch in chars:
        srcs = get_sources_for_char(ch, base_source_spec)
        if not srcs:
            masks.append(None); tags.append(None); kinds.append(""); continue
        src = srcs[0] if len(srcs) == 1 else srcs[int(rng.integers(0, len(srcs)))]
        m = render_mask(ch, src, rng=rng)
        if m is None:
            masks.append(None); tags.append(None); kinds.append(""); continue
        kind = getattr(src, "last_kind", None) or getattr(src, "kind", "") or ""
        masks.append(m); tags.append(src.tag()); kinds.append(kind)
    return masks, tags, kinds


def save_png(img, path: Path) -> None:
    img.save(path, compress_level=1)


def generate(char: str, count: int, config: dict, out_dir: Path,
             seed: int, save_workers: int, write_metadata: bool,
             batch_size: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    device = "cuda"

    chars = [char] * count
    out_dir.mkdir(parents=True, exist_ok=True)
    notation = f"U+{ord(char):04X}"
    sub = out_dir / notation
    sub.mkdir(parents=True, exist_ok=True)

    stats = {"n": count, "saved": 0, "skipped_no_source": 0}
    t_start = time.perf_counter()

    # 1) rasterize all masks on CPU (serial in this single-process driver;
    #    multi-proc lives in 10_generate_corpus_v3.py)
    t0 = time.perf_counter()
    base_spec = config.get("base_source", {"kind": "font"})
    masks, tags, kinds = render_masks_cpu(chars, base_spec, rng)
    valid_idx = [i for i, m in enumerate(masks) if m is not None]
    stats["skipped_no_source"] = count - len(valid_idx)
    if not valid_idx:
        print(f"ERROR: no base source covers {char!r}", file=sys.stderr)
        return stats
    t_mask = time.perf_counter() - t0

    # 2) GPU pipeline in chunks — avoids VRAM pressure + per-layer clone overhead
    #    when a single batch balloons past the sweet spot.
    t_gpu = 0.0
    t_save = 0.0
    all_paths = []

    pool = ThreadPoolExecutor(max_workers=max(1, save_workers))
    save_futures = []

    for chunk_start in range(0, len(valid_idx), batch_size):
        chunk_idx = valid_idx[chunk_start:chunk_start + batch_size]
        n_chunk = len(chunk_idx)

        tg0 = time.perf_counter()
        mask_t = masks_to_tensor([masks[i] for i in chunk_idx], device=device)
        canvas = torch.ones(n_chunk, 3, CANVAS, CANVAS, device=device)
        # fresh generator per chunk for reproducibility across --batch-size values
        gen = torch.Generator(device=device).manual_seed(seed + chunk_start)
        ctx = GPUContext(canvas=canvas, mask=mask_t, rng=gen,
                         chars=[chars[i] for i in chunk_idx],
                         source_kinds=[kinds[i] for i in chunk_idx], device=device)
        ctx = run_pipeline(ctx, config)
        final = finalize_center_crop(ctx.canvas)
        torch.cuda.synchronize()
        t_gpu += time.perf_counter() - tg0

        ts0 = time.perf_counter()
        imgs = tensor_to_pil_batch(final)
        for j, i in enumerate(chunk_idx):
            fname = f"{i:04d}_{notation}_{tags[i]}.png"
            path = sub / fname
            all_paths.append(path)
            save_futures.append(pool.submit(save_png, imgs[j], path))
        t_save += time.perf_counter() - ts0

    for f in save_futures:
        f.result()
    pool.shutdown()
    stats["saved"] = len(valid_idx)

    if write_metadata:
        manifest_path = sub / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for j, i in enumerate(valid_idx):
                f.write(json.dumps({
                    "char": char, "codepoint": notation,
                    "path": str(all_paths[j].name), "base_source": tags[i],
                    "source_kind": kinds[i], "seed": seed + j,
                }, ensure_ascii=False) + "\n")

    t_total = time.perf_counter() - t_start
    print(f"char={char} ({notation})  count={count}")
    print(f"  mask raster (serial): {t_mask*1000:7.1f} ms   ({len(valid_idx)/t_mask:7.1f} masks/s)")
    print(f"  GPU pipeline+crop:    {t_gpu*1000:7.1f} ms   ({len(valid_idx)/t_gpu:7.1f} samples/s)")
    print(f"  save ({save_workers} threads):     {t_save*1000:7.1f} ms   ({len(valid_idx)/t_save:7.1f} saves/s)")
    print(f"  total:                {t_total*1000:7.1f} ms   ({len(valid_idx)/t_total:7.1f} samples/s)")
    stats["t_total_s"] = t_total
    stats["samples_per_s"] = len(valid_idx) / t_total
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("char", help="single character literal, e.g. 鑑")
    ap.add_argument("--config", required=True)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out", default="synth_engine_v3/out/05_phase5_generate_v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--metadata", action="store_true")
    args = ap.parse_args()
    config = load_config(Path(args.config))
    generate(args.char, args.count, config, Path(args.out),
             args.seed, args.save_workers, args.metadata, args.batch_size)


if __name__ == "__main__":
    main()
