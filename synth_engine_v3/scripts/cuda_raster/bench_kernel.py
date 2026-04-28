"""Bench the CUDA polygon rasterizer in isolation (Phase OPT-2.4).

Measures pure GPU rasterize_batch throughput: outlines are pre-extracted +
cached, kernel is pre-built. We time only the rasterize_batch call.

This isolates the kernel speed from outline extraction (which has its own
LRU cache). Real production also hits the LRU cache 99 %+ of the time after
warmup, so the kernel-only number is representative of steady-state.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from outline_cache import get_outline                   # noqa: E402
from rasterize import rasterize_batch                   # noqa: E402
from streaming_log import setup_logging                 # noqa: E402

CANVAS = 384
PAD = 48


def _win_fonts() -> Path:
    if sys.platform.startswith("win"):
        return Path("C:/Windows/Fonts")
    if Path("/mnt/c/Windows/Fonts").is_dir():
        return Path("/mnt/c/Windows/Fonts")
    return Path("/usr/share/fonts/truetype")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--font", default="batang.ttc")
    ap.add_argument("--log", default=None,
                    help="optional log path; default = bench_kernel_<timestamp>.log "
                         "next to this script")
    ap.add_argument("--kernel", default="v1", choices=["v1", "v2"],
                    help="raster kernel version (v1=naive, v2=shared-mem+sorted-edges)")
    args = ap.parse_args()

    log_path = args.log
    if log_path is None:
        from datetime import datetime
        log_path = HERE / f"out_bench_{args.kernel}_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)

    fdir = _win_fonts()
    font_path = fdir / args.font

    # Build a pool of typical CJK chars (CJK Unified U+4E00 + offsets).
    # These all get cached on first extraction.
    pool = [chr(cp) for cp in range(0x4E00, 0x4E00 + max(args.batch * 4, 256))]
    print(f"[bench] building outline cache for {len(pool)} chars on {font_path.name}…")
    outlines = []
    skipped = 0
    for ch in pool:
        try:
            od = get_outline(font_path, 0, ch, CANVAS, PAD)
            if od.edges is not None and len(od.edges) > 0:
                outlines.append(od)
        except Exception:
            skipped += 1
    print(f"[bench] cached {len(outlines)} outlines  (skipped {skipped})")

    # Warmup: builds extension + caches kernel
    print(f"[bench] warmup batch (size {args.batch})  kernel={args.kernel}…")
    sample = outlines[: args.batch] * (args.batch // len(outlines) + 1)
    sample = sample[: args.batch]
    _ = rasterize_batch(sample, CANVAS, CANVAS, device="cuda", kernel=args.kernel)
    torch.cuda.synchronize()

    # Bench: iters × (rebuild input list + rasterize_batch)
    print(f"[bench] timing {args.iters} iters × batch {args.batch}…")
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        # different sample each iter to avoid trivial caching
        idx = rng.integers(0, len(outlines), size=args.batch)
        batch = [outlines[i] for i in idx]
        masks = rasterize_batch(batch, CANVAS, CANVAS, device="cuda", kernel=args.kernel)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_glyphs = args.iters * args.batch
    per_glyph_ms = elapsed * 1000 / total_glyphs
    rate = total_glyphs / elapsed
    print()
    print(f"[bench] elapsed       = {elapsed:.2f} s")
    print(f"[bench] total glyphs  = {total_glyphs:,}")
    print(f"[bench] per-glyph     = {per_glyph_ms:.3f} ms")
    print(f"[bench] throughput    = {rate:,.0f} glyphs/s   <-- raster only")
    print()
    # For comparison: CPU PIL workers ran at ~712 masks/s aggregate (8 workers
    # × 89 / s). Single-process PIL ≈ 89 / s. Reference: 24 hour corpus run
    # baseline = 237 samples/s end-to-end (mask + augment + save).


if __name__ == "__main__":
    main()
