"""Build a stratified val pack (images + labels) for Pi accuracy benchmark.

Pulls N samples from production shards (matching the same `val_per_shard`
pattern used in scer_production.yaml so the val set is consistent with
training-time stratification), saves as a single .npz:
    images : (N, 128, 128, 3) uint8
    labels : (N,) int64

Pi-side accuracy script reads this and computes top-1 / top-5 / scer pipeline.

Usage:
    python deploy_pi/44_make_val_pack.py \\
        --shard-dir synth_engine_v3/out/94_production_102k_x200 \\
        --out deploy_pi/export/val_pack_1000.npz \\
        --n 1000
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--val-per-shard", type=int, default=3,
                    help="matches scer_production.yaml (3 per shard, stratified)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    shards = sorted(shard_dir.glob("shard-*.npz"))
    print(f"[pack] {len(shards)} shards in {shard_dir}")

    rng = random.Random(args.seed)
    images_out = []
    labels_out = []
    for shard_path in shards:
        if len(images_out) >= args.n:
            break
        d = np.load(shard_path)
        imgs = d["images"]
        labs = d["labels"]
        avail = list(range(len(imgs)))
        rng.shuffle(avail)
        per = min(args.val_per_shard, len(avail), args.n - len(images_out))
        for i in avail[:per]:
            images_out.append(imgs[i])
            labels_out.append(int(labs[i]))

    images = np.stack(images_out[:args.n], axis=0)        # uint8 NHWC
    labels = np.array(labels_out[:args.n], dtype=np.int64)
    print(f"[pack] images {images.shape} dtype={images.dtype}  "
          f"labels {labels.shape} unique={len(np.unique(labels))}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images=images, labels=labels)
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"[pack] wrote {out_path} ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
