"""Convert PNG corpus to uint8 tensor shards for fast training (Phase 2, doc/13).

Input:  corpus_manifest.jsonl + PNG files (from 10_generate_corpus_v3.py output)
Output: shard-NNNN.npz containing pre-resized uint8 images + class labels
        class_index.json (notation -> idx)
        shards_manifest.json (shard meta)

Training path: synth PNGs → this script once → shards → train_engine_v2 loads
shards directly. PIL decode cost is eliminated from the training loop.

Usage:
  python 12_pack_tensor_shards.py \
      --manifest .../corpus_manifest.jsonl \
      --image-root .../corpus_dir \
      --out-dir   .../shards_dir \
      --shard-size 5000 --input-size 128 --workers 8

Notes:
  - Shuffles manifest once with --seed for shard-level class diversity.
  - Pre-resize matches the M4 val pipeline: Resize(shorter side → input_size)
    then CenterCrop(input_size). Training-time RandomCrop is not pre-baked
    (it runs on GPU inside train loop if needed).
  - Shards are uncompressed (.npz via savez) — compression provides little
    benefit on uint8 images of our density and costs write time.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

# Windows stdout utf-8 (for notation prints)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _resize_center(img: Image.Image, size: int) -> Image.Image:
    """Match the val-path transform: Resize shorter side to size, then CenterCrop."""
    w, h = img.size
    if w == size and h == size:
        return img
    if w < h:
        new_w, new_h = size, max(size, int(round(h * size / w)))
    else:
        new_w, new_h = max(size, int(round(w * size / h))), size
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _decode_one(args):
    path_str, input_size = args
    try:
        with Image.open(path_str) as im:
            im = im.convert("RGB")
            im = _resize_center(im, input_size)
            return np.asarray(im, dtype=np.uint8)
    except Exception as e:
        return RuntimeError(f"{path_str}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="corpus_manifest.jsonl from 10_generate_corpus_v3.py")
    ap.add_argument("--image-root", required=True,
                    help="directory containing the PNG files referenced by manifest")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--input-size", type=int, default=256,
                    help="shard image resolution. Default 256 preserves original. "
                         "Pass 128 for 4x smaller shards at cost of detail on complex chars.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="limit total samples (for smoke testing)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # load manifest
    print(f"[pack] loading manifest {args.manifest}")
    rows = []
    t0 = time.time()
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"[pack] loaded {len(rows):,} rows in {time.time()-t0:.1f}s")

    if args.max_samples and args.max_samples < len(rows):
        rows = rows[: args.max_samples]
        print(f"[pack] truncated to {len(rows):,} samples (max-samples)")

    # class index — integer codepoint sort (not string) so first N are the
    # actual common CJK chars. Fixes the v1 pilot bug where SMP ext B sorted
    # before U+4E00.
    notations = sorted({r["notation"] for r in rows},
                       key=lambda n: int(n[2:], 16))
    class_index = {n: i for i, n in enumerate(notations)}
    (out / "class_index.json").write_text(
        json.dumps(class_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[pack] class_index: {len(class_index):,} classes → {out / 'class_index.json'}")

    # shuffle for per-shard class diversity
    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    image_root = Path(args.image_root)
    n = len(shuffled)
    n_shards = (n + args.shard_size - 1) // args.shard_size
    print(f"[pack] making {n_shards} shards × {args.shard_size} samples "
          f"(input_size={args.input_size}, workers={args.workers})")

    # estimated disk: each shard = shard_size * H * W * 3 bytes
    per_shard_mb = args.shard_size * args.input_size * args.input_size * 3 / 1e6
    total_gb = per_shard_mb * n_shards / 1e3
    print(f"[pack] estimated total shard size: {total_gb:.1f} GB "
          f"(~{per_shard_mb:.0f} MB per shard)")

    t0 = time.time()
    errors = 0
    with Pool(args.workers) as pool:
        for shard_idx in range(n_shards):
            start = shard_idx * args.shard_size
            end = min(start + args.shard_size, n)
            chunk = shuffled[start:end]
            tasks = [(str(image_root / r["filename"]), args.input_size) for r in chunk]
            results = pool.map(_decode_one, tasks)

            ok_mask = [not isinstance(r, Exception) for r in results]
            if not all(ok_mask):
                bad = [r for r in results if isinstance(r, Exception)]
                errors += len(bad)
                print(f"[pack] shard {shard_idx}: {len(bad)} decode errors", file=sys.stderr)
                # filter to successful only
                results = [r for r in results if not isinstance(r, Exception)]
                chunk = [c for c, ok in zip(chunk, ok_mask) if ok]

            images = np.stack(results, axis=0)  # (N, H, W, 3) uint8
            labels = np.array([class_index[r["notation"]] for r in chunk],
                              dtype=np.int64)

            shard_path = out / f"shard-{shard_idx:04d}.npz"
            np.savez(shard_path, images=images, labels=labels)

            elapsed = time.time() - t0
            done = end
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            if (shard_idx + 1) % max(1, n_shards // 20) == 0 or shard_idx == n_shards - 1:
                print(f"[pack] shard {shard_idx+1}/{n_shards}  "
                      f"{shard_path.name}  "
                      f"({elapsed:.0f}s  {rate:.0f} samples/s  eta {eta:.0f}s)")

    meta = {
        "n_shards": n_shards,
        "shard_size": args.shard_size,
        "total_samples": n,
        "input_size": args.input_size,
        "n_classes": len(class_index),
        "format": "npz",
        "schema": {
            "images": "uint8[N, H, W, 3]",
            "labels": "int64[N] (idx into class_index)",
        },
        "decode_errors": errors,
        "seed": args.seed,
    }
    (out / "shards_manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[pack] done. total {time.time()-t0:.1f}s, {errors} errors → {out}")


if __name__ == "__main__":
    main()
