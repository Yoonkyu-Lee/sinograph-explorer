"""Verify PyTorch ↔ Keras forward pass parity (Phase 1 sanity check).

Loads:
  - PyTorch best.pt (multi-head ResNet-18, char head)
  - Keras .keras file (char-only, weight-ported by 40_port_pytorch_to_keras.py)

Runs N samples through both and checks:
  - argmax (top-1 char) match rate ≥ 99%
  - max abs diff of logits ≤ small threshold (typically < 1e-3)

Mismatch is the canary for a layer-mapping bug (esp. BasicBlock downsample,
BN momentum/eps, padding="same" vs PyTorch padding=1 at non-power-of-2 shapes).

Run from lab2-style WSL venv (TF 2.15) but PyTorch must also be importable
in the same env. If PyTorch is not in the lab2 venv, copy logits via .npy:

    # in PyTorch venv
    python -c "..." --save-pt-logits pt_logits.npy
    # in lab2 venv
    python 42_verify_keras_parity.py --pt-logits pt_logits.npy

For convenience this script supports both modes via --pt-logits.

Usage (single venv, PyTorch + TF coexist):
    python train_engine_v3/scripts/42_verify_keras_parity.py \\
        --ckpt train_engine_v3/out/15_t5_light_v2/best.pt \\
        --keras deploy_pi/export/v3_keras_char.keras \\
        --shard-dir synth_engine_v3/out/94_production_102k_x200 \\
        --n 100 --input-size 128
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def collect_samples(shard_dir: Path, n: int, input_size: int, seed: int = 0):
    """Return (nhwc, nchw): both float32 in [-1, 1]."""
    from PIL import Image
    shards = sorted(shard_dir.glob("shard-*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    rng = random.Random(seed)
    rng.shuffle(shards)
    samples_nhwc: list[np.ndarray] = []
    for s in shards:
        if len(samples_nhwc) >= n:
            break
        d = np.load(s)
        imgs = d["images"]
        per = min(len(imgs), n - len(samples_nhwc))
        for i in rng.sample(range(len(imgs)), per):
            arr = imgs[i]
            if arr.shape[0] != input_size:
                arr = np.asarray(
                    Image.fromarray(arr).resize((input_size, input_size), Image.BILINEAR)
                )
            arr = arr.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            samples_nhwc.append(arr)
    nhwc = np.stack(samples_nhwc[:n], axis=0)
    nchw = np.transpose(nhwc, (0, 3, 1, 2)).copy()
    return nhwc, nchw


def run_pytorch_logits(ckpt_path: Path, nchw: np.ndarray, num_classes: int) -> np.ndarray:
    import torch
    from train_engine_v3.modules.model import build_model

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model = build_model("resnet18", num_classes=num_classes)
    model.load_state_dict(sd)
    model.eval()

    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(nchw), 8):
            x = torch.from_numpy(nchw[i:i+8])
            logits = model.forward_char_only(x).cpu().numpy()
            out_chunks.append(logits)
    return np.concatenate(out_chunks, axis=0)


def run_keras_logits(keras_path: Path, nhwc: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    model = tf.keras.models.load_model(str(keras_path))
    out = model.predict(nhwc, batch_size=8, verbose=0)
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="PyTorch best.pt (or use --pt-logits)")
    ap.add_argument("--pt-logits", default=None, help="precomputed PyTorch logits .npy")
    ap.add_argument("--save-pt-logits", default=None,
                    help="run PyTorch only and save logits to this path; skip Keras")
    ap.add_argument("--keras", default=None, help="path to v3_keras_char.keras")
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=98169)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    nhwc, nchw = collect_samples(Path(args.shard_dir), args.n, args.input_size,
                                  args.seed)
    print(f"[parity] {args.n} samples nhwc={nhwc.shape} nchw={nchw.shape}")

    if args.save_pt_logits:
        if not args.ckpt:
            raise SystemExit("--save-pt-logits requires --ckpt")
        print(f"[parity] running PyTorch only → save to {args.save_pt_logits}")
        pt_logits = run_pytorch_logits(Path(args.ckpt), nchw, args.num_classes)
        np.save(args.save_pt_logits, pt_logits)
        print(f"[parity] saved {pt_logits.shape}")
        return

    if args.pt_logits:
        pt_logits = np.load(args.pt_logits)
        print(f"[parity] loaded PT logits {pt_logits.shape} from {args.pt_logits}")
    elif args.ckpt:
        print(f"[parity] running PyTorch...")
        pt_logits = run_pytorch_logits(Path(args.ckpt), nchw, args.num_classes)
        print(f"[parity] PT logits {pt_logits.shape}")
    else:
        raise SystemExit("need --ckpt or --pt-logits")

    if not args.keras:
        raise SystemExit("--keras required for parity check")

    print(f"[parity] running Keras...")
    keras_logits = run_keras_logits(Path(args.keras), nhwc)
    print(f"[parity] Keras logits {keras_logits.shape}")

    if pt_logits.shape != keras_logits.shape:
        raise SystemExit(
            f"shape mismatch: PT {pt_logits.shape} vs Keras {keras_logits.shape}"
        )

    pt_top1 = pt_logits.argmax(axis=1)
    k_top1 = keras_logits.argmax(axis=1)
    match = (pt_top1 == k_top1).mean()
    diff_abs = np.abs(pt_logits - keras_logits)
    diff_rel = diff_abs / (np.abs(pt_logits).max() + 1e-8)

    print()
    print("=" * 70)
    print(f"top-1 match    : {match*100:.2f}%   ({(pt_top1 == k_top1).sum()}/{len(pt_top1)})")
    print(f"max abs diff   : {diff_abs.max():.6f}")
    print(f"mean abs diff  : {diff_abs.mean():.6f}")
    print(f"max rel diff   : {diff_rel.max():.6f}")

    # Top-5 match
    pt_top5 = np.argsort(-pt_logits, axis=1)[:, :5]
    k_top5 = np.argsort(-keras_logits, axis=1)[:, :5]
    same_set = sum(set(pt_top5[i]) == set(k_top5[i]) for i in range(len(pt_top5)))
    print(f"top-5 set match: {same_set / len(pt_top5) * 100:.2f}%")

    print()
    if match >= 0.99 and diff_abs.max() < 1e-2:
        print("✅ PARITY OK")
    elif match >= 0.95:
        print("⚠ partial parity — investigate large-diff samples")
    else:
        print("❌ FAIL — layer mapping likely incorrect")
        sys.exit(1)


if __name__ == "__main__":
    main()
