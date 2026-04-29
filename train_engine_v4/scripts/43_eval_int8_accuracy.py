"""SCER INT8 vs FP32 정확도 게이트 (doc/29 §7.1 P3-G2).

비교 path:
    PT FP32       — train_engine_v4/out/16_scer_v1/best.pt + forward_inference
    TFLite INT8   — deploy_pi/export/scer_int8.tflite (Edge TPU compatible)

같은 val sample 1000 개로:
    1. PT FP32 의 emb_full top-1 (vs anchor DB)
    2. TFLite INT8 의 emb_full top-1 (vs anchor DB)
    3. 두 결과의 top-1 일치율 + 절대 정확도 차이

게이트: TFL-INT8 의 emb top-1 정확도 손실 < 2pp.

Usage (lab2-style WSL):
    python train_engine_v4/scripts/43_eval_int8_accuracy.py \\
        --tflite deploy_pi/export/scer_int8.tflite \\
        --ckpt   train_engine_v4/out/16_scer_v1/best.pt \\
        --anchors deploy_pi/export/scer_anchor_db.npy \\
        --shard-dir synth_engine_v3/out/94_production_102k_x200 \\
        --max-samples 1000
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def collect_val_samples(shard_dir: Path, n: int, val_per_shard: int,
                         input_size: int = 128, seed: int = 0) -> tuple:
    """Return (images_nhwc fp32 [-1,1], labels) — stratified val from shards."""
    from PIL import Image
    shards = sorted(shard_dir.glob("shard-*.npz"))
    rng = random.Random(seed)
    imgs_out: list[np.ndarray] = []
    labels_out: list[int] = []
    for shard_path in shards:
        if len(imgs_out) >= n:
            break
        d = np.load(shard_path)
        imgs = d["images"]
        labs = d["labels"]
        avail = list(range(len(imgs)))
        rng.shuffle(avail)
        per = min(val_per_shard, len(avail), n - len(imgs_out))
        for i in avail[:per]:
            arr = imgs[i]
            if arr.shape[0] != input_size:
                arr = np.asarray(
                    Image.fromarray(arr).resize(
                        (input_size, input_size), Image.BILINEAR
                    )
                )
            arr = arr.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5                # [-1, 1]
            imgs_out.append(arr)
            labels_out.append(int(labs[i]))
    arr = np.stack(imgs_out[:n], axis=0)            # NHWC
    return arr, np.array(labels_out[:n], dtype=np.int64)


def run_pt(ckpt_path: Path, images_hwc: np.ndarray) -> np.ndarray:
    """PT FP32 — returns L2-normalized embeddings (N, 128)."""
    import torch
    from train_engine_v4.modules.model import build_scer
    n_class = 98169
    model = build_scer(name="resnet18", num_classes=n_class, emb_dim=128)
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    images_chw = np.transpose(images_hwc, (0, 3, 1, 2))     # NHWC → NCHW
    embs: list[np.ndarray] = []
    bs = 32
    with torch.no_grad():
        for s in range(0, len(images_chw), bs):
            x = torch.from_numpy(images_chw[s:s+bs])
            out = model.forward_inference(x)
            embs.append(out["embedding"].numpy())
    return np.concatenate(embs, axis=0)


def run_tflite_int8(tflite_path: Path, images_hwc: np.ndarray) -> np.ndarray:
    """TFLite INT8 — returns L2-normalized embeddings (N, 128) in fp32 after dequant."""
    import tensorflow as tf
    itp = tf.lite.Interpreter(model_path=str(tflite_path))
    itp.allocate_tensors()
    inp_d = itp.get_input_details()[0]
    in_scale, in_zp = inp_d["quantization"]
    # Find the embedding output (128-d) by shape
    emb_d = None
    for d in itp.get_output_details():
        if list(d["shape"][-1:]) == [128]:
            emb_d = d
            break
    if emb_d is None:
        raise RuntimeError("could not find 128-d embedding output")
    out_scale, out_zp = emb_d["quantization"]

    embs: list[np.ndarray] = []
    for i in range(len(images_hwc)):
        x_fp = images_hwc[i:i+1]                              # (1, H, W, 3) fp32
        x_int8 = np.round(x_fp / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        itp.set_tensor(inp_d["index"], x_int8)
        itp.invoke()
        y_int8 = itp.get_tensor(emb_d["index"])               # (1, 128) int8
        y_fp = (y_int8.astype(np.float32) - out_zp) * out_scale
        # Re-L2-normalize after dequant (INT8 quantization can drift the norm
        # away from 1; re-normalize for fair cosine comparison vs anchor DB).
        norm = np.linalg.norm(y_fp, axis=1, keepdims=True).clip(min=1e-8)
        y_fp = y_fp / norm
        embs.append(y_fp[0])
    return np.stack(embs, axis=0)


def cosine_top1(emb: np.ndarray, anchors: np.ndarray, labels: np.ndarray) -> tuple:
    """Returns (top1, top5, top1_pred_for_each_sample)."""
    sims = emb @ anchors.T                                    # (N, C)
    top5 = np.argsort(-sims, axis=1)[:, :5]
    top1 = top5[:, 0]
    is_correct_1 = (top1 == labels).mean()
    is_correct_5 = (top5 == labels[:, None]).any(axis=1).mean()
    return is_correct_1, is_correct_5, top1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--val-per-shard", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate-pp-tol", type=float, default=2.0,
                    help="max allowed top-1 absolute drop in pp (1pp = 0.01)")
    args = ap.parse_args()

    print(f"[gate] tflite      = {args.tflite}")
    print(f"[gate] ckpt        = {args.ckpt}")
    print(f"[gate] anchors     = {args.anchors}")
    print(f"[gate] shard_dir   = {args.shard_dir}")
    print(f"[gate] max_samples = {args.max_samples}")

    print(f"[gate] collecting val samples...")
    images, labels = collect_val_samples(
        Path(args.shard_dir), args.max_samples,
        val_per_shard=args.val_per_shard,
        seed=args.seed,
    )
    print(f"[gate] got {len(images)} samples,  range=[{images.min():.2f}, {images.max():.2f}]")

    anchors = np.load(args.anchors)
    print(f"[gate] anchors {anchors.shape}")

    print(f"[gate] running PT FP32...")
    t0 = time.time()
    pt_emb = run_pt(Path(args.ckpt), images)
    t_pt = time.time() - t0
    print(f"[gate]   PT done in {t_pt:.1f}s,  emb {pt_emb.shape}")

    print(f"[gate] running TFLite INT8...")
    t0 = time.time()
    tfl_emb = run_tflite_int8(Path(args.tflite), images)
    t_tfl = time.time() - t0
    print(f"[gate]   TFL done in {t_tfl:.1f}s,  emb {tfl_emb.shape}")

    print()
    print(f"=== Top-1/5 vs anchor DB ===")
    pt_top1, pt_top5, pt_pred = cosine_top1(pt_emb, anchors, labels)
    tfl_top1, tfl_top5, tfl_pred = cosine_top1(tfl_emb, anchors, labels)
    print(f"  PT  FP32   emb top-1 = {pt_top1*100:6.2f}%   top-5 = {pt_top5*100:6.2f}%")
    print(f"  TFL INT8   emb top-1 = {tfl_top1*100:6.2f}%   top-5 = {tfl_top5*100:6.2f}%")

    drop_top1 = (pt_top1 - tfl_top1) * 100
    drop_top5 = (pt_top5 - tfl_top5) * 100
    print(f"  Δ top-1   = {drop_top1:+.2f} pp   (gate: < {args.gate_pp_tol:.1f} pp)")
    print(f"  Δ top-5   = {drop_top5:+.2f} pp")

    # Pred agreement
    pred_match = (pt_pred == tfl_pred).mean()
    print(f"  PT↔TFL top-1 prediction agreement: {pred_match*100:.1f}%")

    # Embedding cosine similarity (sample-wise)
    pt_norms = np.linalg.norm(pt_emb, axis=1)
    tfl_norms = np.linalg.norm(tfl_emb, axis=1)
    cos_per_sample = (pt_emb * tfl_emb).sum(axis=1) / (pt_norms * tfl_norms).clip(min=1e-8)
    print(f"  PT↔TFL embedding cosine: mean={cos_per_sample.mean():.4f}  "
          f"min={cos_per_sample.min():.4f}  max={cos_per_sample.max():.4f}")

    print()
    if abs(drop_top1) <= args.gate_pp_tol:
        print(f"✅ P3-G2 PASS — INT8 top-1 drop {drop_top1:+.2f}pp within "
              f"±{args.gate_pp_tol:.1f}pp tolerance")
    else:
        print(f"❌ P3-G2 FAIL — INT8 top-1 drop {drop_top1:+.2f}pp exceeds "
              f"±{args.gate_pp_tol:.1f}pp tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
