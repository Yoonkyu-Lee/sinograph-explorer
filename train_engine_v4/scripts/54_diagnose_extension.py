"""Extension 학습 결과 4 진단 (A: anchor cos, C: 1000-pack, D: 20 PNG).

20-epoch SCER extension 의 emb/top1 95.2% 가 *진짜* 학습인지, 아니면 anchor
crowding pathology 인지 확인하기 위한 PT FP32 진단.

A. **Anchor pairwise cosine analysis**
   100k random pair 의 cos 분포 — collapse (mean > 0.3 또는 >0.9 pair 다수)
   여부 판정. 정상은 mean ~0, std ~0.09 (Gaussian on sphere).

C. **1000-pack 정확도 (PT FP32)**
   `val_pack_1000.npz` (Pi epoch 10 baseline 34.30%) 으로 emb top-1/5.
   Sample distribution 무관하게 향상 보이면 진짜 학습.

D. **20 한자 PNG**
   real-world 합성 한자로 cosine NN top-1/5. epoch 10 의 50% top-1 vs.

Usage:
    python train_engine_v4/scripts/54_diagnose_extension.py \\
        --ckpt train_engine_v4/out/16_scer_v1/best.pt \\
        --anchors deploy_pi/export/scer_anchor_db_v20.npy \\
        --val-pack deploy_pi/export/val_pack_1000.npz \\
        --png-dir deploy_pi/test_chars \\
        --shard-dir synth_engine_v3/out/94_production_102k_x200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train_engine_v4.modules.model import build_scer       # noqa: E402


def char_to_codepoint_key(c: str) -> str:
    return f"U+{ord(c):04X}"


def codepoint_key_to_char(k: str) -> str:
    if not k.startswith("U+"):
        return "?"
    try:
        return chr(int(k[2:], 16))
    except (ValueError, OverflowError):
        return "?"


# ----------------------------------------------------------------------
# Diagnostic A: anchor pairwise cosine
# ----------------------------------------------------------------------

def diagnose_anchors(anchors: np.ndarray, n_pairs: int = 100_000,
                      seed: int = 0) -> dict:
    """Random pair sampling. Reports distribution + collapse signal."""
    rng = np.random.default_rng(seed)
    n_class = anchors.shape[0]
    print(f"\n{'='*70}\n[A] Anchor pairwise cosine — {n_pairs} random pairs from "
          f"{n_class} anchors\n{'='*70}")
    i = rng.integers(0, n_class, size=n_pairs)
    j = rng.integers(0, n_class, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    cos = (anchors[i] * anchors[j]).sum(axis=1)            # (~N,)

    # Histogram bins
    bins = np.array([-0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01])
    hist, _ = np.histogram(cos, bins=bins)
    hist_pct = hist / len(cos) * 100

    pcts = {f"p{p}": float(np.percentile(cos, p)) for p in [50, 90, 99, 99.9, 100]}

    n_collapsed = int((cos > 0.9).sum())
    n_orth = int(((cos > -0.1) & (cos < 0.1)).sum())

    print(f"  pairs analyzed   : {len(cos)}")
    print(f"  mean ± std       : {cos.mean():+.4f} ± {cos.std():.4f}")
    print(f"  median           : {np.median(cos):+.4f}")
    print(f"  percentiles      : p50={pcts['p50']:+.3f}  p90={pcts['p90']:+.3f}  "
          f"p99={pcts['p99']:+.3f}  p99.9={pcts['p99.9']:+.3f}  max={pcts['p100']:+.3f}")
    print(f"  near-orthogonal (-0.1, 0.1) : {n_orth} ({n_orth/len(cos)*100:.1f}%)")
    print(f"  collapsed (>0.9) : {n_collapsed} ({n_collapsed/len(cos)*100:.3f}%)")
    print(f"  histogram:")
    for k in range(len(hist)):
        bar = "#" * int(hist_pct[k] * 1.0)
        print(f"    [{bins[k]:+.2f}, {bins[k+1]:+.2f})  "
              f"{hist[k]:7d}  {hist_pct[k]:5.1f}%  {bar}")

    # Diagnosis
    if cos.mean() > 0.3 or n_collapsed / len(cos) > 0.01:
        diagnosis = "COLLAPSE_SUSPECTED"
        print(f"  → ⚠️ {diagnosis}")
    elif abs(cos.mean()) < 0.05 and cos.std() < 0.15:
        diagnosis = "WELL_SEPARATED"
        print(f"  → ✅ {diagnosis} (mean ≈ 0, std small)")
    else:
        diagnosis = "INTERMEDIATE"
        print(f"  → 🟡 {diagnosis}")

    return {
        "n_pairs": int(len(cos)),
        "mean": float(cos.mean()),
        "std": float(cos.std()),
        "median": float(np.median(cos)),
        "percentiles": pcts,
        "near_orthogonal_pct": float(n_orth / len(cos) * 100),
        "collapsed_pct": float(n_collapsed / len(cos) * 100),
        "histogram_bins": bins.tolist(),
        "histogram_counts": hist.tolist(),
        "diagnosis": diagnosis,
    }


# ----------------------------------------------------------------------
# Forward helpers
# ----------------------------------------------------------------------

def make_gpu_transform(input_size: int):
    def _t(x):
        x = x.float().div_(255.0)
        if x.shape[-1] != input_size or x.shape[-2] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear",
                               align_corners=False, antialias=True)
        x.sub_(0.5).div_(0.5)
        return x
    return _t


def forward_emb(model, images_hwc: np.ndarray, device: str,
                  batch: int = 64) -> np.ndarray:
    """Forward NHWC uint8 → L2-norm emb (N, 128)."""
    transform = make_gpu_transform(128)
    embs = []
    with torch.no_grad():
        for s in range(0, len(images_hwc), batch):
            x = images_hwc[s:s + batch]                                  # NHWC uint8
            x = torch.from_numpy(x).to(device, non_blocking=True)
            x = x.permute(0, 3, 1, 2)                                     # NHWC → NCHW
            x = transform(x)
            out = model.forward_inference(x)
            embs.append(out["embedding"].cpu().numpy())
    return np.concatenate(embs, axis=0)


# ----------------------------------------------------------------------
# Diagnostic C: 1000-pack
# ----------------------------------------------------------------------

def diagnose_val_pack(model, val_pack_path: Path, anchors: np.ndarray,
                       device: str) -> dict:
    print(f"\n{'='*70}\n[C] 1000-pack accuracy — {val_pack_path.name}\n{'='*70}")
    pack = np.load(val_pack_path)
    images = pack["images"]                                             # NHWC uint8
    labels = pack["labels"]
    print(f"  loaded {images.shape} dtype={images.dtype}  unique labels="
          f"{len(np.unique(labels))}")

    t0 = time.time()
    emb = forward_emb(model, images, device, batch=64)
    fwd_sec = time.time() - t0
    print(f"  forward {fwd_sec:.1f}s ({len(images)/fwd_sec:.0f} samples/sec)")

    # Cosine NN
    sims = emb @ anchors.T                                              # (N, C)
    top5 = np.argsort(-sims, axis=1)[:, :5]
    top1_correct = (top5[:, 0] == labels).sum()
    top5_correct = (top5 == labels[:, None]).any(axis=1).sum()
    n = len(labels)
    print(f"  top-1 = {top1_correct}/{n} ({top1_correct/n*100:.2f}%)")
    print(f"  top-5 = {top5_correct}/{n} ({top5_correct/n*100:.2f}%)")
    print(f"  baseline (Pi epoch 10): top-1 = 34.30%, top-5 = 48.60%")
    delta_top1 = top1_correct / n * 100 - 34.30
    print(f"  Δ top-1 = {delta_top1:+.2f} pp")
    return {
        "n_samples": int(n),
        "top1_pct": float(top1_correct / n * 100),
        "top5_pct": float(top5_correct / n * 100),
        "delta_top1_vs_e10_baseline_pp": float(delta_top1),
    }


# ----------------------------------------------------------------------
# Diagnostic D: 20 PNG
# ----------------------------------------------------------------------

def diagnose_pngs(model, png_dir: Path, anchors: np.ndarray,
                    class_index: dict, device: str) -> dict:
    print(f"\n{'='*70}\n[D] 20 한자 PNG real samples — {png_dir}\n{'='*70}")
    idx_to_key = {v: k for k, v in class_index.items()}
    pngs = sorted(png_dir.glob("*.png"))
    print(f"  found {len(pngs)} PNG files")

    images = []
    gt_chars = []
    gt_indices = []
    skipped = []
    for png in pngs:
        gt_char = png.stem
        gt_key = char_to_codepoint_key(gt_char)
        gt_idx = class_index.get(gt_key)
        if gt_idx is None:
            skipped.append(gt_char)
            continue
        img = Image.open(png).convert("RGB")
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.BILINEAR)
        images.append(np.array(img, dtype=np.uint8))
        gt_chars.append(gt_char)
        gt_indices.append(gt_idx)
    if not images:
        print("  no usable PNGs"); return {}
    images = np.stack(images, axis=0)                                   # (N, H, W, 3)
    gt_indices = np.array(gt_indices)
    print(f"  usable {len(images)} (skipped: {skipped})")

    emb = forward_emb(model, images, device, batch=64)
    sims = emb @ anchors.T
    top5_idx = np.argsort(-sims, axis=1)[:, :5]
    top5_sim = np.take_along_axis(sims, top5_idx, axis=1)

    # Per-image table + accuracy
    print(f"\n  {'GT':>3}  {'top-5 (cos sim)':<70}  rank")
    print("  " + "-" * 90)
    n = len(gt_chars)
    top1_correct = 0
    top5_correct = 0
    per_image = []
    for i, gt_char in enumerate(gt_chars):
        gt_idx = gt_indices[i]
        top_chars = [codepoint_key_to_char(idx_to_key[int(j)])
                     for j in top5_idx[i]]
        top_str = " ".join(f"{c}({s:.2f})" for c, s in zip(top_chars, top5_sim[i]))
        rank = int((sims[i] > sims[i, gt_idx]).sum())
        is_t1 = int(top5_idx[i, 0] == gt_idx)
        is_t5 = int((top5_idx[i] == gt_idx).any())
        top1_correct += is_t1
        top5_correct += is_t5
        flag = "✓" if is_t1 else ("·" if is_t5 else "✗")
        print(f"  {flag} {gt_char}  {top_str:<70}  {rank}")
        per_image.append({
            "gt": gt_char, "top5": top_chars,
            "top5_sim": [float(s) for s in top5_sim[i]],
            "rank": rank, "top1_correct": bool(is_t1),
            "top5_correct": bool(is_t5),
        })

    print(f"\n  top-1 = {top1_correct}/{n} ({top1_correct/n*100:.1f}%)")
    print(f"  top-5 = {top5_correct}/{n} ({top5_correct/n*100:.1f}%)")
    print(f"  baseline (Pi epoch 10): top-1 = 50.0%, top-5 = 70.0%")
    return {
        "n_images": n, "skipped": skipped,
        "top1_pct": float(top1_correct / n * 100),
        "top5_pct": float(top5_correct / n * 100),
        "per_image": per_image,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--val-pack", required=True)
    ap.add_argument("--png-dir", required=True)
    ap.add_argument("--shard-dir", required=True,
                    help="for class_index.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-pairs", type=int, default=100_000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diag] device = {device}")
    print(f"[diag] ckpt    = {args.ckpt}")
    print(f"[diag] anchors = {args.anchors}")

    # Load
    class_index_path = Path(args.shard_dir) / "class_index.json"
    class_index = json.loads(class_index_path.read_text(encoding="utf-8"))
    n_class = len(class_index)
    print(f"[diag] n_class = {n_class}")

    anchors = np.load(args.anchors)                                   # (C, 128) fp32
    print(f"[diag] anchors {anchors.shape} dtype={anchors.dtype}")

    print(f"[diag] loading model ...")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    emb_dim = int(ck.get("emb_dim", 128))
    model = build_scer(name="resnet18", num_classes=n_class, emb_dim=emb_dim).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[diag]   emb_dim={emb_dim}  ckpt_epoch={ck.get('epoch')}  "
          f"best_metric={ck.get('best_metric_value', 0.0):.4f}")

    # Run diagnostics
    result = {
        "ckpt_epoch": int(ck.get("epoch", -1)),
        "ckpt_best_metric": float(ck.get("best_metric_value", -1.0)),
        "anchors_path": str(args.anchors),
    }
    result["A_anchors"] = diagnose_anchors(anchors, n_pairs=args.n_pairs)
    result["C_val_pack"] = diagnose_val_pack(model, Path(args.val_pack),
                                              anchors, device)
    result["D_pngs"] = diagnose_pngs(model, Path(args.png_dir), anchors,
                                       class_index, device)

    # Save
    out_path = Path(args.out) if args.out else \
        Path(args.ckpt).parent / "diagnose_extension.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[diag] wrote {out_path}")

    # Summary
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    a = result["A_anchors"]
    c = result["C_val_pack"]
    d = result["D_pngs"]
    print(f"  A. anchors  : mean={a['mean']:+.4f} std={a['std']:.4f} "
          f"collapse={a['collapsed_pct']:.3f}%  → {a['diagnosis']}")
    print(f"  C. 1000-pack: top-1={c['top1_pct']:.2f}%  top-5={c['top5_pct']:.2f}%  "
          f"(Δ vs e10: {c['delta_top1_vs_e10_baseline_pp']:+.2f}pp)")
    print(f"  D. 20 PNG   : top-1={d['top1_pct']:.1f}%  top-5={d['top5_pct']:.1f}%")


if __name__ == "__main__":
    main()
