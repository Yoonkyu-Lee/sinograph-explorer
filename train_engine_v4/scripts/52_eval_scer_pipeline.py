"""SCER deploy-time pipeline evaluator (doc/28 §6.2).

Measures the actual deploy path: structure soft filter → cosine NN search.

Pipeline per sample (doc/28 §4.2):
    1. backbone + heads forward
    2. structure prefilter:
         candidates = { c : c.radical    ∈ rad_top3(rad_logits)
                          ∧ c.idc        ∈ idc_top2(idc_logits)
                          ∧ |c.total_strokes − pred_strokes| ≤ 2 }
       (intersection of three sets per sample)
    3. cosine sim between embedding and anchor[candidates] → top-k
    4. final prediction = top-1 within filtered candidates

Reports four numbers (doc/28 §6.3 gates):
    char/full       — char head top-1/5 over full 98169 (legacy v3-style)
    emb/full        — embedding cosine top-1/5 vs full anchor table (no filter)
    scer/filtered   — embedding cosine top-1/5 within structure-filtered candidates
    filter coverage — avg # candidates / sample, % samples whose GT survives the filter

The gap between emb/full and scer/filtered tells us if the structure filter
is dropping the GT. The gap between char/full and emb/full tells us if the
embedding head reranked the same model better than the legacy char head.

Usage:
    python train_engine_v4/scripts/52_eval_scer_pipeline.py \\
        --ckpt train_engine_v4/out/16_scer_v1/best.pt \\
        --anchors deploy_pi/export/scer_anchor_db.npy \\
        --shard-dir synth_engine_v3/out/94_production_102k_x200 \\
        --max-samples 12000 \\
        --rad-topk 3 --idc-topk 2 --stroke-tol 2
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
import yaml
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "synth_engine_v3" / "scripts" / "cuda_raster"))

from train_engine_v3.modules.aux_labels import AuxTable        # noqa: E402
from train_engine_v3.modules.shard_dataset import (             # noqa: E402
    TensorShardDataset, build_stratified_val_split, list_shards,
)
from train_engine_v4.modules.model import build_scer            # noqa: E402


def make_gpu_transform(input_size: int):
    def _transform(x):
        x = x.float().div_(255.0)
        if x.shape[-1] != input_size or x.shape[-2] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear",
                               align_corners=False, antialias=True)
        x.sub_(0.5).div_(0.5)
        return x
    return _transform


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Trained SCER checkpoint")
    ap.add_argument("--anchors", required=True,
                    help="Anchor DB .npy from 51_build_anchor_db.py")
    ap.add_argument("--shard-dir", required=True,
                    help="Validation shards directory (uses class_index.json + aux_labels.npz)")
    ap.add_argument("--aux-labels", default=None,
                    help="aux_labels.npz path (default: <shard-dir>/aux_labels.npz)")
    ap.add_argument("--val-per-shard", type=int, default=3,
                    help="stratified val sampling (matches production config)")
    ap.add_argument("--max-samples", type=int, default=12000,
                    help="Cap evaluation at this many samples (None = all)")
    ap.add_argument("--rad-topk", type=int, default=3)
    ap.add_argument("--idc-topk", type=int, default=2)
    ap.add_argument("--stroke-tol", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="JSON output path (default: <ckpt-dir>/eval_scer_pipeline.json)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out) if args.out else ckpt_path.parent / "eval_scer_pipeline.json"
    print(f"[52_eval] device={device}  ckpt={ckpt_path}  out={out_path}")

    # ----- class_index + aux_labels -----
    shard_dir = Path(args.shard_dir)
    class_index = json.loads((shard_dir / "class_index.json").read_text(encoding="utf-8"))
    n_class = len(class_index)
    aux_path = Path(args.aux_labels) if args.aux_labels else shard_dir / "aux_labels.npz"
    aux = AuxTable.from_npz(aux_path, expected_class_index=class_index, device=device)
    print(f"[52_eval] n_class={n_class}  aux loaded n={aux.n_class}")

    # ----- model -----
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    emb_dim = int(ck.get("emb_dim", 128))
    print(f"[52_eval] ckpt epoch={ck.get('epoch')}  emb_dim={emb_dim}  "
          f"best {ck.get('best_metric_key')}={ck.get('best_metric_value', 'n/a')}")
    model = build_scer(name="resnet18", num_classes=n_class, emb_dim=emb_dim).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    gpu_transform = make_gpu_transform(128)

    # ----- anchor table -----
    anchors = torch.from_numpy(np.load(args.anchors)).to(device)        # (C, D)
    if anchors.shape[0] != n_class or anchors.shape[1] != emb_dim:
        raise RuntimeError(
            f"anchor shape {anchors.shape} != ({n_class}, {emb_dim})"
        )
    # Sanity: rows should be L2-normalized already (51 enforces this)
    print(f"[52_eval] anchors loaded shape={tuple(anchors.shape)}  "
          f"row_norms_mean={anchors.norm(dim=1).mean().item():.4f}")

    # ----- val data -----
    all_shards = list_shards(shard_dir)
    _, ds_val = build_stratified_val_split(
        all_shards, val_per_shard=args.val_per_shard, seed=0, shuffle_buffer=0,
    )
    dl_val = DataLoader(ds_val, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=False, persistent_workers=False)

    # Pre-fetch full per-class structure metadata (on device for fast filter)
    ALL_RAD = aux.radical                                         # (C,) long
    ALL_IDC = aux.idc                                              # (C,) long
    ALL_TOT = aux.total                                            # (C,) float
    ALL_VALID = aux.valid                                          # (C, 4) bool

    # ----- accumulators -----
    n_seen = 0
    char_correct = {1: 0, 5: 0}
    emb_full_correct = {1: 0, 5: 0}
    scer_correct = {1: 0, 5: 0}
    candidate_count_sum = 0                  # avg # candidates after filter
    gt_in_candidates = 0                     # filter recall: GT survived filter
    no_candidate_samples = 0                 # filter dropped everything
    fallback_to_full = 0                     # we fell back to full-table for these

    t_start = time.time()
    with torch.no_grad():
        for x, y in dl_val:
            if args.max_samples and n_seen >= args.max_samples:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = gpu_transform(x)
            out = model.forward_inference(x)

            # char head — only available via training-time forward
            char_out = model.char_head(model.backbone(x))
            _, char_pred = char_out.topk(5, dim=1)
            char_match = (char_pred == y.unsqueeze(1))
            char_correct[1] += char_match[:, :1].any(dim=1).sum().item()
            char_correct[5] += char_match[:, :5].any(dim=1).sum().item()

            # full-table embedding cosine (no filter, oracle)
            emb = out["embedding"]                                  # (B, D) L2-norm
            cos_full = emb @ anchors.t()                            # (B, C)
            _, full_pred = cos_full.topk(5, dim=1)
            full_match = (full_pred == y.unsqueeze(1))
            emb_full_correct[1] += full_match[:, :1].any(dim=1).sum().item()
            emb_full_correct[5] += full_match[:, :5].any(dim=1).sum().item()

            # structure soft filter — per sample
            rad_logits = out["radical"]                             # (B, 214)
            idc_logits = out["ids_top_idc"]                         # (B, 12)
            tot_pred = out["total_strokes"]                         # (B,)

            _, rad_topk_idx = rad_logits.topk(args.rad_topk, dim=1)         # (B, K_r)
            _, idc_topk_idx = idc_logits.topk(args.idc_topk, dim=1)         # (B, K_i)

            B = x.size(0)
            for i in range(B):
                rad_set = rad_topk_idx[i]                            # (K_r,)
                idc_set = idc_topk_idx[i]                            # (K_i,)
                stroke_pred = tot_pred[i].item()

                # boolean mask over (C,) classes
                # any-of: ALL_RAD[c] is in rad_set
                rad_ok = torch.isin(ALL_RAD, rad_set)
                idc_ok = torch.isin(ALL_IDC, idc_set)
                # only apply stroke filter where structure label is valid; else accept
                stroke_diff = (ALL_TOT - stroke_pred).abs()
                stroke_ok = stroke_diff <= float(args.stroke_tol)
                # If the structure label for class c is invalid (missing), don't
                # exclude it on that field — i.e. treat as accept.
                rad_ok = rad_ok | (~ALL_VALID[:, 0])
                idc_ok = idc_ok | (~ALL_VALID[:, 3])
                stroke_ok = stroke_ok | (~ALL_VALID[:, 1])

                cand_mask = rad_ok & idc_ok & stroke_ok               # (C,) bool
                n_cand = int(cand_mask.sum().item())
                candidate_count_sum += n_cand
                gt_idx = int(y[i].item())
                if cand_mask[gt_idx]:
                    gt_in_candidates += 1

                if n_cand == 0:
                    no_candidate_samples += 1
                    fallback_to_full += 1
                    # Fall back to full-table top-k (no filter)
                    pred_top5 = full_pred[i]
                else:
                    # cosine sim restricted to candidates
                    cand_idx = cand_mask.nonzero(as_tuple=True)[0]    # (n_cand,)
                    cand_anchors = anchors[cand_idx]                  # (n_cand, D)
                    cos_cand = emb[i:i+1] @ cand_anchors.t()          # (1, n_cand)
                    _, top_local = cos_cand.topk(min(5, n_cand), dim=1)
                    pred_top5 = cand_idx[top_local[0]]
                    # If we have fewer than 5 candidates, top-5 is shorter — pad
                    # to 5 with -1 so equality check below is well-defined
                    if pred_top5.numel() < 5:
                        pad = torch.full((5 - pred_top5.numel(),), -1,
                                          device=device, dtype=pred_top5.dtype)
                        pred_top5 = torch.cat([pred_top5, pad])

                if pred_top5[0] == y[i]:
                    scer_correct[1] += 1
                if (pred_top5 == y[i]).any():
                    scer_correct[5] += 1

            n_seen += B
            if (n_seen // args.batch_size) % 5 == 0:
                t = time.time() - t_start
                print(f"[52_eval] {n_seen} samples / "
                      f"{(args.max_samples or len(ds_val))}  ({t:.0f}s)", flush=True)

    t_total = time.time() - t_start

    def pct(num, den):
        return float(num) / max(den, 1)

    metrics = {
        "n_samples": n_seen,
        "wall_clock_sec": round(t_total, 2),

        "char/top1": pct(char_correct[1], n_seen),
        "char/top5": pct(char_correct[5], n_seen),

        "emb_full/top1": pct(emb_full_correct[1], n_seen),
        "emb_full/top5": pct(emb_full_correct[5], n_seen),

        "scer/top1": pct(scer_correct[1], n_seen),
        "scer/top5": pct(scer_correct[5], n_seen),

        "filter/avg_candidates": candidate_count_sum / max(n_seen, 1),
        "filter/gt_recall": pct(gt_in_candidates, n_seen),
        "filter/no_candidate_samples": no_candidate_samples,
        "filter/fallback_to_full": fallback_to_full,
        "filter/fallback_rate": pct(fallback_to_full, n_seen),
    }

    # Banner print
    print()
    print(f"=== SCER pipeline eval — {n_seen} samples in {t_total:.0f}s ===")
    print(f"  char/top1                = {metrics['char/top1']:.4f}     "
          f"(legacy 50MB FC head; v3 baseline 0.3899)")
    print(f"  char/top5                = {metrics['char/top5']:.4f}")
    print(f"  emb_full/top1            = {metrics['emb_full/top1']:.4f}     "
          f"(cosine vs all 98169 anchors)")
    print(f"  emb_full/top5            = {metrics['emb_full/top5']:.4f}")
    print(f"  scer/top1                = {metrics['scer/top1']:.4f}     "
          f"(structure prefilter + cosine)")
    print(f"  scer/top5                = {metrics['scer/top5']:.4f}")
    print(f"  filter/avg_candidates    = {metrics['filter/avg_candidates']:.1f}     "
          f"(out of {n_class})")
    print(f"  filter/gt_recall         = {metrics['filter/gt_recall']:.4f}     "
          f"(GT survives filter)")
    print(f"  filter/fallback_rate     = {metrics['filter/fallback_rate']:.4f}     "
          f"({fallback_to_full}/{n_seen} → full table)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[52_eval] wrote {out_path}")


if __name__ == "__main__":
    main()
