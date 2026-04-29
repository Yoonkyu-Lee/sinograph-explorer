"""Anchor DB builder (doc/28 §6.1).

Takes a trained SCER checkpoint and exports the per-class anchor embedding
database that powers deploy-time cosine NN search.

The ArcFace classifier weight is, by construction, the per-class anchor in
the (L2-normalized) embedding space. We just normalize it once and dump it
as `deploy_pi/export/scer_anchor_db.npy`.

Modes (doc/28 §6.1):
    weight  (default) — ArcFace classifier weight directly; cheap and clean
    mean              — per-class mean of training-set embeddings;
                        requires a forward pass over the corpus
    blend             — 0.5*weight + 0.5*mean, then re-L2

For Phase 2 launch we ship `weight` only. `mean` and `blend` can be added
later when we want to compare; the script accepts the flag but errors out.

Usage:
    python train_engine_v4/scripts/51_build_anchor_db.py \\
        --ckpt train_engine_v4/out/16_scer_v1/best.pt \\
        --out  deploy_pi/export/scer_anchor_db.npy \\
        --mode weight
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to trained SCER checkpoint (best.pt or last.pt)")
    ap.add_argument("--out", required=True,
                    help="Output .npy path (anchor table)")
    ap.add_argument("--mode", default="weight",
                    choices=["weight", "mean", "blend"],
                    help="Anchor source mode (doc/28 §6.1). Phase 2 ships weight.")
    ap.add_argument("--device", default="cpu",
                    help="Torch device for ckpt load (cpu is fine, no compute)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode != "weight":
        raise NotImplementedError(
            f"mode={args.mode!r} requires forward pass over training corpus; "
            f"Phase 2 ships mode=weight. Add 'mean' / 'blend' if/when comparing."
        )

    print(f"[51_anchor] loading ckpt: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    state = ck["model"]

    # The Parameter is registered as `arc_classifier.weight` shape (n_class, emb_dim)
    weight_key = "arc_classifier.weight"
    if weight_key not in state:
        # support possible "_orig_mod." prefix from torch.compile
        compiled_key = "_orig_mod." + weight_key
        if compiled_key in state:
            weight_key = compiled_key
        else:
            available = [k for k in state.keys() if "arc_classifier" in k]
            raise KeyError(
                f"arc_classifier.weight not found in ckpt. "
                f"Available arc_classifier* keys: {available}"
            )
    W = state[weight_key].to(torch.float32)        # (n_class, emb_dim)
    n_class, emb_dim = W.shape
    print(f"[51_anchor] arc_classifier.weight  shape=({n_class}, {emb_dim})")

    # L2-normalize (ArcFace forward already operates on normalized weight, but
    # the stored Parameter is unnormalized — normalization is applied in forward).
    W_norm = F.normalize(W, dim=1)                 # (n_class, emb_dim)

    # Sanity: every row should now have unit L2 norm
    row_norms = W_norm.norm(dim=1)
    if not torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5):
        max_dev = (row_norms - 1.0).abs().max().item()
        raise RuntimeError(
            f"L2 normalization failed: max row-norm deviation {max_dev:.6f}"
        )
    print(f"[51_anchor] L2 normalized, row norms ≈ 1 (max deviation "
          f"{(row_norms - 1.0).abs().max().item():.2e})")

    arr = W_norm.cpu().numpy().astype(np.float32)
    np.save(str(out_path), arr)
    print(f"[51_anchor] wrote {out_path}  ({arr.nbytes / 1024 / 1024:.2f} MiB)")

    # Metadata sidecar — gives the deploy code the info it needs to validate
    meta_path = out_path.with_suffix(".json")
    h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    meta = {
        "mode": args.mode,
        "n_class": int(n_class),
        "emb_dim": int(emb_dim),
        "dtype": "float32",
        "l2_normalized": True,
        "source_ckpt": str(ckpt_path.resolve()),
        "source_epoch": int(ck.get("epoch", -1)),
        "source_metrics": ck.get("metrics", {}),
        "source_best_metric_value": float(ck.get("best_metric_value", -1.0)),
        "source_best_metric_key": str(ck.get("best_metric_key", "")),
        "sha256_prefix": h,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"[51_anchor] wrote {meta_path}")
    print(f"[51_anchor] DONE — anchor DB ready for 52_eval_scer_pipeline.py")


if __name__ == "__main__":
    main()
