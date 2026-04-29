"""PyTorch ↔ Keras forward parity check for SCER (doc/29 §7.1 P3-G1).

Loads the same checkpoint as 40_port_pytorch_to_keras.py and runs both the
PyTorch deploy graph (forward_inference) and the Keras model on the same
random batch, then compares the 5 outputs:

    embedding         max abs diff < 1e-5  (L2-normalized 128-d)
    radical           max abs diff < 1e-5
    total_strokes     max abs diff < 1e-5
    residual_strokes  max abs diff < 1e-5
    idc               max abs diff < 1e-5

Also reports radical/idc top-1 agreement on a small synthetic batch.

Usage:
    python train_engine_v4/scripts/42_verify_keras_parity.py \\
        --ckpt train_engine_v4/out/16_scer_v1/best.pt \\
        --keras deploy_pi/export/scer_keras.keras
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# These imports are heavy; load lazily inside main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--keras", required=True)
    ap.add_argument("--n", type=int, default=8, help="batch size for parity test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=98169)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--gate-tol", type=float, default=1e-5,
                    help="max abs diff threshold for PASS")
    args = ap.parse_args()

    print(f"[parity] ckpt  = {args.ckpt}")
    print(f"[parity] keras = {args.keras}")
    print(f"[parity] n={args.n}  seed={args.seed}  tol={args.gate_tol:.0e}")

    # ---- random batch (NCHW for PyTorch, NHWC for Keras) ----
    rng = np.random.default_rng(args.seed)
    x_chw = rng.standard_normal(
        (args.n, 3, args.input_size, args.input_size)
    ).astype(np.float32)                                       # PT input
    x_hwc = np.transpose(x_chw, (0, 2, 3, 1))                  # Keras input

    # ---- PyTorch forward ----
    print(f"[parity] PyTorch forward...")
    import torch
    from train_engine_v4.modules.model import build_scer
    pt_model = build_scer(
        name="resnet18", num_classes=args.num_classes,
        emb_dim=args.emb_dim,
    )
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pt_model.load_state_dict(ck["model"])
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model.forward_inference(torch.from_numpy(x_chw))
    pt_emb = pt_out["embedding"].numpy()              # (N, 128) L2-norm
    pt_rad = pt_out["radical"].numpy()                # (N, 214)
    pt_tot = pt_out["total_strokes"].numpy()          # (N,)
    pt_res = pt_out["residual_strokes"].numpy()       # (N,)
    pt_idc = pt_out["ids_top_idc"].numpy()            # (N, 12)
    print(f"[parity]   PT shapes: emb {pt_emb.shape}  rad {pt_rad.shape}  "
          f"tot {pt_tot.shape}  res {pt_res.shape}  idc {pt_idc.shape}")

    # ---- Keras forward ----
    print(f"[parity] Keras forward...")
    import tensorflow as tf
    # safe_mode=False allows Lambda layers with Python lambda to deserialize.
    # The Lambda we use is the L2 normalize wrapper from keras_scer.py — trusted.
    keras_model = tf.keras.models.load_model(
        args.keras, compile=False, safe_mode=False,
    )
    keras_outs = keras_model(x_hwc, training=False)
    if isinstance(keras_outs, (list, tuple)):
        k_emb, k_rad, k_tot, k_res, k_idc = (o.numpy() for o in keras_outs)
    else:
        raise RuntimeError("Keras model expected to return 5 outputs as list")
    print(f"[parity]   Keras shapes: emb {k_emb.shape}  rad {k_rad.shape}  "
          f"tot {k_tot.shape}  res {k_res.shape}  idc {k_idc.shape}")

    # ---- Compare ----
    def _cmp(name, pt, k, tol):
        diff = np.abs(pt - k)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        ok = max_d < tol
        marker = "✓" if ok else "✗"
        print(f"[parity] {marker} {name:20s}  max={max_d:.3e}  mean={mean_d:.3e}  "
              f"tol={tol:.0e}  {'PASS' if ok else 'FAIL'}")
        return ok

    print()
    print(f"=== Output parity ===")
    ok_emb = _cmp("embedding (L2-norm)", pt_emb, k_emb, args.gate_tol)
    ok_rad = _cmp("radical (logits)",    pt_rad, k_rad, args.gate_tol)
    ok_tot = _cmp("total_strokes",       pt_tot, k_tot, args.gate_tol)
    ok_res = _cmp("residual_strokes",    pt_res, k_res, args.gate_tol)
    ok_idc = _cmp("idc (logits)",        pt_idc, k_idc, args.gate_tol)

    # ---- top-1 agreement (rad / idc only — they're classification heads) ----
    rad_pt_top1 = pt_rad.argmax(axis=1)
    rad_k_top1 = k_rad.argmax(axis=1)
    idc_pt_top1 = pt_idc.argmax(axis=1)
    idc_k_top1 = k_idc.argmax(axis=1)
    rad_match = (rad_pt_top1 == rad_k_top1).mean()
    idc_match = (idc_pt_top1 == idc_k_top1).mean()
    print()
    print(f"=== Top-1 agreement ===")
    print(f"  radical top-1 match   = {rad_match*100:.1f}% ({(rad_pt_top1==rad_k_top1).sum()}/{args.n})")
    print(f"  idc     top-1 match   = {idc_match*100:.1f}% ({(idc_pt_top1==idc_k_top1).sum()}/{args.n})")

    # ---- L2 norm sanity on embedding output ----
    pt_norms = np.linalg.norm(pt_emb, axis=1)
    k_norms = np.linalg.norm(k_emb, axis=1)
    print(f"  pt    embedding L2 norm: min={pt_norms.min():.6f}  max={pt_norms.max():.6f}  mean={pt_norms.mean():.6f}")
    print(f"  keras embedding L2 norm: min={k_norms.min():.6f}  max={k_norms.max():.6f}  mean={k_norms.mean():.6f}")

    print()
    all_ok = all([ok_emb, ok_rad, ok_tot, ok_res, ok_idc])
    if all_ok:
        print(f"✅ PARITY OK — all 5 outputs within tol {args.gate_tol:.0e}")
    else:
        failed = [n for n, ok in zip(
            ["emb", "rad", "tot", "res", "idc"],
            [ok_emb, ok_rad, ok_tot, ok_res, ok_idc],
        ) if not ok]
        print(f"❌ PARITY FAIL — {failed} exceed tol")
        sys.exit(1)


if __name__ == "__main__":
    main()
