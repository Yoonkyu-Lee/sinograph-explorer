"""Port best.pt (PyTorch SCER) → Keras SCER deploy model (doc/29 §7.1).

Reads `train_engine_v4/out/16_scer_v1/best.pt`, extracts the deploy-only
weights (backbone + 4 structure heads + embedding head), transposes Conv
kernels (out,in,h,w) → (h,w,in,out), and writes a Keras model file.

Explicitly DROPS:
  - char_head     (deploy 안 됨, training-only warmup signal)
  - arc_classifier (anchor DB 로 별도 export — `51_build_anchor_db.py`)

Usage (from lab2-style WSL venv with TF 2.15):
    python train_engine_v4/scripts/40_port_pytorch_to_keras.py \\
        --ckpt train_engine_v4/out/16_scer_v1/best.pt \\
        --out  deploy_pi/export/scer_keras.keras
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train_engine_v4.modules.keras_scer import (   # noqa: E402
    all_layer_name_mapping, build_keras_scer,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Names of dense heads in the Keras model (deploy-only)
DENSE_HEADS = {
    "radical_head", "total_strokes_head", "residual_head",
    "idc_head", "embedding_head",
}


def load_pytorch_state(ckpt_path: Path) -> dict:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state["model"] if "model" in state else state
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def transfer_conv(keras_layer, pt_weight_oihw: np.ndarray) -> None:
    """Conv2D kernel: PyTorch (out, in, h, w) → Keras (h, w, in, out)."""
    k = np.transpose(pt_weight_oihw, (2, 3, 1, 0))
    expected = tuple(keras_layer.weights[0].shape)
    if tuple(k.shape) != expected:
        raise ValueError(
            f"Conv shape mismatch for {keras_layer.name}: "
            f"got {k.shape}, expected {expected}"
        )
    keras_layer.set_weights([k])


def transfer_bn(keras_layer, pt_weight: np.ndarray, pt_bias: np.ndarray,
                pt_run_mean: np.ndarray, pt_run_var: np.ndarray) -> None:
    weights = [pt_weight, pt_bias, pt_run_mean, pt_run_var]
    expected = [tuple(w.shape) for w in keras_layer.weights]
    got = [tuple(w.shape) for w in weights]
    if got != expected:
        raise ValueError(
            f"BN shape mismatch for {keras_layer.name}: got {got}, expected {expected}"
        )
    keras_layer.set_weights(weights)


def transfer_dense(keras_layer, pt_weight: np.ndarray, pt_bias: np.ndarray) -> None:
    """Dense kernel: PyTorch (out, in) → Keras (in, out)."""
    k = np.transpose(pt_weight, (1, 0))
    expected_k = tuple(keras_layer.weights[0].shape)
    expected_b = tuple(keras_layer.weights[1].shape)
    if tuple(k.shape) != expected_k or tuple(pt_bias.shape) != expected_b:
        raise ValueError(
            f"Dense shape mismatch for {keras_layer.name}: "
            f"kernel {k.shape} vs {expected_k}, bias {pt_bias.shape} vs {expected_b}"
        )
    keras_layer.set_weights([k, pt_bias])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-radicals", type=int, default=214)
    ap.add_argument("--num-idc", type=int, default=12)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--no-l2-in-graph", action="store_true",
                    help="If set, build Keras with raw embedding output (no L2 norm "
                         "in graph). For Edge TPU compatibility fallback.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Fixed batch size for deploy (Edge TPU requires fixed). "
                         "Default None = dynamic batch (for parity/training).")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[port] ckpt = {ckpt_path}")
    print(f"[port] out  = {out_path}")
    print(f"[port] num_radicals={args.num_radicals} num_idc={args.num_idc} "
          f"emb_dim={args.emb_dim} input_size={args.input_size} "
          f"l2_in_graph={not args.no_l2_in_graph}")

    pt = load_pytorch_state(ckpt_path)
    print(f"[port] loaded {len(pt)} PyTorch tensors")

    print(f"[port] building Keras SCER...")
    model = build_keras_scer(
        num_radicals=args.num_radicals,
        num_idc=args.num_idc,
        emb_dim=args.emb_dim,
        input_size=args.input_size,
        l2_normalize_in_graph=not args.no_l2_in_graph,
        batch_size=args.batch_size,
    )
    print(f"[port] keras model: {len(model.layers)} layers, "
          f"{model.count_params() / 1e6:.2f} M params, "
          f"{len(model.outputs)} outputs")

    mapping = all_layer_name_mapping()
    print(f"[port] mapping {len(mapping)} layer prefixes")

    transferred_keys: set[str] = set()

    for pt_prefix, keras_name in mapping.items():
        try:
            klayer = model.get_layer(keras_name)
        except ValueError:
            print(f"[port] WARN keras layer '{keras_name}' not found, skip")
            continue

        if "conv" in keras_name and "bn" not in keras_name:
            w = pt[f"{pt_prefix}.weight"]
            transfer_conv(klayer, w)
            transferred_keys.add(f"{pt_prefix}.weight")
            print(f"[port] conv  {pt_prefix:40s} → {keras_name:32s} {tuple(w.shape)}")
        elif "bn" in keras_name:
            w = pt[f"{pt_prefix}.weight"]
            b = pt[f"{pt_prefix}.bias"]
            rm = pt[f"{pt_prefix}.running_mean"]
            rv = pt[f"{pt_prefix}.running_var"]
            transfer_bn(klayer, w, b, rm, rv)
            for s in ("weight", "bias", "running_mean", "running_var"):
                transferred_keys.add(f"{pt_prefix}.{s}")
            print(f"[port] bn    {pt_prefix:40s} → {keras_name:32s}")
        elif keras_name in DENSE_HEADS:
            w = pt[f"{pt_prefix}.weight"]
            b = pt[f"{pt_prefix}.bias"]
            transfer_dense(klayer, w, b)
            transferred_keys.add(f"{pt_prefix}.weight")
            transferred_keys.add(f"{pt_prefix}.bias")
            print(f"[port] dense {pt_prefix:40s} → {keras_name:32s} {tuple(w.shape)}")
        else:
            print(f"[port] WARN unhandled keras layer '{keras_name}'")

    # Report any PyTorch tensors we did NOT consume.
    # Expected drops: char_head.*, arc_classifier.weight, num_batches_tracked
    expected_drop_prefixes = ("char_head.", "arc_classifier.")
    untouched = [
        k for k in pt
        if k not in transferred_keys
        and not k.endswith("num_batches_tracked")
        and not k.startswith(expected_drop_prefixes)
    ]
    if untouched:
        print(f"[port] WARN unexpected unconsumed tensors: {untouched}")
    else:
        print(f"[port] all backbone+head tensors consumed; char_head + "
              f"arc_classifier dropped (expected)")

    print(f"[port] saving Keras model to {out_path}")
    model.save(str(out_path))
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"[port] DONE — wrote {out_path} ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
