"""Port best.pt (PyTorch multi-head) → Keras char-only ResNet-18.

Reads `train_engine_v3/out/15_t5_light_v2/best.pt`, extracts the backbone
+ char_head weights, transposes Conv kernels (out,in,h,w) → (h,w,in,out),
and writes a Keras model file (.keras).

Aux heads (radical / idc / strokes) are dropped — Phase 1 is char-only.

Usage (from lab2-style WSL venv with TF 2.15 installed):
    python train_engine_v3/scripts/40_port_pytorch_to_keras.py \\
        --ckpt train_engine_v3/out/15_t5_light_v2/best.pt \\
        --out  deploy_pi/export/v3_keras_char.keras \\
        --num-classes 98169 --input-size 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train_engine_v3.modules.keras_resnet18 import (   # noqa: E402
    all_layer_name_mapping, build_keras_resnet18_char,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_pytorch_state(ckpt_path: Path) -> dict:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state["model"] if "model" in state else state
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def transfer_conv(keras_layer, pt_weight_oihw: np.ndarray) -> None:
    """Conv2D kernel: PyTorch (out, in, h, w) → Keras (h, w, in, out)."""
    k = np.transpose(pt_weight_oihw, (2, 3, 1, 0))
    expected = keras_layer.weights[0].shape
    if tuple(k.shape) != tuple(expected):
        raise ValueError(
            f"Conv shape mismatch for {keras_layer.name}: "
            f"got {k.shape}, expected {expected}"
        )
    keras_layer.set_weights([k])


def transfer_bn(keras_layer, pt_weight: np.ndarray, pt_bias: np.ndarray,
                pt_run_mean: np.ndarray, pt_run_var: np.ndarray) -> None:
    """BatchNormalization: gamma, beta, moving_mean, moving_variance."""
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
    ap.add_argument("--num-classes", type=int, default=98169)
    ap.add_argument("--input-size", type=int, default=128)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[port] ckpt = {ckpt_path}")
    print(f"[port] out  = {out_path}")
    print(f"[port] num_classes={args.num_classes} input_size={args.input_size}")

    pt = load_pytorch_state(ckpt_path)
    print(f"[port] loaded {len(pt)} PyTorch tensors")

    print(f"[port] building Keras model...")
    model = build_keras_resnet18_char(
        num_classes=args.num_classes, input_size=args.input_size,
    )
    print(f"[port] keras model has {len(model.layers)} layers, "
          f"{model.count_params() / 1e6:.2f} M params")

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
            print(f"[port] conv  {pt_prefix:50s} → {keras_name:30s} {tuple(w.shape)}")
        elif "bn" in keras_name:
            w = pt[f"{pt_prefix}.weight"]
            b = pt[f"{pt_prefix}.bias"]
            rm = pt[f"{pt_prefix}.running_mean"]
            rv = pt[f"{pt_prefix}.running_var"]
            transfer_bn(klayer, w, b, rm, rv)
            for s in ("weight", "bias", "running_mean", "running_var"):
                transferred_keys.add(f"{pt_prefix}.{s}")
            print(f"[port] bn    {pt_prefix:50s} → {keras_name:30s}")
        elif keras_name == "char_head":
            w = pt[f"{pt_prefix}.weight"]
            b = pt[f"{pt_prefix}.bias"]
            transfer_dense(klayer, w, b)
            transferred_keys.add(f"{pt_prefix}.weight")
            transferred_keys.add(f"{pt_prefix}.bias")
            print(f"[port] dense {pt_prefix:50s} → {keras_name:30s} {tuple(w.shape)}")
        else:
            print(f"[port] WARN unhandled keras layer '{keras_name}'")

    # Report any PyTorch tensors we did NOT consume (aux heads + bn num_batches_tracked)
    untouched = [k for k in pt
                 if k not in transferred_keys
                 and not k.endswith("num_batches_tracked")
                 and not k.startswith(("radical_head.", "total_strokes_head.",
                                        "residual_head.", "idc_head."))]
    if untouched:
        print(f"[port] WARN {len(untouched)} unconsumed tensor(s):")
        for k in untouched:
            print(f"          {k}")
    else:
        print(f"[port] all backbone + char_head tensors consumed")

    # aux head tensors are expected to be unconsumed in Phase 1
    aux = [k for k in pt
           if k.startswith(("radical_head.", "total_strokes_head.",
                            "residual_head.", "idc_head."))]
    print(f"[port] dropped {len(aux)} aux-head tensor(s) (Phase 1 = char-only)")

    print(f"[port] saving Keras model → {out_path}")
    model.save(str(out_path))
    sz_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6 \
            if out_path.is_dir() else out_path.stat().st_size / 1e6
    print(f"[port] wrote ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
