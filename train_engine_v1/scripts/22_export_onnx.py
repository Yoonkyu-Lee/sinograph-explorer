"""Export a trained checkpoint to ONNX for downstream TFLite conversion.

Runs on CPU. Produces a single-input, single-output ONNX graph with input
shape (N, 3, H, H) where H = config.model.input_size. Batch dim is dynamic.

Usage:
  python 22_export_onnx.py --ckpt .../ckpt_epoch_20.pth \
                           --class-index .../class_index.json \
                           --config .../resnet18_t1_full.yaml \
                           --out .../model_t1.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--class-index", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    class_index = json.load(open(args.class_index, encoding="utf-8"))
    num_classes = len(class_index)
    input_size = cfg["model"]["input_size"]
    print(f"[export] classes={num_classes}  input_size={input_size}  opset={args.opset}")

    model = build_model(cfg["model"]["name"], num_classes)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[export] wrote {out_path}  ({size_mb:.1f} MB)")

    # parity check: torch vs onnxruntime
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] onnxruntime not installed; skipping parity check")
        return

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    import numpy as np
    diff = np.abs(torch_out - onnx_out).max()
    print(f"[export] torch vs onnx max abs diff: {diff:.2e}")
    assert diff < 1e-3, f"torch/onnx mismatch too large: {diff}"
    print("[export] parity ok")


if __name__ == "__main__":
    main()
