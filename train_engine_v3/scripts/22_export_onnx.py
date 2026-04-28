"""Export v3 multi-head ckpt to ONNX (char head only).

Multi-head model (char + 4 aux heads) at training time, but inference only
needs char head — aux heads are training-time regularizers (doc/19 §3).
This script wraps the model in a `forward_char_only` adapter and exports a
single-input single-output ONNX graph.

Output: ONNX with input (N, 3, 128, 128), output (N, n_class) char logits.

Usage:
  python train_engine_v3/scripts/22_export_onnx.py \
      --ckpt train_engine_v3/out/15_t5_light_v2/best.pt \
      --class-index synth_engine_v3/out/94_production_102k_x200/class_index.json \
      --out deploy_pi/export/v3_char.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train_engine_v3.modules.model import build_model     # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class CharOnlyWrapper(nn.Module):
    """Strip aux heads — exposes (input → char logits) only."""
    def __init__(self, multi_head):
        super().__init__()
        self.backbone = multi_head.backbone
        self.char_head = multi_head.char_head

    def forward(self, x):
        f = self.backbone(x)
        return self.char_head(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--class-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    class_index = json.loads(Path(args.class_index).read_text(encoding="utf-8"))
    n_class = len(class_index)
    print(f"[export] n_class={n_class}  input_size={args.input_size}  opset={args.opset}")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    model = build_model("resnet18", num_classes=n_class)
    model.load_state_dict(sd)
    model.eval()

    wrapper = CharOnlyWrapper(model).eval()
    dummy = torch.randn(1, 3, args.input_size, args.input_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export] wrote {out_path} ({size_mb:.1f} MB)")

    # parity check: torch vs onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("[export] onnxruntime not installed; skipping parity check")
        return

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = wrapper(dummy).numpy()
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    diff = np.abs(torch_out - onnx_out).max()
    print(f"[export] torch vs onnx max abs diff: {diff:.2e}")
    if diff > 1e-3:
        print(f"[export] WARNING: diff > 1e-3, but continuing", file=sys.stderr)
    else:
        print(f"[export] parity ok")


if __name__ == "__main__":
    main()
