"""Export ONLY the backbone (no char head, no aux heads) to ONNX.

For Edge TPU PoC: verify that ResNet-18 backbone alone (~11M params, no
98k FC) can be quantized + Edge TPU compiled. If yes, SCER is viable —
the deployable model becomes backbone + small heads + embedding linear,
all together <15 MB.

Output: ONNX with input (N, 3, 128, 128), output (N, 512) — backbone
features only. No classifier.

Usage:
  python train_engine_v3/scripts/22b_export_backbone_only.py \\
      --ckpt train_engine_v3/out/15_t5_light_v2/best.pt \\
      --out deploy_pi/export/v3_backbone.onnx \\
      --opset 11
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train_engine_v3.modules.model import build_model     # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class BackboneOnly(nn.Module):
    """Wraps multi-head model to expose only backbone (512-d feature)."""
    def __init__(self, multi_head):
        super().__init__()
        self.backbone = multi_head.backbone

    def forward(self, x):
        return self.backbone(x)             # (N, 512)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--num-classes", type=int, default=98169,
                    help="for build_model() — doesn't affect backbone export")
    args = ap.parse_args()

    print(f"[export] backbone only  input_size={args.input_size}  opset={args.opset}")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    model = build_model("resnet18", num_classes=args.num_classes)
    model.load_state_dict(sd)
    model.eval()

    wrapper = BackboneOnly(model).eval()
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f"[export] backbone params = {n_params/1e6:.2f} M")

    dummy = torch.randn(1, 3, args.input_size, args.input_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=["input"],
        output_names=["feature"],
        dynamic_axes={"input": {0: "N"}, "feature": {0: "N"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export] wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
