"""Export a trained checkpoint to ONNX for downstream TFLite conversion.

Runs on CPU. Produces a single-input, single-output ONNX graph with input
shape (N, 3, H, H) where H = config.model.input_size. Batch dim is dynamic.

Usage:
  # explicit ckpt
  python 22_export_onnx.py --ckpt .../ckpt_epoch_20.pth \
                           --class-index .../class_index.json \
                           --config .../resnet18_t1_full.yaml \
                           --out .../model_t1.onnx

  # auto: pick best.pth + class_index.json + config from a run dir
  python 22_export_onnx.py --run-dir .../out/03_v3r_prod_t1 --out .../model.onnx
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


def _resolve_from_run_dir(run_dir: Path) -> tuple[Path, Path, Path]:
    """Auto-locate ckpt + class_index + embedded config from a run directory.

    Prefers best.pth (val top-1 winner) over epoch ckpts. Config is read from
    the ckpt dict (saved by 20_train.py) — no external yaml needed."""
    ckpt_path = run_dir / "best.pth"
    if not ckpt_path.exists():
        epoch_ckpts = sorted(run_dir.glob("ckpt_epoch_*.pth"))
        if not epoch_ckpts:
            raise FileNotFoundError(f"no ckpts found in {run_dir}")
        ckpt_path = epoch_ckpts[-1]
    class_index_path = run_dir / "class_index.json"
    if not class_index_path.exists():
        raise FileNotFoundError(f"no class_index.json in {run_dir}")
    return ckpt_path, class_index_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="auto-pick best.pth + class_index.json + embedded config "
                         "from a training output directory (overrides explicit args)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--class-index", default=None)
    ap.add_argument("--config", default=None,
                    help="training config yaml. Optional if --run-dir (config is "
                         "read from ckpt embedded state).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    # Resolve ckpt / class_index via --run-dir if provided.
    if args.run_dir:
        run_dir = Path(args.run_dir)
        rd_ckpt, rd_ci = _resolve_from_run_dir(run_dir)
        args.ckpt = args.ckpt or str(rd_ckpt)
        args.class_index = args.class_index or str(rd_ci)
        print(f"[export] run_dir={run_dir}  → ckpt={Path(args.ckpt).name}")
    if not args.ckpt or not args.class_index:
        ap.error("must provide --run-dir or both --ckpt + --class-index")

    class_index = json.load(open(args.class_index, encoding="utf-8"))
    num_classes = len(class_index)

    # Prefer cfg embedded in the ckpt (set by 20_train.py) over external yaml.
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "config" in state:
        cfg = state["config"]
    elif args.config:
        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    else:
        raise SystemExit("ckpt has no embedded config and --config not given")

    input_size = cfg["model"]["input_size"]
    print(f"[export] classes={num_classes}  input_size={input_size}  opset={args.opset}")

    model = build_model(cfg["model"]["name"], num_classes)
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
