"""Raspberry Pi inference (ONNX runtime path) — single-image OCR.

Path A of 2-stage v3 deploy: load ONNX model directly with onnxruntime.
No quantization, FP32 inference. Slower than INT8 TFLite (~300-600ms vs
~150-400ms on Pi 4) but vastly simpler — no TF/onnx2tf install needed.

Pi setup:
  pip install onnxruntime numpy Pillow

Usage (single image):
  python3 infer_pi_onnx.py \\
      --model v3_char.onnx \\
      --class-index class_index.json \\
      --image test/標.png \\
      --topk 5

Usage (batch — directory of single-char images):
  python3 infer_pi_onnx.py \\
      --model v3_char.onnx \\
      --class-index class_index.json \\
      --images "test/*.png" \\
      --topk 5

For multi-head fusion (use radical/idc heads to rerank), run on PC with
30_predict.py — this script is char-head-only (matches the ONNX export).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def preprocess(img_path: Path, input_size: int = 128) -> np.ndarray:
    """Match train-time GPU transform exactly:
       PIL → RGB → pad-to-square (white) → resize bilinear → normalize [-1,1].

    Returns NCHW (1, 3, input_size, input_size) float32.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), color=(255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0    # (H, W, 3)
    arr = (arr - 0.5) / 0.5                                # [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))                     # (3, H, W)
    arr = np.expand_dims(arr, 0)                           # (1, 3, H, W)
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ONNX model file")
    ap.add_argument("--class-index", required=True)
    ap.add_argument("--image", default=None, help="single image (or use --images glob)")
    ap.add_argument("--images", default=None, help='glob, e.g. "test/*.png"')
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--input-size", type=int, default=128)
    args = ap.parse_args()

    if not args.image and not args.images:
        ap.error("must provide --image or --images")

    # Load class_index → idx → char
    class_index = json.load(open(args.class_index, encoding="utf-8"))
    n_class = len(class_index)
    classes_sorted = sorted(class_index.keys(), key=lambda k: class_index[k])
    print(f"[infer] classes = {n_class}")

    # Load ONNX session
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print(f"[infer] model: {args.model}")
    print(f"[infer] input: {sess.get_inputs()[0].shape}  output: {sess.get_outputs()[0].shape}")

    # Build image list
    if args.images:
        paths = sorted(Path().glob(args.images))
    else:
        paths = [Path(args.image)]
    if not paths:
        print(f"[infer] no images matched", file=sys.stderr)
        sys.exit(1)
    print(f"[infer] {len(paths)} image(s)\n")

    stats = {"n": 0, "top1": 0, "top5": 0, "t_pre_total_ms": 0.0, "t_inf_total_ms": 0.0}

    for path in paths:
        # Ground truth from filename stem (single char only)
        gt = path.stem if len(path.stem) == 1 else None

        t0 = time.perf_counter()
        arr = preprocess(path, args.input_size)
        t_pre = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        logits = sess.run(None, {inp_name: arr})[0][0]    # (n_class,)
        t_inf = (time.perf_counter() - t0) * 1000

        probs = softmax(logits)
        top_idx = np.argsort(-probs)[:args.topk]

        stats["n"] += 1
        stats["t_pre_total_ms"] += t_pre
        stats["t_inf_total_ms"] += t_inf

        gt_str = f" gt={gt}" if gt else ""
        print(f"  {path.name}{gt_str}  pre={t_pre:.1f}ms  inf={t_inf:.1f}ms")
        for rank, i in enumerate(top_idx):
            notation = classes_sorted[i]
            try:
                ch = chr(int(notation[2:], 16))
            except Exception:
                ch = "?"
            mark = "  ←GT" if (gt and ch == gt) else ""
            print(f"    #{rank + 1}  {notation}  {ch}   p={probs[i] * 100:.2f}%{mark}")
            if gt and ch == gt:
                if rank == 0: stats["top1"] += 1
                stats["top5"] += 1
        print()

    if stats["n"] > 1:
        n = stats["n"]
        avg_pre = stats["t_pre_total_ms"] / n
        avg_inf = stats["t_inf_total_ms"] / n
        gt_imgs = sum(1 for p in paths if len(p.stem) == 1)
        print("=" * 70)
        print(f"  AGGREGATE — {n} images, {gt_imgs} with single-char GT")
        print(f"  avg preprocess: {avg_pre:.1f}ms")
        print(f"  avg inference:  {avg_inf:.1f}ms")
        print(f"  avg total:      {avg_pre + avg_inf:.1f}ms ({1000/(avg_pre+avg_inf):.1f} img/s)")
        if gt_imgs > 0:
            print(f"  top-1 hit:  {stats['top1']:>3d}/{gt_imgs}  ({100*stats['top1']/gt_imgs:5.1f}%)")
            print(f"  top-{args.topk} hit:  {stats['top5']:>3d}/{gt_imgs}  ({100*stats['top5']/gt_imgs:5.1f}%)")


if __name__ == "__main__":
    main()
