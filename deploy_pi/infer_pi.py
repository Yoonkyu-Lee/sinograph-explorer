"""Raspberry Pi inference — standalone single-image OCR.

Uses `tflite_runtime` (lightweight, no full TF install needed).
Preprocessing MUST match train_engine_v2's GPU transform:
  float / 255 → resize(input_size, bilinear) → (x - 0.5) / 0.5 → [-1, 1]

For INT8 TFLite:
  quantize input via (x / input_scale + input_zero_point).astype(int8)
  dequantize output via (y - output_zero_point) * output_scale

Usage (on Pi):
  python3 infer_pi.py --model model_int8.tflite \\
                      --class-index class_index.json \\
                      --image photo.jpg \\
                      --topk 5 \\
                      [--family-db canonical_v2.sqlite]

For Coral Edge TPU:
  --model model_int8_edgetpu.tflite    (post edgetpu_compiler step)
  --use-coral                           (loads Coral delegate)

Dependencies on Pi:
  pip install tflite-runtime numpy Pillow
  # Coral TPU additionally: pip install pycoral  OR  libedgetpu1-std
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Interpreter priority: tflite_runtime (Pi, lightweight) → ai_edge_litert (modern) → TF
load_delegate = None
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate  # Pi default
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter            # modern LiteRT
        try:
            from ai_edge_litert.interpreter import load_delegate
        except ImportError:
            pass
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter    # legacy fallback

# Windows stdout utf-8 for CJK
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def preprocess(img_path: str, input_size: int, layout: str = "nhwc") -> np.ndarray:
    """Load + resize + normalize → (1, 3, H, H) / (1, H, H, 3) float32 in [-1, 1].

    Matches train_engine_v2 make_gpu_transform:
      - PIL.Image.open → RGB
      - resize shorter side to input_size (bilinear)
      - center crop to (input_size, input_size)
      - float / 255 → sub 0.5 → div 0.5

    layout: "nhwc" (TFLite default after onnx2tf conversion) or "nchw" (raw PyTorch/ONNX).
    """
    img = Image.open(img_path).convert("RGB")
    # resize shorter side
    w, h = img.size
    if w < h:
        new_w, new_h = input_size, int(h * input_size / w)
    else:
        new_w, new_h = int(w * input_size / h), input_size
    img = img.resize((new_w, new_h), Image.BILINEAR)
    # center crop
    left = (new_w - input_size) // 2
    top = (new_h - input_size) // 2
    img = img.crop((left, top, left + input_size, top + input_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
    arr = (arr - 0.5) / 0.5                            # [-1, 1]
    if layout.lower() == "nchw":
        arr = np.transpose(arr, (2, 0, 1))            # (3, H, W)
    # NHWC: leave as (H, W, 3)
    arr = np.expand_dims(arr, 0)                      # (1, *, *, *)
    return arr


def quantize_if_int8(arr: np.ndarray, input_detail: dict) -> np.ndarray:
    """If model expects int8 input, quantize float32 → int8."""
    if input_detail["dtype"] == np.float32:
        return arr
    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        # Not actually quantized (edge case)
        return arr.astype(input_detail["dtype"])
    q = (arr / scale + zero_point).round().astype(input_detail["dtype"])
    return q


def dequantize_if_int8(out: np.ndarray, output_detail: dict) -> np.ndarray:
    """If model output is int8, dequantize to float32."""
    if output_detail["dtype"] == np.float32:
        return out
    scale, zero_point = output_detail["quantization"]
    if scale == 0:
        return out.astype(np.float32)
    return (out.astype(np.float32) - zero_point) * scale


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tflite model file")
    ap.add_argument("--class-index", required=True, help="class_index.json")
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--input-size", type=int, default=128,
                    help="must match training config.model.input_size")
    ap.add_argument("--use-coral", action="store_true",
                    help="load Coral Edge TPU delegate (requires libedgetpu)")
    ap.add_argument("--family-db", default=None,
                    help="canonical_v2.sqlite for variant family lookup")
    ap.add_argument("--no-xnnpack", action="store_true",
                    help="force TF interpreter without default delegates "
                         "(dev-machine workaround when XNNPack crashes)")
    args = ap.parse_args()

    # Load class_index
    class_index = json.load(open(args.class_index, encoding="utf-8"))
    classes = sorted(class_index.keys(), key=lambda k: class_index[k])

    # Load interpreter
    delegates = []
    if args.use_coral:
        if load_delegate is None:
            raise SystemExit("Coral delegate requires tflite_runtime.load_delegate")
        # Linux/Pi: /usr/lib/x86_64-linux-gnu/libedgetpu.so.1
        #  or on Pi:  libedgetpu.so.1
        delegates.append(load_delegate("libedgetpu.so.1"))
    if args.no_xnnpack:
        # Dev-machine path: XNNPack delegate on some Windows/TF builds crashes
        # on specific ops; forcing TF interpreter without defaults works.
        import tensorflow as tf
        interp = tf.lite.Interpreter(
            model_path=args.model,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
    else:
        interp = Interpreter(model_path=args.model,
                             experimental_delegates=delegates or None)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"[infer] model: {args.model}")
    print(f"[infer] input: shape={inp['shape'].tolist()} dtype={inp['dtype'].__name__}")
    print(f"[infer] output: shape={out['shape'].tolist()} dtype={out['dtype'].__name__}")
    print(f"[infer] classes: {len(classes)}")
    print(f"[infer] coral: {args.use_coral}")

    # Auto-detect layout from input tensor shape: NHWC if last dim==3, else NCHW
    in_shape = inp["shape"].tolist()
    layout = "nhwc" if in_shape[-1] == 3 else "nchw"

    # Preprocess
    t0 = time.perf_counter()
    arr = preprocess(args.image, args.input_size, layout=layout)
    arr_q = quantize_if_int8(arr, inp)
    t_pre = (time.perf_counter() - t0) * 1000

    # Inference
    interp.set_tensor(inp["index"], arr_q)
    t0 = time.perf_counter()
    interp.invoke()
    t_inf = (time.perf_counter() - t0) * 1000

    logits = interp.get_tensor(out["index"])[0]   # (n_classes,)
    logits = dequantize_if_int8(logits, out)

    probs = softmax(logits)
    top_idx = np.argsort(-probs)[:args.topk]

    print(f"\n[timing] preprocess={t_pre:.1f}ms  invoke={t_inf:.1f}ms  "
          f"total={t_pre + t_inf:.1f}ms")
    print(f"\n=== Top-{args.topk} predictions ===")
    top_notations = []
    for rank, i in enumerate(top_idx):
        notation = classes[i]
        try:
            ch = chr(int(notation[2:], 16))
        except Exception:
            ch = "?"
        top_notations.append(notation)
        print(f"  #{rank + 1}  {notation}  '{ch}'  prob={probs[i] * 100:.1f}%")

    # Optional canonical DB lookup
    if args.family_db and Path(args.family_db).exists():
        con = sqlite3.connect(args.family_db)
        row = con.execute(
            "SELECT family_members_json FROM variant_components WHERE codepoint=?",
            (top_notations[0],),
        ).fetchone()
        con.close()
        if row and row[0]:
            members = json.loads(row[0])
            chars = []
            for m in members:
                try:
                    chars.append(chr(int(m[2:], 16)))
                except Exception:
                    chars.append("?")
            print(f"\n[family] top-1 {top_notations[0]} variant family ({len(members)}): "
                  f"{' '.join(chars)}")


if __name__ == "__main__":
    main()
