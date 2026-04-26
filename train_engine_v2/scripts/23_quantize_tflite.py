"""Convert ONNX → TFLite INT8 for edge deployment (Pi / Coral).

Pipeline:
  .pth  →  .onnx  (22_export_onnx.py)
     ↓
   onnx2tf  (ONNX → TF SavedModel, NHWC layout)
     ↓
  TFLiteConverter with representative dataset (this script)
     ↓
  .tflite INT8 quantized
     ↓
  (optional) edgetpu_compiler  (Coral Edge TPU — Linux only)

Calibration uses our tensor shards: we pull N random samples, preprocess to
exactly the training distribution (float/255 → (x-0.5)/0.5 → [-1, 1]), and
yield them through the converter's representative_dataset hook.

**IMPORTANT:** onnx2tf needs TensorFlow which requires Python <= 3.12.
Run this script with a dedicated venv, e.g. `d:/.../.venv_tflite/`.

Usage:
  python 23_quantize_tflite.py \\
    --onnx .../model.onnx \\
    --shard-dir .../synth_engine_v3/out/80_production_v3r_shard256 \\
    --out .../model_int8.tflite \\
    [--calib-count 300] [--input-size 128]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def collect_calib_samples(shard_dir: Path, n: int, input_size: int,
                           seed: int = 0) -> np.ndarray:
    """Pull N samples from shards, resize to input_size, normalize to [-1, 1].

    Returns: (N, H, H, 3) float32 in NHWC layout (TF/TFLite native).
    """
    import random
    from PIL import Image

    shards = sorted(shard_dir.glob("shard-*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    rng = random.Random(seed)
    rng.shuffle(shards)

    samples: list[np.ndarray] = []
    for s in shards:
        if len(samples) >= n:
            break
        d = np.load(s)
        imgs = d["images"]  # (N, H, W, 3) uint8 @ 256
        per_shard = min(len(imgs), n - len(samples))
        idx = rng.sample(range(len(imgs)), per_shard)
        for i in idx:
            arr = imgs[i]
            if arr.shape[0] != input_size:
                pil = Image.fromarray(arr).resize(
                    (input_size, input_size), Image.BILINEAR
                )
                arr = np.asarray(pil)
            arr = arr.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            samples.append(arr)
    out = np.stack(samples[:n], axis=0)
    print(f"[calib] collected {out.shape} range=[{out.min():.3f},{out.max():.3f}]")
    return out


def onnx_to_savedmodel(onnx_path: Path, saved_model_dir: Path,
                        input_size: int) -> None:
    """Convert ONNX (NCHW) → TF SavedModel (NHWC)."""
    import onnx2tf

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[onnx2tf] converting {onnx_path.name} → SavedModel")
    # onnx2tf auto-detects NCHW input and transposes to NHWC.
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    print(f"[onnx2tf] wrote {saved_model_dir}")


def tflite_int8_quantize(saved_model_dir: Path, calib_samples: np.ndarray,
                          out_path: Path, input_size: int) -> None:
    """Full-integer INT8 quantization via tf.lite.TFLiteConverter."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    # Full-integer quantization (Coral-compatible).
    def representative_dataset():
        for i in range(len(calib_samples)):
            s = calib_samples[i:i + 1]  # (1, H, H, 3)
            yield [s.astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("[tflite] quantizing (int8, Coral-compatible)...")
    t0 = time.perf_counter()
    tflite_bytes = converter.convert()
    dt = time.perf_counter() - t0
    out_path.write_bytes(tflite_bytes)
    sz = len(tflite_bytes) / 1024 / 1024
    print(f"[tflite] wrote {out_path} ({sz:.1f} MB) in {dt:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--shard-dir", required=True,
                    help="dir with shard-*.npz for INT8 calibration")
    ap.add_argument("--out", required=True, help="output .tflite path")
    ap.add_argument("--calib-count", type=int, default=300)
    ap.add_argument("--input-size", type=int, default=128,
                    help="model input H=W (must match training config)")
    ap.add_argument("--saved-model-dir", default=None,
                    help="intermediate TF SavedModel dir (default: sibling of out)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    shard_dir = Path(args.shard_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sm_dir = (Path(args.saved_model_dir)
              if args.saved_model_dir
              else out_path.parent / (out_path.stem + "_sm"))

    print(f"[run] onnx={onnx_path}")
    print(f"[run] shard_dir={shard_dir}")
    print(f"[run] out={out_path}")
    print(f"[run] calib_count={args.calib_count}  input_size={args.input_size}")

    calib = collect_calib_samples(shard_dir, args.calib_count,
                                   args.input_size, seed=args.seed)
    onnx_to_savedmodel(onnx_path, sm_dir, args.input_size)
    tflite_int8_quantize(sm_dir, calib, out_path, args.input_size)

    # Summary
    print()
    print("=" * 60)
    print(f"TFLite INT8 ready: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("Next (optional, Coral TPU — Linux only):")
    print(f"  edgetpu_compiler {out_path}")


if __name__ == "__main__":
    main()
