"""Keras char-only ResNet-18 → INT8 TFLite (Edge TPU compatible).

Same path as Lab2 cell 23/35/49 (the FaceNet path that compiled 181/181 ops):
  Keras model
    ↓ tf.lite.TFLiteConverter.from_keras_model()
    ↓ + representative_dataset (300 samples NHWC, [-1, 1])
    ↓ + target_spec.supported_ops = [TFLITE_BUILTINS_INT8]
    ↓ + inference_input_type / output_type = tf.int8
  INT8 TFLite (Edge TPU compatible)
    ↓ edgetpu_compiler v3_keras_char_int8.tflite -o deploy_pi/export/

Run from lab2-style WSL venv (Python 3.11, TF 2.15.0).

Usage:
    python train_engine_v3/scripts/41_export_keras_tflite.py \\
        --keras deploy_pi/export/v3_keras_char.keras \\
        --shard-dir synth_engine_v3/out/94_production_102k_x200 \\
        --out deploy_pi/export/v3_keras_char_int8.tflite \\
        --calib-count 300 --input-size 128
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def collect_calib_nhwc(shard_dir: Path, n: int, input_size: int,
                        seed: int = 0) -> np.ndarray:
    """Pull N samples → (N, H, H, 3) float32 in [-1, 1] (NHWC, Keras-native)."""
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
        imgs = d["images"]
        per = min(len(imgs), n - len(samples))
        for i in rng.sample(range(len(imgs)), per):
            arr = imgs[i]
            if arr.shape[0] != input_size:
                arr = np.asarray(
                    Image.fromarray(arr).resize((input_size, input_size), Image.BILINEAR)
                )
            arr = arr.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            samples.append(arr)
    out = np.stack(samples[:n], axis=0)
    print(f"[calib] collected {out.shape} range=[{out.min():.3f},{out.max():.3f}] (NHWC)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", required=True, help="path to v3_keras_char.keras")
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-count", type=int, default=300)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keras_path = Path(args.keras)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run] keras      = {keras_path}")
    print(f"[run] shard_dir  = {args.shard_dir}")
    print(f"[run] out        = {out_path}")
    print(f"[run] calib_count= {args.calib_count}  input_size={args.input_size}")

    import tensorflow as tf

    print(f"[tf]    version = {tf.__version__}")
    print(f"[tf]    loading Keras model...")
    keras_model = tf.keras.models.load_model(str(keras_path))
    print(f"[tf]    loaded — input shape {keras_model.input_shape} "
          f"output {keras_model.output_shape}")

    calib = collect_calib_nhwc(Path(args.shard_dir), args.calib_count,
                               args.input_size, args.seed)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    def representative_data_gen():
        for i in range(len(calib)):
            yield [calib[i:i+1].astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print(f"[tflite] converting (this may take a few minutes)...")
    t0 = time.perf_counter()
    tflite_bytes = converter.convert()
    dt = time.perf_counter() - t0

    out_path.write_bytes(tflite_bytes)
    sz = len(tflite_bytes) / 1e6
    print(f"[tflite] wrote {out_path} ({sz:.1f} MB) in {dt:.1f}s")

    print()
    print("=" * 70)
    print(f"INT8 TFLite ready: {out_path} ({sz:.1f} MB)")
    print()
    print(f"Next:")
    print(f"  edgetpu_compiler {out_path} -o {out_path.parent}")


if __name__ == "__main__":
    main()
