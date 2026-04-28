"""[OBSOLETE — superseded by Phase 1 Keras-native path, see doc/27]

Historical note:
  Filename + initial docstring claim "onnx_tf + Lab2 pipeline" but the actual
  implementation here uses `onnx2tf.convert(..., output_keras_v3=True)` —
  it was renamed mid-experiment and the docstring lagged behind. The strict
  onnx_tf-based variant lives in 23c_quantize_tflite_lab2_strict.py.

  Both this script (23b) and the strict onnx_tf path (23c) failed to produce
  Edge-TPU-compilable INT8: edgetpu_compiler v16 reported 0/N ops mapped
  ("unsupported data type") regardless of which ONNX-derived intermediate
  was used. doc/25 documents the failure mode.

  The working path is Phase 1 (doc/26-27): rebuild the model in Keras-native
  layers and run TF's native TFLiteConverter on the Keras model — see
  train_engine_v3/scripts/41_export_keras_tflite.py. This script is kept
  only for historical context; do not use it for new deploys.

ONNX → TFLite INT8 via onnx2tf (Keras v3) → tf.lite.TFLiteConverter.from_keras_model.

Pipeline:
  ONNX
    ↓ onnx_tf.backend.prepare()
  TF SavedModel
    ↓ tf.lite.TFLiteConverter.from_saved_model()
    ↓ + representative_dataset (calibration)
    ↓ + target_spec.supported_ops = [TFLITE_BUILTINS_INT8]
    ↓ + inference_input_type / output_type = tf.int8
  INT8 TFLite (Edge TPU compatible)
    ↓ edgetpu_compiler
  Edge TPU compiled

Calibration: 300 samples from synth corpus, normalized to [-1, 1] (matching
training preprocess), yielded in NCHW format (PyTorch native — onnx_tf
preserves NCHW unlike onnx2tf which auto-transposes to NHWC).

Usage:
    python train_engine_v3/scripts/23b_quantize_tflite_lab2.py \\
        --onnx deploy_pi/export/v3_char.onnx \\
        --shard-dir synth_engine_v3/out/94_production_102k_x200 \\
        --out deploy_pi/export/v3_char_int8_lab2.tflite \\
        --calib-count 300 --input-size 128
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

# Lab2's environment quirks — TF legacy keras + protobuf python impl
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def collect_calib_samples_nhwc(shard_dir: Path, n: int, input_size: int,
                                seed: int = 0) -> np.ndarray:
    """Pull N samples from shards → (N, H, H, 3) float32 in [-1, 1] (NHWC).

    onnx2tf converts NCHW (ONNX) → NHWC (TF native), so the SavedModel
    expects NHWC input. Calibration data must match.
    """
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
        imgs = d["images"]               # (N, H, W, 3) uint8 NHWC
        per = min(len(imgs), n - len(samples))
        for i in rng.sample(range(len(imgs)), per):
            arr = imgs[i]                # (H, W, 3) uint8
            if arr.shape[0] != input_size:
                arr = np.asarray(
                    Image.fromarray(arr).resize((input_size, input_size), Image.BILINEAR)
                )
            arr = arr.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5      # → [-1, 1]
            samples.append(arr)          # keep NHWC
    out = np.stack(samples[:n], axis=0)  # (N, H, H, 3)
    print(f"[calib] collected {out.shape} range=[{out.min():.3f},{out.max():.3f}] (NHWC)")
    return out


def onnx_to_keras(onnx_path: Path, work_dir: Path) -> Path:
    """ONNX → Keras v3 (.keras) via onnx2tf.

    Originally tried `onnx_tf.backend.prepare()` (Lab2 path) but onnx_tf
    is broken with ONNX >= 1.15. Tried onnx2tf `output_signaturedefs=True`
    but that's a metadata flag, not SavedModel output. Settled on
    `output_keras_v3=True` which produces a .keras file that TF's
    TFLiteConverter.from_keras_model() can consume.

    Returns: path to the .keras file.
    """
    import onnx2tf

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[onnx2tf] converting {onnx_path.name} → Keras v3...")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(work_dir),
        output_keras_v3=True,              # ★ Keras v3 (.keras) 출력
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    # find the .keras file
    keras_files = list(work_dir.glob("*.keras"))
    if not keras_files:
        raise FileNotFoundError(f"no .keras file produced in {work_dir}")
    keras_path = keras_files[0]
    print(f"[onnx2tf] wrote {keras_path}")
    return keras_path


def tflite_int8_from_keras(keras_path: Path, calib: np.ndarray,
                            out_path: Path) -> None:
    """Keras v3 model → INT8 TFLite via tf.lite.TFLiteConverter.

    Same constraints as Lab2 cell 23/35/49:
      optimizations DEFAULT, representative_dataset, BUILTINS_INT8,
      inference_input_type=int8, inference_output_type=int8.

    This produces Edge TPU compatible INT8 (TF native converter spec).
    """
    import tensorflow as tf

    print(f"[tflite] loading Keras model from {keras_path}...")
    keras_model = tf.keras.models.load_model(str(keras_path))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True, help="output .tflite path")
    ap.add_argument("--saved-model-dir", default=None,
                    help="intermediate SavedModel dir (default: sibling of out)")
    ap.add_argument("--calib-count", type=int, default=300)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-savedmodel", action="store_true",
                    help="keep intermediate SavedModel dir (default: cleanup)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = (Path(args.saved_model_dir) if args.saved_model_dir
                else out_path.parent / (out_path.stem + "_intermediate"))

    print(f"[run] onnx       = {onnx_path}")
    print(f"[run] shard_dir  = {args.shard_dir}")
    print(f"[run] work_dir   = {work_dir}")
    print(f"[run] out        = {out_path}")
    print(f"[run] calib_count= {args.calib_count}  input_size={args.input_size}")
    print()

    # Step 1: ONNX → Keras v3 (.keras file)
    keras_path = onnx_to_keras(onnx_path, work_dir)

    # Step 2: collect calibration data (NHWC, matches model input)
    calib = collect_calib_samples_nhwc(Path(args.shard_dir), args.calib_count,
                                        args.input_size, args.seed)

    # Step 3: Keras → INT8 TFLite (Edge TPU compatible spec)
    tflite_int8_from_keras(keras_path, calib, out_path)

    print()
    print("=" * 70)
    print(f"INT8 TFLite ready: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print()
    print(f"Next:")
    print(f"  edgetpu_compiler {out_path} -o {out_path.parent}")
    print()

    if not args.keep_savedmodel:
        try:
            shutil.rmtree(work_dir)
            print(f"[cleanup] removed intermediate dir")
        except Exception as e:
            print(f"[cleanup] could not remove {work_dir}: {e}")


if __name__ == "__main__":
    main()
