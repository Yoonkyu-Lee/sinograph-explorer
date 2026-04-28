"""[OBSOLETE — superseded by Phase 1 Keras-native path, see doc/27]

Historical note: this strict onnx_tf path produced an INT8 TFLite that
edgetpu_compiler v16 silent-failed on (no log, no ops mapped). The working
path is train_engine_v3/scripts/41_export_keras_tflite.py (doc/27).
Kept for historical context only.

ONNX → TFLite INT8 — STRICT Lab2 path (onnx_tf + tf.lite.TFLiteConverter).

Identical to Lab2 cells 20/23/35/49. Requires a venv with:
    onnx==1.14.1
    onnx-tf==1.10.0
    tensorflow==2.15.0
    protobuf<4

Usage:
    source ~/lab2-style-venv/bin/activate
    python train_engine_v3/scripts/23c_quantize_tflite_lab2_strict.py \\
        --onnx deploy_pi/export/v3_char.onnx \\
        --shard-dir synth_engine_v3/out/94_production_102k_x200 \\
        --out deploy_pi/export/v3_char_int8_lab2_strict.tflite \\
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

# Lab2 cell 19 의 환경 변수
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def collect_calib_samples_nchw(shard_dir: Path, n: int, input_size: int,
                                seed: int = 0) -> np.ndarray:
    """Pull N samples from shards → (N, 3, H, H) float32 in [-1, 1] (NCHW).

    onnx_tf preserves NCHW from ONNX (unlike onnx2tf which auto-transposes).
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
            arr = np.transpose(arr, (2, 0, 1))   # NHWC → CHW
            samples.append(arr)
    out = np.stack(samples[:n], axis=0)
    print(f"[calib] collected {out.shape} range=[{out.min():.3f},{out.max():.3f}] (NCHW)")
    return out


def onnx_to_savedmodel(onnx_path: Path, sm_dir: Path) -> None:
    """Lab2 cell 20/31/34: onnx_tf.backend.prepare() → SavedModel."""
    import onnx
    from onnx_tf.backend import prepare

    if sm_dir.exists():
        shutil.rmtree(sm_dir)

    print(f"[onnx_tf] loading {onnx_path.name}...")
    onnx_model = onnx.load(str(onnx_path))
    print(f"[onnx_tf] preparing TF representation...")
    tf_rep = prepare(onnx_model)
    print(f"[onnx_tf] exporting SavedModel to {sm_dir}...")
    tf_rep.export_graph(str(sm_dir))
    print(f"[onnx_tf] done")


def tflite_int8_from_savedmodel(sm_dir: Path, calib: np.ndarray,
                                 out_path: Path) -> None:
    """Lab2 cell 23/35/49: TFLiteConverter.from_saved_model + INT8 spec."""
    import tensorflow as tf

    print(f"[tflite] loading SavedModel from {sm_dir}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(sm_dir))

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
    ap.add_argument("--out", required=True)
    ap.add_argument("--saved-model-dir", default=None)
    ap.add_argument("--calib-count", type=int, default=300)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-savedmodel", action="store_true")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sm_dir = (Path(args.saved_model_dir) if args.saved_model_dir
              else out_path.parent / (out_path.stem + "_sm"))

    print(f"[run] onnx       = {onnx_path}")
    print(f"[run] shard_dir  = {args.shard_dir}")
    print(f"[run] sm_dir     = {sm_dir}")
    print(f"[run] out        = {out_path}")
    print(f"[run] calib_count= {args.calib_count}  input_size={args.input_size}")
    print()

    onnx_to_savedmodel(onnx_path, sm_dir)
    calib = collect_calib_samples_nchw(Path(args.shard_dir), args.calib_count,
                                        args.input_size, args.seed)
    tflite_int8_from_savedmodel(sm_dir, calib, out_path)

    print()
    print("=" * 70)
    print(f"INT8 TFLite ready: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"\nNext: edgetpu_compiler {out_path} -o {out_path.parent}")

    if not args.keep_savedmodel:
        try:
            shutil.rmtree(sm_dir)
            print(f"[cleanup] removed intermediate SavedModel dir")
        except Exception as e:
            print(f"[cleanup] could not remove {sm_dir}: {e}")


if __name__ == "__main__":
    main()
