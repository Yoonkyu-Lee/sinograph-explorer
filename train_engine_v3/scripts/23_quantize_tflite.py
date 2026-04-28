"""ONNX → TFLite INT8 (v3 multi-head, char head only).

Modern onnx2tf (>=1.20) writes TFLite directly via flatbuffer_direct, skipping
SavedModel intermediate. So we use onnx2tf's BUILT-IN INT8 quantization with
calibration data passed as a .npy file. One-step, no SavedModel needed.

Calibration: pull N samples from synth corpus shards, normalize to [-1,1]
(matching train preprocess), save as a .npy. onnx2tf reads this for per-tensor
quantization range estimation.

Usage:
  python train_engine_v3/scripts/23_quantize_tflite.py \\
      --onnx deploy_pi/export/v3_char.onnx \\
      --shard-dir synth_engine_v3/out/94_production_102k_x200 \\
      --out-dir deploy_pi/export/ \\
      --calib-count 300

Output (in --out-dir):
  v3_char_full_integer_quant.tflite     ← INT8, Coral-ready
  v3_char_integer_quant.tflite          ← INT8 weights, FP32 IO (slower)
  v3_char_float32.tflite                ← FP32 baseline (always)
  v3_char_float16.tflite                ← FP16 (always)
"""
from __future__ import annotations

import argparse
import random
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
    """Pull N samples from shards → (N, H, H, 3) float32 in [-1, 1] (NHWC)."""
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
        imgs = d["images"]               # (N, H, W, 3) uint8
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
    print(f"[calib] collected {out.shape} range=[{out.min():.3f},{out.max():.3f}]")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="onnx2tf writes multiple .tflite files here")
    ap.add_argument("--calib-count", type=int, default=300)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] onnx       = {onnx_path}")
    print(f"[run] shard_dir  = {args.shard_dir}")
    print(f"[run] out_dir    = {out_dir}")
    print(f"[run] calib_count= {args.calib_count}  input_size={args.input_size}")

    # Step 1: build calibration .npy in NHWC (TFLite native layout)
    calib_npy = out_dir / "calib_samples.npy"
    calib = collect_calib_samples(Path(args.shard_dir), args.calib_count,
                                   args.input_size, args.seed)
    np.save(calib_npy, calib)
    print(f"[calib] wrote {calib_npy} ({calib_npy.stat().st_size / 1e6:.1f} MB)")

    # Step 2: onnx2tf with INT8 quantization
    import onnx2tf
    print(f"[onnx2tf] converting + quantizing INT8 (full integer)...")
    t0 = time.perf_counter()
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(out_dir),
        output_integer_quantized_tflite=True,
        quant_type="per-channel",                  # better accuracy than per-tensor
        copy_onnx_input_output_names_to_tflite=True,
        # NOTE: calib data is ALREADY normalized to [-1,1] by collect_calib_samples.
        # onnx2tf applies (x - mean) / std internally, so use identity mean/std
        # to avoid double-normalization. Earlier 0.5/0.5 was wrong → produced
        # garbage int8 output (constant logits, all probs ≈ 1/N).
        custom_input_op_name_np_data_path=[
            ["input", str(calib_npy), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ],
        non_verbose=True,
    )
    dt = time.perf_counter() - t0
    print(f"[onnx2tf] done in {dt:.1f}s")

    # Step 3: list outputs
    print(f"\n[output] files in {out_dir}:")
    for p in sorted(out_dir.glob("*.tflite")):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:50s}  {size_mb:7.2f} MB")

    print()
    print("=" * 70)
    print("Recommended for Pi/Coral:")
    int8_full = out_dir / "v3_char_full_integer_quant.tflite"
    if int8_full.exists():
        print(f"  → {int8_full}")
        print(f"\nNext (Coral, Linux only):")
        print(f"  edgetpu_compiler {int8_full} -o {out_dir}")
    else:
        print("  (look for *full_integer_quant*.tflite)")


if __name__ == "__main__":
    main()
