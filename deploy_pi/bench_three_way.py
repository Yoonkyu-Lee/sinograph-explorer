"""3-way Pi/Coral latency benchmark — Phase 1 closing experiment.

Runs the same image through three configurations and reports per-step latency:
  (1) ONNX FP32      via onnxruntime (the working baseline, doc/24)
  (2) TFLite INT8    CPU only (XNNPack disabled if it crashes)
  (3) TFLite INT8    + Coral Edge TPU delegate (the Phase 1 artifact)

This addresses Codex's feedback (forwarded 2026-04-28): "compile mapping ≠
fast inference". Even with 36/36 ops mapped, the 50 MB FC streams from
off-chip USB so real Coral latency may not beat CPU. We need the actual
numbers to know.

Pass criteria for Phase 1 closure:
  - All 3 configs produce the same top-1 (within INT8 tolerance: TFLite top-1
    should match ONNX FP32 in ≥ 95% of trial inputs)
  - Coral latency < CPU TFLite latency (otherwise TPU isn't actually helping)
  - All three latencies should be reportable for the Final Demo slide

Usage on Pi (after copying deploy_pi/export/* to ~/lab3/export/):
    python3 bench_three_way.py \\
        --onnx export/v3_char.onnx \\
        --tflite-cpu export/v3_keras_char_int8.tflite \\
        --tflite-edgetpu export/v3_keras_char_int8_edgetpu.tflite \\
        --class-index export/class_index.json \\
        --image sample.jpg \\
        --warmup 3 --iters 30

Dependencies on Pi:
    pip install tflite-runtime onnxruntime numpy Pillow
    # For Coral:
    sudo apt install libedgetpu1-std   # OR libedgetpu1-max (faster, hotter)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def preprocess(img_path: str, input_size: int):
    """Load → resize-shorter-side → center-crop → /255 → ((-)0.5)/0.5.

    Returns (nhwc, nchw) — both float32 in [-1, 1], shape (1, *, *, *).
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < h:
        nw, nh = input_size, int(h * input_size / w)
    else:
        nw, nh = int(w * input_size / h), input_size
    img = img.resize((nw, nh), Image.BILINEAR)
    L = (nw - input_size) // 2
    T = (nh - input_size) // 2
    img = img.crop((L, T, L + input_size, T + input_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    nhwc = arr[None, ...]
    nchw = np.transpose(nhwc, (0, 3, 1, 2)).copy()
    return nhwc, nchw


def time_block(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return {
        "n": iters,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples) if iters > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def fmt(stat: dict) -> str:
    return (f"{stat['mean']:7.2f} ± {stat['stdev']:5.2f} ms  "
            f"(median {stat['median']:6.2f}, "
            f"min {stat['min']:6.2f}, max {stat['max']:6.2f}, n={stat['n']})")


def bench_onnx(model_path: str, nchw: np.ndarray, warmup: int, iters: int):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def _run():
        return sess.run(None, {in_name: nchw.astype(np.float32)})[0]

    out = _run()
    stats = time_block(_run, warmup, iters)
    top1 = int(np.argmax(out[0]))
    return stats, top1, out[0]


def make_tflite(model_path: str, *, use_coral: bool, no_xnnpack: bool):
    """Return (interp, in_det, out_det). Falls back across runtimes."""
    delegates = []
    Interp = None
    load_delegate = None
    try:
        from tflite_runtime.interpreter import Interpreter as TFLR
        from tflite_runtime.interpreter import load_delegate as TFLR_LD
        Interp = TFLR
        load_delegate = TFLR_LD
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter as AIEL
            Interp = AIEL
            try:
                from ai_edge_litert.interpreter import load_delegate as AIEL_LD
                load_delegate = AIEL_LD
            except ImportError:
                pass
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter as TFI
            Interp = TFI

    if use_coral:
        if load_delegate is None:
            raise SystemExit("Coral delegate not available (no load_delegate)")
        delegates.append(load_delegate("libedgetpu.so.1"))

    if no_xnnpack:
        import tensorflow as tf
        interp = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
    else:
        interp = Interp(model_path=model_path,
                        experimental_delegates=delegates or None)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]


def bench_tflite(model_path: str, nhwc: np.ndarray,
                 *, use_coral: bool, no_xnnpack: bool,
                 warmup: int, iters: int):
    interp, in_det, out_det = make_tflite(
        model_path, use_coral=use_coral, no_xnnpack=no_xnnpack,
    )
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    if in_det["dtype"] == np.int8:
        x = np.round(nhwc / in_scale + in_zp).clip(-128, 127).astype(np.int8)
    else:
        x = nhwc.astype(in_det["dtype"])

    def _run():
        interp.set_tensor(in_det["index"], x)
        interp.invoke()
        return interp.get_tensor(out_det["index"])

    out_q = _run()
    if out_det["dtype"] == np.int8:
        out = (out_q.astype(np.float32) - out_zp) * out_scale
    else:
        out = out_q.astype(np.float32)

    stats = time_block(_run, warmup, iters)
    top1 = int(np.argmax(out[0]))
    return stats, top1, out[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tflite-cpu", required=True)
    ap.add_argument("--tflite-edgetpu", required=True)
    ap.add_argument("--class-index", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-xnnpack", action="store_true",
                    help="dev workaround if XNNPack crashes; not needed on Pi")
    args = ap.parse_args()

    classes = json.load(open(args.class_index, encoding="utf-8"))
    classes_sorted = sorted(classes.keys(), key=lambda k: classes[k])

    print(f"[bench] preprocessing {args.image}")
    nhwc, nchw = preprocess(args.image, args.input_size)

    print()
    print("=" * 72)
    print(f"  Bench config: warmup={args.warmup}  iters={args.iters}")
    print(f"  Image:        {args.image}")
    print(f"  Input size:   {args.input_size}×{args.input_size}")
    print("=" * 72)
    print()

    print(f"[1/3] ONNX FP32 (onnxruntime CPU)...")
    onnx_stats, onnx_top1, _ = bench_onnx(args.onnx, nchw, args.warmup, args.iters)

    print(f"[2/3] TFLite INT8 (CPU, no Coral)...")
    cpu_stats, cpu_top1, _ = bench_tflite(
        args.tflite_cpu, nhwc, use_coral=False,
        no_xnnpack=args.no_xnnpack, warmup=args.warmup, iters=args.iters,
    )

    print(f"[3/3] TFLite INT8 + Coral Edge TPU...")
    try:
        tpu_stats, tpu_top1, _ = bench_tflite(
            args.tflite_edgetpu, nhwc, use_coral=True,
            no_xnnpack=args.no_xnnpack, warmup=args.warmup, iters=args.iters,
        )
    except Exception as e:
        print(f"[3/3] Coral delegate FAILED: {e}")
        tpu_stats = None
        tpu_top1 = -1

    print()
    print("=" * 72)
    print(f"  Latency (lower is better)")
    print("=" * 72)
    print(f"  ONNX FP32       : {fmt(onnx_stats)}")
    print(f"  TFLite INT8 CPU : {fmt(cpu_stats)}")
    if tpu_stats:
        print(f"  TFLite INT8 TPU : {fmt(tpu_stats)}")
        speedup_cpu = cpu_stats['mean'] / tpu_stats['mean']
        speedup_onnx = onnx_stats['mean'] / tpu_stats['mean']
        print()
        print(f"  TPU vs CPU TFLite : {speedup_cpu:.2f}× faster")
        print(f"  TPU vs ONNX FP32  : {speedup_onnx:.2f}× faster")

    print()
    print("=" * 72)
    print(f"  Top-1 agreement")
    print("=" * 72)
    def _label(idx):
        try:
            n = classes_sorted[idx]
            ch = chr(int(n[2:], 16))
            return f"{n} '{ch}'"
        except Exception:
            return f"#{idx}"
    print(f"  ONNX FP32       : {_label(onnx_top1)}")
    print(f"  TFLite INT8 CPU : {_label(cpu_top1)} "
          f"({'match' if cpu_top1 == onnx_top1 else 'DIFFER'})")
    if tpu_stats:
        print(f"  TFLite INT8 TPU : {_label(tpu_top1)} "
              f"({'match' if tpu_top1 == onnx_top1 else 'DIFFER'})")
        if cpu_top1 != tpu_top1:
            print(f"  ! CPU vs TPU on same TFLite differ — possible delegate bug")


if __name__ == "__main__":
    main()
