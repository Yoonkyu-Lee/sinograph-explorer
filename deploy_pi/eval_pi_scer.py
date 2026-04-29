"""End-to-end accuracy + latency evaluation on Pi.

Reads val_pack_*.npz (real images + labels) and runs the full SCER deploy
pipeline (forward + cosine NN), comparing CPU and Coral if both available.

Reports per-mode:
    forward latency (mean / p99)
    cosine NN latency (mean / p99)
    total per-char latency
    emb_full top-1 / top-5  (cosine vs all 98169 anchors)

Usage:
    python deploy_pi/eval_pi_scer.py \\
        --val-pack scer/val_pack_1000.npz \\
        --tflite-cpu scer/scer_int8.tflite \\
        --tflite-coral scer/scer_int8_edgetpu.tflite \\
        --anchors scer/scer_anchor_db.npy \\
        --modes cpu,coral
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from ai_edge_litert.interpreter import Interpreter, load_delegate


def percentile(xs, p):
    if not xs:
        return 0.0
    xs_s = sorted(xs)
    k = (len(xs_s) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs_s) - 1)
    return xs_s[f] + (xs_s[c] - xs_s[f]) * (k - f)


def report(name, ts):
    if not ts:
        return {}
    return {
        "n": len(ts),
        "mean_ms": stats.mean(ts),
        "p50_ms": percentile(ts, 0.5),
        "p99_ms": percentile(ts, 0.99),
        "min_ms": min(ts),
        "max_ms": max(ts),
    }


def quantize_input(x_uint8: np.ndarray, in_d) -> np.ndarray:
    """uint8 NHWC [0,255] → INT8 NHWC matching input scale/zp."""
    # Pre-process: float [-1,1] then quantize to int8 with model's scale/zp
    f = x_uint8.astype(np.float32) / 255.0
    f = (f - 0.5) / 0.5                                       # [-1, 1]
    scale, zp = in_d["quantization"]
    q = np.round(f / scale + zp).clip(-128, 127).astype(np.int8)
    return q


def dequantize_emb(y_int8: np.ndarray, out_d) -> np.ndarray:
    """INT8 emb → fp32, then re-L2-normalize."""
    scale, zp = out_d["quantization"]
    y = (y_int8.astype(np.float32) - zp) * scale
    norm = np.linalg.norm(y, axis=-1, keepdims=True).clip(min=1e-8)
    return y / norm


def find_emb_output(interp):
    """Find the 128-d embedding output among the 5 model outputs."""
    for d in interp.get_output_details():
        if list(d["shape"][-1:]) == [128]:
            return d
    raise RuntimeError("no 128-d output found in model")


def run_pipeline(model_path: Path, delegate, images_uint8: np.ndarray,
                  labels: np.ndarray, anchors: np.ndarray) -> dict:
    """Run forward + cosine NN per sample, measure latency, compute accuracy."""
    delegates = [load_delegate("libedgetpu.so.1")] if delegate == "coral" else []
    interp = Interpreter(model_path=str(model_path),
                          experimental_delegates=delegates)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    emb_d = find_emb_output(interp)

    n = len(images_uint8)
    fwd_times = []
    cos_times = []
    top1_correct = 0
    top5_correct = 0

    # Quantize all inputs once (off the hot path)
    x_q_all = quantize_input(images_uint8, in_d)

    # Warmup (5 samples)
    for i in range(min(5, n)):
        interp.set_tensor(in_d["index"], x_q_all[i:i+1])
        interp.invoke()

    for i in range(n):
        # Forward
        t0 = time.perf_counter()
        interp.set_tensor(in_d["index"], x_q_all[i:i+1])
        interp.invoke()
        emb_int8 = interp.get_tensor(emb_d["index"])
        fwd_times.append((time.perf_counter() - t0) * 1000)

        # Dequant + cosine NN
        t0 = time.perf_counter()
        emb = dequantize_emb(emb_int8, emb_d)                 # (1, 128) L2-norm
        sims = emb @ anchors.T                                 # (1, 98169)
        top5 = np.argsort(-sims, axis=1)[:, :5]
        cos_times.append((time.perf_counter() - t0) * 1000)

        gt = labels[i]
        if top5[0, 0] == gt:
            top1_correct += 1
        if (top5[0] == gt).any():
            top5_correct += 1

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{n}", flush=True)

    return {
        "forward": report("forward", fwd_times),
        "cosine": report("cosine", cos_times),
        "top1_acc": top1_correct / n,
        "top5_acc": top5_correct / n,
        "n_samples": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-pack", required=True)
    ap.add_argument("--tflite-cpu", default=None)
    ap.add_argument("--tflite-coral", default=None)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--modes", default="cpu,coral")
    ap.add_argument("--max-n", type=int, default=1000)
    args = ap.parse_args()

    print(f"[pi-eval] loading val_pack {args.val_pack}")
    pack = np.load(args.val_pack)
    images = pack["images"]                                    # uint8 NHWC
    labels = pack["labels"]
    if args.max_n and len(images) > args.max_n:
        images = images[:args.max_n]
        labels = labels[:args.max_n]
    print(f"[pi-eval] val: images {images.shape} labels {labels.shape}")

    print(f"[pi-eval] loading anchors {args.anchors}")
    anchors = np.load(args.anchors)
    print(f"[pi-eval] anchors {anchors.shape}")

    modes = {m.strip() for m in args.modes.split(",")}
    results = {}

    if "cpu" in modes and args.tflite_cpu:
        print(f"\n=== CPU pipeline ({args.tflite_cpu}) ===")
        results["cpu"] = run_pipeline(Path(args.tflite_cpu), None,
                                       images, labels, anchors)
        r = results["cpu"]
        fw, co = r["forward"], r["cosine"]
        print(f"  forward: mean {fw['mean_ms']:6.2f}ms  p99 {fw['p99_ms']:6.2f}")
        print(f"  cosine : mean {co['mean_ms']:6.2f}ms  p99 {co['p99_ms']:6.2f}")
        print(f"  total  : {fw['mean_ms'] + co['mean_ms']:6.2f}ms / char  "
              f"({1000.0 / (fw['mean_ms'] + co['mean_ms']):5.1f} chars/sec)")
        print(f"  top-1  = {r['top1_acc']*100:.2f}%  top-5 = {r['top5_acc']*100:.2f}%")

    if "coral" in modes and args.tflite_coral:
        print(f"\n=== Coral pipeline ({args.tflite_coral}) ===")
        results["coral"] = run_pipeline(Path(args.tflite_coral), "coral",
                                         images, labels, anchors)
        r = results["coral"]
        fw, co = r["forward"], r["cosine"]
        print(f"  forward: mean {fw['mean_ms']:6.2f}ms  p99 {fw['p99_ms']:6.2f}")
        print(f"  cosine : mean {co['mean_ms']:6.2f}ms  p99 {co['p99_ms']:6.2f}")
        print(f"  total  : {fw['mean_ms'] + co['mean_ms']:6.2f}ms / char  "
              f"({1000.0 / (fw['mean_ms'] + co['mean_ms']):5.1f} chars/sec)")
        print(f"  top-1  = {r['top1_acc']*100:.2f}%  top-5 = {r['top5_acc']*100:.2f}%")

    # Save
    out_path = Path(args.val_pack).parent / "eval_pi_scer.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[pi-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
