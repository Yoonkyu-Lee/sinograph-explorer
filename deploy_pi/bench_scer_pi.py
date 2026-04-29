"""SCER end-to-end Pi/Coral latency benchmark — Phase 4.

Measures per-character inference latency for the SCER deploy pipeline:
    image → tflite forward (CPU or Edge TPU) → anchor cosine NN → top-1

Three modes (any subset, comma-separated):
    cpu      — scer_int8.tflite on CPU via ai-edge-litert
    coral    — scer_int8_edgetpu.tflite on Coral USB via libedgetpu
    cosine   — anchor cosine NN search on CPU (just the post-processing)

Outputs per-stage latency (mean + p50 + p90 + p99) over N timed runs after
W warmup runs. Reports the total deploy pipeline latency = forward + cosine.

Synthetic random INT8 inputs by default (latency only — no accuracy check).
For accuracy with real images, see infer_pi_scer.py (separate).

Usage on Pi:
    cd ~/ece479
    .venv/bin/python ~/ece479/scer/bench_scer_pi.py \\
        --tflite-cpu   scer/scer_int8.tflite \\
        --tflite-coral scer/scer_int8_edgetpu.tflite \\
        --anchors      scer/scer_anchor_db.npy \\
        --modes cpu,coral,cosine \\
        --warmup 10 --runs 50
"""
from __future__ import annotations

import argparse
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np


def percentile(xs: list, p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def report(name: str, times_ms: list) -> dict:
    if not times_ms:
        print(f"  {name:30s}  no data")
        return {}
    n = len(times_ms)
    mean = stats.mean(times_ms)
    p50 = percentile(times_ms, 0.5)
    p90 = percentile(times_ms, 0.9)
    p99 = percentile(times_ms, 0.99)
    mn = min(times_ms)
    mx = max(times_ms)
    print(f"  {name:30s}  n={n:3d}  mean={mean:7.2f}ms  p50={p50:7.2f}  "
          f"p90={p90:7.2f}  p99={p99:7.2f}  min={mn:6.2f}  max={mx:6.2f}")
    return {"name": name, "n": n, "mean_ms": mean, "p50_ms": p50,
            "p90_ms": p90, "p99_ms": p99, "min_ms": mn, "max_ms": mx}


def bench_tflite(tflite_path: Path, n_warmup: int, n_runs: int,
                  delegate: str | None = None) -> tuple[list, dict]:
    """Run TFLite inference on synthetic input. delegate='coral' loads libedgetpu."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
    print(f"[bench] loading {tflite_path.name}  delegate={delegate}")
    delegates = []
    if delegate == "coral":
        delegates = [load_delegate("libedgetpu.so.1")]
    interp = Interpreter(model_path=str(tflite_path),
                          experimental_delegates=delegates)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    in_shape = in_d["shape"]
    in_dtype = in_d["dtype"]
    print(f"[bench]   input  {in_d['name']}  shape={in_shape}  dtype={in_dtype}  "
          f"scale={in_d['quantization'][0]:.4f} zp={in_d['quantization'][1]}")
    out_details = interp.get_output_details()
    print(f"[bench]   {len(out_details)} outputs")

    # Synthetic INT8 input — uniformly random (won't predict anything sensible
    # but we only measure forward latency)
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=in_shape, dtype=np.int8)

    # Warmup
    for _ in range(n_warmup):
        interp.set_tensor(in_d["index"], x)
        interp.invoke()

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        # also fetch outputs (as caller would)
        for d in out_details:
            interp.get_tensor(d["index"])
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)
    return times_ms, {"input_shape": in_shape.tolist(),
                       "n_outputs": len(out_details)}


def bench_cosine(anchors_path: Path, emb_dim: int, n_warmup: int,
                  n_runs: int) -> list:
    """Anchor cosine NN search on CPU with random embeddings."""
    print(f"[bench] loading anchors {anchors_path.name}")
    anchors = np.load(str(anchors_path))                            # (C, D) fp32
    print(f"[bench]   anchors shape={anchors.shape}  "
          f"dtype={anchors.dtype}")
    rng = np.random.default_rng(0)
    # Generate L2-normalized query embeddings
    q = rng.standard_normal((1, emb_dim)).astype(np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    # Warmup
    for _ in range(n_warmup):
        sims = q @ anchors.T
        _ = sims.argmax(axis=1)

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sims = q @ anchors.T                                          # (1, C)
        top1 = sims.argmax(axis=1)
        top5 = np.argsort(-sims, axis=1)[:, :5]
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)
    return times_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite-cpu", default=None)
    ap.add_argument("--tflite-coral", default=None)
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--modes", default="cpu,coral,cosine",
                    help="comma-separated subset of {cpu,coral,cosine}")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    print(f"=== SCER Pi/Coral latency benchmark ===")
    print(f"modes={sorted(modes)}  warmup={args.warmup}  runs={args.runs}")
    print()

    results = {}

    if "cpu" in modes:
        if not args.tflite_cpu:
            print("[bench] skip cpu mode — --tflite-cpu not given")
        else:
            print(f"--- CPU (ai-edge-litert) ---")
            times, _info = bench_tflite(Path(args.tflite_cpu),
                                          args.warmup, args.runs,
                                          delegate=None)
            results["cpu"] = report("CPU forward", times)
            print()

    if "coral" in modes:
        if not args.tflite_coral:
            print("[bench] skip coral mode — --tflite-coral not given")
        else:
            print(f"--- Coral USB (libedgetpu) ---")
            times, _info = bench_tflite(Path(args.tflite_coral),
                                          args.warmup, args.runs,
                                          delegate="coral")
            results["coral"] = report("Coral forward", times)
            print()

    if "cosine" in modes:
        if not args.anchors:
            print("[bench] skip cosine mode — --anchors not given")
        else:
            print(f"--- CPU anchor cosine NN ---")
            times = bench_cosine(Path(args.anchors), args.emb_dim,
                                  args.warmup, args.runs)
            results["cosine"] = report("Cosine NN search", times)
            print()

    # Pipeline summary — best-case (Coral + cosine) and CPU baseline
    print("=" * 70)
    print(f"=== End-to-end pipeline (forward + cosine NN) ===")
    if "cpu" in results and "cosine" in results:
        cpu_total = results["cpu"]["mean_ms"] + results["cosine"]["mean_ms"]
        print(f"  CPU only   : {results['cpu']['mean_ms']:6.2f} + "
              f"{results['cosine']['mean_ms']:5.2f} = {cpu_total:6.2f} ms / char  "
              f"({1000.0/cpu_total:5.1f} chars/sec)")
    if "coral" in results and "cosine" in results:
        coral_total = results["coral"]["mean_ms"] + results["cosine"]["mean_ms"]
        print(f"  Coral+CPU  : {results['coral']['mean_ms']:6.2f} + "
              f"{results['cosine']['mean_ms']:5.2f} = {coral_total:6.2f} ms / char  "
              f"({1000.0/coral_total:5.1f} chars/sec)")
        if "cpu" in results:
            speedup = (results["cpu"]["mean_ms"] / results["coral"]["mean_ms"])
            print(f"  Coral speedup over CPU: {speedup:.1f}× (forward only)")
    print()

    import json
    out_json = Path(args.tflite_cpu or args.tflite_coral or args.anchors).parent / "scer_bench_pi.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out_json}")


if __name__ == "__main__":
    main()
