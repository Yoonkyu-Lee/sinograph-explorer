"""Phase 12 — measure mask-source cost by kind.

For each v2 source kind, render the same character many times and report:
  - cold time (first call, includes file scan / parse)
  - warm time (subsequent calls, should hit caches)
  - warm-warm time (after 100 warmups, steady-state CPU cost only)

Also reports CPU core count and suggested worker cap.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mask_adapter import get_sources_for_char, render_mask


SPECS = {
    "font": {"kind": "font"},
    "svg_stroke": {
        "kind": "svg_stroke", "base_width": 48,
        "stroke_ops": [
            {"op": "width_jitter", "std": 5},
            {"op": "control_jitter", "std_ratio": 0.020, "std_min": 0.8, "std_max": 10.0},
            {"op": "endpoint_jitter", "std_ratio": 0.030, "std_min": 1.5, "std_max": 15.0},
        ],
    },
    "ehanja_median": {
        "kind": "ehanja_median", "width_scale": 1.5,
        "stroke_ops": [
            {"op": "width_jitter", "std": 5},
            {"op": "control_jitter", "std_ratio": 0.020, "std_min": 0.8, "std_max": 10.0},
        ],
    },
    "kanjivg_median": {
        "kind": "kanjivg_median", "width_scale": 1.7,
        "stroke_ops": [
            {"op": "control_jitter", "std_ratio": 0.020, "std_min": 0.1, "std_max": 1.2},
        ],
    },
    "ehanja_stroke": {
        "kind": "ehanja_stroke",
        "stroke_ops": [
            {"op": "stroke_rotate", "angle_std": 1.5},
            {"op": "stroke_translate", "std_ratio": 0.010, "std_min": 0.5, "std_max": 6.0},
        ],
    },
    "mmh_stroke": {
        "kind": "mmh_stroke",
        "stroke_ops": [
            {"op": "stroke_rotate", "angle_std": 1.5},
            {"op": "stroke_translate", "std_ratio": 0.010, "std_min": 0.5, "std_max": 6.0},
        ],
    },
}

# chars picked so all sources cover at least one
COVERED_CHARS = {
    "font": "鑑",
    "svg_stroke": "鑑",
    "ehanja_median": "鑑",
    "kanjivg_median": "鑑",
    "ehanja_stroke": "鑑",
    "mmh_stroke": "鑑",
}


def time_call(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def profile_kind(kind: str, spec: dict, char: str, warmup: int = 5, reps: int = 50) -> dict:
    """Return cold / mean_warm / p50 / p95 ms for a kind."""
    rng = np.random.default_rng(0)
    # cold — first source resolve + first render
    t0 = time.perf_counter()
    sources = get_sources_for_char(char, spec)
    t_resolve_cold = (time.perf_counter() - t0) * 1000
    if not sources:
        return {"kind": kind, "skipped": "no_source"}
    src = sources[0]
    t0 = time.perf_counter()
    m = render_mask(char, src, rng=rng)
    t_render_cold = (time.perf_counter() - t0) * 1000
    if m is None:
        return {"kind": kind, "skipped": "render_failed"}

    # warmup (re-resolve + re-render) — post-cache behavior
    for _ in range(warmup):
        srcs = get_sources_for_char(char, spec)
        render_mask(char, srcs[0], rng=rng)

    # timed reps — warm steady state
    resolve_times = []
    render_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        srcs = get_sources_for_char(char, spec)
        resolve_times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        render_mask(char, srcs[0], rng=rng)
        render_times.append((time.perf_counter() - t0) * 1000)

    arr_res = np.array(resolve_times)
    arr_ren = np.array(render_times)
    return {
        "kind": kind, "char": char,
        "cold_resolve_ms": round(t_resolve_cold, 2),
        "cold_render_ms": round(t_render_cold, 2),
        "warm_resolve_mean_ms": round(arr_res.mean(), 2),
        "warm_resolve_p95_ms": round(float(np.percentile(arr_res, 95)), 2),
        "warm_render_mean_ms": round(arr_ren.mean(), 2),
        "warm_render_p50_ms": round(float(np.percentile(arr_ren, 50)), 2),
        "warm_render_p95_ms": round(float(np.percentile(arr_ren, 95)), 2),
        "warm_total_mean_ms": round((arr_res + arr_ren).mean(), 2),
        "reps": reps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_engine_v3/out/12_mask_profile")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # env info
    cpu_logical = os.cpu_count()
    cpu_phys = None
    try:
        import psutil
        cpu_phys = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    env = {
        "cpu_count_logical": cpu_logical,
        "cpu_count_physical": cpu_phys,
        "mp_cpu_count": mp.cpu_count(),
    }
    print(f"ENV: {env}")
    print()

    rows = []
    for kind, spec in SPECS.items():
        ch = COVERED_CHARS[kind]
        print(f"profiling {kind}  char={ch} ...")
        row = profile_kind(kind, spec, ch, reps=args.reps)
        rows.append(row)
        if "skipped" in row:
            print(f"  SKIPPED: {row['skipped']}")
            continue
        print(f"  cold: resolve {row['cold_resolve_ms']:>7.2f} ms + render {row['cold_render_ms']:>7.2f} ms")
        print(f"  warm: resolve {row['warm_resolve_mean_ms']:>7.2f} ms (p95 {row['warm_resolve_p95_ms']})"
              f" + render {row['warm_render_mean_ms']:>7.2f} ms (p50 {row['warm_render_p50_ms']}, p95 {row['warm_render_p95_ms']})"
              f"   total {row['warm_total_mean_ms']:>7.2f} ms/sample   → {1000/row['warm_total_mean_ms']:.0f} masks/s (1 worker)")
    print()

    # save
    with open(out / "profile.json", "w", encoding="utf-8") as f:
        json.dump({"env": env, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"saved: {out/'profile.json'}")

    # suggested worker cap
    if cpu_phys:
        print(f"suggested worker cap: {cpu_phys} physical (or {cpu_logical} logical for I/O-bound)")


if __name__ == "__main__":
    main()
