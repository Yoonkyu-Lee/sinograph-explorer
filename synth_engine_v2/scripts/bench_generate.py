"""Benchmark per-sample generation time, broken down by source kind."""
from __future__ import annotations
import argparse, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import generate  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def stats(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return None
    return {
        "n": int(len(xs)),
        "mean_ms": float(xs.mean() * 1000),
        "median_ms": float(np.median(xs) * 1000),
        "p10_ms": float(np.percentile(xs, 10) * 1000),
        "p90_ms": float(np.percentile(xs, 90) * 1000),
        "min_ms": float(xs.min() * 1000),
        "max_ms": float(xs.max() * 1000),
        "std_ms": float(xs.std() * 1000),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--coverage",
                    default=str(_HERE.parent / "out" / "coverage_per_char.jsonl"))
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5,
                    help="throw-away samples (cache warmup)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = generate.load_config(Path(args.config))

    # Load a sample of chars from coverage
    import json
    all_chars = []
    with open(args.coverage, encoding="utf-8") as f:
        for line in f:
            all_chars.append(json.loads(line)["char"])
    rng = np.random.default_rng(args.seed)
    chars = rng.choice(all_chars, size=args.total + args.warmup, replace=True).tolist()

    fonts_dir = Path(generate.DEFAULT_FONTS_DIR)
    times_by_kind = defaultdict(list)
    total_times = []
    resolve_times = []

    print(f"warmup {args.warmup} samples (cache population)...")
    t_global = time.perf_counter()
    for i, char in enumerate(chars):
        t0 = time.perf_counter()
        sources = generate.resolve_base_sources(
            config.get("base_source", {}), char, fonts_dir
        )
        t1 = time.perf_counter()
        if not sources:
            continue
        sample_rng = np.random.default_rng(args.seed + 1_000_000 + i)
        src = (sources[0] if len(sources) == 1
               else sources[int(sample_rng.integers(0, len(sources)))])
        img = generate.render_one(char, config, src, sample_rng)
        t2 = time.perf_counter()
        if img is None:
            continue
        if i < args.warmup:
            continue
        kind = getattr(src, "kind", "unknown")
        total_times.append(t2 - t0)
        resolve_times.append(t1 - t0)
        times_by_kind[kind].append(t2 - t1)
    t_global = time.perf_counter() - t_global

    print()
    print(f"total {args.total} samples in {t_global:.1f}s "
          f"→ {args.total/t_global:.1f} samples/s overall")
    print(f"  resolve (dispatcher): {stats(resolve_times)}")
    print(f"  end-to-end (per sample): {stats(total_times)}")
    print()
    print("by kind (render_one only, excludes resolve):")
    for kind in sorted(times_by_kind):
        s = stats(times_by_kind[kind])
        print(f"  {kind:<20} n={s['n']:>4}  "
              f"median={s['median_ms']:>6.1f}ms  "
              f"mean={s['mean_ms']:>6.1f}ms  "
              f"p10/p90={s['p10_ms']:>5.1f}/{s['p90_ms']:>6.1f}ms  "
              f"max={s['max_ms']:>6.1f}ms")


if __name__ == "__main__":
    main()
