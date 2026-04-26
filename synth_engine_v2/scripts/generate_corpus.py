"""
Corpus-level driver — generate training data across many characters.

Where `generate.py` fires samples for ONE character, this script iterates over
a character pool and emits N total samples. Characters are drawn by a
configurable strategy (uniform or stratified by Unicode block). Each sample
then flows through the usual base_source → style → augment pipeline; if the
config uses `kind: multi`, per-sample source selection happens inside.

Inputs
------
- Coverage JSONL (`coverage_per_char.jsonl`) — maps char → available sources
  (mmh / ehanja / kanjivg). Produced by `coverage_report.py`.
- YAML config — same shape as `generate.py`, typically `kind: multi` with a
  weighted source list.

Sampling strategies
-------------------
- `uniform`          — random-with-replacement over the pool.
- `stratified_by_block` — bucket by Unicode block, sample each block with
  a per-block weight (from CLI or config `corpus.block_weights`).
  Useful to avoid SMP tail from swamping the corpus (BMP has 20k+ chars,
  SMP has just a few thousand); or conversely to up-weight Hiragana so
  the 90-char block gets enough samples.

Char pool filters (`--pool`)
----------------------------
- `union`        — every char in any of the 3 SVG sources (18,982)
- `intersection` — chars present in ALL 3 sources (5,029)
- `mmh` / `ehanja` / `kanjivg` — single-source pool
- `font_only`    — pool from a passed file of chars (any chars; font fallback)

Output
------
`out/corpus_<config_tag>/` with:
- `{idx:06d}_{notation}_{picked_source}.png` — each sample
- `corpus_manifest.jsonl` (if --metadata) — one record per sample

Usage
-----
    python generate_corpus.py --config configs/multi_source_handwriting.yaml \
        --total 1000 --strategy stratified_by_block --metadata

    python generate_corpus.py --config configs/full_random_multi.yaml \
        --total 50000 --pool union --strategy stratified_by_block \
        --block-weights-json '{"Hiragana": 3.0, "Katakana": 3.0, "Ext_B_SMP": 0.3}'
"""

from __future__ import annotations

# IMPORTANT — set BLAS/OpenMP thread caps *before* numpy / PIL / scipy import.
# When running N worker processes, each process's native libs default to
# spawning `cpu_count()` threads → N × cpu_count() threads contending for the
# same cores → catastrophic slowdown (parallel path actually *slower* than
# serial). Capping to 1 thread per process gives each worker predictable
# single-thread speed, so N workers ≈ N× throughput.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Share code with the single-char generator.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import generate  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


BLOCKS = [
    ("CJK_Unified",      0x4E00,  0x9FFF),
    ("CJK_Ext_A",        0x3400,  0x4DBF),
    ("CJK_Compat",       0xF900,  0xFAFF),
    ("Radicals_Supp",    0x2E80,  0x2EFF),
    ("Kangxi_Radicals",  0x2F00,  0x2FDF),
    ("Ext_B_SMP",        0x20000, 0x2A6DF),
    ("Ext_C_SMP",        0x2A700, 0x2B73F),
    ("Ext_D_SMP",        0x2B740, 0x2B81F),
    ("Ext_E_SMP",        0x2B820, 0x2CEAF),
    ("Hiragana",         0x3040,  0x309F),
    ("Katakana",         0x30A0,  0x30FF),
]


def block_of(cp: int) -> str:
    for name, lo, hi in BLOCKS:
        if lo <= cp <= hi:
            return name
    return "Other"


def load_char_pool(coverage_jsonl: Path, pool: str) -> list[tuple[str, set]]:
    chars: list[tuple[str, set]] = []
    with coverage_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            c = rec["char"]
            srcs = set(rec["sources"])
            if pool == "union":
                chars.append((c, srcs))
            elif pool == "intersection":
                if len(srcs) >= 3:  # all three
                    chars.append((c, srcs))
            elif pool in ("mmh", "ehanja", "kanjivg"):
                if pool in srcs:
                    chars.append((c, srcs))
            else:
                raise ValueError(f"unknown pool: {pool!r}")
    return chars


def sample_chars(chars: list[tuple[str, set]],
                  rng: np.random.Generator,
                  strategy: str,
                  total: int,
                  block_weights: dict) -> list[str]:
    """Return a list of length `total` with character choices."""
    if strategy == "uniform":
        idx = rng.integers(0, len(chars), total)
        return [chars[int(i)][0] for i in idx]

    if strategy == "stratified_by_block":
        buckets: dict[str, list[str]] = defaultdict(list)
        for c, _ in chars:
            buckets[block_of(ord(c))].append(c)

        blocks = sorted(buckets.keys())
        w = np.array([float(block_weights.get(b, 1.0)) for b in blocks])
        # Drop zero-weight blocks entirely (prevents allocation pretending
        # they're sampled).
        keep = w > 0
        blocks = [b for b, k in zip(blocks, keep) if k]
        w = w[keep]
        if len(blocks) == 0:
            raise ValueError("no blocks have positive weight")
        w = w / w.sum()

        # Proportional allocation with rounding residual redistributed
        # among blocks weighted by their remainders.
        raw = w * total
        alloc = np.floor(raw).astype(int)
        rem = total - alloc.sum()
        if rem > 0:
            fracs = raw - alloc
            # Assign remaining samples to blocks with largest fractional parts.
            extra_order = np.argsort(-fracs)
            for i in range(rem):
                alloc[extra_order[i % len(alloc)]] += 1

        picks: list[str] = []
        for b, n in zip(blocks, alloc):
            pool_list = buckets[b]
            idx = rng.integers(0, len(pool_list), int(n))
            picks.extend(pool_list[int(j)] for j in idx)
        rng.shuffle(picks)
        return picks

    raise ValueError(f"unknown strategy: {strategy!r}")


# ---------- worker (shared by serial and multiprocessing paths) -------------


# Per-process state, populated by `_init_worker`. Module-level so the Pool's
# worker function `_render_one_task` can read without re-passing each call.
_WORKER_STATE: dict = {}


def _init_worker(config: dict, fonts_dir: str, out_dir: str, seed: int):
    """Run once per worker (or once in the main process for serial mode).

    Also warms the font scan cache so the first sample doesn't pay the 3.5s
    Windows-Fonts walk. JSONL caches are lazy — they warm on first lookup,
    which is fast enough (<50ms).
    """
    _WORKER_STATE["config"] = config
    _WORKER_STATE["fonts_dir"] = Path(fonts_dir)
    _WORKER_STATE["out_dir"] = Path(out_dir)
    _WORKER_STATE["seed"] = int(seed)
    try:
        import base_source
        base_source.discover_font_sources(_WORKER_STATE["fonts_dir"])
    except Exception:
        # Non-fatal — font-less configs should still work.
        pass


def _render_one_task(task):
    """task = (idx, char). Returns a dict with `status` + provenance."""
    idx, char = task
    cfg = _WORKER_STATE["config"]
    fonts_dir = _WORKER_STATE["fonts_dir"]
    out_dir = _WORKER_STATE["out_dir"]
    seed = _WORKER_STATE["seed"]

    notation = f"U+{ord(char):04X}"
    sample_rng = np.random.default_rng(seed + 1_000_000 + idx)

    try:
        sources = generate.resolve_base_sources(
            cfg.get("base_source", {}), char, fonts_dir
        )
    except SystemExit as e:
        return {"status": "resolve_error", "idx": idx, "char": char,
                "msg": str(e)}

    if not sources:
        return {"status": "no_source", "idx": idx, "char": char}

    src = (sources[0] if len(sources) == 1
           else sources[int(sample_rng.integers(0, len(sources)))])

    img = generate.render_one(char, cfg, src, sample_rng)
    if img is None:
        return {"status": "render_failed", "idx": idx, "char": char}

    picked = getattr(src, "last_picked", None) or src.tag()
    picked_name = picked.rsplit("-", 1)[0] if "-" in picked else picked
    safe = picked_name.replace("/", "_").replace("\\", "_")
    filename = f"{idx:06d}_{notation}_{safe}.png"
    img.save(out_dir / filename)

    return {
        "status": "ok",
        "idx": idx, "char": char, "cp": ord(char),
        "notation": notation,
        "block": block_of(ord(char)),
        "picked_source": picked,
        "filename": filename,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--coverage",
                   default=str(_HERE.parent / "out" / "coverage_per_char.jsonl"),
                   help="coverage_per_char.jsonl from coverage_report.py")
    p.add_argument("--pool", default="union",
                   choices=["union", "intersection", "mmh", "ehanja", "kanjivg"])
    p.add_argument("--strategy", default="uniform",
                   choices=["uniform", "stratified_by_block"])
    p.add_argument("--total", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--metadata", action="store_true",
                   help="write corpus_manifest.jsonl with per-sample provenance")
    p.add_argument("--fonts-dir", default=str(generate.DEFAULT_FONTS_DIR))
    p.add_argument("--block-weights-json", default=None,
                   help="JSON of {block_name: weight} overrides; merged with "
                        "config 'corpus.block_weights'")
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--workers", type=int, default=1,
                   help="worker processes. 1 (default) = serial. "
                        "Use e.g. --workers 8 for 5~7x speedup on 8-core CPU. "
                        "Each worker pays a one-time ~3.5s font-scan warmup "
                        "(in parallel), then runs at full rate.")
    args = p.parse_args()

    config_path = Path(args.config)
    config = generate.load_config(config_path)
    config_tag = config_path.stem

    out_dir = (Path(args.out) if args.out
               else _HERE.parent / "out" / f"corpus_{config_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Block weights: config first, CLI overrides.
    block_weights: dict = {}
    if isinstance(config.get("corpus"), dict):
        block_weights.update(config["corpus"].get("block_weights", {}) or {})
    if args.block_weights_json:
        block_weights.update(json.loads(args.block_weights_json))

    coverage_path = Path(args.coverage)
    print(f"char pool: {args.pool}  coverage={coverage_path}")
    chars = load_char_pool(coverage_path, pool=args.pool)
    print(f"  pool size: {len(chars):,}")

    pick_rng = np.random.default_rng(args.seed)
    picks = sample_chars(chars, pick_rng, args.strategy, args.total,
                          block_weights=block_weights)
    print(f"strategy={args.strategy}  total={args.total}  out={out_dir}")
    # Sanity: report pick distribution by block
    dist = defaultdict(int)
    for c in picks:
        dist[block_of(ord(c))] += 1
    print("pick distribution:")
    for b, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {b:<20} {n:>6,}")

    t0 = time.time()
    written = 0
    skipped = 0
    manifest: list[dict] = []

    tasks = list(enumerate(picks))
    n_workers = max(1, int(args.workers))
    init_args = (config, str(args.fonts_dir), str(out_dir), args.seed)

    if n_workers <= 1:
        print(f"workers={n_workers} (serial). Warming font cache…")
        _init_worker(*init_args)
        results_iter = (_render_one_task(t) for t in tasks)
        pool = None
    else:
        print(f"workers={n_workers} (multiprocessing). "
              f"Each worker warms font cache in parallel on startup.")
        pool = mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=init_args,
        )
        # chunksize trade-off: too small → per-task IPC overhead dominates;
        # too large → last chunk finishes late, stragglers delay shutdown.
        chunksize = max(4, min(64, len(tasks) // (n_workers * 8) or 1))
        results_iter = pool.imap_unordered(
            _render_one_task, tasks, chunksize=chunksize
        )

    try:
        for i, res in enumerate(results_iter, 1):
            if res.get("status") == "ok":
                written += 1
                if args.metadata:
                    manifest.append({
                        k: v for k, v in res.items() if k != "status"
                    })
            else:
                skipped += 1

            if i % args.progress_every == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.01)
                eta = (len(tasks) - i) / max(rate, 0.01)
                print(f"  {i:,}/{len(tasks):,}  written={written:,}  "
                      f"skipped={skipped:,}  rate={rate:.1f}/s  eta={eta:.0f}s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if args.metadata:
        # imap_unordered returns results out of order; sort by idx so the
        # manifest is deterministic and aligns with filename ordering.
        manifest.sort(key=lambda m: m.get("idx", 0))
        mpath = out_dir / "corpus_manifest.jsonl"
        with mpath.open("w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"wrote manifest: {mpath}  ({len(manifest):,} records)")

    print()
    print(f"done  elapsed={(time.time()-t0):.1f}s  "
          f"written={written:,}  skipped={skipped:,}")


if __name__ == "__main__":
    main()
