"""Trim class_index + remap shards to drop dead (no-sample) classes.

Reads:  <corpus>/class_index.json + <corpus>/shard-*.npz
Writes: <corpus>/class_index_full.json   (backup of original)
        <corpus>/class_index.json        (new, contiguous indices)
        <corpus>/shard-*.npz             (in-place rewrite via temp+rename)

Atomic per-shard: writes shard-NNNNN.npz.tmp then renames over the original.
Crash-safe — interruption leaves either the old or the new shard intact.

After this script, regenerate aux_labels.npz against the new class_index.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True,
                    help="corpus directory containing class_index.json + shard-*.npz")
    ap.add_argument("--scan-only", action="store_true",
                    help="only scan + print stats, don't write anything")
    ap.add_argument("--resume-from", type=int, default=0,
                    help="resume mode (manual): skip first N shards. "
                         "Builds remap from class_index_full.json + class_index.json directly.")
    ap.add_argument("--resume-auto", action="store_true",
                    help="resume mode (auto): detect which shards are already remapped "
                         "by comparing mtime to class_index.json mtime. Robust to "
                         "uncertain Ctrl-C cutoff.")
    ap.add_argument("--scan-progress-every", type=int, default=500)
    ap.add_argument("--write-progress-every", type=int, default=100)
    args = ap.parse_args()

    corpus = Path(args.corpus_dir)
    ci_path = corpus / "class_index.json"
    backup_path = corpus / "class_index_full.json"

    # ========================================================================
    # RESUME MODE
    # Both --resume-from N (manual) and --resume-auto (mtime-based) require:
    #   - class_index.json = NEW (trimmed) one — leave it as written by prev run
    #   - class_index_full.json = original backup, untouched
    #   - shards: some already remapped, some not
    # Build remap directly from both files, skip alive-scan + class_index
    # rewrite, then process only shards that need remapping.
    # ========================================================================
    if args.resume_auto:
        if not backup_path.exists():
            print(f"[trim] --resume-auto needs {backup_path.name}", file=sys.stderr)
            sys.exit(1)
        ci_full = json.loads(backup_path.read_text(encoding="utf-8"))
        ci_new = json.loads(ci_path.read_text(encoding="utf-8"))
        n_full = len(ci_full)
        n_alive = len(ci_new)
        ci_mtime = ci_path.stat().st_mtime
        print(f"[trim] RESUME-AUTO mode (mtime-based)")
        print(f"[trim]   original class_index (backup) : {n_full:,}")
        print(f"[trim]   new class_index (current)     : {n_alive:,}")
        print(f"[trim]   class_index.json mtime cutoff = {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ci_mtime))}")

        remap = np.full(n_full, -1, dtype=np.int64)
        for notation, old_idx in ci_full.items():
            new_idx = ci_new.get(notation, -1)
            remap[int(old_idx)] = int(new_idx)
        n_alive_remap = int((remap >= 0).sum())
        assert n_alive_remap == n_alive, f"remap built {n_alive_remap}, ci_new has {n_alive}"

        all_shards = sorted(corpus.glob("shard-*.npz"))
        already_done: list[Path] = []
        pending: list[Path] = []
        for p in all_shards:
            if p.stat().st_mtime > ci_mtime:
                already_done.append(p)
            else:
                pending.append(p)
        # Sanity-verify a few "already done" shards: their max label must be < n_alive
        if already_done:
            sample = already_done[:5] + already_done[-5:]
            for p in sample:
                d = np.load(p)
                mx = int(d["labels"].max())
                d.close()
                if mx >= n_alive:
                    print(f"[trim] FATAL: {p.name} marked done by mtime but has max label "
                          f"{mx} >= n_alive {n_alive}. Aborting.", file=sys.stderr)
                    sys.exit(2)
            print(f"[trim]   sanity check OK on {len(sample)} already-done shards")

        print(f"[trim]   already remapped : {len(already_done):,}")
        print(f"[trim]   pending          : {len(pending):,}")
        if not pending:
            print(f"[trim] all shards already remapped — nothing to do")
            return
        print(f"[trim] rewriting {len(pending):,} pending shards…")
        _rewrite_shards(pending, remap, args.write_progress_every,
                        idx_offset=len(already_done), total=len(all_shards))
        print(f"[trim] DONE — resume-auto rewrite complete")
        return

    if args.resume_from > 0:
        if not backup_path.exists():
            print(f"[trim] --resume-from set but {backup_path.name} missing", file=sys.stderr)
            sys.exit(1)
        ci_full = json.loads(backup_path.read_text(encoding="utf-8"))
        ci_new = json.loads(ci_path.read_text(encoding="utf-8"))
        n_full = len(ci_full)
        n_alive = len(ci_new)
        print(f"[trim] RESUME mode: --resume-from {args.resume_from}")
        print(f"[trim]   original class_index (backup) : {n_full:,}")
        print(f"[trim]   new class_index (current)     : {n_alive:,}")

        remap = np.full(n_full, -1, dtype=np.int64)
        for notation, old_idx in ci_full.items():
            new_idx = ci_new.get(notation, -1)
            remap[int(old_idx)] = int(new_idx)
        n_alive_remap = int((remap >= 0).sum())
        assert n_alive_remap == n_alive, f"remap built {n_alive_remap}, ci_new has {n_alive}"

        shards = sorted(corpus.glob("shard-*.npz"))
        if args.resume_from >= len(shards):
            print(f"[trim] --resume-from {args.resume_from} >= total shards {len(shards)}, nothing to do")
            return
        shards_remaining = shards[args.resume_from:]
        print(f"[trim] skipping shards 0..{args.resume_from-1} (assumed already remapped)")
        print(f"[trim] rewriting {len(shards_remaining):,} remaining shards "
              f"({args.resume_from}..{len(shards)-1})…")
        _rewrite_shards(shards_remaining, remap, args.write_progress_every,
                        idx_offset=args.resume_from, total=len(shards))
        print(f"[trim] DONE — resume rewrite complete")
        return

    # --- 1. Load original class_index ---------------------------------------
    ci_full = json.loads(ci_path.read_text(encoding="utf-8"))
    n_full = len(ci_full)
    print(f"[trim] original class_index : {n_full:,} entries")

    # --- 2. Scan all shards to find which classes have any samples ----------
    shards = sorted(corpus.glob("shard-*.npz"))
    if not shards:
        print(f"[trim] no shard-*.npz found in {corpus}", file=sys.stderr)
        sys.exit(1)
    print(f"[trim] scanning {len(shards):,} shards…")
    seen = np.zeros(n_full, dtype=bool)
    t0 = time.perf_counter()
    n_samples = 0
    for i, p in enumerate(shards):
        d = np.load(p)
        lbls = d["labels"]
        seen[lbls] = True
        n_samples += len(lbls)
        d.close()
        if (i + 1) % args.scan_progress_every == 0:
            print(f"  scan {i+1:>5}/{len(shards)}  "
                  f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    n_alive = int(seen.sum())
    n_dead = n_full - n_alive
    print(f"[trim] scan done  ({time.perf_counter()-t0:.1f}s)")
    print(f"[trim]   alive classes : {n_alive:,}")
    print(f"[trim]   dead  classes : {n_dead:,}")
    print(f"[trim]   total samples : {n_samples:,}")

    if args.scan_only:
        print("[trim] --scan-only set, exiting before any writes")
        return

    # --- 3. Build remap old_idx → new_idx (contiguous 0..n_alive-1) ---------
    remap = np.full(n_full, -1, dtype=np.int64)
    new_idx = 0
    alive_old_indices = np.where(seen)[0]
    for old_idx in alive_old_indices:
        remap[int(old_idx)] = new_idx
        new_idx += 1
    assert new_idx == n_alive, (new_idx, n_alive)

    # --- 4. Build new class_index dict ---------------------------------------
    idx_to_notation = {i: n for n, i in ci_full.items()}
    new_ci: dict[str, int] = {}
    for old_idx in alive_old_indices:
        notation = idx_to_notation[int(old_idx)]
        new_ci[notation] = int(remap[int(old_idx)])
    assert len(new_ci) == n_alive

    # --- 5. Backup original + write new class_index --------------------------
    if not backup_path.exists():
        backup_path.write_text(
            json.dumps(ci_full, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[trim] backup written : {backup_path.name}")
    else:
        print(f"[trim] backup already exists, leaving as-is : {backup_path.name}")

    ci_path.write_text(
        json.dumps(new_ci, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[trim] new class_index ({n_alive:,}) written : {ci_path.name}")

    # --- 6. Rewrite all shards (atomic temp+rename) -------------------------
    print(f"[trim] rewriting {len(shards):,} shards (atomic temp+rename)…")
    _rewrite_shards(shards, remap, args.write_progress_every,
                    idx_offset=0, total=len(shards))
    print(f"[trim] DONE — {n_full:,} → {n_alive:,} classes ({n_dead:,} dropped)")
    print(f"[trim] next: regenerate aux_labels.npz against new class_index")


def _rewrite_shards(shards, remap, progress_every: int,
                    idx_offset: int, total: int) -> None:
    """Rewrite each shard's labels via remap (atomic temp+rename).

    `idx_offset` and `total` are for progress display only — when resuming, we
    want to show e.g. "rewrite  300/3927" rather than "rewrite  100/3727".
    """
    t0 = time.perf_counter()
    bytes_written = 0
    for i, p in enumerate(shards):
        d = np.load(p)
        old_labels = d["labels"]
        new_labels = remap[old_labels]
        if (new_labels < 0).any():
            bad = int((new_labels < 0).sum())
            raise RuntimeError(
                f"{p.name}: {bad} dead labels found — class_index/shard mismatch"
            )
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, images=d["images"], labels=new_labels)
        d.close()
        tmp.replace(p)
        bytes_written += p.stat().st_size
        if (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate_gb = bytes_written / elapsed / 1e9
            absolute_done = idx_offset + (i + 1)
            done_pct = 100 * absolute_done / total
            eta = (len(shards) - (i + 1)) * elapsed / (i + 1)
            print(f"  rewrite {absolute_done:>5}/{total} ({done_pct:5.1f}%)  "
                  f"elapsed={elapsed:.1f}s  rate={rate_gb:.2f}GB/s  "
                  f"eta={eta:.0f}s", flush=True)
    print(f"[trim] segment done in {time.perf_counter()-t0:.1f}s, "
          f"{bytes_written/1e9:.1f} GB touched")


if __name__ == "__main__":
    main()
