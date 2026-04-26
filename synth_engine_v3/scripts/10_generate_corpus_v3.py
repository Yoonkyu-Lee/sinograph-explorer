"""Corpus-level v3 driver.

Architecture (per V3_DESIGN):
  CPU mask workers (multiprocessing.Pool) — render PIL mask per char
       |
       v  (imap_unordered)
  Main process — accumulate masks, run GPU batch style+augment
       |
       v  (ThreadPoolExecutor)
  Save workers — finalize + PIL + PNG write to disk

v2's `generate_corpus.py` char-pool / sample_chars / block_weights are
imported as-is (same JSONL, same semantics).
"""
from __future__ import annotations

# Thread caps BEFORE numpy import. Workers each cap to 1 thread so N workers
# get ~N× total throughput rather than N × cpu_count() thread contention.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import queue as _queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# v2 reuse — char pool + block-weighted sampling + BLOCKS.
_V2_SCRIPTS = _HERE.parent.parent / "synth_engine_v2" / "scripts"
sys.path.insert(0, str(_V2_SCRIPTS))
import generate_corpus as v2_corpus  # noqa: E402

from mask_adapter import CANVAS, get_sources_for_char, masks_to_tensor, render_mask  # noqa: E402
from pipeline_gpu import (  # noqa: E402
    GPUContext, finalize_center_crop, run_pipeline, tensor_to_pil_batch,
)
import style_gpu    # noqa: F401, E402
import augment_gpu  # noqa: F401, E402


# ---------- mask worker ----------

_WORKER_STATE: dict = {}


def _init_mask_worker(base_source_spec: dict, fonts_dir: str, seed: int) -> None:
    _WORKER_STATE["spec"] = base_source_spec
    _WORKER_STATE["fonts_dir"] = Path(fonts_dir)
    _WORKER_STATE["seed"] = int(seed)
    # Pre-warm EVERY cache the spec will touch at sample time. One
    # resolve+render per sub-kind pulls the font scan, MMH / e-hanja / KanjiVG
    # JSONL indices, outline JSONL indices, and parsed-polygon caches into
    # memory so the first real sample doesn't pay them on the hot path.
    warm_spec = base_source_spec or {"kind": "font"}
    sub_specs = (warm_spec.get("sources") or [warm_spec]
                 if warm_spec.get("kind") == "multi" else [warm_spec])
    warmup_char = "人"  # covered by every base_source kind
    warm_rng = np.random.default_rng(0)
    for sub in sub_specs:
        try:
            srcs = get_sources_for_char(warmup_char, sub, _WORKER_STATE["fonts_dir"])
            if srcs:
                render_mask(warmup_char, srcs[0], rng=warm_rng)
        except Exception:
            pass


def _render_mask_task(task):
    idx, char = task
    spec = _WORKER_STATE["spec"]
    fonts_dir = _WORKER_STATE["fonts_dir"]
    seed = _WORKER_STATE["seed"]
    sample_rng = np.random.default_rng(seed + 1_000_000 + idx)
    try:
        sources = get_sources_for_char(char, spec, fonts_dir)
    except Exception as e:
        return {"idx": idx, "char": char, "err": f"resolve:{e}"}
    if not sources:
        return {"idx": idx, "char": char, "err": "no_source"}
    src = (sources[0] if len(sources) == 1
           else sources[int(sample_rng.integers(0, len(sources)))])
    mask = render_mask(char, src, rng=sample_rng)
    if mask is None:
        return {"idx": idx, "char": char, "err": "render_failed"}
    kind = getattr(src, "last_kind", None) or getattr(src, "kind", "") or ""
    tag = src.tag()
    picked = getattr(src, "last_picked", None) or tag
    return {
        "idx": idx, "char": char,
        "mask": np.asarray(mask, dtype=np.uint8),
        "tag": tag, "kind": kind, "picked": picked,
        "seed": seed + 1_000_000 + idx,
    }


# ---------- save worker ----------

def _save_png(arr_hwc: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(arr_hwc).save(path, compress_level=1)


def _save_shard(path: Path, images: np.ndarray, labels: np.ndarray) -> None:
    """Save one tensor shard (uncompressed npz) — images: uint8[N,H,W,3], labels: int64[N]."""
    np.savez(path, images=images, labels=labels)


# ---------- manual mask worker loop (backpressure-aware) ----------

def _mask_worker_loop(task_q: "mp.Queue", result_q: "mp.Queue",
                       base_spec: dict, fonts_dir: str, seed: int):
    """Dedicated worker process. Pulls tasks (or task batches) from task_q,
    renders mask(s), puts result(s) into result_q. When result_q is full
    (maxsize reached), the put() blocks — this is how we enforce backpressure
    against main's consumption rate.

    Task formats accepted:
      - `None`                         → poison pill, exit
      - `(idx, char)` tuple            → single render
      - `list[(idx, char), ...]`       → batched render, returns list of results

    Batched path amortizes mp.Queue pickle overhead by N× (Phase B of doc/14).
    """
    _init_mask_worker(base_spec, fonts_dir, seed)
    while True:
        task = task_q.get()
        if task is None:
            break
        if isinstance(task, list):
            # batched — render all, push as list
            results = [_render_mask_task(t) for t in task]
            result_q.put(results)
        else:
            # single (legacy)
            result = _render_mask_task(task)
            result_q.put(result)


# ---------- main ----------

def run_corpus(cfg: dict, picks: list[str], out_dir: Path,
                seed: int, n_mask_workers: int, n_save_workers: int,
                batch_size: int, write_metadata: bool,
                fonts_dir: Path, progress_every: int = 500,
                output_format: str = "png",
                shard_size: int = 5000,
                shard_input_size: int = 128,
                mask_batch: int = 1) -> dict:
    """Feed picks through mask workers → GPU pipeline → save threads.

    `output_format`:
      - "png"          : one PNG per sample (default, backward compat)
      - "tensor_shard" : u8 images batched into .npz shards at shard_input_size.
                         Eliminates per-sample PNG encode + later decode in train loop.

    Returns a stats dict with counts + timing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    total = len(picks)
    tasks = list(enumerate(picks))
    init_args = (cfg.get("base_source", {}), str(fonts_dir), seed)

    # --- class_index (only needed for tensor_shard mode) -------------------
    class_index = None
    if output_format == "tensor_shard":
        notations = sorted({f"U+{ord(c):04X}" for c in picks},
                           key=lambda n: int(n[2:], 16))
        class_index = {n: i for i, n in enumerate(notations)}
        (out_dir / "class_index.json").write_text(
            json.dumps(class_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[shard] output_format=tensor_shard  shard_size={shard_size}  "
              f"input_size={shard_input_size}  classes={len(class_index)}")

    # --- mask workers with explicit backpressure ---------------------------
    # mp.Pool.imap_unordered has no backpressure: workers keep producing even
    # when main can't consume, and the internal result cache grows unbounded,
    # leaking memory on long runs. Switch to a pair of bounded Queues so the
    # mask workers block on put() when main is behind. Keeps RSS flat.
    mask_procs: list = []
    task_q: "mp.Queue" = None  # type: ignore[assignment]
    result_q: "mp.Queue" = None  # type: ignore[assignment]
    mask_iter_legacy = None  # only set in serial (workers=1) mode

    if n_mask_workers <= 1:
        print(f"mask workers: 1 (serial, in-process)")
        _init_mask_worker(*init_args)
        mask_iter_legacy = (_render_mask_task(t) for t in tasks)
    else:
        # task_q is bounded so tasks are produced lazily (lock-step with
        # consumption); result_q is bounded so workers block on put() when
        # GPU is behind. Both bounds are modest → total pending-results
        # memory is O(batch_size × something small × mask_bytes).
        task_q_cap = max(batch_size * 8, 256)
        result_q_cap = max(batch_size * 4, 128)
        task_q = mp.Queue(maxsize=task_q_cap)
        result_q = mp.Queue(maxsize=result_q_cap)
        print(f"mask workers: {n_mask_workers}  task_q={task_q_cap}  result_q={result_q_cap}")
        for _ in range(n_mask_workers):
            p = mp.Process(target=_mask_worker_loop,
                            args=(task_q, result_q,
                                   cfg.get("base_source", {}),
                                   str(fonts_dir), seed),
                            daemon=False)
            p.start()
            mask_procs.append(p)

    save_pool = ThreadPoolExecutor(max_workers=max(1, n_save_workers))
    # Bounded pending-futures deque. When it fills we drain the oldest half
    # so save-side memory + GC pressure stays flat across a multi-million run.
    from collections import deque
    save_futures: deque = deque()
    # Cap depends on output format — each in-flight future holds its payload:
    #  - PNG mode: ~40 KB per future → generous cap fine
    #  - tensor_shard mode: shard_size × H × W × 3 bytes (up to ~1 GB) → tight cap required
    if output_format == "tensor_shard":
        # max (n_workers * 2) shards in flight, typically ~n_workers GB memory.
        _SAVE_FUTURES_SOFT_CAP = max(n_save_workers * 2, 4)
    else:
        _SAVE_FUTURES_SOFT_CAP = max(n_save_workers * 256, 1024)
    # Manifest is streamed to disk instead of accumulating a 5M+ entry list.
    manifest_file = None
    if write_metadata:
        manifest_path = out_dir / "corpus_manifest.jsonl"
        manifest_file = open(manifest_path, "w", encoding="utf-8", newline="\n")

    t_start = time.perf_counter()
    stats = {"total": total, "written": 0, "skipped": 0, "errors": {}}

    # instrumentation — wall time spent in each stage
    t_gpu_total = 0.0
    t_save_dispatch_total = 0.0
    t_mask_wait_total = 0.0
    n_batches = 0

    # System monitoring — same SysMon as train_engine_v1 (sampled GPU util +
    # VRAM + RSS). Initialized after CUDA is ready (model/pipeline construction
    # forces context). Guard import so serial CPU mode still works.
    sysmon = None
    try:
        from sysmon import SysMon, format_snapshot
        sysmon = SysMon()
    except Exception as e:
        print(f"[warn] SysMon unavailable: {e}")

    # buffers (per GPU batch)
    buf_masks: list[np.ndarray] = []
    buf_meta: list[dict] = []

    # shard accumulator (only used when output_format="tensor_shard")
    shard_images: list[np.ndarray] = []
    shard_metas: list[dict] = []
    shard_idx = 0

    def flush_batch():
        nonlocal t_gpu_total, t_save_dispatch_total, n_batches, shard_idx
        if not buf_masks:
            return
        n = len(buf_masks)
        tg0 = time.perf_counter()
        stacked = np.stack(buf_masks, axis=0)
        mask_t = torch.from_numpy(stacked).to(device, dtype=torch.float32).unsqueeze(1) / 255.0
        chars_b = [m["char"] for m in buf_meta]
        kinds_b = [m["kind"] for m in buf_meta]
        gen = torch.Generator(device=device).manual_seed(seed + 2_000_000 + buf_meta[0]["idx"])
        canvas = torch.ones(n, 3, CANVAS, CANVAS, device=device)
        ctx = GPUContext(canvas=canvas, mask=mask_t, rng=gen,
                         chars=chars_b, source_kinds=kinds_b, device=device)
        ctx = run_pipeline(ctx, cfg)
        final = finalize_center_crop(ctx.canvas)
        # When output_format=tensor_shard, resize on GPU to training input size
        # BEFORE the cpu/numpy transfer — cuts CPU/IPC payload 4× (e.g. 256→128).
        if output_format == "tensor_shard" and final.shape[-1] != shard_input_size:
            import torch.nn.functional as _F
            final = _F.interpolate(final, size=shard_input_size, mode="bilinear",
                                    align_corners=False, antialias=True)
        u8 = (final.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        torch.cuda.synchronize()
        t_gpu_total += time.perf_counter() - tg0
        n_batches += 1

        ts0 = time.perf_counter()
        if output_format == "tensor_shard":
            # accumulate per-sample uint8 images into shard buffer; when full,
            # submit a batched shard write to the save pool.
            for j, meta in enumerate(buf_meta):
                shard_images.append(u8[j])  # (H, W, 3) uint8
                shard_metas.append(meta)
                stats["written"] += 1
            # flush full shards (multiple may accumulate if batch > shard_size)
            while len(shard_images) >= shard_size:
                chunk_imgs = np.stack(shard_images[:shard_size], axis=0)
                chunk_metas = shard_metas[:shard_size]
                del shard_images[:shard_size]
                del shard_metas[:shard_size]
                labels = np.array(
                    [class_index[f"U+{ord(m['char']):04X}"] for m in chunk_metas],
                    dtype=np.int64,
                )
                shard_fname = f"shard-{shard_idx:05d}.npz"
                shard_path = out_dir / shard_fname
                save_futures.append(save_pool.submit(_save_shard, shard_path, chunk_imgs, labels))
                if manifest_file is not None:
                    for j, meta in enumerate(chunk_metas):
                        notation = f"U+{ord(meta['char']):04X}"
                        manifest_file.write(json.dumps({
                            "idx": meta["idx"], "char": meta["char"],
                            "notation": notation,
                            "block": v2_corpus.block_of(ord(meta["char"])),
                            "base_source_kind": meta["kind"],
                            "picked_source": meta["picked"],
                            "tag": meta["tag"],
                            "seed": meta["seed"],
                            "shard_filename": shard_fname,
                            "shard_index": j,
                        }, ensure_ascii=False) + "\n")
                shard_idx += 1
        else:
            # png mode (legacy)
            for j, meta in enumerate(buf_meta):
                notation = f"U+{ord(meta['char']):04X}"
                safe = str(meta["picked"]).replace("/", "_").replace("\\", "_")
                fname = f"{meta['idx']:06d}_{notation}_{safe}.png"
                path = out_dir / fname
                save_futures.append(save_pool.submit(_save_png, u8[j], path))
                if manifest_file is not None:
                    manifest_file.write(json.dumps({
                        "idx": meta["idx"], "char": meta["char"],
                        "notation": notation,
                        "block": v2_corpus.block_of(ord(meta["char"])),
                        "base_source_kind": meta["kind"],
                        "picked_source": meta["picked"],
                        "tag": meta["tag"],
                        "seed": meta["seed"],
                        "filename": fname,
                    }, ensure_ascii=False) + "\n")
                stats["written"] += 1
        t_save_dispatch_total += time.perf_counter() - ts0

        # Drain save futures when pending queue grows beyond soft cap. This
        # keeps total memory flat (Futures hold refs to completed results
        # indefinitely, and a 5M+ list was causing a linear slowdown in the
        # original implementation).
        while len(save_futures) > _SAVE_FUTURES_SOFT_CAP:
            save_futures.popleft().result()

        buf_masks.clear()
        buf_meta.clear()

    # --- feeder + consumer iterator ---------------------------------------
    if mask_procs:
        mb = max(1, int(mask_batch))
        # Bounded producer thread — fills task_q with task(s). If mask_batch>1
        # we chunk tasks into mini-batches, amortizing mp.Queue pickle overhead.
        def _task_feeder():
            if mb <= 1:
                for t in tasks:
                    task_q.put(t)
            else:
                for i in range(0, len(tasks), mb):
                    task_q.put(tasks[i:i + mb])
            # poison pills — one per worker so they exit cleanly
            for _ in range(len(mask_procs)):
                task_q.put(None)
        feeder_thread = threading.Thread(target=_task_feeder, daemon=True)
        feeder_thread.start()

        def _result_iter():
            # Yields exactly `total` individual results. When mb>1, workers
            # return lists — we flatten them here.
            got = 0
            while got < total:
                item = result_q.get()
                if isinstance(item, list):
                    for r in item:
                        yield r
                        got += 1
                        if got >= total:
                            return
                else:
                    yield item
                    got += 1
        buffered_iter = _result_iter()
    else:
        buffered_iter = mask_iter_legacy  # serial mode

    try:
        _t_last_mask = time.perf_counter()
        for i, res in enumerate(buffered_iter, 1):
            t_mask_wait_total += time.perf_counter() - _t_last_mask
            if "err" in res:
                stats["skipped"] += 1
                stats["errors"][res["err"]] = stats["errors"].get(res["err"], 0) + 1
            else:
                buf_masks.append(res["mask"])
                buf_meta.append(res)
                if len(buf_masks) >= batch_size:
                    flush_batch()
            if i % progress_every == 0 or i == total:
                elapsed = time.perf_counter() - t_start
                rate = i / max(elapsed, 1e-6)
                eta = (total - i) / max(rate, 1e-6)
                ts = time.strftime("%H:%M:%S")
                msg = (
                    f"[{ts}] {i:,}/{total:,}  written={stats['written']:,}  "
                    f"skipped={stats['skipped']:,}  "
                    f"rate={rate:.1f}/s  t={elapsed:.0f}s  eta={eta:.0f}s"
                )
                if sysmon is not None:
                    snap = sysmon.snapshot()
                    msg += "  " + format_snapshot(snap)
                print(msg)
            _t_last_mask = time.perf_counter()
        # tail
        flush_batch()
        # flush any remaining partial shard (tensor_shard mode only)
        if output_format == "tensor_shard" and shard_images:
            chunk_imgs = np.stack(shard_images, axis=0)
            labels = np.array(
                [class_index[f"U+{ord(m['char']):04X}"] for m in shard_metas],
                dtype=np.int64,
            )
            shard_fname = f"shard-{shard_idx:05d}.npz"
            shard_path = out_dir / shard_fname
            save_futures.append(save_pool.submit(_save_shard, shard_path, chunk_imgs, labels))
            if manifest_file is not None:
                for j, meta in enumerate(shard_metas):
                    notation = f"U+{ord(meta['char']):04X}"
                    manifest_file.write(json.dumps({
                        "idx": meta["idx"], "char": meta["char"],
                        "notation": notation,
                        "block": v2_corpus.block_of(ord(meta["char"])),
                        "base_source_kind": meta["kind"],
                        "picked_source": meta["picked"],
                        "tag": meta["tag"],
                        "seed": meta["seed"],
                        "shard_filename": shard_fname,
                        "shard_index": j,
                    }, ensure_ascii=False) + "\n")
            shard_idx += 1
            shard_images.clear()
            shard_metas.clear()
    finally:
        # Shut down mask workers cleanly. feeder_thread already put the poison
        # pills at the end of tasks, so workers exit once they drain. Just
        # join with a timeout in case of anomalies.
        for p in mask_procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        # wait for in-flight saves (drain remaining bounded deque)
        while save_futures:
            save_futures.popleft().result()
        save_pool.shutdown()
        if manifest_file is not None:
            manifest_file.close()

    elapsed = time.perf_counter() - t_start
    stats["elapsed_s"] = elapsed
    stats["rate_samples_per_s"] = stats["written"] / max(elapsed, 1e-6)
    stats["t_mask_wait_s"] = t_mask_wait_total
    stats["t_gpu_s"] = t_gpu_total
    stats["t_save_dispatch_s"] = t_save_dispatch_total
    stats["n_batches"] = n_batches
    stats["ms_per_batch_gpu"] = (t_gpu_total / max(n_batches, 1)) * 1000
    print(f"[instrument] mask_wait={t_mask_wait_total:.1f}s  "
          f"gpu={t_gpu_total:.1f}s ({stats['ms_per_batch_gpu']:.1f} ms/batch × {n_batches})  "
          f"save_dispatch={t_save_dispatch_total:.1f}s")
    # per-step profiler dump (set V3_PROFILE_STEPS=1)
    from pipeline_gpu import PROFILE_STEPS, STEP_PROFILE, STEP_CALLS, STEP_SAMPLES
    if PROFILE_STEPS and STEP_PROFILE:
        print("[per-step] (sorted by total ms)")
        rows = sorted(STEP_PROFILE.items(), key=lambda x: -x[1])
        for name, ms in rows:
            calls = STEP_CALLS[name]
            samples = STEP_SAMPLES[name]
            per_call = ms / max(calls, 1)
            per_sample = ms / max(samples, 1)
            print(f"   {name:<35} total={ms:7.0f} ms  calls={calls:4d}  samples={samples:6d}  "
                  f"per_call={per_call:6.2f} ms  per_sample={per_sample:6.3f} ms")

    # Manifest was streamed directly during the run (see above). It is NOT
    # sorted by idx — entries appear in completion order. If idx-sorted
    # order matters downstream, sort at load time.
    if write_metadata:
        print(f"wrote manifest: {out_dir / 'corpus_manifest.jsonl'} (streamed)")

    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    # --- per-class-quota mode (preferred for production): takes a JSONL where
    # each line has {"codepoint": "U+XXXX", "char": "鑑", "target_samples": N}.
    # Generates exactly N samples for each class (char). Overrides pool /
    # strategy / total / coverage / block-weights.
    p.add_argument("--class-list", default=None,
                    help="path to class_list JSONL from select_class_list.py")
    p.add_argument("--samples-scale", type=float, default=1.0,
                    help="multiplier on each class's target_samples (e.g. 0.1 for pilot)")
    # --- legacy pool-based sampling ---
    p.add_argument("--coverage",
                    default=str(_V2_SCRIPTS.parent / "out" / "coverage_per_char.jsonl"))
    p.add_argument("--pool", default="union",
                    choices=["union", "intersection", "mmh", "ehanja", "kanjivg"])
    p.add_argument("--strategy", default="uniform",
                    choices=["uniform", "stratified_by_block"])
    p.add_argument("--total", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--metadata", action="store_true")
    p.add_argument("--fonts-dir", default="C:/Windows/Fonts")
    p.add_argument("--block-weights-json", default=None)
    p.add_argument("--mask-workers", type=int, default=4)
    p.add_argument("--mask-batch", type=int, default=1,
                   help="tasks per mp.Queue item for mask workers. Default 1 "
                        "(legacy per-task). Values 8-32 amortize pickle overhead "
                        "(Windows IPC), reducing mask_wait when it dominates gpu.")
    p.add_argument("--save-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--output-format", choices=["png", "tensor_shard"], default="png",
                   help="png: one file per sample (legacy). "
                        "tensor_shard: batch samples into uncompressed npz shards "
                        "(pre-resized to --shard-input-size). Eliminates PNG encode/decode "
                        "round-trip — feeds train_engine_v2 directly.")
    p.add_argument("--shard-size", type=int, default=5000,
                   help="samples per tensor shard (output-format=tensor_shard only)")
    p.add_argument("--shard-input-size", type=int, default=256,
                   help="shard image resolution. Default 256 preserves full CANVAS "
                        "(same as PNG output). Pass 128 for ~4x smaller shards matching "
                        "the M4 training input_size — but loses detail on complex Ext A/B "
                        "rare chars. At 256: ~1 TB for 5.45M samples uncompressed. "
                        "At 128: ~267 GB.")
    p.add_argument("--profile-steps", action="store_true",
                    help="enable per-layer timing in pipeline_gpu (main process only)")
    p.add_argument("--log-file", default=None,
                    help="also tee all stdout/stderr to this file. Defaults to "
                         "<out>/run.log if --out is set.")
    args = p.parse_args()
    if args.profile_steps:
        import pipeline_gpu as _pg
        _pg.set_profile_steps(True)

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config_tag = cfg_path.stem

    out_dir = (Path(args.out) if args.out
               else _HERE.parent / "out" / f"corpus_{config_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr to a log file for later inspection.
    log_path = Path(args.log_file) if args.log_file else (out_dir / "run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, s: str) -> int:
            for st in self._streams:
                try:
                    st.write(s)
                except Exception:
                    pass
            # Force flush-on-newline so run.log tail is usable in real time.
            # Without this, Python's default block buffering holds progress
            # lines in memory until the buffer fills (kilobytes), so the
            # on-disk file lags minutes behind the terminal.
            if "\n" in s:
                for st in self._streams:
                    try:
                        st.flush()
                    except Exception:
                        pass
            return len(s)
        def flush(self) -> None:
            for st in self._streams:
                try:
                    st.flush()
                except Exception:
                    pass

    _log_fh = open(log_path, "w", encoding="utf-8", newline="\n", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, _log_fh)
    sys.stderr = _Tee(_orig_stderr, _log_fh)
    print(f"[log] tee-ing stdout/stderr to {log_path}")

    # block weights: cfg first, CLI overrides
    block_weights: dict = {}
    if isinstance(cfg.get("corpus"), dict):
        block_weights.update(cfg["corpus"].get("block_weights", {}) or {})
    if args.block_weights_json:
        block_weights.update(json.loads(args.block_weights_json))

    pick_rng = np.random.default_rng(args.seed)
    from collections import defaultdict
    dist: dict = defaultdict(int)

    if args.class_list:
        # per-class quota mode
        class_list_path = Path(args.class_list)
        print(f"class-list mode: {class_list_path}  scale={args.samples_scale}")
        picks: list[str] = []
        class_count = 0
        tier_class_counts: dict = defaultdict(int)
        tier_sample_counts: dict = defaultdict(int)
        with open(class_list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                ch = entry["char"]
                n = max(1, int(round(entry["target_samples"] * args.samples_scale)))
                picks.extend([ch] * n)
                class_count += 1
                tier = entry.get("tier_picked", "?")
                tier_class_counts[tier] += 1
                tier_sample_counts[tier] += n
                dist[v2_corpus.block_of(ord(ch))] += n
        print(f"  classes: {class_count:,}   total samples: {len(picks):,}")
        print("  per-tier subtotals:")
        for tier in sorted(tier_class_counts):
            print(f"    {tier}: classes={tier_class_counts[tier]:>7,}  "
                  f"samples={tier_sample_counts[tier]:>10,}")
        pick_rng.shuffle(picks)
        print(f"  shuffled. out={out_dir}")
    else:
        # legacy pool-based sampling
        coverage_path = Path(args.coverage)
        print(f"char pool: {args.pool}  coverage={coverage_path}")
        chars = v2_corpus.load_char_pool(coverage_path, pool=args.pool)
        print(f"  pool size: {len(chars):,}")
        picks = v2_corpus.sample_chars(chars, pick_rng, args.strategy, args.total,
                                         block_weights=block_weights)
        print(f"strategy={args.strategy}  total={args.total}  out={out_dir}")
        for c in picks:
            dist[v2_corpus.block_of(ord(c))] += 1

    print("pick distribution (by block):")
    for b, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {b:<20} {n:>9,}")

    stats = run_corpus(
        cfg=cfg, picks=picks, out_dir=out_dir,
        seed=args.seed, n_mask_workers=args.mask_workers,
        n_save_workers=args.save_workers, batch_size=args.batch_size,
        write_metadata=args.metadata, fonts_dir=Path(args.fonts_dir),
        progress_every=args.progress_every,
        output_format=args.output_format,
        shard_size=args.shard_size,
        shard_input_size=args.shard_input_size,
        mask_batch=args.mask_batch,
    )
    print()
    print(f"done  elapsed={stats['elapsed_s']:.1f}s  "
          f"written={stats['written']:,}  skipped={stats['skipped']:,}  "
          f"rate={stats['rate_samples_per_s']:.1f} samples/s")
    if stats["errors"]:
        print("error breakdown:", stats["errors"])


if __name__ == "__main__":
    mp.freeze_support()
    main()
