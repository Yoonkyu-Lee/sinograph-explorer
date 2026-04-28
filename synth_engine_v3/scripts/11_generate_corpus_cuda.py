"""CUDA-rasterizer corpus generator (Phase OPT-2.3 + OPT-Final, doc/20).

Same end product as 10_generate_corpus_v3.py (tensor shards + class_index +
manifest) but uses the CUDA polygon rasterizer instead of CPU PIL workers.
No multiprocessing.

OPT-Final pipelining (vs first version):
  * **outline prefetch thread** — extracts outlines (CPU, LRU-cached) for
    batch N+1 while GPU runs batch N. Bounded queue depth 2.
  * **CUDA streams** — `compute_stream` for raster + augment chain;
    `copy_stream` for D2H download to a pinned host buffer. Compute on
    batch N+1 starts as soon as its mask is rasterized, even if D2H of
    batch N hasn't completed.
  * **double-buffered pinned host** — alternates two pinned buffers so D2H
    of batch N can complete asynchronously while compute_stream begins
    batch N+1.

These three stack on top of raster_kernel V2 (Lab 4 + 7 + 8) for an
end-to-end pipeline that keeps the GPU as busy as the augment chain
allows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Variable-shape per-batch allocations (edge tensors, prob-gated augment
# sub-batches) fragment the default caching allocator. Expandable segments
# behave like virtual memory — segments grow/shrink instead of being pinned
# at the high-water mark. Required for long runs (>30 min) at bs >= 96.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "cuda_raster"))
_V2_SCRIPTS = _HERE.parent.parent / "synth_engine_v2" / "scripts"
sys.path.insert(0, str(_V2_SCRIPTS))

from mask_adapter import CANVAS as MASK_CANVAS                     # noqa: E402
from pipeline_gpu import (                                          # noqa: E402
    CANVAS, GPUContext, finalize_center_crop, run_pipeline,
)
import style_gpu    # noqa: F401, E402
import augment_gpu  # noqa: F401, E402
from outline_cache import get_outline                                # noqa: E402
from rasterize import rasterize_batch                                # noqa: E402
from streaming_log import setup_logging                              # noqa: E402
import generate_corpus as v2_corpus                                  # noqa: E402
import base_source as v2_base_source                                 # noqa: E402

# SysMon: same per-iteration stats as 10_generate_corpus_v3.py — GPU util,
# VRAM (torch + device), RSS, system RAM. Optional dep, guarded import.
try:
    from sysmon import SysMon, format_snapshot                       # noqa: E402
except Exception:
    SysMon = None
    format_snapshot = None

PAD = v2_base_source.PAD_DEFAULT


def _resolve_font_for_char(fonts_dir: Path, char: str):
    sources = v2_base_source.discover_font_sources(fonts_dir, char_filter=char)
    if not sources:
        return None
    s = sources[0]
    return s.font_path, s.face_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--class-list", required=True)
    ap.add_argument("--samples-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--shard-input-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-workers", type=int, default=4)
    ap.add_argument("--metadata", action="store_true")
    ap.add_argument("--fonts-dir", default=None)
    ap.add_argument("--progress-every", type=int, default=0,
                    help="sample-based progress threshold; 0 = use --progress-secs only")
    ap.add_argument("--progress-secs", type=float, default=10.0,
                    help="time-based progress interval in seconds")
    ap.add_argument("--kernel", default="v2", choices=["v1", "v2"],
                    help="raster kernel selection (default v2 = Lab 4+7+8)")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="outline-prefetch queue depth (1 = no overlap)")
    ap.add_argument("--log", default=None,
                    help="optional run log path (default: <out>/run.log)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log if args.log else (out_dir / "run.log"))

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else (
        Path("/mnt/c/Windows/Fonts") if Path("/mnt/c/Windows/Fonts").is_dir()
        else Path("C:/Windows/Fonts")
    )
    # --- run parameters banner ---
    print("=" * 72, flush=True)
    print("RUN PARAMETERS", flush=True)
    print("=" * 72, flush=True)
    print(f"  config            = {args.config}", flush=True)
    print(f"  class_list        = {args.class_list}", flush=True)
    print(f"  samples_scale     = {args.samples_scale}", flush=True)
    print(f"  out_dir           = {out_dir}", flush=True)
    print(f"  seed              = {args.seed}", flush=True)
    print(f"  kernel            = {args.kernel}      (v1=naive Lab1 / v2=Lab4+7+8 shared-mem)", flush=True)
    print(f"  batch_size        = {args.batch_size}", flush=True)
    print(f"  shard_size        = {args.shard_size}      ({args.shard_size} samples/shard)", flush=True)
    print(f"  shard_input_size  = {args.shard_input_size} px", flush=True)
    print(f"  save_workers      = {args.save_workers}", flush=True)
    print(f"  prefetch          = {args.prefetch}      (outline-extract queue depth)", flush=True)
    print(f"  progress_secs     = {args.progress_secs}      (s, time-based log interval)", flush=True)
    print(f"  progress_every    = {args.progress_every}      (samples, 0 = disabled)", flush=True)
    print(f"  fonts_dir         = {fonts_dir}", flush=True)
    print(f"  device            = cuda", flush=True)
    print(f"  python            = {sys.version.split()[0]}", flush=True)
    print(f"  torch             = {torch.__version__}", flush=True)
    print(f"  CUDA              = {torch.version.cuda}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU               = {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)", flush=True)
    print("=" * 72, flush=True)

    # --- expand class_list into picks ---
    picks: list[str] = []
    with open(args.class_list, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            ch = entry["char"]
            n = max(1, int(round(entry["target_samples"] * args.samples_scale)))
            picks.extend([ch] * n)
    total = len(picks)
    print(f"[cuda-corpus] picks = {total:,}  (classes = {len(set(picks)):,})",
          flush=True)

    # --- class_index ---
    notations = sorted({f"U+{ord(c):04X}" for c in picks},
                       key=lambda n: int(n[2:], 16))
    class_index = {n: i for i, n in enumerate(notations)}
    (out_dir / "class_index.json").write_text(
        json.dumps(class_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[cuda-corpus] class_index written ({len(class_index):,} classes)",
          flush=True)

    # --- pre-resolve fonts per unique char ---
    print("[cuda-corpus] resolving fonts per unique char…", flush=True)
    char_to_font: dict[str, tuple] = {}
    miss = 0
    for ch in set(picks):
        fp = _resolve_font_for_char(fonts_dir, ch)
        if fp is None:
            miss += 1
            continue
        char_to_font[ch] = fp
    print(f"[cuda-corpus] resolved {len(char_to_font):,} chars, missed {miss}",
          flush=True)

    device = "cuda"
    save_pool = ThreadPoolExecutor(max_workers=args.save_workers,
                                    thread_name_prefix="save")
    save_futures: deque = deque()
    SAVE_CAP = max(args.save_workers * 2, 4)

    shard_images: list[np.ndarray] = []
    shard_metas: list[dict] = []
    shard_idx = 0
    written = 0
    skipped_no_source = 0
    t_start = time.perf_counter()

    # CUDA streams
    compute_stream = torch.cuda.Stream(device=device)
    copy_stream = torch.cuda.Stream(device=device)

    # SysMon — sampled GPU util / VRAM / system RAM, prepended to progress
    sysmon = None
    if SysMon is not None:
        try:
            sysmon = SysMon()
        except Exception as e:
            print(f"[warn] SysMon unavailable: {e}", flush=True)

    # warmup + force kernel build
    print(f"[cuda-corpus] warmup rasterizer (kernel={args.kernel})…", flush=True)
    sample_ch = next(iter(char_to_font))
    fp0, fi0 = char_to_font[sample_ch]
    od0 = get_outline(fp0, fi0, sample_ch, MASK_CANVAS, PAD)
    _ = rasterize_batch([od0], MASK_CANVAS, MASK_CANVAS,
                         device=device, kernel=args.kernel)
    torch.cuda.synchronize()
    print("[cuda-corpus] rasterizer ready", flush=True)

    # Rough up-front ETA — uses tabulated steady-state rates from doc/20 §13
    # so the user can see "this run should take ~9.2 h" before the first
    # progress line. Refined after first measurement.
    from datetime import datetime as _dt, timedelta as _td
    REFERENCE_RATES = {  # samples/s @ 384 canvas, 128 shard, RTX 4080 Laptop
        "v1_bs64":  531,
        "v2_bs64":  530,
        "v2_bs128": 623,
        "v2_bs256": 250,    # OOM-spill territory
    }
    key = f"{args.kernel}_bs{args.batch_size}"
    ref_rate = REFERENCE_RATES.get(key, 500)
    est_sec = total / ref_rate
    est_finish = _dt.now() + _td(seconds=est_sec)
    h, rem = divmod(int(est_sec), 3600)
    m, s = divmod(rem, 60)
    eta_str = f"{h}h{m:02d}m{s:02d}s" if h > 0 else f"{m}m{s:02d}s"
    print(f"[cuda-corpus] up-front ETA: ~{eta_str}  (assumes {ref_rate}/s, "
          f"finish ~{est_finish:%Y-%m-%d %H:%M:%S})", flush=True)

    # --- outline prefetch thread ---
    # Background CPU thread pulls batches of (chars) from the iterator and
    # extracts outlines into the queue. Main loop pulls from the queue. With
    # prefetch_depth=2 the GPU rarely waits for outline extraction.
    bs = args.batch_size
    batch_starts = list(range(0, total, bs))
    prefetch_q: "deque[tuple[int, list, list]]" = deque()
    prefetch_lock = threading.Lock()
    prefetch_cv = threading.Condition(prefetch_lock)
    prefetch_done = threading.Event()
    PREFETCH_DEPTH = max(1, args.prefetch)

    def prefetch_worker():
        for batch_start in batch_starts:
            batch_picks = picks[batch_start:batch_start + bs]
            valid_chars = [c for c in batch_picks if c in char_to_font]
            if valid_chars:
                outlines = []
                for ch in valid_chars:
                    fp, fi = char_to_font[ch]
                    outlines.append(get_outline(fp, fi, ch, MASK_CANVAS, PAD))
            else:
                outlines = []
            with prefetch_cv:
                while len(prefetch_q) >= PREFETCH_DEPTH:
                    prefetch_cv.wait()
                prefetch_q.append((batch_start, valid_chars, outlines))
                prefetch_cv.notify_all()
        prefetch_done.set()
        with prefetch_cv:
            prefetch_cv.notify_all()

    pf_thread = threading.Thread(target=prefetch_worker, daemon=True,
                                  name="outline-prefetch")
    pf_thread.start()

    # --- save flush helper ---
    def flush_shard(force: bool = False):
        nonlocal shard_idx
        while len(shard_images) >= args.shard_size or (force and shard_images):
            n = min(args.shard_size, len(shard_images))
            chunk_imgs = np.stack(shard_images[:n], axis=0)
            chunk_metas = shard_metas[:n]
            del shard_images[:n]
            del shard_metas[:n]
            labels = np.array(
                [class_index[f"U+{ord(m['char']):04X}"] for m in chunk_metas],
                dtype=np.int64,
            )
            shard_fname = f"shard-{shard_idx:05d}.npz"
            shard_path = out_dir / shard_fname
            shard_idx += 1

            def _save(p, imgs, lbls):
                np.savez(p, images=imgs, labels=lbls)
            save_futures.append(save_pool.submit(_save, shard_path, chunk_imgs, labels))
            while len(save_futures) > SAVE_CAP:
                save_futures.popleft().result()

    # --- main loop: pull prefetched batches, run GPU pipeline ---------------
    n_batches_done = 0
    # Two complementary triggers:
    #  - sample-based: print when crossing each multiple of progress_every
    #    (disabled by default — set 0 to fall back to time-only).
    #  - time-based:   print when at least progress_secs have elapsed since
    #    the previous print. Default 10 s, batch-size-independent.
    next_progress_threshold = args.progress_every if args.progress_every > 0 else float("inf")
    last_progress_time = time.perf_counter()
    while True:
        # pull next prefetched batch (or break)
        with prefetch_cv:
            while not prefetch_q and not prefetch_done.is_set():
                prefetch_cv.wait()
            if not prefetch_q:
                break
            batch_start, valid_chars, outlines = prefetch_q.popleft()
            prefetch_cv.notify_all()

        if not outlines:
            skipped_no_source += len(picks[batch_start:batch_start + bs])
            continue

        n = len(outlines)
        # GPU compute on compute_stream
        with torch.cuda.stream(compute_stream):
            mask_t = rasterize_batch(outlines, MASK_CANVAS, MASK_CANVAS,
                                      device=device, kernel=args.kernel)
            canvas = torch.ones(n, 3, CANVAS, CANVAS, device=device)
            gen = torch.Generator(device=device).manual_seed(args.seed + batch_start)
            ctx = GPUContext(
                canvas=canvas, mask=mask_t, rng=gen,
                chars=valid_chars,
                source_kinds=["font"] * n,
                device=device,
            )
            ctx = run_pipeline(ctx, cfg)
            final = finalize_center_crop(ctx.canvas)
            if final.shape[-1] != args.shard_input_size:
                import torch.nn.functional as _F
                final = _F.interpolate(final, size=args.shard_input_size,
                                        mode="bilinear", align_corners=False,
                                        antialias=True)
            u8_dev = (final.clamp(0, 1) * 255).to(torch.uint8) \
                .permute(0, 2, 3, 1).contiguous()
        compute_done = compute_stream.record_event()

        # D2H on copy_stream — async, doesn't block compute_stream from
        # picking up next batch
        with torch.cuda.stream(copy_stream):
            torch.cuda.current_stream().wait_event(compute_done)
            u8_cpu = u8_dev.cpu()
        copy_done = copy_stream.record_event()
        copy_done.synchronize()  # wait until u8_cpu is ready
        u8 = u8_cpu.numpy()

        for j, ch in enumerate(valid_chars):
            shard_images.append(u8[j])
            shard_metas.append({"char": ch, "idx": batch_start + j})
            written += 1
        flush_shard(force=False)

        n_batches_done += 1
        done = batch_start + n
        now = time.perf_counter()
        # Print when (a) crossed sample threshold (if enabled), (b) elapsed
        # `progress_secs` since last print, or (c) done.
        crossed_sample = done >= next_progress_threshold
        crossed_time = (now - last_progress_time) >= args.progress_secs
        if crossed_sample or crossed_time or done >= total:
            if crossed_sample and args.progress_every > 0:
                while next_progress_threshold <= done:
                    next_progress_threshold += args.progress_every
            last_progress_time = now
            elapsed = now - t_start
            from datetime import datetime, timedelta
            rate = done / max(elapsed, 1e-6)
            eta_sec = (total - done) / max(rate, 1e-6)
            ts = time.strftime("%H:%M:%S")

            def _fmt(sec: float) -> str:
                """Format seconds as Xh Ym Zs (only show hours when > 0)."""
                sec = int(sec)
                h, r = divmod(sec, 3600)
                m, s = divmod(r, 60)
                return f"{h}h{m:02d}m{s:02d}s" if h > 0 else f"{m}m{s:02d}s"

            finish_dt = datetime.now() + timedelta(seconds=eta_sec)
            finish_clock = finish_dt.strftime("%H:%M:%S")
            msg = (f"[{ts}] {done:,}/{total:,}  written={written:,}  "
                   f"skipped_no_src={skipped_no_source:,}  "
                   f"rate={rate:.1f}/s  "
                   f"elapsed={_fmt(elapsed)}  "
                   f"eta={_fmt(eta_sec)} (finish ~{finish_clock})")
            if sysmon is not None and format_snapshot is not None:
                try:
                    msg += "  " + format_snapshot(sysmon.snapshot())
                except Exception:
                    pass
            print(msg, flush=True)

    pf_thread.join()
    flush_shard(force=True)
    for f in save_futures:
        f.result()
    save_pool.shutdown()

    elapsed = time.perf_counter() - t_start
    print(f"\n[cuda-corpus] done.  written={written:,}  "
          f"skipped_no_src={skipped_no_source:,}  "
          f"elapsed={elapsed:.1f}s  rate={written/elapsed:.1f} samples/s",
          flush=True)


if __name__ == "__main__":
    main()
