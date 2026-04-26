"""Phase 0 smoke bench.

random (N, 3, 256, 256) on GPU -> kornia gaussian_blur2d -> uint8 -> PNG save.
Measures per-stage ms and samples/s to confirm the v3 approach is viable.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import kornia.filters as KF
from PIL import Image


def bench(batch: int, batches: int, out_dir: Path, save: bool, device: str) -> None:
    g = torch.Generator(device=device).manual_seed(0)

    t_total0 = time.perf_counter()
    t_gpu = 0.0
    t_dl = 0.0
    t_save = 0.0
    n = 0

    # warmup
    x = torch.rand(batch, 3, 256, 256, device=device, generator=g)
    _ = KF.gaussian_blur2d(x, (7, 7), (2.0, 2.0))
    if device == "cuda":
        torch.cuda.synchronize()

    for b in range(batches):
        t0 = time.perf_counter()
        x = torch.rand(batch, 3, 256, 256, device=device, generator=g)
        y = KF.gaussian_blur2d(x, (7, 7), (2.0, 2.0))
        y = (y * 255).clamp(0, 255).to(torch.uint8)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_gpu += t1 - t0

        arr = y.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        t2 = time.perf_counter()
        t_dl += t2 - t1

        if save:
            for i in range(batch):
                Image.fromarray(arr[i]).save(out_dir / f"b{b:03d}_{i:02d}.png", compress_level=1)
        t3 = time.perf_counter()
        t_save += t3 - t2
        n += batch

    t_total = time.perf_counter() - t_total0
    print(f"device={device} batch={batch} batches={batches} total_samples={n}")
    print(f"  gpu (gen+blur+u8):  {t_gpu*1000/batches:7.2f} ms/batch   ({n/t_gpu:7.1f} samples/s)")
    print(f"  gpu->cpu download:  {t_dl*1000/batches:7.2f} ms/batch   ({n/t_dl:7.1f} samples/s)")
    if save:
        print(f"  png save (serial):  {t_save*1000/batches:7.2f} ms/batch   ({n/t_save:7.1f} samples/s)")
    print(f"  end-to-end:         {t_total*1000/batches:7.2f} ms/batch   ({n/t_total:7.1f} samples/s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--out", type=str, default="synth_engine_v3/out/01_phase0_smoke")
    ap.add_argument("--save", action="store_true", help="also write PNGs (serial, measures save cost)")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench(args.batch, args.batches, out_dir, args.save, args.device)


if __name__ == "__main__":
    main()
