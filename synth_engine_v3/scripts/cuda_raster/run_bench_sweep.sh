#!/usr/bin/env bash
set -e
cd "/mnt/d/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3"
source ~/lab3-venv/bin/activate
for bs in 16 64 128 256 512 1024; do
    echo "=== batch=${bs} ==="
    python synth_engine_v3/scripts/cuda_raster/bench_kernel.py --batch ${bs} --iters 100 2>&1 | grep -E "throughput|per-glyph"
done
