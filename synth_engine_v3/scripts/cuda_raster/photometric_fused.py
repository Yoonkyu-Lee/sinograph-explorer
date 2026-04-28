"""Python facade over the fused photometric CUDA kernel (Phase OPT-3).

Wraps the C++/CUDA build behind a single function `apply_fused_photometric`
that takes a (N, 3, H, W) canvas tensor and the per-sample parameters,
runs the fused kernel in-place, and returns the canvas.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_ext = None


def _ensure_ninja_on_path() -> None:
    scripts_dir = Path(sys.executable).parent
    p = os.environ.get("PATH", "")
    if str(scripts_dir) not in p.split(os.pathsep):
        os.environ["PATH"] = str(scripts_dir) + os.pathsep + p


def _load() -> "module":
    global _ext
    if _ext is not None:
        return _ext
    _ensure_ninja_on_path()
    is_win = sys.platform.startswith("win")
    cflags = ["/O2"] if is_win else ["-O3"]
    cuda_flags = ["-O3", "--use_fast_math"]
    if is_win:
        cuda_flags += [
            "--allow-unsupported-compiler",
            "-Xcompiler", "/D__NV_NO_HOST_COMPILER_CHECK=1",
        ]
    _ext = load(
        name="photometric_fused_ext",
        sources=[str(_HERE / "photometric_fused.cu")],
        verbose=True,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
    )
    return _ext


def apply_fused_photometric(
    canvas: torch.Tensor,         # (N, 3, H, W) float32 CUDA, mutated in-place
    *,
    brightness: torch.Tensor,     # (N,) float32
    contrast: torch.Tensor,       # (N,) float32
    noise_std_pixel: torch.Tensor,  # (N,) float32 (already normalized to [0,1] scale)
    invert_flag: torch.Tensor,    # (N,) uint8
    rng: torch.Generator,         # for noise generation
) -> torch.Tensor:
    """Run the fused photometric kernel. Modifies `canvas` in-place and
    returns it. Per-sample mean is computed via torch.mean (already optimal
    via cuDNN); the noise tensor is drawn here once and consumed by the
    kernel — single-launch fusion over the rest of the chain."""
    ext = _load()
    N = canvas.size(0)
    # per-sample mean over (C, H, W) — cheap reduction
    mean_per_sample = canvas.mean(dim=(1, 2, 3))      # (N,)
    # standard normal noise (one launch, fused into kernel via multiply)
    noise_seed = torch.randn(canvas.shape, generator=rng, device=canvas.device)
    ext.photometric_fused(
        canvas, noise_seed,
        brightness.to(torch.float32),
        contrast.to(torch.float32),
        mean_per_sample.to(torch.float32),
        noise_std_pixel.to(torch.float32),
        invert_flag.to(torch.uint8),
    )
    return canvas
