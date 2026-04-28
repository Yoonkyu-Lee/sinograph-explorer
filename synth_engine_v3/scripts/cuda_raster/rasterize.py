"""Python facade over the CUDA polygon rasterizer (Phase OPT-2.2, doc/20).

Compiles `raster_kernel.cu` on first import via torch.utils.cpp_extension.
Subsequent imports use the cached `.pyd` (Windows) / `.so` (Linux). First
build takes ~30-60 s; cached builds are instant.

Public API:
    rasterize_batch(outlines, H, W, device='cuda') → torch.Tensor (B, 1, H, W)

`outlines` is a list of `OutlineData` (or any object with an `edges` (M, 4)
np.ndarray attribute). Empty / glyphless entries pad as zero edges.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent


def _ensure_ninja_on_path() -> None:
    """torch.utils.cpp_extension shell-execs `ninja` for the build but doesn't
    consult `sys.executable`-relative `Scripts/`. On Windows the venv `Scripts`
    dir holds `ninja.exe` but isn't on the parent `PATH`. Patch it in."""
    scripts_dir = Path(sys.executable).parent
    p = os.environ.get("PATH", "")
    if str(scripts_dir) not in p.split(os.pathsep):
        os.environ["PATH"] = str(scripts_dir) + os.pathsep + p


# Lazily-built per-version extensions: "v1" (naive Lab-1) and "v2"
# (shared-mem Lab-4 + sorted-edges Lab-8 + scan Lab-7).
_ext_cache: dict = {}


def _load_ext(kernel: str = "v1"):
    if kernel in _ext_cache:
        return _ext_cache[kernel]
    _ensure_ninja_on_path()
    is_win = sys.platform.startswith("win")
    cflags = ["/O2"] if is_win else ["-O3"]
    cuda_flags = ["-O3", "--use_fast_math"]
    if is_win:
        cuda_flags += [
            "--allow-unsupported-compiler",
            "-Xcompiler", "/D__NV_NO_HOST_COMPILER_CHECK=1",
        ]
    if kernel == "v1":
        sources = [str(_HERE / "raster_kernel.cu")]
        name = "cuda_raster_ext_v1"
    elif kernel == "v2":
        sources = [str(_HERE / "raster_kernel_v2.cu")]
        name = "cuda_raster_ext_v2"
    else:
        raise ValueError(f"unknown kernel: {kernel!r}")
    ext = load(
        name=name,
        sources=sources,
        verbose=True,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
    )
    _ext_cache[kernel] = ext
    return ext


def rasterize_batch(outlines, H: int, W: int,
                     device: str | torch.device = "cuda",
                     kernel: str = "v2") -> torch.Tensor:
    """Rasterize a list of glyph outlines into a (B, 1, H, W) float mask.

    `kernel` selects the CUDA implementation:
        "v1" — Lab-1 naive (per-pixel-per-edge global memory loop)
        "v2" — Lab 4 shared memory + Lab 8 sorted-edge coalesced load +
               Lab 7 active-edge tile pruning. Default.

    Each outline is expected to expose an `.edges` ndarray of shape (M, 4)
    sorted by y_min if kernel == "v2" (outline_cache emits y-sorted by
    default). None / missing entries become all-zero glyphs.
    """
    ext = _load_ext(kernel=kernel)
    B = len(outlines)
    if B == 0:
        return torch.zeros((0, 1, H, W), dtype=torch.float32, device=device)

    edges_list = []
    offsets = [0]
    for od in outlines:
        if od is None or od.edges is None or len(od.edges) == 0:
            offsets.append(offsets[-1])
            continue
        edges_list.append(od.edges)
        offsets.append(offsets[-1] + len(od.edges))
    if edges_list:
        all_edges = np.concatenate(edges_list, axis=0).astype(np.float32, copy=False)
    else:
        all_edges = np.zeros((0, 4), dtype=np.float32)

    edges_t = torch.from_numpy(all_edges).to(device, non_blocking=True).contiguous()
    offsets_t = torch.tensor(offsets, dtype=torch.int32, device=device)

    fn = ext.raster_forward_v2 if kernel == "v2" else ext.raster_forward
    return fn(edges_t, offsets_t, int(H), int(W))
