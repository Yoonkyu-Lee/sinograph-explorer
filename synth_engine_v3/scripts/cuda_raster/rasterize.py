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


_ext = None


def _load_ext():
    global _ext
    if _ext is None:
        _ensure_ninja_on_path()
        _ext = load(
            name="cuda_raster_ext",
            sources=[str(_HERE / "raster_kernel.cu")],
            verbose=True,
            extra_cflags=["/O2", "-D__NV_NO_HOST_COMPILER_CHECK=1"],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math",
                "--allow-unsupported-compiler",
                "-Xcompiler", "/D__NV_NO_HOST_COMPILER_CHECK=1",
            ],
        )
    return _ext


def rasterize_batch(outlines, H: int, W: int,
                     device: str | torch.device = "cuda") -> torch.Tensor:
    """Rasterize a list of glyph outlines into a (B, 1, H, W) float mask.

    Each outline is expected to expose an `.edges` ndarray of shape (M, 4)
    in pixel coords (top-left origin, y-down). None / missing entries become
    all-zero glyphs.
    """
    ext = _load_ext()
    B = len(outlines)
    if B == 0:
        return torch.zeros((0, 1, H, W), dtype=torch.float32, device=device)

    # Build packed edges + per-glyph offsets
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

    return ext.raster_forward(edges_t, offsets_t, int(H), int(W))
