"""GPU-batched pipeline engine for synth_engine_v3.

Mirrors v2's `pipeline.py` Context + REGISTRY + run_block architecture, but:
  - canvas/mask are (N, 3, H, W) / (N, 1, H, W) torch tensors on CUDA
  - `prob` is applied per-sample via torch.where (not per-batch dropout)
  - `skip_if_kinds` / `only_if_kinds` gate per-sample against ctx.source_kinds

Parameter sampling rules mirror v2 (_sample):
  scalar           -> scalar (batch-common)
  [lo, hi] 2-nums  -> uniform sample (batch-common)
  [a, b, c, ...]   -> discrete choice (batch-common)

Per-sample variation for a single parameter is deferred: the GPU layer can
broadcast a batch-common scalar efficiently. When per-sample is needed a layer
can call `_sample_batch(v, n, rng)` to get a (N,) tensor.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

# Optional per-step profiler. Set V3_PROFILE_STEPS=1 to enable; totals live in
# `STEP_PROFILE` keyed by layer name (sum of wall ms over the run).
PROFILE_STEPS = os.environ.get("V3_PROFILE_STEPS", "0") == "1"
STEP_PROFILE: dict[str, float] = defaultdict(float)
STEP_CALLS: dict[str, int] = defaultdict(int)
STEP_SAMPLES: dict[str, int] = defaultdict(int)


def set_profile_steps(enabled: bool) -> None:
    """Toggle per-step profiling at runtime (CLI-friendly)."""
    global PROFILE_STEPS
    PROFILE_STEPS = bool(enabled)

CANVAS = 384
OUTPUT = 256
PAD = 80

REGISTRY: dict[str, Callable[..., "GPUContext"]] = {}


@dataclass
class GPUContext:
    canvas: torch.Tensor              # (N, 3, H, W) float [0, 1] RGB
    mask: torch.Tensor                # (N, 1, H, W) float [0, 1] — glyph white=1
    rng: torch.Generator              # on same device as canvas/mask
    chars: list[str] = field(default_factory=list)
    source_kinds: list[str] = field(default_factory=list)  # len == N
    # Per-sample total_strokes from canonical_v3.characters_structure. Used
    # by stroke_weight.dilate and friends to cap operation amount on high-
    # stroke glyphs (prevents strokes from merging into a blob). 0 / None
    # entries mean "unknown" → treat as low-stroke (no cap).
    total_strokes: list[int] = field(default_factory=list)
    device: str = "cuda"

    @property
    def n(self) -> int:
        return self.canvas.shape[0]


def register_layer(name: str):
    def deco(fn):
        if name in REGISTRY:
            raise ValueError(f"layer already registered: {name!r}")
        REGISTRY[name] = fn
        return fn
    return deco


def _sample(v, rng: torch.Generator):
    """Batch-common parameter resolution (matches v2 semantics)."""
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int):
                return int(torch.randint(lo, hi + 1, (1,), generator=rng, device=rng.device).item())
            return float(torch.empty(1, device=rng.device).uniform_(float(lo), float(hi), generator=rng).item())
        # discrete
        idx = int(torch.randint(0, len(v), (1,), generator=rng, device=rng.device).item())
        return v[idx]
    return v


def _sample_batch_uniform(lo: float, hi: float, n: int, rng: torch.Generator) -> torch.Tensor:
    """Draw (n,) uniform in [lo, hi]."""
    return torch.empty(n, device=rng.device).uniform_(float(lo), float(hi), generator=rng)


def _coerce_color_tensor(c, device: str, dtype=torch.float32) -> torch.Tensor:
    """RGB triple or (N,3) list -> tensor on device, values in [0, 1]."""
    t = torch.as_tensor(c, device=device, dtype=dtype)
    if t.max() > 1.5:  # assume 0..255
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def _gate_mask(ctx: GPUContext, skip_if_kinds, only_if_kinds) -> torch.Tensor:
    """Per-sample (N,) boolean: True = run the layer on this sample."""
    n = ctx.n
    apply = torch.ones(n, dtype=torch.bool, device=ctx.device)
    if skip_if_kinds:
        skip_set = set(skip_if_kinds)
        mask = torch.tensor(
            [sk in skip_set for sk in ctx.source_kinds],
            dtype=torch.bool, device=ctx.device,
        )
        apply &= ~mask
    if only_if_kinds is not None:
        only_set = set(only_if_kinds)
        mask = torch.tensor(
            [(sk in only_set) for sk in ctx.source_kinds],
            dtype=torch.bool, device=ctx.device,
        )
        apply &= mask
    return apply


def run_block(ctx: GPUContext, specs: list[dict], default_prefix: str = "") -> GPUContext:
    """Per-step dispatcher.

    Phase 17 optimization: when apply_mask is not all-True we slice the batch
    down to the active samples, run the layer on just those, and scatter the
    result back. This avoids:
      - running heavy layers (e.g. background.scene with a Python loop over N)
        on samples that will be discarded anyway, and
      - the (N,3,H,W)+(N,1,H,W) pre-call clone that the old whole-batch-with-
        restore approach required.
    """
    for raw in specs:
        step = dict(raw)
        name = step.pop("layer", None) or step.pop("op", None)
        if name is None:
            raise ValueError(f"step missing 'layer'/'op' key: {raw}")
        if default_prefix and "." not in name:
            name = f"{default_prefix}.{name}"
        if name not in REGISTRY:
            raise ValueError(f"unknown layer: {name!r}. known: {sorted(REGISTRY)}")

        skip_if_kinds = step.pop("skip_if_kinds", None)
        only_if_kinds = step.pop("only_if_kinds", None)
        prob = float(step.pop("prob", 1.0))

        apply_mask = _gate_mask(ctx, skip_if_kinds, only_if_kinds)  # (N,) bool
        if prob < 1.0:
            prob_mask = torch.rand(ctx.n, generator=ctx.rng, device=ctx.device) < prob
            apply_mask = apply_mask & prob_mask

        if not apply_mask.any():
            continue

        if bool(apply_mask.all().item()):
            if PROFILE_STEPS:
                if ctx.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            ctx = REGISTRY[name](ctx, **step)
            if PROFILE_STEPS:
                if ctx.device == "cuda":
                    torch.cuda.synchronize()
                STEP_PROFILE[name] += (time.perf_counter() - t0) * 1000
                STEP_CALLS[name] += 1
                STEP_SAMPLES[name] += ctx.n
            continue

        # Sub-batch: slice down to the rows where this step should apply,
        # run the layer on that slice, scatter the outputs back.
        idx = torch.nonzero(apply_mask, as_tuple=False).squeeze(1)
        sub_kinds = [ctx.source_kinds[i] for i in idx.tolist()]
        sub_chars = [ctx.chars[i] for i in idx.tolist()]
        sub_strokes = (
            [ctx.total_strokes[i] for i in idx.tolist()]
            if ctx.total_strokes else []
        )
        sub_ctx = GPUContext(
            canvas=ctx.canvas[idx].contiguous(),
            mask=ctx.mask[idx].contiguous(),
            rng=ctx.rng,
            chars=sub_chars,
            source_kinds=sub_kinds,
            total_strokes=sub_strokes,
            device=ctx.device,
        )
        if PROFILE_STEPS:
            if ctx.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        sub_ctx = REGISTRY[name](sub_ctx, **step)
        if PROFILE_STEPS:
            if ctx.device == "cuda":
                torch.cuda.synchronize()
            STEP_PROFILE[name] += (time.perf_counter() - t0) * 1000
            STEP_CALLS[name] += 1
            STEP_SAMPLES[name] += sub_ctx.n
        ctx.canvas[idx] = sub_ctx.canvas
        ctx.mask[idx] = sub_ctx.mask
    return ctx


def _apply_glyph_scale(ctx: GPUContext, scale: float) -> GPUContext:
    """Shrink the glyph mask by `scale` and center-paste into fresh canvas.

    Motivation (Codex follow-up, 2026-04-23): v2 renders the glyph at 288×288
    inside a 384 canvas (PAD=48). After center-crop 256 the glyph overflows by
    16 px/side — augments like perspective / lens_distort then push strokes
    outside the crop. Shrinking the mask here reserves a proper margin so
    downstream transforms stay inside the final crop.
    """
    if scale is None or not (0.0 < scale < 1.0):
        return ctx
    H, W = ctx.mask.shape[2:]
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    shrunk = F.interpolate(ctx.mask, size=(new_h, new_w),
                            mode="bilinear", align_corners=False)
    padded = torch.zeros_like(ctx.mask)
    top = (H - new_h) // 2
    left = (W - new_w) // 2
    padded[:, :, top:top + new_h, left:left + new_w] = shrunk
    ctx.mask = padded
    return ctx


def run_pipeline(ctx: GPUContext, spec: dict) -> GPUContext:
    glyph_scale = spec.get("glyph_scale")
    if glyph_scale is not None:
        ctx = _apply_glyph_scale(ctx, float(glyph_scale))
    if spec.get("style"):
        ctx = run_block(ctx, spec["style"])
    aug_spec = spec.get("augment")
    if aug_spec:
        # Two formats accepted:
        #   list[dict]              — legacy (all samples run the whole block)
        #   {clean_prob, ops: [..]} — Codex Tier 1 clean corridor. Per-sample
        #                             Bernoulli(clean_prob): winners bypass the
        #                             entire augment block (pristine anchor).
        if isinstance(aug_spec, dict):
            clean_prob = float(aug_spec.get("clean_prob", 0.0))
            ops = aug_spec.get("ops") or aug_spec.get("layers") or []
        else:
            clean_prob = 0.0
            ops = aug_spec

        if clean_prob <= 0.0 or not ops:
            ctx = run_block(ctx, ops, default_prefix="augment")
        else:
            n = ctx.n
            clean = torch.rand(n, generator=ctx.rng, device=ctx.device) < clean_prob
            if bool(clean.all().item()):
                return ctx  # everyone pristine, skip the block
            if not bool(clean.any().item()):
                ctx = run_block(ctx, ops, default_prefix="augment")
            else:
                dirty_idx = torch.nonzero(~clean, as_tuple=False).squeeze(1)
                sub_strokes = (
                    [ctx.total_strokes[i] for i in dirty_idx.tolist()]
                    if ctx.total_strokes else []
                )
                sub_ctx = GPUContext(
                    canvas=ctx.canvas[dirty_idx].contiguous(),
                    mask=ctx.mask[dirty_idx].contiguous(),
                    rng=ctx.rng,
                    chars=[ctx.chars[i] for i in dirty_idx.tolist()],
                    source_kinds=[ctx.source_kinds[i] for i in dirty_idx.tolist()],
                    total_strokes=sub_strokes,
                    device=ctx.device,
                )
                sub_ctx = run_block(sub_ctx, ops, default_prefix="augment")
                ctx.canvas[dirty_idx] = sub_ctx.canvas
                ctx.mask[dirty_idx] = sub_ctx.mask
    return ctx


def finalize_center_crop(canvas: torch.Tensor, out: int = OUTPUT) -> torch.Tensor:
    """(N, 3, H, W) -> (N, 3, out, out) center crop."""
    _, _, h, w = canvas.shape
    top = (h - out) // 2
    left = (w - out) // 2
    return canvas[:, :, top:top + out, left:left + out].contiguous()


def fresh_canvas(n: int, device: str = "cuda") -> torch.Tensor:
    """White (N, 3, CANVAS, CANVAS) canvas."""
    return torch.ones(n, 3, CANVAS, CANVAS, device=device)


def tensor_to_pil_batch(canvas: torch.Tensor):
    """(N,3,H,W) float [0,1] -> list of PIL RGB."""
    from PIL import Image
    arr = (canvas.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return [Image.fromarray(arr[i], mode="RGB") for i in range(arr.shape[0])]
