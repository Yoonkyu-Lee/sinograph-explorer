"""GPU-batched style layers — v2 style block ported to (N,3,H,W) tensors.

Implemented (Phase 2 + Phase 7 — matches v2 catalog):
  background.solid, background.gradient, background.noise,
    background.stripe, background.lines, background.scene
  fill.solid, fill.hsv_contrast, fill.gradient, fill.stripe,
    fill.contrast, fill.radial
  outline.simple, outline.double
  shadow.drop, shadow.soft, shadow.long
  stroke_weight.dilate, stroke_weight.erode
  glow.outer, glow.inner, glow.neon
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import kornia.filters as KF
import kornia.morphology as KM

from pathlib import Path

from pipeline_gpu import (
    CANVAS, GPUContext, REGISTRY, _coerce_color_tensor, _sample,
    _sample_batch_uniform, register_layer,
)


def _is_color_triple(x) -> bool:
    return isinstance(x, (list, tuple)) and len(x) == 3 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in x
    )


def _sample_color_tensor(c, n: int, rng: torch.Generator, device: str,
                          dtype=torch.float32) -> torch.Tensor:
    """Return (N, 3) color tensor in [0, 1].

    c may be:
      - single (R,G,B) → broadcast to (N, 3)
      - [(R,G,B), (R,G,B), ...] palette → per-sample random choice
    """
    if _is_color_triple(c):
        t = torch.as_tensor(c, device=device, dtype=dtype)
        if t.max() > 1.5:
            t = t / 255.0
        return t.clamp(0, 1).view(1, 3).expand(n, 3).contiguous()
    if isinstance(c, (list, tuple)) and all(_is_color_triple(x) for x in c):
        palette = torch.as_tensor(c, device=device, dtype=dtype)
        if palette.max() > 1.5:
            palette = palette / 255.0
        idx = torch.randint(0, palette.shape[0], (n,), generator=rng, device=device)
        return palette[idx].clamp(0, 1)
    raise ValueError(f"unsupported color spec: {c!r}")


def _composite(canvas: torch.Tensor, fill: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """alpha-composite: canvas*(1-mask) + fill*mask. Broadcasts OK."""
    return canvas * (1.0 - mask) + fill * mask


# ---------- background ----------

@register_layer("background.solid")
def background_solid(ctx: GPUContext, *, color=(255, 255, 255)) -> GPUContext:
    # color may be single (R,G,B) or palette [[R,G,B], ...] — latter picks per sample.
    c = _sample_color_tensor(color, ctx.n, ctx.rng, ctx.device)  # (N, 3) in [0,1]
    ctx.canvas = c.view(ctx.n, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS).clone()
    return ctx


@register_layer("background.gradient")
def background_gradient(ctx: GPUContext, *, start=(0, 0, 0), end=(255, 255, 255),
                         direction="vertical") -> GPUContext:
    c0 = _coerce_color_tensor(start, ctx.device).view(3, 1, 1)  # (3,1,1)
    c1 = _coerce_color_tensor(end, ctx.device).view(3, 1, 1)
    H = ctx.canvas.shape[2]; W = ctx.canvas.shape[3]
    if direction == "vertical":
        t = torch.linspace(0, 1, H, device=ctx.device).view(1, H, 1).expand(1, H, W)
    else:
        t = torch.linspace(0, 1, W, device=ctx.device).view(1, 1, W).expand(1, H, W)
    grad = c0 * (1 - t) + c1 * t  # (3, H, W)
    ctx.canvas = grad.unsqueeze(0).expand(ctx.n, 3, H, W).clone()
    return ctx


@register_layer("background.noise")
def background_noise(ctx: GPUContext, *, scale=0.6, smooth=0.0) -> GPUContext:
    s = float(_sample(scale, ctx.rng))
    sm = float(_sample(smooth, ctx.rng))
    H, W = ctx.canvas.shape[2:]
    # grayscale noise expanded to RGB
    n = torch.rand(ctx.n, 1, H, W, generator=ctx.rng, device=ctx.device) * s
    if sm > 0:
        k = max(3, int(sm * 4) | 1)  # odd kernel size
        n = KF.gaussian_blur2d(n, (k, k), (sm, sm))
    ctx.canvas = n.expand(-1, 3, -1, -1).contiguous()
    return ctx


@register_layer("background.one_of")
def background_one_of(ctx: GPUContext, *, choices) -> GPUContext:
    """Categorical background selector: pick exactly one child spec per sample.

    Replaces the overwrite-prone independent-Bernoulli pattern (Codex Tier 1).
    Each entry in `choices` is a regular layer spec dict plus an optional
    `weight` (default 1.0). Per-sample multinomial draw over weights → assign
    each sample to one choice → run sub-layer on its assigned slice.
    """
    if not choices:
        return ctx
    weights = torch.tensor([float(c.get("weight", 1.0)) for c in choices],
                            device=ctx.device, dtype=torch.float32)
    probs = weights / weights.sum()
    pick = torch.multinomial(probs, ctx.n, replacement=True, generator=ctx.rng)
    for ci, spec in enumerate(choices):
        sample_mask = pick == ci
        if not bool(sample_mask.any().item()):
            continue
        sub_idx = torch.nonzero(sample_mask, as_tuple=False).squeeze(1)
        sub_spec = dict(spec)
        sub_spec.pop("weight", None)
        name = sub_spec.pop("layer", None) or sub_spec.pop("op", None)
        if name is None or name not in REGISTRY:
            raise ValueError(f"background.one_of: bad child spec {spec!r}")
        if bool(sample_mask.all().item()):
            ctx = REGISTRY[name](ctx, **sub_spec)
            continue
        sub_strokes = (
            [ctx.total_strokes[i] for i in sub_idx.tolist()]
            if ctx.total_strokes else []
        )
        sub_ctx = GPUContext(
            canvas=ctx.canvas[sub_idx].contiguous(),
            mask=ctx.mask[sub_idx].contiguous(),
            rng=ctx.rng,
            chars=[ctx.chars[i] for i in sub_idx.tolist()],
            source_kinds=[ctx.source_kinds[i] for i in sub_idx.tolist()],
            total_strokes=sub_strokes,
            device=ctx.device,
        )
        sub_ctx = REGISTRY[name](sub_ctx, **sub_spec)
        ctx.canvas[sub_idx] = sub_ctx.canvas
        ctx.mask[sub_idx] = sub_ctx.mask
    return ctx


# ---------- fill ----------

@register_layer("fill.solid")
def fill_solid(ctx: GPUContext, *, color=(0, 0, 0)) -> GPUContext:
    # color may be single (R,G,B) or palette [[R,G,B], ...] — latter picks per sample.
    c = _sample_color_tensor(color, ctx.n, ctx.rng, ctx.device)  # (N, 3) in [0,1]
    fill = c.view(ctx.n, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ctx.canvas = _composite(ctx.canvas, fill, ctx.mask)
    return ctx


def _canvas_luma(canvas: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) -> (N,1,H,W) luma in [0,1]."""
    w = torch.tensor([0.299, 0.587, 0.114], device=canvas.device).view(1, 3, 1, 1)
    return (canvas * w).sum(dim=1, keepdim=True)


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(*,) hsv in [0,1] -> (*, 3) rgb in [0,1]. Fully vectorized (no Python loop)."""
    k = ((h * 6.0).floor().long() % 6)
    f = h * 6.0 - k.float()
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = torch.where(k == 0, v,
        torch.where(k == 1, q,
        torch.where(k == 2, p,
        torch.where(k == 3, p,
        torch.where(k == 4, t, v)))))
    g = torch.where(k == 0, t,
        torch.where(k == 1, v,
        torch.where(k == 2, v,
        torch.where(k == 3, q,
        torch.where(k == 4, p, p)))))
    b = torch.where(k == 0, p,
        torch.where(k == 1, p,
        torch.where(k == 2, t,
        torch.where(k == 3, v,
        torch.where(k == 4, v, q)))))
    return torch.stack([r, g, b], dim=-1)


@register_layer("fill.hsv_contrast")
def fill_hsv_contrast(ctx: GPUContext, *,
                       threshold=160, saturation=(0.5, 1.0), value=(0.0, 1.0),
                       min_contrast=60, max_attempts=8,
                       sample_region="mask_dilated", use_median=False) -> GPUContext:
    """Per-sample HSV random color with signed luma contrast vs background.

    Computes mask-dilated background luma per sample, picks HSV candidate,
    retries failures up to `max_attempts`, falls back to black/white.
    min_contrast is in 0..255 space (matches v2); we convert to 0..1 internally.
    """
    N, _, H, W = ctx.canvas.shape
    lum = _canvas_luma(ctx.canvas)  # (N,1,H,W)
    k = torch.ones(1, 1, 11, 11, device=ctx.device)
    dilated = (F.conv2d(ctx.mask, k, padding=5) > 0).float()
    # mean luma over dilated area (median would need a Python loop; mean is a
    # valid v2-supported mode (`use_median=False`) and vectorizes cleanly.
    area_sum = dilated.sum(dim=(1, 2, 3)).clamp(min=1.0)            # (N,)
    lum_sum = (lum * dilated).sum(dim=(1, 2, 3))                    # (N,)
    stat = lum_sum / area_sum                                        # (N,) in [0,1]

    thr = threshold / 255.0
    mc = min_contrast / 255.0
    is_bright_bg = stat > thr                                        # (N,)

    sat_lo, sat_hi = (saturation if isinstance(saturation, (list, tuple)) else (saturation, saturation))
    val_lo, val_hi = (value if isinstance(value, (list, tuple)) else (value, value))
    K = int(max_attempts)
    lw = torch.tensor([0.299, 0.587, 0.114], device=ctx.device)

    # Phase 20: draw all N*K candidates in one shot, select first-valid per row.
    h = torch.rand(N, K, generator=ctx.rng, device=ctx.device)
    s = torch.empty(N, K, device=ctx.device).uniform_(sat_lo, sat_hi, generator=ctx.rng)
    v = torch.empty(N, K, device=ctx.device).uniform_(val_lo, val_hi, generator=ctx.rng)
    cand = _hsv_to_rgb(h, s, v)                                      # (N, K, 3)
    cand_luma = (cand * lw).sum(dim=-1)                              # (N, K)
    stat_ = stat.unsqueeze(1)                                        # (N, 1)
    bright_ = is_bright_bg.unsqueeze(1)                              # (N, 1)
    gap = torch.where(bright_, stat_ - cand_luma, cand_luma - stat_)
    ok = gap >= mc                                                   # (N, K) bool

    any_ok = ok.any(dim=1)                                           # (N,)
    first_idx = ok.int().argmax(dim=1)                               # (N,)
    color = cand[torch.arange(N, device=ctx.device), first_idx]      # (N, 3)

    fb_black = torch.zeros(N, 3, device=ctx.device)
    fb_white = torch.ones(N, 3, device=ctx.device)
    fb = torch.where(is_bright_bg.unsqueeze(1), fb_black, fb_white)
    color = torch.where(any_ok.unsqueeze(1), color, fb)

    fill = color.view(N, 3, 1, 1).expand(N, 3, H, W)
    ctx.canvas = _composite(ctx.canvas, fill, ctx.mask)
    return ctx


# ---------- outline ----------

@register_layer("outline.simple")
def outline_simple(ctx: GPUContext, *, color=(0, 0, 0), width=2) -> GPUContext:
    w = max(1, int(_sample(width, ctx.rng)))
    k = torch.ones(2 * w + 1, 2 * w + 1, device=ctx.device)
    dil = KM.dilation(ctx.mask, k)
    outline = (dil - ctx.mask).clamp(0, 1)
    c = _coerce_color_tensor(color, ctx.device).view(1, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ctx.canvas = _composite(ctx.canvas, c, outline)
    return ctx


# ---------- shadow ----------

@register_layer("shadow.drop")
def shadow_drop(ctx: GPUContext, *, offset=(6, 6), blur=3.0, opacity=0.5,
                 color=(0, 0, 0)) -> GPUContext:
    dx = int(_sample(offset[0] if isinstance(offset, (list, tuple)) else offset, ctx.rng))
    dy = int(_sample(offset[1] if isinstance(offset, (list, tuple)) else offset, ctx.rng))
    b = float(_sample(blur, ctx.rng))
    op = float(_sample(opacity, ctx.rng))
    shadow = torch.roll(ctx.mask, shifts=(dy, dx), dims=(2, 3))
    if b > 0:
        k = max(3, int(b * 4) | 1)
        shadow = KF.gaussian_blur2d(shadow, (k, k), (b, b))
    shadow = shadow * op
    c = _coerce_color_tensor(color, ctx.device).view(1, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ctx.canvas = _composite(ctx.canvas, c, shadow)
    return ctx


# ---------- stroke weight ----------

def _dilate_cap_for_strokes(total_strokes: int, r_max: int) -> int:
    """Per-char cap on dilate radius. Above ~25 strokes any dilation starts
    to merge adjacent strokes; above ~30 it renders as a blob. Gate bands:
      unknown / 0      → no cap
      strokes < 15     → no cap (full r_max)
      15 ≤ strokes < 25→ cap 1 (allow a small thicken)
      strokes ≥ 25     → 0 (skip entirely)
    """
    if total_strokes <= 0:
        return r_max
    if total_strokes < 15:
        return r_max
    if total_strokes < 25:
        return min(r_max, 1)
    return 0


def _erode_cap_for_strokes(total_strokes: int, r_max: int) -> int:
    """Per-char cap on erode radius. Erosion removes strokes; on very-thin
    glyphs (1–3 strokes) even r=1 can erase them. Gate bands:
      unknown / 0    → no cap
      strokes ≤ 3    → 0 (skip)
      strokes ≤ 6    → cap 1
      else           → no cap
    """
    if total_strokes <= 0:
        return r_max
    if total_strokes <= 3:
        return 0
    if total_strokes <= 6:
        return min(r_max, 1)
    return r_max


@register_layer("stroke_weight.dilate")
def stroke_weight_dilate(ctx: GPUContext, *, radius=None, amount=1) -> GPUContext:
    """v2 uses `radius`; `amount` kept as v3 alias.

    Per-sample cap (2026-04-23): when ctx.total_strokes is populated we cap
    the dilate radius per sample via `_dilate_cap_for_strokes`. Samples
    group by their effective radius and dilate in sub-batches; unknown /
    low-stroke samples keep the full range.
    """
    spec = radius if radius is not None else amount
    r_max = int(_sample(spec, ctx.rng))
    if r_max <= 0:
        return ctx

    if ctx.total_strokes:
        effective = torch.tensor(
            [_dilate_cap_for_strokes(ts, r_max) for ts in ctx.total_strokes],
            device=ctx.device, dtype=torch.int32,
        )
    else:
        effective = torch.full((ctx.n,), r_max, device=ctx.device, dtype=torch.int32)

    for r in sorted({int(v) for v in effective.tolist() if v > 0}):
        idx = torch.nonzero(effective == r, as_tuple=False).squeeze(1)
        k = torch.ones(2 * r + 1, 2 * r + 1, device=ctx.device)
        ctx.mask[idx] = KM.dilation(ctx.mask[idx].contiguous(), k)
    return ctx


@register_layer("stroke_weight.erode")
def stroke_weight_erode(ctx: GPUContext, *, radius=None, amount=1) -> GPUContext:
    spec = radius if radius is not None else amount
    r_max = int(_sample(spec, ctx.rng))
    if r_max <= 0:
        return ctx

    if ctx.total_strokes:
        effective = torch.tensor(
            [_erode_cap_for_strokes(ts, r_max) for ts in ctx.total_strokes],
            device=ctx.device, dtype=torch.int32,
        )
    else:
        effective = torch.full((ctx.n,), r_max, device=ctx.device, dtype=torch.int32)

    for r in sorted({int(v) for v in effective.tolist() if v > 0}):
        idx = torch.nonzero(effective == r, as_tuple=False).squeeze(1)
        k = torch.ones(2 * r + 1, 2 * r + 1, device=ctx.device)
        ctx.mask[idx] = KM.erosion(ctx.mask[idx].contiguous(), k)
    return ctx


# ---------- glow ----------

@register_layer("glow.outer")
def glow_outer(ctx: GPUContext, *, color=(255, 255, 255), dilate=None, radius=6,
               strength=0.6, blur=4.0) -> GPUContext:
    """Outer glow. v2 uses `dilate` keyword; `radius` is kept as alias."""
    r = max(1, int(_sample(dilate if dilate is not None else radius, ctx.rng)))
    b = float(_sample(blur, ctx.rng))
    st = float(_sample(strength, ctx.rng))
    k = torch.ones(2 * r + 1, 2 * r + 1, device=ctx.device)
    dil = KM.dilation(ctx.mask, k)
    glow = (dil - ctx.mask).clamp(0, 1)
    if b > 0:
        ks = max(3, int(b * 4) | 1)
        glow = KF.gaussian_blur2d(glow, (ks, ks), (b, b))
    glow = glow * st
    c = _coerce_color_tensor(color, ctx.device).view(1, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ctx.canvas = (ctx.canvas + c * glow).clamp(0, 1)
    return ctx


# ---------- Phase 7: additional backgrounds ----------

def _build_stripe_pattern(h: int, w: int, thickness: int,
                           angle: float, ca: torch.Tensor, cb: torch.Tensor,
                           device: str) -> torch.Tensor:
    """(3, H, W) stripe pattern rotated by `angle` degrees.

    Phase 21: compute directly at (H, W) via per-pixel rotated coordinate
    instead of building a 2× canvas + kornia rotate (~3× faster, ~4× less mem).
    """
    import math
    ys = torch.arange(h, device=device, dtype=torch.float32).view(h, 1) - (h - 1) / 2
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, w) - (w - 1) / 2
    rad = math.radians(angle)
    coord = ys * math.cos(rad) + xs * math.sin(rad)              # (H, W)
    band = ((coord / max(1, thickness)).floor().long() % 2).to(torch.float32).unsqueeze(0)
    ca3 = ca.view(3, 1, 1); cb3 = cb.view(3, 1, 1)
    return ((1 - band) * ca3 + band * cb3).contiguous()


@register_layer("background.stripe")
def background_stripe(ctx: GPUContext, *, thickness=20, angle=0,
                       color_a=(255, 255, 255), color_b=(230, 235, 240)) -> GPUContext:
    t = max(1, int(_sample(thickness, ctx.rng)))
    a = float(_sample(angle, ctx.rng))
    ca = _coerce_color_tensor(color_a, ctx.device)
    cb = _coerce_color_tensor(color_b, ctx.device)
    H, W = ctx.canvas.shape[2:]
    pattern = _build_stripe_pattern(H, W, t, a, ca, cb, ctx.device)      # (3, H, W)
    ctx.canvas = pattern.unsqueeze(0).expand(ctx.n, 3, H, W).contiguous()
    return ctx


@register_layer("background.lines")
def background_lines(ctx: GPUContext, *, spacing=25, line_width=1, angle=0,
                      base_color=(255, 255, 255), line_color=(200, 210, 230)) -> GPUContext:
    import math
    sp = max(2, int(_sample(spacing, ctx.rng)))
    lw = max(1, int(_sample(line_width, ctx.rng)))
    a = float(_sample(angle, ctx.rng))
    bc = _coerce_color_tensor(base_color, ctx.device)
    lc = _coerce_color_tensor(line_color, ctx.device)
    H, W = ctx.canvas.shape[2:]
    # Phase 21: compute directly at (H, W) from rotated coord (no big canvas).
    ys = torch.arange(H, device=ctx.device, dtype=torch.float32).view(H, 1) - (H - 1) / 2
    xs = torch.arange(W, device=ctx.device, dtype=torch.float32).view(1, W) - (W - 1) / 2
    rad = math.radians(a)
    coord = ys * math.cos(rad) + xs * math.sin(rad)
    on_line = ((coord.floor().long() % sp) < lw).to(torch.float32).unsqueeze(0)
    bc3 = bc.view(3, 1, 1); lc3 = lc.view(3, 1, 1)
    single = (1 - on_line) * bc3 + on_line * lc3                 # (3, H, W)
    ctx.canvas = single.unsqueeze(0).expand(ctx.n, 3, H, W).contiguous()
    return ctx


# background.scene — image pool loaded ONCE per (folder, device) into a single
# (K, 3, H_pool, W_pool) GPU tensor; per-sample picks are then pure tensor
# indexing + F.grid_sample random crop. No Python loop, no PIL on the hot path.
# Added at Phase 2, vectorized at Phase 18.

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# (folder_abspath, device_str) → (K, 3, H, W) uint8 tensor
_SCENE_POOLS: dict[tuple[str, str], torch.Tensor] = {}
# (folder_abspath) → list of file Paths (for lazy discovery)
_SCENE_FILES: dict[str, list[Path]] = {}
# cap for the pool — load this many images, random-pick per sample at runtime
_SCENE_POOL_MAX = 128
# internal resolution each image is resized to before uploading to GPU
_SCENE_POOL_RES = 512


def _list_scene_files(folder: str) -> list[Path]:
    key = str(Path(folder).resolve())
    files = _SCENE_FILES.get(key)
    if files is not None:
        return files
    fp = Path(key)
    if not fp.is_dir():
        _SCENE_FILES[key] = []
        return []
    files = sorted(p for p in fp.rglob("*")
                   if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    _SCENE_FILES[key] = files
    return files


def _get_scene_pool(folder: str, device: str) -> torch.Tensor | None:
    """Lazy-load up to _SCENE_POOL_MAX images into a (K,3,H,W) uint8 tensor on `device`."""
    key = (str(Path(folder).resolve()), device)
    pool = _SCENE_POOLS.get(key)
    if pool is not None:
        return pool
    files = _list_scene_files(folder)
    if not files:
        _SCENE_POOLS[key] = torch.empty(0, 3, 1, 1, device=device, dtype=torch.uint8)
        return _SCENE_POOLS[key]
    from PIL import Image
    import numpy as np
    picked = files[:_SCENE_POOL_MAX]
    res = _SCENE_POOL_RES
    arrs = np.zeros((len(picked), res, res, 3), dtype=np.uint8)
    ok = 0
    for i, p in enumerate(picked):
        try:
            img = Image.open(p).convert("RGB").resize((res, res), Image.LANCZOS)
            arrs[ok] = np.asarray(img)
            ok += 1
        except Exception:
            continue
    if ok == 0:
        _SCENE_POOLS[key] = torch.empty(0, 3, 1, 1, device=device, dtype=torch.uint8)
        return _SCENE_POOLS[key]
    t = torch.from_numpy(arrs[:ok]).to(device).permute(0, 3, 1, 2).contiguous()
    _SCENE_POOLS[key] = t
    return t


@register_layer("background.scene")
def background_scene(ctx: GPUContext, *, folder, mode="random_crop",
                     scale_jitter=(1.0, 1.5), dim=0.0, desaturate=0.0,
                     blur=0.0) -> GPUContext:
    """Vectorized scene-pool background.

    First call lazy-loads up to _SCENE_POOL_MAX resized images into a (K,3,res,res)
    uint8 tensor on `device`. Per-batch: N independent random indices + random
    crop offsets; everything downstream is pure tensor ops on GPU. Matches v2's
    semantics (random_crop with scale_jitter, dim, desaturate, blur) without
    the per-sample PIL round-trip.
    """
    pool = _get_scene_pool(folder, ctx.device)
    if pool is None or pool.numel() == 0 or pool.shape[0] == 0:
        return ctx  # silent no-op (v2 parity when folder empty/missing)
    K, _, PH, PW = pool.shape
    N, _, H, W = ctx.canvas.shape

    # per-sample image index
    pick = torch.randint(0, K, (N,), generator=ctx.rng, device=ctx.device)
    src = pool[pick].float() / 255.0   # (N, 3, PH, PW)

    lo, hi = scale_jitter if isinstance(scale_jitter, (list, tuple)) else (scale_jitter, scale_jitter)
    if mode == "random_crop":
        # Each sample picks a zoom factor z in [lo, hi] and a random crop
        # offset. Implemented as affine_grid — pure GPU, N-vectorized.
        z = _sample_batch_uniform(float(lo), float(hi), N, ctx.rng).clamp(min=1.0)  # (N,)
        # desired window in source coords: window size = (H/z_y, W/z_x) on target,
        # mapped to src. Since we want a zoomed CROP, the effective theta is:
        #   theta = [[1/z, 0, tx], [0, 1/z, ty]]  (normalized -1..1 coords)
        # where tx, ty ∈ [-(1 - 1/z), (1 - 1/z)] ensures crop stays within source.
        inv_z = 1.0 / z                                                         # (N,)
        max_shift = (1.0 - inv_z).clamp(min=0.0)                                # (N,)
        tx = (torch.rand(N, generator=ctx.rng, device=ctx.device) * 2 - 1) * max_shift
        ty = (torch.rand(N, generator=ctx.rng, device=ctx.device) * 2 - 1) * max_shift
        theta = torch.zeros(N, 2, 3, device=ctx.device)
        theta[:, 0, 0] = inv_z
        theta[:, 1, 1] = inv_z
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, (N, 3, H, W), align_corners=False)
        bg_t = F.grid_sample(src, grid, mode="bilinear",
                              padding_mode="border", align_corners=False)
    else:  # mode == "resize"
        bg_t = F.interpolate(src, size=(H, W), mode="bilinear", align_corners=False)

    d_amt = float(_sample(desaturate, ctx.rng))
    if d_amt > 0:
        lw = torch.tensor([0.299, 0.587, 0.114], device=ctx.device).view(1, 3, 1, 1)
        gray = (bg_t * lw).sum(dim=1, keepdim=True).expand_as(bg_t)
        bg_t = bg_t * (1 - d_amt) + gray * d_amt
    dim_amt = float(_sample(dim, ctx.rng))
    if dim_amt > 0:
        bg_t = bg_t * (1 - dim_amt)
    b = float(_sample(blur, ctx.rng))
    if b > 0:
        k = max(3, int(b * 4) | 1)
        bg_t = KF.gaussian_blur2d(bg_t, (k, k), (b, b))

    ctx.canvas = bg_t.clamp(0, 1)
    return ctx


# ---------- Phase 7: additional fills ----------

@register_layer("fill.gradient")
def fill_gradient(ctx: GPUContext, *, start=(0, 0, 0), end=(255, 255, 255),
                   direction="vertical") -> GPUContext:
    c0 = _coerce_color_tensor(start, ctx.device).view(3, 1, 1)
    c1 = _coerce_color_tensor(end, ctx.device).view(3, 1, 1)
    H, W = ctx.canvas.shape[2:]
    if direction == "vertical":
        t = torch.linspace(0, 1, H, device=ctx.device).view(1, H, 1).expand(1, H, W)
    else:
        t = torch.linspace(0, 1, W, device=ctx.device).view(1, 1, W).expand(1, H, W)
    grad = c0 * (1 - t) + c1 * t
    fill = grad.unsqueeze(0).expand(ctx.n, 3, H, W)
    ctx.canvas = _composite(ctx.canvas, fill, ctx.mask)
    return ctx


@register_layer("fill.stripe")
def fill_stripe(ctx: GPUContext, *, thickness=10, angle=0,
                 color_a=(0, 0, 0), color_b=(255, 255, 255)) -> GPUContext:
    t = max(1, int(_sample(thickness, ctx.rng)))
    a = float(_sample(angle, ctx.rng))
    ca = _coerce_color_tensor(color_a, ctx.device)
    cb = _coerce_color_tensor(color_b, ctx.device)
    H, W = ctx.canvas.shape[2:]
    pattern = _build_stripe_pattern(H, W, t, a, ca, cb, ctx.device)
    fill = pattern.unsqueeze(0).expand(ctx.n, 3, H, W)
    ctx.canvas = _composite(ctx.canvas, fill, ctx.mask)
    return ctx


@register_layer("fill.radial")
def fill_radial(ctx: GPUContext, *, inner=(255, 255, 255), outer=(0, 0, 0),
                 radius=None) -> GPUContext:
    c0 = _coerce_color_tensor(inner, ctx.device).view(3, 1, 1)
    c1 = _coerce_color_tensor(outer, ctx.device).view(3, 1, 1)
    H, W = ctx.canvas.shape[2:]
    ys = torch.arange(H, device=ctx.device).view(H, 1).expand(H, W).float() - (H - 1) / 2
    xs = torch.arange(W, device=ctx.device).view(1, W).expand(H, W).float() - (W - 1) / 2
    d = torch.sqrt(xs * xs + ys * ys)
    max_r = float(_sample(radius, ctx.rng)) if radius is not None else float(
        (torch.hypot(torch.tensor(float(H), device=ctx.device),
                      torch.tensor(float(W), device=ctx.device)) / 2).item()
    )
    t = (d / max(max_r, 1e-6)).clamp(0, 1).unsqueeze(0)  # (1, H, W)
    grad = c0 * (1 - t) + c1 * t  # (3, H, W)
    fill = grad.unsqueeze(0).expand(ctx.n, 3, H, W)
    ctx.canvas = _composite(ctx.canvas, fill, ctx.mask)
    return ctx


@register_layer("fill.contrast")
def fill_contrast(ctx: GPUContext, *, threshold=128,
                   dark_color=(20, 20, 20), light_color=(240, 240, 240),
                   jitter=30, sample_region="mask_dilated",
                   use_median=False, min_contrast=90) -> GPUContext:
    """Palette-driven contrast fill (v2 parity).

    v2 supports median stats; v3 uses mean (area-weighted) for the same reason
    as fill.hsv_contrast — vectorizes per-sample cleanly. `use_median` is
    accepted but ignored here (documented deviation in V3_DESIGN).
    """
    N, _, H, W = ctx.canvas.shape
    lum = _canvas_luma(ctx.canvas)  # (N,1,H,W)
    if sample_region == "mask_dilated":
        k = torch.ones(1, 1, 11, 11, device=ctx.device)
        area = (F.conv2d(ctx.mask, k, padding=5) > 0).float()
    elif sample_region == "full":
        area = torch.ones_like(ctx.mask)
    else:
        area = (ctx.mask > 0).float()
    area_sum = area.sum(dim=(1, 2, 3)).clamp(min=1.0)
    lum_sum = (lum * area).sum(dim=(1, 2, 3))
    stat_255 = (lum_sum / area_sum) * 255.0

    is_bright_bg = stat_255 > float(threshold)  # (N,)

    dark = _sample_color_tensor(dark_color, N, ctx.rng, ctx.device)
    light = _sample_color_tensor(light_color, N, ctx.rng, ctx.device)
    base = torch.where(is_bright_bg.unsqueeze(1), dark, light)  # (N, 3)

    j = int(_sample(jitter, ctx.rng))
    if j > 0:
        shift = torch.randint(-j, j + 1, (N, 3), generator=ctx.rng, device=ctx.device).float() / 255.0
        color = (base + shift).clamp(0, 1)
    else:
        color = base

    fill_luma = (color * torch.tensor([0.299, 0.587, 0.114], device=ctx.device)).sum(dim=1) * 255.0
    gap = (fill_luma - stat_255).abs()
    need_fallback = gap < float(min_contrast)
    fb = torch.where(is_bright_bg.unsqueeze(1),
                     torch.zeros(N, 3, device=ctx.device),
                     torch.ones(N, 3, device=ctx.device))
    color = torch.where(need_fallback.unsqueeze(1), fb, color)

    fill_c = color.view(N, 3, 1, 1).expand(N, 3, H, W)
    ctx.canvas = _composite(ctx.canvas, fill_c, ctx.mask)
    return ctx


# ---------- Phase 7: additional outline / shadow / glow ----------

def _dilate_k(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device)
    return KM.dilation(mask, k)


def _ring(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return (_dilate_k(mask, radius) - mask).clamp(0, 1)


# Rename existing outline.simple's `width` to also accept `radius` (v2 name).
# The current implementation already lives above; we override here with v2
# parity + `radius` alias support.
# (To avoid double registration conflict we `pop` then re-register.)
from pipeline_gpu import REGISTRY as _REGISTRY  # noqa: E402
if "outline.simple" in _REGISTRY:
    _REGISTRY.pop("outline.simple")


@register_layer("outline.simple")
def outline_simple(ctx: GPUContext, *, radius=None, width=2, color=(0, 0, 0)) -> GPUContext:
    r = int(_sample(radius if radius is not None else width, ctx.rng))
    if r <= 0:
        return ctx
    ring = _ring(ctx.mask, r)
    c = _sample_color_tensor(color, ctx.n, ctx.rng, ctx.device)
    c = c.view(ctx.n, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ctx.canvas = _composite(ctx.canvas, c, ring)
    return ctx


@register_layer("outline.double")
def outline_double(ctx: GPUContext, *, outer_offset=4, outer_width=2,
                    inner_width=1, outer_color=None, inner_color=None,
                    color=(0, 0, 0)) -> GPUContext:
    # Legacy `color` applies to both if outer/inner colors not specified.
    # Allows signboard-style double outline: outer light (white) + inner dark (black).
    off = int(_sample(outer_offset, ctx.rng))
    ow = int(_sample(outer_width, ctx.rng))
    iw = int(_sample(inner_width, ctx.rng))
    oc = outer_color if outer_color is not None else color
    ic = inner_color if inner_color is not None else color
    inner_ring = _ring(ctx.mask, iw)
    outer_ring = _ring(_dilate_k(ctx.mask, off), ow)
    oc_t = _sample_color_tensor(oc, ctx.n, ctx.rng, ctx.device).view(ctx.n, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ic_t = _sample_color_tensor(ic, ctx.n, ctx.rng, ctx.device).view(ctx.n, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ctx.canvas = _composite(ctx.canvas, oc_t, outer_ring)
    ctx.canvas = _composite(ctx.canvas, ic_t, inner_ring)
    return ctx


@register_layer("shadow.soft")
def shadow_soft(ctx: GPUContext, *, blur=12, color=(160, 160, 160)) -> GPUContext:
    b = float(_sample(blur, ctx.rng))
    halo = ctx.mask
    if b > 0:
        k = max(3, int(b * 4) | 1)
        halo = KF.gaussian_blur2d(halo, (k, k), (b, b))
    c = _coerce_color_tensor(color, ctx.device).view(1, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ctx.canvas = _composite(ctx.canvas, c, halo)
    return ctx


@register_layer("shadow.long")
def shadow_long(ctx: GPUContext, *, step=(1, 1), length=20,
                 color=(180, 180, 180)) -> GPUContext:
    sx = int(_sample(step[0] if isinstance(step, (list, tuple)) else step, ctx.rng))
    sy = int(_sample(step[1] if isinstance(step, (list, tuple)) else step, ctx.rng))
    n_steps = int(_sample(length, ctx.rng))
    acc = torch.zeros_like(ctx.mask)
    for i in range(1, n_steps + 1):
        shifted = torch.roll(ctx.mask, shifts=(sy * i, sx * i), dims=(2, 3))
        acc = torch.maximum(acc, shifted)
    c = _coerce_color_tensor(color, ctx.device).view(1, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ctx.canvas = _composite(ctx.canvas, c, acc)
    return ctx


@register_layer("glow.inner")
def glow_inner(ctx: GPUContext, *, color=(255, 255, 255), blur=3) -> GPUContext:
    b = float(_sample(blur, ctx.rng))
    soft = ctx.mask
    if b > 0:
        k = max(3, int(b * 4) | 1)
        soft = KF.gaussian_blur2d(soft, (k, k), (b, b))
    c = _coerce_color_tensor(color, ctx.device).view(1, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    ctx.canvas = _composite(ctx.canvas, c, soft)
    return ctx


@register_layer("glow.neon")
def glow_neon(ctx: GPUContext, *, color=(0, 220, 255),
               outer_dilate=3, outer_blur=10, inner_blur=3,
               core_color=(255, 255, 255)) -> GPUContext:
    outer_r = int(_sample(outer_dilate, ctx.rng))
    outer_b = float(_sample(outer_blur, ctx.rng))
    inner_b = float(_sample(inner_blur, ctx.rng))
    glow_c = _sample_color_tensor(color, ctx.n, ctx.rng, ctx.device).view(ctx.n, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)
    core_c = _sample_color_tensor(core_color, ctx.n, ctx.rng, ctx.device).view(ctx.n, 3, 1, 1).expand(ctx.n, 3, CANVAS, CANVAS)

    halo = _dilate_k(ctx.mask, outer_r)
    if outer_b > 0:
        k = max(3, int(outer_b * 4) | 1)
        halo = KF.gaussian_blur2d(halo, (k, k), (outer_b, outer_b))
    ctx.canvas = _composite(ctx.canvas, glow_c, halo)

    inner = ctx.mask
    if inner_b > 0:
        k = max(3, int(inner_b * 4) | 1)
        inner = KF.gaussian_blur2d(inner, (k, k), (inner_b, inner_b))
    ctx.canvas = _composite(ctx.canvas, glow_c, inner)

    ctx.canvas = _composite(ctx.canvas, core_c, ctx.mask)
    return ctx
