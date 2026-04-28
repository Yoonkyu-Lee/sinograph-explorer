"""GPU-batched augment ops — full v2 catalog (Phase 3 MVP + Phase 8 infill).

Implemented (matches v2 `augment.py`):
  geometric:   rotate, perspective, scale_translate, shear
  photometric: brightness, contrast, gamma, saturation, color_jitter, invert
  degradation: gaussian_blur, motion_blur, gaussian_noise, salt_pepper_noise,
               downscale_upscale, jpeg
  scan_sim:    paper_texture, ink_bleed, binarize, shadow_gradient, vignette
  camera_sim:  defocus, low_light, chromatic_aberration, lens_distort
  nonrigid:    elastic

Parameters that were batch-common `[lo, hi]` ranges in v2 are sampled PER
SAMPLE on GPU here — so each sample in the batch gets an independent rotation
angle etc. That is how v2 behaves semantically (each call to the op resolves
_sample once per image); we preserve the same by drawing (N,) tensors.
"""
from __future__ import annotations

import io
import math

import torch
import torch.nn.functional as F
import kornia.filters as KF
import kornia.geometry.transform as KT

from pipeline_gpu import (
    GPUContext, _sample, _sample_batch_uniform, register_layer,
)

# Optional: fused photometric (brightness + contrast + invert + gaussian_noise)
# CUDA kernel (Phase OPT-3). Falls back to per-op chain when extension is
# unavailable (Windows native build w/ MSVC OS check, etc.).
try:
    import sys as _sys
    from pathlib import Path as _Path
    _CR_DIR = _Path(__file__).resolve().parent / "cuda_raster"
    if str(_CR_DIR) not in _sys.path:
        _sys.path.insert(0, str(_CR_DIR))
    from photometric_fused import apply_fused_photometric    # noqa: E402
except Exception:
    apply_fused_photometric = None


def _range(v):
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return float(v[0]), float(v[1])
    f = float(v)
    return f, f


# ---------- geometric ----------

def _warp_canvas_and_mask(ctx: GPUContext, warp_fn, fill) -> GPUContext:
    """Apply `warp_fn` to canvas (3ch) AND mask (1ch) in a single pass so
    the mask stays spatially aligned with the canvas after the transform.

    Bug context (2026-04-23): previously each geometric augment warped only
    the canvas; mask stayed at the pre-transform position. Any downstream
    op that used ctx.mask (e.g. visibility_guard) mis-attributed pixels →
    a "ghost" of the glyph at its original position got painted over the
    actually-rotated canvas. Fix is to warp both as one 4-ch tensor.
    """
    combined = torch.cat([ctx.canvas, ctx.mask], dim=1)         # (N, 4, H, W)
    warped = warp_fn(combined)
    new_canvas = warped[:, :3].contiguous()
    new_mask = warped[:, 3:4].clamp(0.0, 1.0).contiguous()

    fc = torch.as_tensor(fill, device=ctx.device, dtype=torch.float32)
    if fc.max() > 1.5:
        fc = fc / 255.0
    H, W = ctx.canvas.shape[2:]
    bg_mask = (new_canvas.sum(dim=1, keepdim=True) < 1e-6).float()
    fill_canvas = fc.view(1, 3, 1, 1).expand(ctx.n, 3, H, W)
    ctx.canvas = new_canvas + bg_mask * fill_canvas
    ctx.mask = new_mask
    return ctx


@register_layer("augment.rotate")
def aug_rotate(ctx: GPUContext, *, angle=0.0, fill=(1.0, 1.0, 1.0)) -> GPUContext:
    lo, hi = _range(angle)
    angles = _sample_batch_uniform(lo, hi, ctx.n, ctx.rng)  # (N,) degrees
    return _warp_canvas_and_mask(
        ctx,
        lambda x: KT.rotate(x, angles, mode="bilinear", padding_mode="zeros"),
        fill,
    )


@register_layer("augment.perspective")
def aug_perspective(ctx: GPUContext, *, strength=0.1, fill=(1.0, 1.0, 1.0)) -> GPUContext:
    lo, hi = _range(strength)
    s = _sample_batch_uniform(lo, hi, ctx.n, ctx.rng)  # (N,)
    H, W = ctx.canvas.shape[2:]
    off = s * min(H, W)  # (N,)
    base = torch.tensor([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                         dtype=torch.float32, device=ctx.device)
    src = base.unsqueeze(0).expand(ctx.n, 4, 2).clone()
    jitter = torch.empty(ctx.n, 4, 2, device=ctx.device).uniform_(0.0, 1.0, generator=ctx.rng)
    signs = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32, device=ctx.device)
    dst = src + signs * jitter * off.view(ctx.n, 1, 1)
    M = KT.get_perspective_transform(src, dst)
    return _warp_canvas_and_mask(
        ctx,
        lambda x: KT.warp_perspective(x, M, dsize=(H, W), mode="bilinear", padding_mode="zeros"),
        fill,
    )


@register_layer("augment.scale_translate")
def aug_scale_translate(ctx: GPUContext, *, scale=1.0, translate=0.0,
                         fill=(1.0, 1.0, 1.0)) -> GPUContext:
    s_lo, s_hi = _range(scale)
    t_lo, t_hi = _range(translate)
    s = _sample_batch_uniform(s_lo, s_hi, ctx.n, ctx.rng)
    tx = _sample_batch_uniform(t_lo, t_hi, ctx.n, ctx.rng)
    ty = _sample_batch_uniform(t_lo, t_hi, ctx.n, ctx.rng)
    H, W = ctx.canvas.shape[2:]
    center = torch.tensor([W / 2, H / 2], device=ctx.device).unsqueeze(0).expand(ctx.n, 2)
    angle = torch.zeros(ctx.n, device=ctx.device)
    scale_t = torch.stack([s, s], dim=-1)
    M = KT.get_rotation_matrix2d(center, angle, scale_t)
    M[:, 0, 2] += tx * W
    M[:, 1, 2] += ty * H
    return _warp_canvas_and_mask(
        ctx,
        lambda x: KT.warp_affine(x, M, dsize=(H, W), mode="bilinear", padding_mode="zeros"),
        fill,
    )


@register_layer("augment.shear")
def aug_shear(ctx: GPUContext, *, sx=0.0, sy=0.0, fill=(1.0, 1.0, 1.0)) -> GPUContext:
    sx_lo, sx_hi = _range(sx)
    sy_lo, sy_hi = _range(sy)
    ax = _sample_batch_uniform(sx_lo, sx_hi, ctx.n, ctx.rng)
    ay = _sample_batch_uniform(sy_lo, sy_hi, ctx.n, ctx.rng)
    H, W = ctx.canvas.shape[2:]
    M = torch.zeros(ctx.n, 2, 3, device=ctx.device)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    M[:, 0, 1] = ax
    M[:, 1, 0] = ay
    M[:, 0, 2] = -ax * W / 2
    M[:, 1, 2] = -ay * H / 2
    return _warp_canvas_and_mask(
        ctx,
        lambda x: KT.warp_affine(x, M, dsize=(H, W), mode="bilinear", padding_mode="zeros"),
        fill,
    )


# ---------- photometric ----------

def _to_batch_factor(v, n: int, rng: torch.Generator) -> torch.Tensor:
    lo, hi = _range(v)
    return _sample_batch_uniform(lo, hi, n, rng).view(n, 1, 1, 1)


@register_layer("augment.brightness")
def aug_brightness(ctx: GPUContext, *, factor=1.0) -> GPUContext:
    f = _to_batch_factor(factor, ctx.n, ctx.rng)
    ctx.canvas = (ctx.canvas * f).clamp(0, 1)
    return ctx


@register_layer("augment.contrast")
def aug_contrast(ctx: GPUContext, *, factor=1.0) -> GPUContext:
    f = _to_batch_factor(factor, ctx.n, ctx.rng)
    mean = ctx.canvas.mean(dim=(2, 3), keepdim=True)
    ctx.canvas = ((ctx.canvas - mean) * f + mean).clamp(0, 1)
    return ctx


@register_layer("augment.gamma")
def aug_gamma(ctx: GPUContext, *, gamma=1.0) -> GPUContext:
    g = _to_batch_factor(gamma, ctx.n, ctx.rng)
    ctx.canvas = ctx.canvas.clamp(1e-6, 1).pow(g)
    return ctx


@register_layer("augment.saturation")
def aug_saturation(ctx: GPUContext, *, factor=1.0) -> GPUContext:
    f = _to_batch_factor(factor, ctx.n, ctx.rng)
    # luma-weighted desat/sat
    w = torch.tensor([0.299, 0.587, 0.114], device=ctx.device).view(1, 3, 1, 1)
    luma = (ctx.canvas * w).sum(dim=1, keepdim=True).expand_as(ctx.canvas)
    ctx.canvas = (luma + (ctx.canvas - luma) * f).clamp(0, 1)
    return ctx


@register_layer("augment.color_jitter")
def aug_color_jitter(ctx: GPUContext, *, brightness=1.0, contrast=1.0, saturation=1.0) -> GPUContext:
    ctx = aug_brightness(ctx, factor=brightness)
    ctx = aug_contrast(ctx, factor=contrast)
    ctx = aug_saturation(ctx, factor=saturation)
    return ctx


@register_layer("augment.invert")
def aug_invert(ctx: GPUContext, *, prob=1.0) -> GPUContext:
    # note: the surrounding run_block already handles per-sample `prob` gating;
    # this function just inverts everything it sees.
    ctx.canvas = 1.0 - ctx.canvas
    return ctx


@register_layer("augment.photometric_fused")
def aug_photometric_fused(
    ctx: GPUContext, *,
    brightness=1.0,
    contrast=1.0,
    invert_prob: float = 0.0,
    noise_std=0.0,    # 0..255 scale to match aug_gaussian_noise
    brightness_prob: float = 1.0,
    contrast_prob: float = 1.0,
    noise_prob: float = 1.0,
) -> GPUContext:
    """Single CUDA-kernel substitute for brightness+contrast+invert+noise
    chain (OPT-3, Lab 1 + register tiling). All four params are sampled per
    sample, then fed to a fused kernel that does one read+write pass.

    On first call this triggers a JIT build (~30 s). Caches afterwards.
    Falls back to per-op chain when the extension fails to build (e.g. Win
    native MSVC OS check).
    """
    n = ctx.n
    rng = ctx.rng
    device = ctx.device

    # Per-sample params
    br_lo, br_hi = _range(brightness)
    ct_lo, ct_hi = _range(contrast)
    ns_lo, ns_hi = _range(noise_std)
    br = _sample_batch_uniform(br_lo, br_hi, n, rng)
    ct = _sample_batch_uniform(ct_lo, ct_hi, n, rng)
    ns_255 = _sample_batch_uniform(ns_lo, ns_hi, n, rng)
    ns_pixel = ns_255 / 255.0
    inv_flag = (torch.rand(n, generator=rng, device=device) < float(invert_prob)).to(torch.uint8)

    # Per-sample prob gating — for samples where prob fails, force the
    # parameter to identity (br=1, ct=1, ns=0). Preserves per-op semantics
    # of the legacy unfused pipeline.
    if float(brightness_prob) < 1.0:
        keep = torch.rand(n, generator=rng, device=device) < float(brightness_prob)
        br = torch.where(keep, br, torch.ones_like(br))
    if float(contrast_prob) < 1.0:
        keep = torch.rand(n, generator=rng, device=device) < float(contrast_prob)
        ct = torch.where(keep, ct, torch.ones_like(ct))
    if float(noise_prob) < 1.0:
        keep = torch.rand(n, generator=rng, device=device) < float(noise_prob)
        ns_pixel = torch.where(keep, ns_pixel, torch.zeros_like(ns_pixel))

    if apply_fused_photometric is None:
        # Fallback: legacy per-op chain (also exercises ctx mutations correctly)
        f_br = br.view(n, 1, 1, 1)
        f_ct = ct.view(n, 1, 1, 1)
        mean = ctx.canvas.mean(dim=(1, 2, 3), keepdim=True)
        canvas = (ctx.canvas - mean) * f_ct + mean
        canvas = canvas * f_br
        f_ns = ns_pixel.view(n, 1, 1, 1)
        if (ns_pixel > 0).any():
            noise = torch.randn(canvas.shape, generator=rng, device=device)
            canvas = canvas + noise * f_ns
        inv = inv_flag.view(n, 1, 1, 1).float()
        canvas = inv * (1 - canvas) + (1 - inv) * canvas
        ctx.canvas = canvas.clamp(0, 1)
        return ctx

    # Fused CUDA kernel path
    apply_fused_photometric(
        ctx.canvas,
        brightness=br, contrast=ct,
        noise_std_pixel=ns_pixel,
        invert_flag=inv_flag,
        rng=rng,
    )
    return ctx


# ---------- degradation ----------

@register_layer("augment.gaussian_blur")
def aug_gaussian_blur(ctx: GPUContext, *, sigma=0.0) -> GPUContext:
    s = float(_sample(sigma, ctx.rng))  # batch-common to keep kornia kernel single
    if s <= 0:
        return ctx
    k = max(3, int(s * 4) | 1)
    ctx.canvas = KF.gaussian_blur2d(ctx.canvas, (k, k), (s, s))
    return ctx


@register_layer("augment.gaussian_noise")
def aug_gaussian_noise(ctx: GPUContext, *, std=0.0) -> GPUContext:
    # std in v2 was on 0..255 scale — convert
    s = float(_sample(std, ctx.rng)) / 255.0
    if s <= 0:
        return ctx
    noise = torch.randn(ctx.canvas.shape, generator=ctx.rng, device=ctx.device) * s
    ctx.canvas = (ctx.canvas + noise).clamp(0, 1)
    return ctx


@register_layer("augment.downscale_upscale")
def aug_downscale_upscale(ctx: GPUContext, *, factor=1.0) -> GPUContext:
    f = float(_sample(factor, ctx.rng))
    if f >= 1.0:
        return ctx
    H, W = ctx.canvas.shape[2:]
    nh, nw = max(1, int(H * f)), max(1, int(W * f))
    low = F.interpolate(ctx.canvas, size=(nh, nw), mode="bilinear", align_corners=False)
    ctx.canvas = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
    return ctx


_JPEG_THREAD_POOL = None


def _get_jpeg_thread_pool():
    global _JPEG_THREAD_POOL
    if _JPEG_THREAD_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        import os
        _JPEG_THREAD_POOL = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))
    return _JPEG_THREAD_POOL


def _aug_jpeg_pil(ctx: "GPUContext", quality) -> "GPUContext":
    """Legacy PIL + ThreadPool path. GIL-free libjpeg, parallel across CPU threads."""
    from PIL import Image
    import numpy as np

    q_lo, q_hi = _range(quality)
    q_lo_i = max(1, min(100, int(q_lo)))
    q_hi_i = max(1, min(100, int(q_hi)))
    u8 = (ctx.canvas.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
    if q_lo_i == q_hi_i:
        q_vals = [q_lo_i] * ctx.n
    else:
        q_vals = torch.randint(q_lo_i, q_hi_i + 1, (ctx.n,),
                                generator=ctx.rng, device=ctx.device).cpu().tolist()

    def _encode_decode(i: int) -> "np.ndarray":
        buf = io.BytesIO()
        Image.fromarray(u8[i]).save(buf, format="JPEG", quality=int(q_vals[i]))
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"))

    pool = _get_jpeg_thread_pool()
    out_arrs = list(pool.map(_encode_decode, range(ctx.n)))
    stacked = np.stack(out_arrs, axis=0)
    ctx.canvas = torch.from_numpy(stacked).to(ctx.device).permute(0, 3, 1, 2).contiguous().float() / 255.0
    return ctx


def _aug_jpeg_cuda(ctx: "GPUContext", quality) -> "GPUContext":
    """CUDA nvJPEG path via torchvision.io. Per-sample encode+decode loop."""
    from torchvision.io import encode_jpeg, decode_jpeg, ImageReadMode

    q_lo, q_hi = _range(quality)
    q_lo_i = max(1, min(100, int(q_lo)))
    q_hi_i = max(1, min(100, int(q_hi)))
    if q_lo_i == q_hi_i:
        q_vals = [q_lo_i] * ctx.n
    else:
        q_vals = torch.randint(q_lo_i, q_hi_i + 1, (ctx.n,),
                                generator=ctx.rng, device=ctx.device).cpu().tolist()

    u8 = (ctx.canvas.clamp(0, 1) * 255).to(torch.uint8)  # (N, 3, H, W) on CUDA
    decoded = []
    for i in range(ctx.n):
        enc = encode_jpeg(u8[i], quality=int(q_vals[i]))  # bytes on CUDA
        dec = decode_jpeg(enc.cpu(), mode=ImageReadMode.RGB, device=ctx.device)
        decoded.append(dec)
    out = torch.stack(decoded, dim=0)  # (N, 3, H, W) uint8
    ctx.canvas = out.float() / 255.0
    return ctx


# Toggle for A/B testing. Set via JPEG_BACKEND env var: "pil" (default) or "cuda".
import os as _os
_JPEG_BACKEND = _os.environ.get("JPEG_BACKEND", "pil").lower()


@register_layer("augment.jpeg")
def aug_jpeg(ctx: GPUContext, *, quality=90) -> GPUContext:
    """JPEG compression simulation. Backend selectable via JPEG_BACKEND env
    (pil | cuda). doc/14 Phase A has steady-state benchmark notes.
    """
    if _JPEG_BACKEND == "cuda":
        return _aug_jpeg_cuda(ctx, quality)
    return _aug_jpeg_pil(ctx, quality)


# ---------- scan_sim ----------

@register_layer("augment.vignette")
def aug_vignette(ctx: GPUContext, *, strength=0.3) -> GPUContext:
    s_lo, s_hi = _range(strength)
    s = _sample_batch_uniform(s_lo, s_hi, ctx.n, ctx.rng).view(ctx.n, 1, 1, 1)
    H, W = ctx.canvas.shape[2:]
    ys = torch.linspace(-1, 1, H, device=ctx.device).view(1, 1, H, 1)
    xs = torch.linspace(-1, 1, W, device=ctx.device).view(1, 1, 1, W)
    r2 = xs * xs + ys * ys
    mult = (1.0 - s * r2).clamp(0, 1)
    ctx.canvas = ctx.canvas * mult
    return ctx


@register_layer("augment.shadow_gradient")
def aug_shadow_gradient(ctx: GPUContext, *, strength=0.3, direction=0.0) -> GPUContext:
    s_lo, s_hi = _range(strength)
    d_lo, d_hi = _range(direction)
    s = _sample_batch_uniform(s_lo, s_hi, ctx.n, ctx.rng).view(ctx.n, 1, 1, 1)
    d = _sample_batch_uniform(d_lo, d_hi, ctx.n, ctx.rng)
    H, W = ctx.canvas.shape[2:]
    ys = torch.linspace(-1, 1, H, device=ctx.device).view(1, H, 1)
    xs = torch.linspace(-1, 1, W, device=ctx.device).view(1, 1, W)
    rad = d * math.pi / 180.0
    # grad per-sample: (N, H, W)
    grad = xs * torch.cos(rad).view(ctx.n, 1, 1) + ys * torch.sin(rad).view(ctx.n, 1, 1)
    gmin = grad.amin(dim=(1, 2), keepdim=True)
    gmax = grad.amax(dim=(1, 2), keepdim=True)
    grad = (grad - gmin) / (gmax - gmin + 1e-6)  # 0..1
    mult = (1.0 - grad.unsqueeze(1) * s)
    ctx.canvas = (ctx.canvas * mult).clamp(0, 1)
    return ctx


# ---------- camera_sim ----------

@register_layer("augment.defocus")
def aug_defocus(ctx: GPUContext, *, radius=0.0) -> GPUContext:
    r = float(_sample(radius, ctx.rng))
    if r <= 0:
        return ctx
    k1 = max(3, int(r * 2.8) | 1)
    k2 = max(3, int(r * 1.2) | 1)
    x = KF.gaussian_blur2d(ctx.canvas, (k1, k1), (r * 0.7, r * 0.7))
    x = KF.gaussian_blur2d(x, (k2, k2), (r * 0.3, r * 0.3))
    ctx.canvas = x
    return ctx


@register_layer("augment.low_light")
def aug_low_light(ctx: GPUContext, *, brightness=0.4, noise_std=10.0) -> GPUContext:
    b_lo, b_hi = _range(brightness)
    b = _sample_batch_uniform(b_lo, b_hi, ctx.n, ctx.rng).view(ctx.n, 1, 1, 1)
    n_lo, n_hi = _range(noise_std)
    ns = _sample_batch_uniform(n_lo, n_hi, ctx.n, ctx.rng).view(ctx.n, 1, 1, 1) / 255.0
    darkened = ctx.canvas * b
    noise = torch.randn(ctx.canvas.shape, generator=ctx.rng, device=ctx.device) * ns
    ctx.canvas = (darkened + noise).clamp(0, 1)
    return ctx


@register_layer("augment.chromatic_aberration")
def aug_chromatic_aberration(ctx: GPUContext, *, shift=0) -> GPUContext:
    sh = int(_sample(shift, ctx.rng))
    if sh == 0:
        return ctx
    r = torch.roll(ctx.canvas[:, 0:1], shifts=sh, dims=3)
    g = ctx.canvas[:, 1:2]
    b = torch.roll(ctx.canvas[:, 2:3], shifts=-sh, dims=3)
    ctx.canvas = torch.cat([r, g, b], dim=1)
    return ctx


@register_layer("augment.lens_distort")
def aug_lens_distort(ctx: GPUContext, *, k=0.0) -> GPUContext:
    kv_lo, kv_hi = _range(k)
    kv = _sample_batch_uniform(kv_lo, kv_hi, ctx.n, ctx.rng)  # (N,)
    H, W = ctx.canvas.shape[2:]
    ys = torch.linspace(-1, 1, H, device=ctx.device).view(1, H, 1).expand(ctx.n, H, W)
    xs = torch.linspace(-1, 1, W, device=ctx.device).view(1, 1, W).expand(ctx.n, H, W)
    r2 = xs * xs + ys * ys
    factor = 1 + kv.view(ctx.n, 1, 1) * r2
    gx = xs * factor
    gy = ys * factor
    grid = torch.stack([gx, gy], dim=-1)  # (N, H, W, 2)
    combined = torch.cat([ctx.canvas, ctx.mask], dim=1)
    warped = F.grid_sample(combined, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
    ctx.canvas = warped[:, :3].contiguous()
    ctx.mask = warped[:, 3:4].clamp(0.0, 1.0).contiguous()
    return ctx


# ---------- nonrigid ----------

# ---------- Phase 8: motion_blur, salt_pepper, paper_texture, ink_bleed, binarize ----------

@register_layer("augment.motion_blur")
def aug_motion_blur(ctx: GPUContext, *, kernel=1, angle=0.0) -> GPUContext:
    """Directional blur via a k×k line kernel rotated by `angle` degrees."""
    k = int(_sample(kernel, ctx.rng))
    a = float(_sample(angle, ctx.rng))
    k = max(1, k | 1)  # force odd
    if k <= 1:
        return ctx
    k = min(k, 9)  # prevent extreme kernel; v2 clamped to 5 for PIL, we allow 9
    # build directional kernel on CPU then upload
    import numpy as np
    ker = np.zeros((k, k), dtype=np.float32)
    cx = cy = k // 2
    rad = math.radians(a)
    dx, dy = math.cos(rad), math.sin(rad)
    for t in np.linspace(-cx, cx, k * 4):
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if 0 <= x < k and 0 <= y < k:
            ker[y, x] = 1
    ker /= max(ker.sum(), 1.0)
    ker_t = torch.from_numpy(ker).to(ctx.device, dtype=ctx.canvas.dtype)
    # apply same kernel to each of 3 channels
    ker_t = ker_t.view(1, 1, k, k).expand(3, 1, k, k).contiguous()
    pad = k // 2
    x = F.pad(ctx.canvas, (pad, pad, pad, pad), mode="reflect")
    ctx.canvas = F.conv2d(x, ker_t, groups=3).clamp(0, 1)
    return ctx


@register_layer("augment.salt_pepper_noise")
def aug_salt_pepper_noise(ctx: GPUContext, *, amount=0.0) -> GPUContext:
    p = float(_sample(amount, ctx.rng))
    if p <= 0:
        return ctx
    N, C, H, W = ctx.canvas.shape
    mask = torch.rand(N, 1, H, W, generator=ctx.rng, device=ctx.device)
    salt = (mask > 1 - p / 2).float()
    pepper = (mask < p / 2).float()
    canvas = ctx.canvas * (1 - salt - pepper) + salt * 1.0 + pepper * 0.0
    ctx.canvas = canvas.clamp(0, 1)
    return ctx


@register_layer("augment.paper_texture")
def aug_paper_texture(ctx: GPUContext, *, strength=0.1) -> GPUContext:
    """Low-frequency grain overlay. v2: downsample random noise + upsample + blur
    + subtract (tex - 0.5) * 255 * strength from image."""
    s = float(_sample(strength, ctx.rng))
    if s <= 0:
        return ctx
    N, C, H, W = ctx.canvas.shape
    low_h, low_w = max(1, H // 16), max(1, W // 16)
    low = torch.rand(N, 1, low_h, low_w, generator=ctx.rng, device=ctx.device)
    tex = F.interpolate(low, size=(H, W), mode="bicubic", align_corners=False)
    tex = KF.gaussian_blur2d(tex, (5, 5), (1.5, 1.5)).clamp(0, 1)
    # v2: arr -= (tex - 0.5) * 255 * strength
    ctx.canvas = (ctx.canvas - (tex - 0.5) * s).clamp(0, 1)
    return ctx


@register_layer("augment.ink_bleed")
def aug_ink_bleed(ctx: GPUContext, *, radius=1.0) -> GPUContext:
    """Darken + spread ink. v2 converts to grayscale, dilates inverted (= erode
    dark), blurs, blends 60% with original. We implement that on GPU."""
    r = float(_sample(radius, ctx.rng))
    if r <= 0:
        return ctx
    w = torch.tensor([0.299, 0.587, 0.114], device=ctx.device).view(1, 3, 1, 1)
    gray = (ctx.canvas * w).sum(dim=1, keepdim=True)     # (N,1,H,W)
    inv = 1.0 - gray
    # dilate inv => spread dark regions; use max pooling via conv trick or kornia
    k = max(1, int(round(r)))
    ker = torch.ones(1, 1, 2 * k + 1, 2 * k + 1, device=ctx.device)
    inv_dil = (F.conv2d(inv, ker, padding=k) > 0).float()
    # blend dilation with continuous dilation (so shape isn't binary)
    dilated = torch.maximum(inv, inv_dil)
    blurred = KF.gaussian_blur2d(dilated, (max(3, int(r * 4) | 1),) * 2, (r, r))
    bled = 1.0 - blurred
    bled_rgb = bled.expand(-1, 3, -1, -1)
    ctx.canvas = (ctx.canvas * 0.4 + bled_rgb * 0.6).clamp(0, 1)
    return ctx


@register_layer("augment.binarize")
def aug_binarize(ctx: GPUContext, *, threshold=128) -> GPUContext:
    t = float(_sample(threshold, ctx.rng)) / 255.0
    w = torch.tensor([0.299, 0.587, 0.114], device=ctx.device).view(1, 3, 1, 1)
    gray = (ctx.canvas * w).sum(dim=1, keepdim=True)
    bw = (gray > t).float().expand(-1, 3, -1, -1)
    ctx.canvas = bw
    return ctx


@register_layer("augment.elastic")
def aug_elastic(ctx: GPUContext, *, alpha=10.0, sigma=4.0) -> GPUContext:
    a_lo, a_hi = _range(alpha)
    s_lo, s_hi = _range(sigma)
    # per-sample alpha, batch-common sigma (kornia gaussian kernel is single-sized)
    a = _sample_batch_uniform(a_lo, a_hi, ctx.n, ctx.rng)
    sig = float(_sample_batch_uniform(s_lo, s_hi, 1, ctx.rng).item())
    H, W = ctx.canvas.shape[2:]
    # random field [-1, 1] -> smooth -> normalize per-sample so max disp = a
    dx = torch.empty(ctx.n, 1, H, W, device=ctx.device).uniform_(-1, 1, generator=ctx.rng)
    dy = torch.empty(ctx.n, 1, H, W, device=ctx.device).uniform_(-1, 1, generator=ctx.rng)
    if sig > 0:
        k = max(3, int(sig * 4) | 1)
        dx = KF.gaussian_blur2d(dx, (k, k), (sig, sig))
        dy = KF.gaussian_blur2d(dy, (k, k), (sig, sig))
    max_mag = torch.maximum(dx.abs().amax(dim=(1, 2, 3)), dy.abs().amax(dim=(1, 2, 3))).clamp(min=1e-6)
    scale = (a / max_mag).view(ctx.n, 1, 1, 1)
    dx = dx * scale
    dy = dy * scale
    # build sampling grid (normalized -1..1)
    ys = torch.linspace(-1, 1, H, device=ctx.device).view(1, H, 1).expand(ctx.n, H, W)
    xs = torch.linspace(-1, 1, W, device=ctx.device).view(1, 1, W).expand(ctx.n, H, W)
    # dx/dy are in pixel units → convert to normalized
    gx = xs + dx.squeeze(1) * (2.0 / max(W - 1, 1))
    gy = ys + dy.squeeze(1) * (2.0 / max(H - 1, 1))
    grid = torch.stack([gx, gy], dim=-1)
    combined = torch.cat([ctx.canvas, ctx.mask], dim=1)
    warped = F.grid_sample(combined, grid, mode="bilinear",
                            padding_mode="reflection", align_corners=True)
    ctx.canvas = warped[:, :3].contiguous()
    ctx.mask = warped[:, 3:4].clamp(0.0, 1.0).contiguous()
    return ctx


# ---------- safety nets ----------

@register_layer("augment.visibility_guard")
def visibility_guard(ctx: GPUContext, *, min_gap: float = 0.12,
                     force_extreme: bool = True) -> GPUContext:
    """Final safety net (2026-04-23, Codex fix): detect samples where the
    glyph is visually invisible (fill luma ≈ background luma, regardless of
    color) and rescue them by stamping a high-contrast fill over the mask.

    Rationale: fill.hsv_contrast guarantees per-sample contrast at *fill*
    time, but subsequent augments (invert + low_light + brightness + defocus)
    can compound into a near-uniform tile. A tiny fraction of samples can
    emerge completely black. This guard fires post-augment on just those.

    Implementation: compute mean luma inside mask vs outside mask. If the
    absolute gap is < `min_gap` (0..1 scale), force rescue:
        - outside bright → paint mask region black
        - outside dark   → paint mask region white
    Samples with healthy contrast pass through untouched.
    """
    import torch as _torch
    N, _, H, W = ctx.canvas.shape
    lw = _torch.tensor([0.299, 0.587, 0.114], device=ctx.device).view(1, 3, 1, 1)
    lum = (ctx.canvas * lw).sum(dim=1, keepdim=True)           # (N,1,H,W)

    # mask needs to be non-empty; use it as the glyph region indicator.
    # Canvas mask may be 0..1 (soft), so threshold.
    m = (ctx.mask > 0.5).float()
    m_area = m.sum(dim=(1, 2, 3)).clamp(min=1.0)
    b = 1.0 - m
    b_area = b.sum(dim=(1, 2, 3)).clamp(min=1.0)

    inside_luma = (lum * m).sum(dim=(1, 2, 3)) / m_area       # (N,)
    outside_luma = (lum * b).sum(dim=(1, 2, 3)) / b_area

    gap = (inside_luma - outside_luma).abs()
    low = gap < float(min_gap)                                 # (N,) bool

    if not bool(low.any().item()):
        return ctx

    if not force_extreme:
        return ctx

    # Rescue: paint mask region with extreme opposite of outside luma.
    # outside > 0.5 → use black; outside ≤ 0.5 → use white.
    rescue_black = _torch.zeros(N, 3, 1, 1, device=ctx.device)
    rescue_white = _torch.ones(N, 3, 1, 1, device=ctx.device)
    rescue = _torch.where(outside_luma.view(N, 1, 1, 1) > 0.5,
                          rescue_black, rescue_white)
    rescue_fill = rescue.expand(N, 3, H, W)

    apply = low.view(N, 1, 1, 1).float() * m                   # (N,1,H,W)
    ctx.canvas = ctx.canvas * (1 - apply) + rescue_fill * apply
    return ctx
