"""
Fill layer — paints the glyph mask interior onto the canvas.

Registered layers:
    fill.solid     — single color
    fill.gradient  — two-color linear gradient inside the glyph
    fill.stripe    — alternating stripes inside the glyph
    fill.radial    — radial gradient centered on the canvas
    fill.contrast  — color chosen to contrast with canvas brightness where
                     the glyph will sit (dark bg → bright fill, vice versa)
"""

from __future__ import annotations

import colorsys
import math

import numpy as np
from PIL import Image, ImageDraw

from pipeline import CANVAS, Context, _coerce_color, _sample, _sample_color, register_layer


def _linear_gradient_rgb(size, c0, c1, vertical=True):
    w, h = size
    img = Image.new("RGB", size)
    px = img.load()
    for j in range(h):
        for i in range(w):
            t = (j / max(1, h - 1)) if vertical else (i / max(1, w - 1))
            r = int(c0[0] * (1 - t) + c1[0] * t)
            g = int(c0[1] * (1 - t) + c1[1] * t)
            b = int(c0[2] * (1 - t) + c1[2] * t)
            px[i, j] = (r, g, b)
    return img


@register_layer("fill.solid")
def fill_solid(ctx: Context, *, color=(0, 0, 0)) -> Context:
    c = _coerce_color(color)
    fg = Image.new("RGB", ctx.canvas.size, c)
    ctx.canvas.paste(fg, (0, 0), ctx.mask)
    return ctx


@register_layer("fill.gradient")
def fill_gradient(ctx: Context, *, start=(0, 0, 0), end=(255, 255, 255),
                   direction="vertical") -> Context:
    c0 = _coerce_color(start)
    c1 = _coerce_color(end)
    vertical = (direction == "vertical")
    grad = _linear_gradient_rgb(ctx.canvas.size, c0, c1, vertical=vertical)
    ctx.canvas.paste(grad, (0, 0), ctx.mask)
    return ctx


@register_layer("fill.stripe")
def fill_stripe(ctx: Context, *, thickness=10, angle=0,
                 color_a=(0, 0, 0), color_b=(255, 255, 255)) -> Context:
    """Alternating diagonal stripes constrained to the mask interior."""
    t = int(_sample(thickness, ctx.rng))
    a = float(_sample(angle, ctx.rng))
    ca = _coerce_color(color_a)
    cb = _coerce_color(color_b)
    w, h = ctx.canvas.size
    big = max(w, h) * 2
    stripe = Image.new("RGB", (big, big), ca)
    d = ImageDraw.Draw(stripe)
    y = 0
    toggle = False
    while y < big:
        if toggle:
            d.rectangle([0, y, big, y + t], fill=cb)
        y += t
        toggle = not toggle
    stripe = stripe.rotate(a, resample=Image.BICUBIC)
    left = (big - w) // 2
    top = (big - h) // 2
    stripe = stripe.crop((left, top, left + w, top + h))
    ctx.canvas.paste(stripe, (0, 0), ctx.mask)
    return ctx


@register_layer("fill.contrast")
def fill_contrast(ctx: Context, *,
                   threshold=128,
                   dark_color=(20, 20, 20),
                   light_color=(240, 240, 240),
                   jitter=30,
                   sample_region="mask_dilated",
                   use_median=True,
                   min_contrast=90) -> Context:
    """Contrast-aware fill: pick a color that stands out against the current canvas.

    Steps
    -----
    1. Compute a robust brightness statistic (median by default, mean optional)
       of the canvas where the glyph will sit.
    2. If that region is bright (stat > threshold), use `dark_color`; else `light_color`.
    3. Apply random ±`jitter` per channel for natural variation.
    4. Verify |fill_luma − bg_stat| ≥ min_contrast. If not, fall back to pure
       black/white to guarantee readability.
    5. Paint the resulting color through the mask.

    Why median by default
    ---------------------
    A glyph often spans mixed bright/dark regions (e.g. photo with rocks
    overlapping the bottom half of the character). Mean pulls toward the
    middle — can flip the decision to the wrong side. Median is determined
    by the majority pixels, so a small dark area inside a mostly-bright
    region won't flip to "light text".

    Parameters
    ----------
    threshold : int in [0, 255]
        Mid-point between "bright" and "dark" regions. 128 = neutral.
    dark_color : RGB tuple
        Base color used when background is bright.
    light_color : RGB tuple
        Base color used when background is dark.
    jitter : int (or range)
        Per-channel ± perturbation for color variation.
    sample_region : 'mask' | 'mask_dilated' | 'full'
        Which pixels of the canvas to sample.
    use_median : bool
        True → median (robust to mixed regions; default).
        False → mean (simpler but flips on mixed bgs).
    min_contrast : int
        Minimum luma gap required between fill and background. If jittered
        fill falls below this gap, snap to pure black/white.
    """
    gray = np.asarray(ctx.canvas.convert("L"))
    mask_arr = np.asarray(ctx.mask)

    if sample_region == "full":
        area = np.ones_like(mask_arr, dtype=bool)
    elif sample_region == "mask_dilated":
        from PIL import ImageFilter
        dilated = ctx.mask.filter(ImageFilter.MaxFilter(11))
        area = np.asarray(dilated) > 0
    else:  # mask
        area = mask_arr > 0

    if use_median:
        stat = float(np.median(gray[area])) if area.any() else 128.0
    else:
        stat = float(gray[area].mean()) if area.any() else 128.0

    is_bright_bg = stat > threshold
    # dark_color / light_color may be a single RGB triple OR a pool of triples
    base = _sample_color(dark_color if is_bright_bg else light_color, ctx.rng)

    j = int(_sample(jitter, ctx.rng))
    if j > 0:
        shift = ctx.rng.integers(-j, j + 1, 3)
        color = tuple(int(np.clip(base[c] + int(shift[c]), 0, 255)) for c in range(3))
    else:
        color = tuple(int(x) for x in base)

    # Safety fallback: enforce minimum contrast gap.
    fill_luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    if abs(fill_luma - stat) < min_contrast:
        color = (0, 0, 0) if is_bright_bg else (255, 255, 255)

    fg = Image.new("RGB", ctx.canvas.size, color)
    ctx.canvas.paste(fg, (0, 0), ctx.mask)
    return ctx


@register_layer("fill.hsv_contrast")
def fill_hsv_contrast(ctx: Context, *,
                       threshold=160,
                       saturation=(0.5, 1.0),
                       value=(0.0, 1.0),
                       sample_region="mask_dilated",
                       use_median=True,
                       min_contrast=60,
                       max_attempts=8) -> Context:
    """Contrast-aware HSV-random fill.

    Samples H uniformly, S/V from user ranges, then checks signed luma gap
    against background. Retries up to `max_attempts` before falling back to
    pure black/white. Unlike `fill.contrast`, the color space is continuous
    (no palette) — every sample is a fresh RGB point.

    Signed contrast ensures direction:
      bright bg → (bg_luma − fill_luma) ≥ min_contrast   (text darker)
      dark   bg → (fill_luma − bg_luma) ≥ min_contrast   (text lighter)

    Parameters
    ----------
    threshold : int   — median luma cutoff between bright/dark background.
    saturation, value : [lo, hi] in [0, 1]  — HSV ranges.
    min_contrast : int  — required luma gap (signed, toward readable side).
    max_attempts : int  — retries before B/W fallback.
    """
    gray = np.asarray(ctx.canvas.convert("L"))
    mask_arr = np.asarray(ctx.mask)

    if sample_region == "full":
        area = np.ones_like(mask_arr, dtype=bool)
    elif sample_region == "mask_dilated":
        from PIL import ImageFilter
        dilated = ctx.mask.filter(ImageFilter.MaxFilter(11))
        area = np.asarray(dilated) > 0
    else:
        area = mask_arr > 0

    if use_median:
        stat = float(np.median(gray[area])) if area.any() else 128.0
    else:
        stat = float(gray[area].mean()) if area.any() else 128.0

    is_bright_bg = stat > threshold

    color = None
    for _ in range(int(max_attempts)):
        h = float(ctx.rng.uniform(0.0, 1.0))
        s = float(_sample(saturation, ctx.rng))
        v = float(_sample(value, ctx.rng))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cand = (int(r * 255), int(g * 255), int(b * 255))
        fill_luma = 0.299 * cand[0] + 0.587 * cand[1] + 0.114 * cand[2]
        gap = (stat - fill_luma) if is_bright_bg else (fill_luma - stat)
        if gap >= min_contrast:
            color = cand
            break

    if color is None:
        color = (0, 0, 0) if is_bright_bg else (255, 255, 255)

    fg = Image.new("RGB", ctx.canvas.size, color)
    ctx.canvas.paste(fg, (0, 0), ctx.mask)
    return ctx


@register_layer("fill.radial")
def fill_radial(ctx: Context, *, inner=(255, 255, 255), outer=(0, 0, 0),
                 center=None, radius=None) -> Context:
    """Radial gradient fill centered on the glyph bounding box."""
    c0 = _coerce_color(inner)
    c1 = _coerce_color(outer)
    w, h = ctx.canvas.size
    cx, cy = (w / 2, h / 2) if center is None else (float(center[0]), float(center[1]))
    max_r = float(_sample(radius, ctx.rng)) if radius is not None else math.hypot(w, h) / 2
    grad = Image.new("RGB", (w, h))
    px = grad.load()
    for j in range(h):
        for i in range(w):
            d = math.hypot(i - cx, j - cy)
            t = min(1.0, d / max_r)
            r = int(c0[0] * (1 - t) + c1[0] * t)
            g = int(c0[1] * (1 - t) + c1[1] * t)
            b = int(c0[2] * (1 - t) + c1[2] * t)
            px[i, j] = (r, g, b)
    ctx.canvas.paste(grad, (0, 0), ctx.mask)
    return ctx
