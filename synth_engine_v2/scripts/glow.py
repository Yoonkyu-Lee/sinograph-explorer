"""
Glow layer — paints an emissive halo around or inside the glyph.

Registered layers:
    glow.outer  — colored halo spreading outward from the glyph
    glow.inner  — soft highlight tracing the glyph shape
    glow.neon   — composite outer + core (one-shot neon look)
"""

from __future__ import annotations

from PIL import Image, ImageFilter

from pipeline import Context, _coerce_color, _sample, register_layer


def _dilate(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))


@register_layer("glow.outer")
def glow_outer(ctx: Context, *, color=(255, 255, 255),
                dilate=3, blur=10) -> Context:
    """Outward halo. Dilate mask, blur, paint color."""
    r = int(_sample(dilate, ctx.rng))
    b = float(_sample(blur, ctx.rng))
    c = _coerce_color(color)
    halo = _dilate(ctx.mask, r).filter(ImageFilter.GaussianBlur(b))
    layer = Image.new("RGB", ctx.canvas.size, c)
    ctx.canvas.paste(layer, (0, 0), halo)
    return ctx


@register_layer("glow.inner")
def glow_inner(ctx: Context, *, color=(255, 255, 255), blur=3) -> Context:
    """Softened copy of the glyph used as an inner highlight."""
    b = float(_sample(blur, ctx.rng))
    c = _coerce_color(color)
    soft = ctx.mask.filter(ImageFilter.GaussianBlur(b))
    layer = Image.new("RGB", ctx.canvas.size, c)
    ctx.canvas.paste(layer, (0, 0), soft)
    return ctx


@register_layer("glow.neon")
def glow_neon(ctx: Context, *, color=(0, 220, 255),
               outer_dilate=3, outer_blur=10,
               inner_blur=3, core_color=(255, 255, 255)) -> Context:
    """One-shot neon look: outer halo + inner halo + bright core.

    Does NOT paint the glyph body itself — add a `fill.solid` afterwards if
    you want a hard-edged core on top of the glow.
    """
    outer_r = int(_sample(outer_dilate, ctx.rng))
    outer_b = float(_sample(outer_blur, ctx.rng))
    inner_b = float(_sample(inner_blur, ctx.rng))
    glow_c = _coerce_color(color)
    core_c = _coerce_color(core_color)
    w, h = ctx.canvas.size

    halo = _dilate(ctx.mask, outer_r).filter(ImageFilter.GaussianBlur(outer_b))
    ctx.canvas.paste(Image.new("RGB", (w, h), glow_c), (0, 0), halo)

    inner = ctx.mask.filter(ImageFilter.GaussianBlur(inner_b))
    ctx.canvas.paste(Image.new("RGB", (w, h), glow_c), (0, 0), inner)

    ctx.canvas.paste(Image.new("RGB", (w, h), core_c), (0, 0), ctx.mask)
    return ctx
