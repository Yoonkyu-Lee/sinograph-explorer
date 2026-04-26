"""
Outline layer — paints a ring around the glyph onto the canvas.

Registered layers:
    outline.simple  — single ring of given radius and color
    outline.double  — two concentric rings with a gap
"""

from __future__ import annotations

from PIL import Image, ImageChops, ImageFilter

from pipeline import Context, _coerce_color, _sample, register_layer


def _dilate(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))


def _ring(mask: Image.Image, radius: int) -> Image.Image:
    """Mask of pixels that are within `radius` of the glyph but not inside it."""
    return ImageChops.subtract(_dilate(mask, radius), mask)


@register_layer("outline.simple")
def outline_simple(ctx: Context, *, radius=1, color=(0, 0, 0)) -> Context:
    r = int(_sample(radius, ctx.rng))
    if r <= 0:
        return ctx
    c = _coerce_color(color)
    ring = _ring(ctx.mask, r)
    fg = Image.new("RGB", ctx.canvas.size, c)
    ctx.canvas.paste(fg, (0, 0), ring)
    return ctx


@register_layer("outline.double")
def outline_double(ctx: Context, *, outer_offset=4, outer_width=2,
                    inner_width=1, color=(0, 0, 0)) -> Context:
    """Two rings: inner ring around glyph, outer ring offset further out.

    outer_offset: how far the outer ring is pushed away from the glyph
    outer_width : thickness of the outer ring
    inner_width : thickness of the inner ring
    """
    off = int(_sample(outer_offset, ctx.rng))
    ow = int(_sample(outer_width, ctx.rng))
    iw = int(_sample(inner_width, ctx.rng))
    c = _coerce_color(color)
    fg = Image.new("RGB", ctx.canvas.size, c)
    inner_ring = _ring(ctx.mask, iw)
    outer_ring = _ring(_dilate(ctx.mask, off), ow)
    ctx.canvas.paste(fg, (0, 0), outer_ring)
    ctx.canvas.paste(fg, (0, 0), inner_ring)
    return ctx
