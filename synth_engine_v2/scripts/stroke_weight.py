"""
Stroke weight layer — modifies the mask itself (not the canvas).

Applied before fill/outline so the rest of the pipeline sees the adjusted
glyph shape.

Registered layers:
    stroke_weight.dilate  — thicker strokes
    stroke_weight.erode   — thinner strokes
"""

from __future__ import annotations

from PIL import ImageFilter, ImageOps

from pipeline import Context, _sample, register_layer


@register_layer("stroke_weight.dilate")
def stroke_weight_dilate(ctx: Context, *, radius=1) -> Context:
    r = int(_sample(radius, ctx.rng))
    if r <= 0:
        return ctx
    k = r * 2 + 1
    ctx.mask = ctx.mask.filter(ImageFilter.MaxFilter(k))
    return ctx


@register_layer("stroke_weight.erode")
def stroke_weight_erode(ctx: Context, *, radius=1) -> Context:
    r = int(_sample(radius, ctx.rng))
    if r <= 0:
        return ctx
    k = r * 2 + 1
    # erosion = invert → dilate → invert (MaxFilter on inverted mask)
    inv = ImageOps.invert(ctx.mask)
    inv = inv.filter(ImageFilter.MaxFilter(k))
    ctx.mask = ImageOps.invert(inv)
    return ctx
