"""
Shadow layer — paints a shadow on the canvas beneath where the glyph will go.

Apply shadow BEFORE fill / outline so it ends up behind the glyph.

Registered layers:
    shadow.drop  — offset + Gaussian-blurred silhouette
    shadow.soft  — centered, heavy blur (diffuse ambient shadow)
    shadow.long  — stacked offset silhouettes creating a long-shadow effect
"""

from __future__ import annotations

from PIL import Image, ImageChops, ImageFilter

from pipeline import Context, _coerce_color, _sample, register_layer


@register_layer("shadow.drop")
def shadow_drop(ctx: Context, *, offset=(6, 6), blur=6,
                 color=(120, 120, 120)) -> Context:
    """Classic drop shadow: blur the glyph silhouette and offset it."""
    if isinstance(offset[0], (list, tuple)):
        dx = int(_sample(offset[0], ctx.rng))
        dy = int(_sample(offset[1], ctx.rng))
    else:
        dx, dy = int(offset[0]), int(offset[1])
    b = float(_sample(blur, ctx.rng))
    c = _coerce_color(color)
    w, h = ctx.canvas.size
    shadow = Image.new("L", (w, h), 0)
    shadow.paste(ctx.mask.filter(ImageFilter.GaussianBlur(b)), (dx, dy))
    shadow_rgb = Image.new("RGB", (w, h), c)
    ctx.canvas.paste(shadow_rgb, (0, 0), shadow)
    return ctx


@register_layer("shadow.soft")
def shadow_soft(ctx: Context, *, blur=12, color=(160, 160, 160)) -> Context:
    """Centered diffuse shadow — no offset, strong blur."""
    b = float(_sample(blur, ctx.rng))
    c = _coerce_color(color)
    w, h = ctx.canvas.size
    halo = ctx.mask.filter(ImageFilter.GaussianBlur(b))
    layer = Image.new("RGB", (w, h), c)
    ctx.canvas.paste(layer, (0, 0), halo)
    return ctx


@register_layer("shadow.long")
def shadow_long(ctx: Context, *, step=(1, 1), length=20,
                 color=(180, 180, 180)) -> Context:
    """Long shadow built by stacking shifted silhouettes along one direction.

    step  : (dx, dy) per layer
    length: how many stacked copies
    """
    sx = int(_sample(step[0], ctx.rng)) if isinstance(step[0], (list, tuple)) else int(step[0])
    sy = int(_sample(step[1], ctx.rng)) if isinstance(step[1], (list, tuple)) else int(step[1])
    n = int(_sample(length, ctx.rng))
    c = _coerce_color(color)
    w, h = ctx.canvas.size
    acc = Image.new("L", (w, h), 0)
    for i in range(1, n + 1):
        shifted = Image.new("L", (w, h), 0)
        shifted.paste(ctx.mask, (sx * i, sy * i))
        acc = ImageChops.lighter(acc, shifted)
    layer = Image.new("RGB", (w, h), c)
    ctx.canvas.paste(layer, (0, 0), acc)
    return ctx
