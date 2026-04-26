"""
Background layer — paints the canvas before the glyph is composited.

Registered layers:
    background.solid     — uniform color fill
    background.noise     — random grayscale noise
    background.gradient  — two-color linear gradient
    background.stripe    — alternating diagonal/horizontal stripe bands
    background.lines     — thin regular lines on base color (notebook paper)
    background.scene     — random crop from an image folder (photos, illustrations, UI)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pipeline import CANVAS, Context, _coerce_color, _sample, register_layer


# Cache image listing per folder to avoid re-walking on every call.
_SCENE_CACHE: dict[str, list[Path]] = {}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _scene_images(folder: str) -> list[Path]:
    """List all image files in folder (recursive). Cached per folder path."""
    key = str(Path(folder).resolve())
    if key in _SCENE_CACHE:
        return _SCENE_CACHE[key]
    fp = Path(key)
    if not fp.is_dir():
        _SCENE_CACHE[key] = []
        return []
    files = [p for p in fp.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    _SCENE_CACHE[key] = files
    return files


@register_layer("background.solid")
def background_solid(ctx: Context, *, color=(255, 255, 255)) -> Context:
    """Fill canvas with a single RGB color."""
    c = _coerce_color(color)
    ctx.canvas = Image.new("RGB", ctx.canvas.size, c)
    return ctx


@register_layer("background.noise")
def background_noise(ctx: Context, *, scale=0.6, smooth=0.0) -> Context:
    """Grayscale random noise.

    scale: 0..1, mean brightness (higher = lighter average)
    smooth: optional Gaussian blur sigma applied to the noise field
    """
    s = float(_sample(scale, ctx.rng))
    sm = float(_sample(smooth, ctx.rng))
    w, h = ctx.canvas.size
    arr = (ctx.rng.random((h, w)) * 255 * s).clip(0, 255).astype(np.uint8)
    noise = Image.fromarray(arr, mode="L")
    if sm > 0:
        noise = noise.filter(ImageFilter.GaussianBlur(sm))
    ctx.canvas = Image.merge("RGB", (noise, noise, noise))
    return ctx


@register_layer("background.stripe")
def background_stripe(ctx: Context, *, thickness=20, angle=0,
                       color_a=(255, 255, 255), color_b=(230, 235, 240)) -> Context:
    """Alternating colored bands as background.

    For subtle paper/texture: thick bands (20~40) + low-contrast colors.
    For game UI / decorative: thinner bands + higher contrast.
    Does NOT touch the glyph mask; glyph stays intact.
    """
    t = max(1, int(_sample(thickness, ctx.rng)))
    a = float(_sample(angle, ctx.rng))
    ca = _coerce_color(color_a)
    cb = _coerce_color(color_b)
    w, h = ctx.canvas.size
    big = max(w, h) * 2
    pattern = Image.new("RGB", (big, big), ca)
    d = ImageDraw.Draw(pattern)
    y = 0
    toggle = False
    while y < big:
        if toggle:
            d.rectangle([0, y, big, y + t], fill=cb)
        y += t
        toggle = not toggle
    pattern = pattern.rotate(a, resample=Image.BICUBIC)
    left = (big - w) // 2
    top = (big - h) // 2
    ctx.canvas = pattern.crop((left, top, left + w, top + h))
    return ctx


@register_layer("background.lines")
def background_lines(ctx: Context, *, spacing=25, line_width=1, angle=0,
                      base_color=(255, 255, 255), line_color=(200, 210, 230)) -> Context:
    """Thin regular lines on base color — notebook / lined paper.

    spacing: vertical gap between lines (larger = fewer lines)
    line_width: line thickness in pixels
    angle: 0 for horizontal lines, 90 for vertical, other for diagonal
    """
    sp = max(2, int(_sample(spacing, ctx.rng)))
    lw = max(1, int(_sample(line_width, ctx.rng)))
    a = float(_sample(angle, ctx.rng))
    bc = _coerce_color(base_color)
    lc = _coerce_color(line_color)
    w, h = ctx.canvas.size
    big = max(w, h) * 2
    pattern = Image.new("RGB", (big, big), bc)
    d = ImageDraw.Draw(pattern)
    y = 0
    while y < big:
        d.rectangle([0, y, big, y + lw], fill=lc)
        y += sp
    pattern = pattern.rotate(a, resample=Image.BICUBIC)
    left = (big - w) // 2
    top = (big - h) // 2
    ctx.canvas = pattern.crop((left, top, left + w, top + h))
    return ctx


@register_layer("background.scene")
def background_scene(ctx: Context, *, folder, mode="random_crop",
                     scale_jitter=(1.0, 1.5), dim=0.0, desaturate=0.0,
                     blur=0.0) -> Context:
    """Random crop from an image in `folder` used as canvas background.

    Intended for real-world-realistic training: dropping glyphs onto photos,
    illustrations, UI screenshots, textures, etc. so the model sees text on
    rich backgrounds (not just solid white).

    Parameters
    ----------
    folder : str
        Path to a directory of JPG/PNG/WEBP images (recursive).
    mode : 'random_crop' | 'resize'
        random_crop: scale up the source image (random factor in scale_jitter),
                     then take a canvas-sized crop from a random position.
        resize:      scale the whole image to canvas size (may distort aspect).
    scale_jitter : (lo, hi)
        Random zoom factor before cropping. Larger values = tighter crop
        (more zoomed in → sees less of the overall scene).
    dim : 0..1 or range
        Blend toward black. Useful to darken busy backgrounds so glyph stands out.
    desaturate : 0..1 or range
        Blend toward grayscale. Reduces color clash with colored glyph fills.
    blur : float or range
        Optional Gaussian blur sigma to soften busy textures.

    If folder is missing or empty, this layer is a no-op (leaves canvas
    untouched) and prints a one-time warning.
    """
    files = _scene_images(folder)
    if not files:
        # silent no-op — caller can log or check empty folder externally
        return ctx

    idx = int(ctx.rng.integers(0, len(files)))
    path = files[idx]
    try:
        bg = Image.open(path).convert("RGB")
    except Exception:
        return ctx  # corrupt image, skip

    w, h = ctx.canvas.size
    bw, bh = bg.size

    if mode == "random_crop":
        lo, hi = scale_jitter if isinstance(scale_jitter, (list, tuple)) else (scale_jitter, scale_jitter)
        scale = float(lo + (hi - lo) * ctx.rng.random())
        # ensure scaled image is at least canvas size
        min_scale = max(w / bw, h / bh)
        scale = max(scale, min_scale)
        bw_new = max(w, int(bw * scale))
        bh_new = max(h, int(bh * scale))
        bg = bg.resize((bw_new, bh_new), Image.LANCZOS)
        x = int(ctx.rng.integers(0, max(1, bw_new - w + 1)))
        y = int(ctx.rng.integers(0, max(1, bh_new - h + 1)))
        bg = bg.crop((x, y, x + w, y + h))
    else:  # resize
        bg = bg.resize((w, h), Image.LANCZOS)

    # desaturate
    d_amount = float(_sample(desaturate, ctx.rng))
    if d_amount > 0:
        gray = bg.convert("L").convert("RGB")
        bg = Image.blend(bg, gray, min(1.0, d_amount))

    # dim toward black
    dim_amount = float(_sample(dim, ctx.rng))
    if dim_amount > 0:
        black = Image.new("RGB", bg.size, (0, 0, 0))
        bg = Image.blend(bg, black, min(1.0, dim_amount))

    # optional blur
    blur_amount = float(_sample(blur, ctx.rng))
    if blur_amount > 0:
        bg = bg.filter(ImageFilter.GaussianBlur(blur_amount))

    ctx.canvas = bg
    return ctx


@register_layer("background.gradient")
def background_gradient(ctx: Context, *, start=(0, 0, 0), end=(255, 255, 255),
                         direction="vertical") -> Context:
    """Linear gradient between two colors.

    direction: 'vertical' (top → bottom) or 'horizontal' (left → right)
    """
    c0 = _coerce_color(start)
    c1 = _coerce_color(end)
    w, h = ctx.canvas.size
    grad = Image.new("RGB", (w, h))
    px = grad.load()
    vertical = (direction == "vertical")
    for j in range(h):
        for i in range(w):
            t = (j / max(1, h - 1)) if vertical else (i / max(1, w - 1))
            r = int(c0[0] * (1 - t) + c1[0] * t)
            g = int(c0[1] * (1 - t) + c1[1] * t)
            b = int(c0[2] * (1 - t) + c1[2] * t)
            px[i, j] = (r, g, b)
    ctx.canvas = grad
    return ctx
