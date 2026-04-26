"""
Effect stacks: transform a glyph mask into a final stylized RGB image.

Each effect is a pure function `(mask: L-mode, rng: np.random.Generator)
-> RGB image` of size (OUTPUT, OUTPUT). They are registered by name and
callable via REGISTRY[name](mask, rng).

'clean' is the trivial stack: solid black glyph on white background (i.e.
what the former render_systemfonts.py produced). It is just one entry among
many; treating clean as a stack instead of a separate code path is the key
structural cleanup for v1 → v2.

v1 note: rotated_italic / rotated_ccw / warp_a / warp_b are effectively
image-space geometric augmentations that happen to be bundled as effect
stacks for parity with the prior prototype. In a later revision they can
migrate out to augment.py.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


CANVAS = 384
OUTPUT = 256

EffectFn = Callable[[Image.Image, np.random.Generator], Image.Image]

REGISTRY: dict[str, EffectFn] = {}


def register(name: str):
    def deco(fn: EffectFn) -> EffectFn:
        REGISTRY[name] = fn
        return fn
    return deco


# ---------- mask / compositing primitives -------------------------------------


def _finalize(rgb: Image.Image) -> Image.Image:
    left = (CANVAS - OUTPUT) // 2
    return rgb.crop((left, left, left + OUTPUT, left + OUTPUT))


def _dilate(mask: Image.Image, radius: int) -> Image.Image:
    return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))


def _outline_ring(mask: Image.Image, radius: int) -> Image.Image:
    dilated = _dilate(mask, radius)
    return ImageChops.subtract(dilated, mask)


def _gradient_image(size: tuple[int, int], c_top, c_bot, vertical: bool = True) -> Image.Image:
    w, h = size
    grad = Image.new("RGB", size)
    px = grad.load()
    for j in range(h):
        t = j / max(1, h - 1) if vertical else 0
        for i in range(w):
            if not vertical:
                t = i / max(1, w - 1)
            r = int(c_top[0] * (1 - t) + c_bot[0] * t)
            g = int(c_top[1] * (1 - t) + c_bot[1] * t)
            b = int(c_top[2] * (1 - t) + c_bot[2] * t)
            px[i, j] = (r, g, b)
    return grad


def _stripe_pattern(size: tuple[int, int], thickness: int, angle: float,
                    color_a, color_b) -> Image.Image:
    w, h = size
    big = max(w, h) * 2
    stripe = Image.new("RGB", (big, big), color_a)
    d = ImageDraw.Draw(stripe)
    y = 0
    toggle = False
    while y < big:
        if toggle:
            d.rectangle([0, y, big, y + thickness], fill=color_b)
        y += thickness
        toggle = not toggle
    stripe = stripe.rotate(angle, resample=Image.BICUBIC)
    left = (big - w) // 2
    top = (big - h) // 2
    return stripe.crop((left, top, left + w, top + h))


def _perspective_coefficients(src, dst):
    A, B = [], []
    for (xs, ys), (xd, yd) in zip(src, dst):
        A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys])
        A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys])
        B.extend([xd, yd])
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


# ---------- effect stacks (registered by name) -------------------------------


@register("clean")
def clean(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), mask)
    return _finalize(bg)


@register("solid_inverse")
def solid_inverse(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    fg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    bg.paste(fg, (0, 0), mask)
    return _finalize(bg)


@register("outline_thin")
def outline_thin(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    ring = _outline_ring(mask, 1)
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), ring)
    return _finalize(bg)


@register("outline_thick")
def outline_thick(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    ring = _outline_ring(mask, 3)
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), ring)
    return _finalize(bg)


@register("double_outline")
def double_outline(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    outer = _outline_ring(_dilate(mask, 4), 2)
    inner = _outline_ring(mask, 1)
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), outer)
    bg.paste(fg, (0, 0), inner)
    return _finalize(bg)


@register("bold")
def bold(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), _dilate(mask, 2))
    return _finalize(bg)


@register("drop_shadow")
def drop_shadow(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    blurred = mask.filter(ImageFilter.GaussianBlur(6))
    shifted = Image.new("L", (CANVAS, CANVAS), 0)
    shifted.paste(blurred, (6, 6))
    shadow_layer = Image.new("RGB", (CANVAS, CANVAS), (120, 120, 120))
    bg.paste(shadow_layer, (0, 0), shifted)
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), mask)
    return _finalize(bg)


@register("long_shadow")
def long_shadow(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    blurred = mask.filter(ImageFilter.GaussianBlur(2))
    shifted = Image.new("L", (CANVAS, CANVAS), 0)
    shifted.paste(blurred, (10, 10))
    shadow_layer = Image.new("RGB", (CANVAS, CANVAS), (180, 180, 180))
    bg.paste(shadow_layer, (0, 0), shifted)
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), mask)
    return _finalize(bg)


@register("gradient_warm")
def gradient_warm(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    grad = _gradient_image((CANVAS, CANVAS), (220, 80, 30), (60, 10, 90), vertical=True)
    bg.paste(grad, (0, 0), mask)
    return _finalize(bg)


@register("gradient_cool")
def gradient_cool(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    grad = _gradient_image((CANVAS, CANVAS), (20, 60, 180), (0, 200, 220), vertical=True)
    bg.paste(grad, (0, 0), mask)
    return _finalize(bg)


@register("neon_cyan")
def neon_cyan(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (10, 10, 20))
    glow_mask = _dilate(mask, 3).filter(ImageFilter.GaussianBlur(10))
    glow_layer = Image.new("RGB", (CANVAS, CANVAS), (0, 220, 255))
    bg.paste(glow_layer, (0, 0), glow_mask)
    inner = mask.filter(ImageFilter.GaussianBlur(3))
    bg.paste(glow_layer, (0, 0), inner)
    core = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    bg.paste(core, (0, 0), mask)
    return _finalize(bg)


@register("neon_pink")
def neon_pink(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (10, 10, 20))
    glow_mask = _dilate(mask, 3).filter(ImageFilter.GaussianBlur(10))
    glow_layer = Image.new("RGB", (CANVAS, CANVAS), (255, 30, 180))
    bg.paste(glow_layer, (0, 0), glow_mask)
    inner = mask.filter(ImageFilter.GaussianBlur(3))
    bg.paste(glow_layer, (0, 0), inner)
    core = Image.new("RGB", (CANVAS, CANVAS), (255, 220, 255))
    bg.paste(core, (0, 0), mask)
    return _finalize(bg)


@register("stripe_30")
def stripe_30(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    pattern = _stripe_pattern((CANVAS, CANVAS), 10, 30, (0, 0, 0), (200, 200, 200))
    bg.paste(pattern, (0, 0), mask)
    ring = _outline_ring(mask, 1)
    bg.paste(Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0)), (0, 0), ring)
    return _finalize(bg)


@register("stripe_90")
def stripe_90(mask, rng):
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    pattern = _stripe_pattern((CANVAS, CANVAS), 8, 90, (0, 0, 0), (200, 200, 200))
    bg.paste(pattern, (0, 0), mask)
    ring = _outline_ring(mask, 1)
    bg.paste(Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0)), (0, 0), ring)
    return _finalize(bg)


@register("noisy_bg")
def noisy_bg(mask, rng):
    arr = (rng.random((CANVAS, CANVAS)) * 255 * 0.6).clip(0, 255).astype(np.uint8)
    noise = Image.fromarray(arr, mode="L")
    bg = Image.merge("RGB", (noise, noise, noise))
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), mask)
    return _finalize(bg)


@register("rotated_italic")
def rotated_italic(mask, rng):
    shear = 0.25
    sheared = mask.transform(
        (CANVAS, CANVAS),
        Image.AFFINE,
        (1, -shear, shear * CANVAS / 2, 0, 1, 0),
        resample=Image.BICUBIC,
    )
    rotated = sheared.rotate(12, resample=Image.BICUBIC)
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), rotated)
    return _finalize(bg)


@register("rotated_ccw")
def rotated_ccw(mask, rng):
    rotated = mask.rotate(-8, resample=Image.BICUBIC)
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), rotated)
    return _finalize(bg)


def _warp(mask, strength, seed):
    r = random.Random(seed)
    s = strength * CANVAS
    src = [(0, 0), (CANVAS, 0), (CANVAS, CANVAS), (0, CANVAS)]
    dst = [
        (r.uniform(0, s),          r.uniform(0, s)),
        (CANVAS - r.uniform(0, s), r.uniform(0, s)),
        (CANVAS - r.uniform(0, s), CANVAS - r.uniform(0, s)),
        (r.uniform(0, s),          CANVAS - r.uniform(0, s)),
    ]
    coeffs = _perspective_coefficients(dst, src)
    warped = mask.transform((CANVAS, CANVAS), Image.PERSPECTIVE, coeffs,
                            resample=Image.BICUBIC)
    bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    bg.paste(fg, (0, 0), warped)
    return _finalize(bg)


@register("warp_a")
def warp_a(mask, rng):
    return _warp(mask, 0.12, 1)


@register("warp_b")
def warp_b(mask, rng):
    return _warp(mask, 0.15, 9)
