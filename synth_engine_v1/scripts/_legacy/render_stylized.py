"""
Programmatic stylized renderer for a single character.

Takes a codepoint (+ optional font path), produces a binary character mask via
font rasterization, then composites several poster-style effects on top of
that mask: outlines, drop shadow, gradient fill, neon glow, stripe fill,
textured background, rotated / italicized / warped variants.

The mask-driven design means the same effect set can later be driven by a
stroke-SVG mask (MakeMeAHanzi) instead of a font, so the DB-only path is
possible for chars with no font coverage.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CANVAS = 384          # work canvas, larger than output to allow effects without cropping
OUTPUT = 256
PAD = 48              # margin inside canvas so shadows/outlines aren't clipped


def render_mask(char: str, font_path: Path, face_index: int = 0) -> Image.Image:
    """Return a grayscale mask (L-mode, white=glyph) of the character at CANVAS size."""
    target = CANVAS - 2 * PAD
    # binary-search font size so the glyph fits target box
    lo, hi = 16, CANVAS
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(str(font_path), mid, index=face_index)
        bbox = font.getbbox(char)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= target and h <= target:
            best = (mid, font, bbox)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        font = ImageFont.truetype(str(font_path), 64, index=face_index)
        best = (64, font, font.getbbox(char))
    _, font, bbox = best

    mask = Image.new("L", (CANVAS, CANVAS), 0)
    d = ImageDraw.Draw(mask)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CANVAS - w) // 2 - bbox[0]
    y = (CANVAS - h) // 2 - bbox[1]
    d.text((x, y), char, fill=255, font=font)
    return mask


def to_alpha(mask: Image.Image) -> Image.Image:
    """Treat grayscale mask as alpha channel."""
    return mask.copy()


def outline_mask(mask: Image.Image, radius: int) -> Image.Image:
    """Ring-shaped mask obtained by dilation - original."""
    dilated = mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    ring = ImageChops.subtract(dilated, mask)
    return ring


def dilate(mask: Image.Image, radius: int) -> Image.Image:
    return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))


def gradient_image(size: tuple[int, int], c_top, c_bot, vertical=True) -> Image.Image:
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


def noise_image(size: tuple[int, int], scale: float = 1.0, seed: int | None = None) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.random(size=(size[1], size[0])) * 255 * scale
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="L")


def stripes(size: tuple[int, int], thickness: int, angle: float = 0.0, color_a=(0, 0, 0), color_b=(255, 255, 255)) -> Image.Image:
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


def finalize(rgb: Image.Image) -> Image.Image:
    """Crop work canvas to OUTPUT size."""
    left = (CANVAS - OUTPUT) // 2
    return rgb.crop((left, left, left + OUTPUT, left + OUTPUT))


# ---------- styles ------------------------------------------------------------


@dataclass
class Style:
    name: str

    def render(self, mask: Image.Image) -> Image.Image:
        raise NotImplementedError


class SolidBlackOnWhite(Style):
    def render(self, mask):
        bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        bg.paste(fg, (0, 0), mask)
        return finalize(bg)


class SolidWhiteOnBlack(Style):
    def render(self, mask):
        bg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        fg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        bg.paste(fg, (0, 0), mask)
        return finalize(bg)


class OutlineOnly(Style):
    def __init__(self, name, radius):
        super().__init__(name)
        self.radius = radius

    def render(self, mask):
        bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        ring = outline_mask(mask, self.radius)
        fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        bg.paste(fg, (0, 0), ring)
        return finalize(bg)


class DoubleOutline(Style):
    def render(self, mask):
        bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        outer_ring = outline_mask(dilate(mask, 4), 2)
        inner_ring = outline_mask(mask, 1)
        fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        bg.paste(fg, (0, 0), outer_ring)
        bg.paste(fg, (0, 0), inner_ring)
        return finalize(bg)


class DropShadow(Style):
    def __init__(self, name, offset=(6, 6), blur=6, shadow_color=(120, 120, 120)):
        super().__init__(name)
        self.offset = offset
        self.blur = blur
        self.shadow_color = shadow_color

    def render(self, mask):
        bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        shadow_mask = mask.filter(ImageFilter.GaussianBlur(self.blur))
        shadow_layer = Image.new("RGB", (CANVAS, CANVAS), self.shadow_color)
        shifted = Image.new("L", (CANVAS, CANVAS), 0)
        shifted.paste(shadow_mask, self.offset)
        bg.paste(shadow_layer, (0, 0), shifted)
        fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        bg.paste(fg, (0, 0), mask)
        return finalize(bg)


class GradientFill(Style):
    def __init__(self, name, c_top, c_bot, bg_color=(255, 255, 255)):
        super().__init__(name)
        self.c_top, self.c_bot, self.bg = c_top, c_bot, bg_color

    def render(self, mask):
        bg = Image.new("RGB", (CANVAS, CANVAS), self.bg)
        grad = gradient_image((CANVAS, CANVAS), self.c_top, self.c_bot, vertical=True)
        bg.paste(grad, (0, 0), mask)
        return finalize(bg)


class NeonGlow(Style):
    def __init__(self, name, glow_color=(0, 200, 255), inner_color=(255, 255, 255)):
        super().__init__(name)
        self.glow_color = glow_color
        self.inner_color = inner_color

    def render(self, mask):
        bg = Image.new("RGB", (CANVAS, CANVAS), (10, 10, 20))
        # outer glow = blurred dilated mask, twice for intensity
        glow_mask = dilate(mask, 3).filter(ImageFilter.GaussianBlur(10))
        glow_layer = Image.new("RGB", (CANVAS, CANVAS), self.glow_color)
        bg.paste(glow_layer, (0, 0), glow_mask)
        # tighter inner glow
        inner_glow_mask = mask.filter(ImageFilter.GaussianBlur(3))
        bg.paste(glow_layer, (0, 0), inner_glow_mask)
        # core white
        core = Image.new("RGB", (CANVAS, CANVAS), self.inner_color)
        bg.paste(core, (0, 0), mask)
        return finalize(bg)


class StripeFill(Style):
    def __init__(self, name, thickness=10, angle=30.0):
        super().__init__(name)
        self.thickness = thickness
        self.angle = angle

    def render(self, mask):
        bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        pattern = stripes((CANVAS, CANVAS), self.thickness, self.angle, (0, 0, 0), (200, 200, 200))
        bg.paste(pattern, (0, 0), mask)
        # add a thin black outline so character silhouette reads
        ring = outline_mask(mask, 1)
        fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        bg.paste(fg, (0, 0), ring)
        return finalize(bg)


class NoisyBackground(Style):
    def __init__(self, name, seed=0):
        super().__init__(name)
        self.seed = seed

    def render(self, mask):
        noise = noise_image((CANVAS, CANVAS), scale=0.6, seed=self.seed)
        bg = Image.merge("RGB", (noise, noise, noise))
        fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        bg.paste(fg, (0, 0), mask)
        return finalize(bg)


class BoldedFill(Style):
    def __init__(self, name, radius=2):
        super().__init__(name)
        self.radius = radius

    def render(self, mask):
        bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        bold = dilate(mask, self.radius)
        fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        bg.paste(fg, (0, 0), bold)
        return finalize(bg)


class RotatedItalic(Style):
    def __init__(self, name, angle=10, shear=0.25):
        super().__init__(name)
        self.angle = angle
        self.shear = shear

    def render(self, mask):
        # shear then rotate
        sheared = mask.transform(
            (CANVAS, CANVAS),
            Image.AFFINE,
            (1, -self.shear, self.shear * CANVAS / 2, 0, 1, 0),
            resample=Image.BICUBIC,
        )
        rotated = sheared.rotate(self.angle, resample=Image.BICUBIC)
        bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        bg.paste(fg, (0, 0), rotated)
        return finalize(bg)


class PerspectiveWarp(Style):
    def __init__(self, name, strength=0.15, seed=1):
        super().__init__(name)
        self.strength = strength
        self.seed = seed

    def render(self, mask):
        rng = random.Random(self.seed)
        s = self.strength * CANVAS
        src = [(0, 0), (CANVAS, 0), (CANVAS, CANVAS), (0, CANVAS)]
        dst = [
            (rng.uniform(0, s), rng.uniform(0, s)),
            (CANVAS - rng.uniform(0, s), rng.uniform(0, s)),
            (CANVAS - rng.uniform(0, s), CANVAS - rng.uniform(0, s)),
            (rng.uniform(0, s), CANVAS - rng.uniform(0, s)),
        ]
        coeffs = _perspective_coefficients(dst, src)
        warped = mask.transform((CANVAS, CANVAS), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)
        bg = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        fg = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
        bg.paste(fg, (0, 0), warped)
        return finalize(bg)


def _perspective_coefficients(src, dst):
    # from PIL docs, solve 8-var system for Image.PERSPECTIVE
    A = []
    B = []
    for (xs, ys), (xd, yd) in zip(src, dst):
        A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys])
        A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys])
        B.extend([xd, yd])
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    return np.linalg.solve(A, B).tolist()


# ---------- driver ------------------------------------------------------------


DEFAULT_FONT = Path("C:/Windows/Fonts/malgun.ttf")


def build_styles() -> list[Style]:
    return [
        SolidBlackOnWhite("01_solid_black"),
        SolidWhiteOnBlack("02_solid_white_on_black"),
        OutlineOnly("03_outline_thin", 1),
        OutlineOnly("04_outline_thick", 3),
        DoubleOutline("05_double_outline"),
        BoldedFill("06_bold", radius=2),
        DropShadow("07_drop_shadow", offset=(6, 6), blur=6),
        DropShadow("08_long_shadow", offset=(10, 10), blur=2, shadow_color=(180, 180, 180)),
        GradientFill("09_gradient_warm", (220, 80, 30), (60, 10, 90)),
        GradientFill("10_gradient_cool", (20, 60, 180), (0, 200, 220)),
        NeonGlow("11_neon_cyan", glow_color=(0, 220, 255)),
        NeonGlow("12_neon_pink", glow_color=(255, 30, 180), inner_color=(255, 220, 255)),
        StripeFill("13_stripe_30", thickness=10, angle=30),
        StripeFill("14_stripe_90", thickness=8, angle=90),
        NoisyBackground("15_noisy_bg", seed=42),
        RotatedItalic("16_rotated_italic", angle=12, shear=0.25),
        RotatedItalic("17_rotated_ccw", angle=-8, shear=0.0),
        PerspectiveWarp("18_warp_a", strength=0.12, seed=1),
        PerspectiveWarp("19_warp_b", strength=0.15, seed=9),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("char")
    parser.add_argument("--font", default=str(DEFAULT_FONT))
    parser.add_argument("--face", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    char = args.char
    if len(char) != 1:
        raise SystemExit(f"expected single character, got {char!r}")
    cp = f"U+{ord(char):04X}"

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else script_dir.parent / "out" / f"{cp}_stylized"
    out_dir.mkdir(parents=True, exist_ok=True)

    font_path = Path(args.font)
    print(f"char={char} ({cp}) base_font={font_path.name}#{args.face}")
    print(f"out: {out_dir}")

    mask = render_mask(char, font_path, args.face)
    for style in build_styles():
        img = style.render(mask)
        out_path = out_dir / f"{style.name}.png"
        img.save(out_path)
        print(f"  ok  {style.name}")


if __name__ == "__main__":
    main()
