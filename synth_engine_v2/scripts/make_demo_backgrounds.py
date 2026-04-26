"""
Generate a handful of demo background images so `background.scene` can be
tested before the user populates samples/backgrounds/ with real photos.

Outputs 5 synthetic backgrounds to samples/backgrounds/demo_*.png.
User is expected to replace these with real photos/illustrations.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent.parent / "samples" / "backgrounds"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 768  # larger than canvas so random_crop has room


def gradient_sunset() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE))
    px = img.load()
    for y in range(SIZE):
        t = y / SIZE
        r = int(255 * (1 - t * 0.3))
        g = int(140 + 60 * (1 - t))
        b = int(90 + 120 * t)
        for x in range(SIZE):
            px[x, y] = (r, g, b)
    return img


def noise_fabric() -> Image.Image:
    rng = np.random.default_rng(7)
    base = rng.integers(180, 240, (SIZE, SIZE, 3), dtype=np.uint8)
    # weave-like structure: add horizontal/vertical brightness modulation
    for i in range(SIZE):
        if i % 8 < 4:
            base[i] = np.clip(base[i] - 15, 0, 255)
        if i % 8 < 4:
            base[:, i] = np.clip(base[:, i] - 10, 0, 255)
    img = Image.fromarray(base)
    return img.filter(ImageFilter.GaussianBlur(0.8))


def geometric_shapes() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), (230, 220, 200))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(3)
    for _ in range(40):
        cx = int(rng.integers(0, SIZE))
        cy = int(rng.integers(0, SIZE))
        r = int(rng.integers(40, 160))
        color = tuple(int(x) for x in rng.integers(80, 220, 3))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return img.filter(ImageFilter.GaussianBlur(2.5))


def paper_texture() -> Image.Image:
    rng = np.random.default_rng(11)
    base = np.full((SIZE, SIZE, 3), 245, dtype=np.uint8)
    # speckled paper fibers
    mask = rng.random((SIZE, SIZE)) > 0.997
    base[mask] = [200, 195, 180]
    # faint color wash
    for y in range(SIZE):
        tint = 0 if y < SIZE * 0.7 else int((y - SIZE * 0.7) / (SIZE * 0.3) * 8)
        base[y] = np.clip(base[y].astype(int) - [tint, tint, 0], 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    return img.filter(ImageFilter.GaussianBlur(0.3))


def abstract_stripes() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(23)
    # random colored diagonal stripes
    colors = [
        (180, 120, 90),  (100, 150, 200), (90, 180, 130),
        (220, 170, 100), (160, 100, 180), (230, 220, 180),
    ]
    y = 0
    while y < SIZE * 2:
        w = int(rng.integers(30, 80))
        c = colors[int(rng.integers(0, len(colors)))]
        d.rectangle([0, y, SIZE * 2, y + w], fill=c)
        y += w
    img = img.rotate(35, resample=Image.BICUBIC)
    # center crop to SIZE×SIZE
    img = img.crop(((img.width - SIZE) // 2, (img.height - SIZE) // 2,
                    (img.width + SIZE) // 2, (img.height + SIZE) // 2))
    return img.filter(ImageFilter.GaussianBlur(3.0))


DEMOS = {
    "demo_sunset_gradient.png": gradient_sunset,
    "demo_fabric_weave.png": noise_fabric,
    "demo_geo_shapes.png": geometric_shapes,
    "demo_paper_texture.png": paper_texture,
    "demo_abstract_stripes.png": abstract_stripes,
}


def main():
    for fname, fn in DEMOS.items():
        p = OUT / fname
        img = fn()
        img.save(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
