"""
Composable image augmentations for synthetic CJK training data.

Each op is a plain function: (img: PIL.Image RGB, rng: np.random.Generator, **params) -> PIL.Image RGB.
Params may be:
  - scalar (deterministic)
  - [lo, hi] pair (uniform random per call)
  - list of options (uniform discrete)

`apply_pipeline(img, pipeline, seed)` runs a list of {"op": name, ...params}
dicts against the OP registry.

Grouped by intent:
  geometric  : rotate, perspective, scale_translate, shear
  photometric: brightness, contrast, gamma, saturation, color_jitter, invert
  degradation: gaussian_blur, motion_blur, gaussian_noise, salt_pepper_noise,
               downscale_upscale, jpeg
  scan_sim   : paper_texture, ink_bleed, binarize, shadow_gradient, vignette
  camera_sim : defocus, chromatic_aberration, lens_distort, low_light
"""

from __future__ import annotations

import io
import math
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


# ---------- param sampling ----------------------------------------------------


def _sample(v, rng: np.random.Generator):
    """Resolve a param value.
    - list of length 2 with numbers -> uniform on [v[0], v[1]]
    - list of length != 2 -> discrete choice
    - tuple of length 2 with numbers -> same as list-2
    - scalar / string / None -> returned as-is
    """
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int):
                return int(rng.integers(lo, hi + 1))
            return float(rng.uniform(lo, hi))
        return v[int(rng.integers(0, len(v)))]
    return v


# ---------- geometric ---------------------------------------------------------


def rotate(img, rng, angle=0.0, fill=(255, 255, 255)):
    a = _sample(angle, rng)
    return img.rotate(a, resample=Image.BICUBIC, fillcolor=tuple(fill))


def _perspective_coefficients(src, dst):
    A, B = [], []
    for (xs, ys), (xd, yd) in zip(src, dst):
        A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys])
        A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys])
        B.extend([xd, yd])
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def perspective(img, rng, strength=0.1, fill=(255, 255, 255)):
    s = _sample(strength, rng)
    w, h = img.size
    off = s * min(w, h)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (rng.uniform(0, off),       rng.uniform(0, off)),
        (w - rng.uniform(0, off),   rng.uniform(0, off)),
        (w - rng.uniform(0, off),   h - rng.uniform(0, off)),
        (rng.uniform(0, off),       h - rng.uniform(0, off)),
    ]
    coeffs = _perspective_coefficients(dst, src)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC, fillcolor=tuple(fill))


def scale_translate(img, rng, scale=1.0, translate=0.0, fill=(255, 255, 255)):
    s = _sample(scale, rng)
    tx = _sample(translate, rng)
    ty = _sample(translate, rng)
    w, h = img.size
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), tuple(fill))
    px = (w - nw) // 2 + int(tx * w)
    py = (h - nh) // 2 + int(ty * h)
    canvas.paste(resized, (px, py))
    return canvas


def shear(img, rng, sx=0.0, sy=0.0, fill=(255, 255, 255)):
    a = _sample(sx, rng)
    b = _sample(sy, rng)
    w, h = img.size
    return img.transform((w, h), Image.AFFINE,
                         (1, a, -a * w / 2, b, 1, -b * h / 2),
                         resample=Image.BICUBIC, fillcolor=tuple(fill))


# ---------- photometric -------------------------------------------------------


def brightness(img, rng, factor=1.0):
    return ImageEnhance.Brightness(img).enhance(_sample(factor, rng))


def contrast(img, rng, factor=1.0):
    return ImageEnhance.Contrast(img).enhance(_sample(factor, rng))


def saturation(img, rng, factor=1.0):
    return ImageEnhance.Color(img).enhance(_sample(factor, rng))


def gamma(img, rng, gamma=1.0):
    g = _sample(gamma, rng)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.power(arr, g) * 255.0
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def color_jitter(img, rng, brightness=1.0, contrast=1.0, saturation=1.0):
    img = ImageEnhance.Brightness(img).enhance(_sample(brightness, rng))
    img = ImageEnhance.Contrast(img).enhance(_sample(contrast, rng))
    img = ImageEnhance.Color(img).enhance(_sample(saturation, rng))
    return img


def invert(img, rng, prob=1.0):
    if rng.random() < prob:
        return ImageOps.invert(img.convert("RGB"))
    return img


# ---------- degradation -------------------------------------------------------


def gaussian_blur(img, rng, sigma=0.0):
    s = _sample(sigma, rng)
    if s <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(s))


def motion_blur(img, rng, kernel=1, angle=0.0):
    k = _sample(kernel, rng)
    a = _sample(angle, rng)
    k = max(1, int(k) | 1)  # force odd
    if k <= 1:
        return img
    # build directional kernel
    ker = np.zeros((k, k), dtype=np.float32)
    cx = cy = k // 2
    rad = math.radians(a)
    dx, dy = math.cos(rad), math.sin(rad)
    for t in np.linspace(-cx, cx, k * 4):
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if 0 <= x < k and 0 <= y < k:
            ker[y, x] = 1
    ker /= max(ker.sum(), 1)
    flat = ker.flatten().tolist()
    return img.filter(ImageFilter.Kernel((k, k), flat, scale=1.0, offset=0))


def gaussian_noise(img, rng, std=0.0):
    s = _sample(std, rng)
    if s <= 0:
        return img
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, s, arr.shape)
    return Image.fromarray((arr + noise).clip(0, 255).astype(np.uint8))


def salt_pepper_noise(img, rng, amount=0.0):
    p = _sample(amount, rng)
    if p <= 0:
        return img
    arr = np.asarray(img).copy()
    mask = rng.random(arr.shape[:2])
    arr[mask < p / 2] = 0
    arr[mask > 1 - p / 2] = 255
    return Image.fromarray(arr)


def downscale_upscale(img, rng, factor=1.0):
    f = _sample(factor, rng)
    if f >= 1.0:
        return img
    w, h = img.size
    nw, nh = max(1, int(w * f)), max(1, int(h * f))
    low = img.resize((nw, nh), Image.BILINEAR)
    return low.resize((w, h), Image.BILINEAR)


def jpeg(img, rng, quality=90):
    q = int(_sample(quality, rng))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=max(1, min(100, q)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------- scan_sim ----------------------------------------------------------


def paper_texture(img, rng, strength=0.1, seed=None):
    s = _sample(strength, rng)
    if s <= 0:
        return img
    w, h = img.size
    # low-frequency paper fibers: downsample noise, upsample smoothly
    small = Image.fromarray(
        (rng.random((h // 16 + 1, w // 16 + 1)) * 255).astype(np.uint8), mode="L"
    ).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(1.5))
    tex = np.asarray(small, dtype=np.float32) / 255.0
    arr = np.asarray(img, dtype=np.float32)
    arr -= (tex[..., None] - 0.5) * 255 * s
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def ink_bleed(img, rng, radius=1.0):
    r = _sample(radius, rng)
    if r <= 0:
        return img
    gray = img.convert("L")
    # invert so dark ink becomes bright, dilate (MaxFilter), invert back
    inv = ImageOps.invert(gray)
    k = max(1, int(round(r)) * 2 + 1)
    spread = inv.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.GaussianBlur(r))
    bled = ImageOps.invert(spread)
    # blend back toward original so color survives
    bled_rgb = Image.merge("RGB", (bled, bled, bled))
    return Image.blend(img, bled_rgb, 0.6)


def binarize(img, rng, threshold=128):
    t = int(_sample(threshold, rng))
    gray = np.asarray(img.convert("L"))
    bw = (gray > t).astype(np.uint8) * 255
    return Image.merge("RGB", [Image.fromarray(bw)] * 3)


def shadow_gradient(img, rng, strength=0.3, direction=0.0):
    s = _sample(strength, rng)
    a = _sample(direction, rng)
    if s <= 0:
        return img
    w, h = img.size
    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(xs, ys)
    rad = math.radians(a)
    grad = X * math.cos(rad) + Y * math.sin(rad)
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    mult = 1.0 - grad * s
    arr = np.asarray(img, dtype=np.float32) * mult[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def vignette(img, rng, strength=0.3):
    s = _sample(strength, rng)
    if s <= 0:
        return img
    w, h = img.size
    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(xs, ys)
    r2 = X**2 + Y**2
    mult = np.clip(1.0 - s * r2, 0.0, 1.0)
    arr = np.asarray(img, dtype=np.float32) * mult[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# ---------- camera_sim --------------------------------------------------------


def defocus(img, rng, radius=0.0):
    r = _sample(radius, rng)
    if r <= 0:
        return img
    # disk-kernel approximation via two Gaussians
    return img.filter(ImageFilter.GaussianBlur(r * 0.7)).filter(ImageFilter.GaussianBlur(r * 0.3))


def chromatic_aberration(img, rng, shift=0):
    s = int(_sample(shift, rng))
    if s == 0:
        return img
    r, g, b = img.convert("RGB").split()
    w, h = img.size
    r2 = Image.new("L", (w, h), 0)
    r2.paste(r, (s, 0))
    b2 = Image.new("L", (w, h), 0)
    b2.paste(b, (-s, 0))
    return Image.merge("RGB", (r2, g, b2))


def lens_distort(img, rng, k=0.0):
    """Radial barrel / pincushion distortion. k>0 barrel, k<0 pincushion."""
    kv = _sample(k, rng)
    if abs(kv) < 1e-6:
        return img
    w, h = img.size
    arr = np.asarray(img)
    ys, xs = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2, h / 2
    nx = (xs - cx) / cx
    ny = (ys - cy) / cy
    r2 = nx**2 + ny**2
    factor = 1 + kv * r2
    sx = (nx * factor * cx + cx).clip(0, w - 1).astype(np.int32)
    sy = (ny * factor * cy + cy).clip(0, h - 1).astype(np.int32)
    out = arr[sy, sx]
    return Image.fromarray(out)


def low_light(img, rng, brightness=0.4, noise_std=10.0):
    b = _sample(brightness, rng)
    n = _sample(noise_std, rng)
    darkened = ImageEnhance.Brightness(img).enhance(b)
    arr = np.asarray(darkened, dtype=np.float32)
    noise = rng.normal(0, n, arr.shape)
    return Image.fromarray((arr + noise).clip(0, 255).astype(np.uint8))


# ---------- registry / driver -------------------------------------------------


OPS: dict[str, Callable[..., Image.Image]] = {
    # geometric
    "rotate": rotate, "perspective": perspective,
    "scale_translate": scale_translate, "shear": shear,
    # photometric
    "brightness": brightness, "contrast": contrast, "saturation": saturation,
    "gamma": gamma, "color_jitter": color_jitter, "invert": invert,
    # degradation
    "gaussian_blur": gaussian_blur, "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise, "salt_pepper_noise": salt_pepper_noise,
    "downscale_upscale": downscale_upscale, "jpeg": jpeg,
    # scan sim
    "paper_texture": paper_texture, "ink_bleed": ink_bleed,
    "binarize": binarize, "shadow_gradient": shadow_gradient, "vignette": vignette,
    # camera sim
    "defocus": defocus, "chromatic_aberration": chromatic_aberration,
    "lens_distort": lens_distort, "low_light": low_light,
}


def apply_pipeline(img: Image.Image, pipeline: list[dict[str, Any]],
                   rng: np.random.Generator) -> Image.Image:
    """Apply a sequence of op dicts. Each dict has 'op' + op-specific params.

    Optional 'prob' in each step: probability that the op runs (default 1.0).
    """
    for step in pipeline:
        params = dict(step)
        name = params.pop("op")
        prob = params.pop("prob", 1.0)
        if rng.random() > prob:
            continue
        fn = OPS[name]
        img = fn(img, rng, **params)
    return img
