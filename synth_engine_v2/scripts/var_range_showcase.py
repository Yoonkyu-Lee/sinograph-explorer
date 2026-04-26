"""
For each tunable variable in the engine, generate a pair of images:
  - <op>_min.png  — lower end of the current recommended range
  - <op>_max.png  — upper end of the current recommended range

Baseline:
  - FontSource: Malgun Gothic (BMP Korean canonical rendering)
  - Single character: 鑑
  - Style block: minimum (background.solid white + fill.solid black)
  - Augment block: ONLY the op under test, at fixed min or max value

Use this to visually audit the parameter ranges: is mild too mild? max too extreme?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import base_source
import svg_stroke
import background  # noqa: F401
import stroke_weight  # noqa: F401
import fill  # noqa: F401
import outline  # noqa: F401
import shadow  # noqa: F401
import glow  # noqa: F401
import augment  # noqa: F401
from pipeline import Context, REGISTRY, finalize, fresh_canvas, run_pipeline


# -------- ranges (single-op at a time) ---------------------------------------
# Each entry: (category, op_fullname, min_params, max_params)
# Paths: "augment.X" for augment ops, "stroke_weight.X" for mask-level ops.

# Existing-range showcase: only ops that have [min, max] defined in some
# config file under configs/. Range values taken as the *widest envelope*
# across configs (if multiple configs define the same op with different
# ranges, we pick min-of-mins and max-of-maxes).
CASES: list[tuple[str, str, dict, dict]] = [
    # ---------- geometric ----------
    # rotate: union of camera_sim (-8,8) / clean_capture (-2,2) / poster_in_dark_room (-6,6)
    ("geometric", "augment.rotate",      {"angle": -8}, {"angle": 8}),
    # perspective: union of camera_sim (0.04,0.15) / poster_in_dark_room (0.05,0.12)
    ("geometric", "augment.perspective", {"strength": 0.04}, {"strength": 0.15}),

    # ---------- photometric ----------
    ("photometric", "augment.brightness", {"factor": 0.6}, {"factor": 1.2}),  # camera_sim
    ("photometric", "augment.contrast",   {"factor": 0.8}, {"factor": 1.2}),  # camera_sim

    # ---------- degradation ----------
    # gaussian_blur: clean_capture (0.0, 0.5)
    ("degradation", "augment.gaussian_blur",  {"sigma": 0.0}, {"sigma": 0.5}),
    # motion_blur: camera_sim kernel (1,7) — but PIL limit 5, so effective (1,5)
    ("degradation", "augment.motion_blur",    {"kernel": 3, "angle": 45}, {"kernel": 5, "angle": 45}),
    # gaussian_noise: union of camera_sim (2,12) / clean_capture (0,4)
    ("degradation", "augment.gaussian_noise", {"std": 0},  {"std": 12}),
    # jpeg: union of clean_capture (80,98) / camera_sim (40,90) / poster_in_dark_room (40,85)
    ("degradation", "augment.jpeg",           {"quality": 98}, {"quality": 40}),

    # ---------- camera sim ----------
    # defocus: union of camera_sim (0.0,1.8) / poster_in_dark_room (0.3,1.5)
    ("camera_sim", "augment.defocus",              {"radius": 0.0}, {"radius": 1.8}),
    ("camera_sim", "augment.chromatic_aberration", {"shift": -2}, {"shift": 2}),   # camera_sim
    ("camera_sim", "augment.lens_distort",         {"k": -0.05},  {"k": 0.05}),    # camera_sim
    # low_light: poster_in_dark_room brightness (0.35,0.65) + noise_std (5,15)
    ("camera_sim", "augment.low_light", {"brightness": 0.65, "noise_std": 5},
                                         {"brightness": 0.35, "noise_std": 15}),

    # ---------- deformation ----------
    # elastic: use mild range (user judged medium+ unsuitable for training)
    ("deformation", "augment.elastic", {"alpha": 4, "sigma": 6}, {"alpha": 8, "sigma": 6}),

    # ---------- mask-level (stroke_weight) ----------
    # stroke_weight.dilate: poster configs (1, 2) — narrow range, representing
    # what's actually used. Showing (0, 2) to also see 'no change' state.
    ("mask", "stroke_weight.dilate", {"radius": 0}, {"radius": 2}),
]


# -------- render helper ------------------------------------------------------


FONTS_DIR = Path("C:/Windows/Fonts")
CHAR = "鑑"
SEED = 42


def make_font_source() -> base_source.FontSource:
    srcs = base_source.discover_font_sources(FONTS_DIR, char_filter=CHAR)
    # Pick malgun regular as the stable baseline
    for s in srcs:
        if s.font_path.stem == "malgun" and s.face_index == 0:
            return s
    return srcs[0]


def render_one(layer_name: str, params: dict, rng: np.random.Generator) -> Image.Image:
    src = make_font_source()
    mask = src.render_mask(CHAR, rng)
    ctx = Context(canvas=fresh_canvas(), mask=mask, rng=rng, char=CHAR)

    # Style block: minimum to render the mask visibly
    style = [
        {"layer": "background.solid", "color": [255, 255, 255]},
    ]
    # If we're testing a mask-level op (stroke_weight.*), run it between bg and fill
    # so it modifies the mask used by fill.
    if layer_name.startswith("stroke_weight."):
        style.append({"layer": layer_name, **params})
        style.append({"layer": "fill.solid", "color": [0, 0, 0]})
        aug = []
    else:
        style.append({"layer": "fill.solid", "color": [0, 0, 0]})
        # augment op under test
        op_name = layer_name.split(".", 1)[1]
        aug = [{"op": op_name, **params}]

    spec = {"style": style, "augment": aug}
    ctx = run_pipeline(ctx, spec)
    return finalize(ctx.canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--category", default=None,
                    help="limit to one category (geometric/photometric/degradation/...)")
    args = ap.parse_args()

    out_root = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "out" / "var_range_showcase"
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for cat, op, pmin, pmax in CASES:
        if args.category and cat != args.category:
            continue
        cat_dir = out_root / cat
        cat_dir.mkdir(exist_ok=True)
        for label, params in [("min", pmin), ("max", pmax)]:
            rng = np.random.default_rng(SEED)
            img = render_one(op, params, rng)
            fname = op.replace(".", "_") + f".{label}.png"
            path = cat_dir / fname
            img.save(path)
            total += 1
            print(f"  {cat:12s}  {op:35s}  {label:3s}  {params}  ->  {path.name}")
    print(f"\nTotal {total} images written under {out_root}")


if __name__ == "__main__":
    main()
