"""
Show ONE image per op at the *proposed* (expanded) max value.
Use to judge if extended ranges are realistic or too extreme.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import base_source
import background  # noqa: F401
import stroke_weight  # noqa: F401
import fill  # noqa: F401
import outline  # noqa: F401
import shadow  # noqa: F401
import glow  # noqa: F401
import augment  # noqa: F401
from pipeline import Context, REGISTRY, finalize, fresh_canvas, run_pipeline


# Proposed expanded maxes (from analysis of realistic conditions).
# Format: (category, op_fullname, params)
PROPOSED_MAX = [
    ("geometric",    "augment.rotate",              {"angle": 15}),
    ("geometric",    "augment.perspective",         {"strength": 0.25}),

    ("photometric",  "augment.brightness",          {"factor": 1.5}),
    ("photometric",  "augment.contrast",            {"factor": 1.5}),

    ("degradation",  "augment.gaussian_blur",       {"sigma": 2.0}),
    # motion_blur stays at PIL hard limit 5 — bug preventing higher
    ("degradation",  "augment.gaussian_noise",      {"std": 25}),
    ("degradation",  "augment.jpeg",                {"quality": 25}),

    ("camera_sim",   "augment.defocus",             {"radius": 3.0}),
    ("camera_sim",   "augment.chromatic_aberration",{"shift": 4}),
    ("camera_sim",   "augment.lens_distort",        {"k": 0.12}),
]


FONTS_DIR = Path("C:/Windows/Fonts")
CHAR = "鑑"
SEED = 42


def make_font_source():
    srcs = base_source.discover_font_sources(FONTS_DIR, char_filter=CHAR)
    for s in srcs:
        if s.font_path.stem == "malgun" and s.face_index == 0:
            return s
    return srcs[0]


def render_one(op_name: str, params: dict, rng) -> any:
    src = make_font_source()
    mask = src.render_mask(CHAR, rng)
    ctx = Context(canvas=fresh_canvas(), mask=mask, rng=rng, char=CHAR)

    style = [
        {"layer": "background.solid", "color": [255, 255, 255]},
        {"layer": "fill.solid", "color": [0, 0, 0]},
    ]
    op = op_name.split(".", 1)[1]
    spec = {"style": style, "augment": [{"op": op, **params}]}
    ctx = run_pipeline(ctx, spec)
    return finalize(ctx.canvas)


def main():
    out_root = Path(__file__).resolve().parent.parent / "out" / "var_range_proposed_max"
    out_root.mkdir(parents=True, exist_ok=True)
    for cat, op, params in PROPOSED_MAX:
        cat_dir = out_root / cat
        cat_dir.mkdir(exist_ok=True)
        rng = np.random.default_rng(SEED)
        img = render_one(op, params, rng)
        fname = op.replace(".", "_") + ".propmax.png"
        path = cat_dir / fname
        img.save(path)
        print(f"  {cat:12s}  {op:35s}  {params}  ->  {path.name}")


if __name__ == "__main__":
    main()
