"""
Composition engine for synth_engine_v2.

- Context: state that flows through the pipeline (canvas, mask, rng, char)
- REGISTRY: name → layer function lookup
- @register_layer(name): decorator used by each dimension module
- run_pipeline(ctx, spec): executes the style block then the augment block
- _sample(v, rng): shared parameter-value sampler

Parameter sampling convention
    scalar              -> returned as-is
    [lo, hi] of numbers -> uniform random (integer if both int)
    other list          -> uniform discrete choice among the elements

Colors should be passed as tuples (r, g, b) or length-3 lists of ints;
layer functions that take color parameters do NOT _sample them by default
(see per-module code). To vary a color, pass a list of colors, e.g.
color=[(255,0,0), (0,255,0)]; layer functions that support variation will
call _sample on that outer list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image


CANVAS = 384   # work canvas size used throughout
OUTPUT = 256   # final crop size produced by finalize()
# PAD 는 glyph box 와 canvas 가장자리 사이 여백.
# 48 → 80 (2026-04-19): PAD=48 일 때 glyph 288 > output 256 → rotate augment
# 에서 코너가 잘림. PAD=80 이면 glyph 224 로 축소, output 대비 12% 여백 확보
# → ~15° 회전까지 거의 완전 담김.
PAD = 80       # margin for shadows/outlines + augment rotation headroom


REGISTRY: dict[str, Callable[..., "Context"]] = {}


@dataclass
class Context:
    canvas: Image.Image           # RGB, accumulating composite
    mask: Image.Image             # L-mode, current glyph shape
    rng: np.random.Generator
    char: str = ""
    # Canonical source-kind name ("font", "svg_stroke", "ehanja_median",
    # "kanjivg_median", "ehanja_stroke", "mmh_stroke"). Set by the driver
    # once the actual source is resolved — for `kind: multi` this is the
    # picked child's kind, so per-sample gating works. Empty string means
    # unknown / not set.
    source_kind: str = ""


def register_layer(name: str):
    """Decorator: add a layer function under `name` in REGISTRY."""
    def deco(fn: Callable[..., Context]) -> Callable[..., Context]:
        if name in REGISTRY:
            raise ValueError(f"layer name already registered: {name!r}")
        REGISTRY[name] = fn
        return fn
    return deco


def _sample(v, rng: np.random.Generator):
    """Resolve a parameter value. See module docstring for rules."""
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int):
                return int(rng.integers(lo, hi + 1))
            return float(rng.uniform(lo, hi))
        return v[int(rng.integers(0, len(v)))]
    return v


def _coerce_color(c) -> tuple[int, int, int]:
    """Turn a color-spec (tuple or length-3 list) into an RGB int tuple."""
    if isinstance(c, (list, tuple)) and len(c) == 3:
        return tuple(int(x) for x in c)
    raise ValueError(f"expected RGB triple, got {c!r}")


def _sample_color(c, rng: np.random.Generator) -> tuple[int, int, int]:
    """Return one RGB triple.

    Accepts either:
      - a single color: tuple or length-3 list of ints/floats  → returned
      - a color pool: list of color triples                    → random one

    Distinguishes by checking whether the length-3 outer is all-numeric
    (single color) or contains nested lists/tuples (pool).
    """
    if not isinstance(c, (list, tuple)):
        raise ValueError(f"expected color or color pool, got {c!r}")
    if len(c) == 3 and all(isinstance(x, (int, float)) for x in c):
        return _coerce_color(c)
    # pool
    if len(c) == 0:
        raise ValueError("empty color pool")
    picked = c[int(rng.integers(0, len(c)))]
    return _coerce_color(picked)


def run_block(ctx: Context, specs: list[dict], default_prefix: str = "") -> Context:
    """Execute a list of layer specs in order.

    Each spec dict must have a 'layer' or 'op' key naming a registered layer.
    If the name has no '.' and `default_prefix` is given, prefix is prepended.

    Common optional keys on any step:
      prob            — 0..1 probability of running this step at all
      skip_if_kinds   — list of source kinds to skip. If `ctx.source_kind`
                         matches any entry (exact string), the step is not run.
      only_if_kinds   — list of source kinds to require. If set and
                         `ctx.source_kind` is NOT in the list, the step is
                         skipped. Useful to gate e.g. elastic only for font
                         samples where per-stroke variation isn't present.

    If `ctx.source_kind` is empty/unset, `skip_if_kinds` is ignored and
    `only_if_kinds` treats it as a miss (step skipped).
    """
    for raw in specs:
        step = dict(raw)
        name = step.pop("layer", None) or step.pop("op", None)
        if name is None:
            raise ValueError(f"step missing 'layer'/'op' key: {raw}")
        if default_prefix and "." not in name:
            name = f"{default_prefix}.{name}"
        if name not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(f"unknown layer: {name!r}. known: {known}")
        skip_if_kinds = step.pop("skip_if_kinds", None)
        only_if_kinds = step.pop("only_if_kinds", None)
        if skip_if_kinds and ctx.source_kind and ctx.source_kind in skip_if_kinds:
            continue
        if only_if_kinds is not None:
            if not ctx.source_kind or ctx.source_kind not in only_if_kinds:
                continue
        prob = step.pop("prob", 1.0)
        if ctx.rng.random() > prob:
            continue
        ctx = REGISTRY[name](ctx, **step)
    return ctx


def run_pipeline(ctx: Context, spec: dict) -> Context:
    """Execute the style block then the augment block of a pipeline spec.

    Style block: layer names used as-is (normally fully qualified, e.g.
    "fill.gradient").
    Augment block: unqualified names auto-prefixed with "augment.", so users
    can write `op: rotate` instead of `op: augment.rotate`.

    `spec["base_source"]` is handled by the caller (generate.py) before this
    function is invoked — Context must already contain a rendered mask and a
    blank canvas.
    """
    if spec.get("style"):
        ctx = run_block(ctx, spec["style"])
    if spec.get("augment"):
        ctx = run_block(ctx, spec["augment"], default_prefix="augment")
    return ctx


def finalize(canvas: Image.Image) -> Image.Image:
    """Center-crop the work canvas down to the final output size."""
    w, h = canvas.size
    left = (w - OUTPUT) // 2
    top = (h - OUTPUT) // 2
    return canvas.crop((left, top, left + OUTPUT, top + OUTPUT))


def fresh_canvas() -> Image.Image:
    """Blank RGB canvas sized for the pipeline. Default white."""
    return Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
