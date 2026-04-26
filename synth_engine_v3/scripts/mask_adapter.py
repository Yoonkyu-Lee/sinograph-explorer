"""Mask-source adapter that reuses v2's base_source / svg_stroke / outline_stroke.

v2 API is Python-level feature-complete (font + svg_stroke + ehanja_median +
kanjivg_median + ehanja_stroke + mmh_stroke + multi, plus stroke_ops). The v3
GPU engine keeps mask generation on the CPU (per V3_DESIGN) and only reuses
that pipeline — so we delegate entirely to v2 here.

v2's render_mask methods require an np.random.Generator, so the v3 callers use
`np.random.default_rng(seed)` rather than Python's `random.Random`.

Added at Phase 1 (font only); expanded at Phase 6 (all SVG kinds + multi).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

_V2_SCRIPTS = Path(__file__).resolve().parents[2] / "synth_engine_v2" / "scripts"
if str(_V2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_V2_SCRIPTS))

import base_source as v2_base_source  # noqa: E402
import svg_stroke as v2_svg_stroke  # noqa: E402
import outline_stroke as v2_outline_stroke  # noqa: E402
# resolve_base_sources + MultiSource live inside v2's generate.py. Importing
# it is safe — v2 has the standard `if __name__ == "__main__": main()` guard,
# so no side effects beyond layer registration in v2's own (separate) REGISTRY.
import generate as v2_generate  # noqa: E402

# Phase 13: monkey-patch v2.outline_stroke file loaders with cached versions.
# Import side-effect installs the patch; workers inherit by importing mask_adapter.
import outline_cache  # noqa: E402, F401

# Phase 83 (Codex follow-up): font policy adds external dir + blacklist + tofu
# filter on top of v2's discover_font_sources. Imported lazily below.
import font_policy  # noqa: E402
import char_meta  # noqa: E402

DEFAULT_FONTS_DIR = Path("C:/Windows/Fonts")
CANVAS = v2_base_source.CANVAS_DEFAULT  # 384


def _resolve_with_policy(spec: dict, char: str, fonts_dir: Path) -> list:
    """Mirror of v2.generate.resolve_base_sources with font-kind routed through
    font_policy.get_font_sources_with_policy (external fonts merge + blacklist
    + tofu filter). Non-font kinds still delegate to v2 unchanged.
    """
    kind = spec.get("kind", "font")

    if kind == "font":
        return font_policy.get_font_sources_with_policy(
            char=char,
            system_fonts_dir=fonts_dir,
            filter_spec=spec.get("filter", "all"),
            drop_tofu=bool(spec.get("drop_tofu", True)),
            total_strokes=char_meta.get_total_strokes(char),
        )

    if kind == "multi":
        # Recurse on sub-specs so child font entries also go through policy.
        # Logic mirrors v2.generate.resolve_base_sources["multi"].
        sub_specs = spec.get("sources", []) or []
        groups, weights = [], []
        for sub in sub_specs:
            if not isinstance(sub, dict):
                continue
            w = float(sub.get("weight", 1.0))
            sub_clean = {k: v for k, v in sub.items() if k != "weight"}
            child = _resolve_with_policy(sub_clean, char, fonts_dir)
            if child:
                groups.append(child)
                weights.append(w)
        if not groups:
            fb_kind = spec.get("fallback")
            if fb_kind:
                return _resolve_with_policy(
                    {"kind": fb_kind, "filter": "all"}, char, fonts_dir
                )
            return []
        # MultiSource lives in v2 — reuse it
        return [v2_generate.MultiSource(groups, weights)]

    # All other kinds (svg_stroke / ehanja_median / kanjivg_median /
    # ehanja_stroke / mmh_stroke) are file-driven, font-dir-independent →
    # delegate to v2 as-is.
    return v2_generate.resolve_base_sources(spec, char, fonts_dir)


def get_sources_for_char(char: str, base_source_spec: dict,
                         fonts_dir: Path = DEFAULT_FONTS_DIR) -> list:
    """Resolve `base_source_spec` (v2 YAML fragment) against a single char.

    Supports every kind v2 does: font, svg_stroke, ehanja_median,
    kanjivg_median, ehanja_stroke, mmh_stroke, multi (with `fallback`).

    Font resolution goes through `font_policy` which adds the external font
    dir (`db_src/fonts/external/`), applies a filename blacklist, and drops
    (font, char) pairs that render as tofu.
    """
    return _resolve_with_policy(
        base_source_spec or {"kind": "font"}, char, fonts_dir
    )


# Back-compat helper kept for existing phase1-5 smoke tests that only dealt
# with fonts.
def get_font_sources_for(char: str, fonts_dir: Path = DEFAULT_FONTS_DIR) -> list:
    return v2_base_source.discover_font_sources(fonts_dir, char_filter=char)


def render_mask(char: str, source, rng: np.random.Generator | None = None) -> Image.Image | None:
    """Delegate to the v2 source. Returns PIL L-mode 384x384 or None.

    rng must be an np.random.Generator for SVG-based sources; font sources
    ignore rng entirely. If rng is None we build a default_rng — useful for
    smoke tests but production callers should pass a seeded generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    return source.render_mask(char, rng=rng)


def masks_to_tensor(
    masks: Iterable[Image.Image | None],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Stack PIL L-mode masks into (N, 1, H, W) tensor in [0, 1] on `device`.

    None / wrong-size masks are replaced with all-zero masks of the first
    valid mask's shape so the batch remains rectangular. Caller filters
    upstream when quality matters.
    """
    arrs: list[np.ndarray | None] = []
    target_shape: tuple[int, int] | None = None
    for m in masks:
        if m is None:
            arrs.append(None)
            continue
        if m.mode != "L":
            m = m.convert("L")
        a = np.asarray(m, dtype=np.uint8)
        if target_shape is None:
            target_shape = a.shape
        arrs.append(a)
    if target_shape is None:
        raise ValueError("all masks are None — cannot infer tensor shape")
    filled = [a if a is not None else np.zeros(target_shape, dtype=np.uint8) for a in arrs]
    stacked = np.stack(filled, axis=0)  # (N, H, W) uint8
    t = torch.from_numpy(stacked).to(device=device, dtype=dtype) / 255.0
    return t.unsqueeze(1)  # (N, 1, H, W)


def batch_render_from_spec(
    chars: list[str],
    base_source_spec: dict,
    rng: np.random.Generator,
    fonts_dir: Path = DEFAULT_FONTS_DIR,
    device: str = "cuda",
) -> tuple[torch.Tensor, list[str | None], list[str]]:
    """Given a list of chars + v2 base_source YAML spec, render all masks.

    Returns (mask_tensor (N,1,H,W), tags, source_kinds). A char with no
    available source contributes a zero placeholder and tag None (so the
    caller can filter).
    """
    tags: list[str | None] = []
    kinds: list[str] = []
    masks: list[Image.Image | None] = []
    for ch in chars:
        srcs = get_sources_for_char(ch, base_source_spec, fonts_dir)
        if not srcs:
            masks.append(None); tags.append(None); kinds.append(""); continue
        # pick one source uniformly per call (MultiSource handles weights itself)
        idx = int(rng.integers(0, len(srcs)))
        src = srcs[idx]
        m = render_mask(ch, src, rng=rng)
        masks.append(m)
        if m is not None:
            # MultiSource sets last_kind after each render; other sources
            # expose kind directly via attribute or property.
            kind = getattr(src, "last_kind", None) or getattr(src, "kind", "") or ""
            tags.append(src.tag())
            kinds.append(kind)
        else:
            tags.append(None); kinds.append("")
    mask_t = masks_to_tensor(masks, device=device)
    return mask_t, tags, kinds
