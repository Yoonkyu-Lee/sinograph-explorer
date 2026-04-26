"""
v2 unified generator — CLI entry point.

Pipeline composition:
    base_source block  -> glyph mask
    style block        -> stylized RGB on work canvas
    augment block      -> captured/degraded RGB
    finalize           -> center-crop to OUTPUT size

Config format (YAML):
    base_source:
      kind: font
      filter: "all"      # or "malgun", "malgun,batang"
    style:
      - layer: background.solid
        color: [255, 255, 255]
      - ...
    augment:
      - op: gaussian_blur
        sigma: [0.3, 1.2]
      - ...

All three blocks are optional. Missing blocks default to no-op.

CLI examples
    generate.py 鑑 --config configs/style/clean.yaml --count 5
    generate.py 鑑 --config configs/combo/poster_in_dark_room.yaml --count 10
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Import all dimension modules so their @register_layer decorators run.
# Order doesn't matter functionally; grouped here for readability.
import base_source
import svg_stroke
import outline_stroke
import background  # noqa: F401  — registers background.* layers
import stroke_weight  # noqa: F401
import fill  # noqa: F401
import outline  # noqa: F401
import shadow  # noqa: F401
import glow  # noqa: F401
import augment  # noqa: F401  — registers augment.* layers

from pipeline import (
    CANVAS,
    Context,
    REGISTRY,
    finalize,
    fresh_canvas,
    run_pipeline,
)


DEFAULT_FONTS_DIR = Path("C:/Windows/Fonts")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


DEFAULT_CANONICAL_JSONL = Path(
    "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/"
    "sinograph_canonical_v1/out/canonical_characters.jsonl"
)


class MultiSource:
    """Wraps several source groups and picks one at each render_mask call.

    A "group" is the list returned by resolve_base_sources for one sub-spec
    (e.g. font returns up to 46 face sources; svg_stroke returns at most one).
    At render time we pick a GROUP by declared weight, then one source within
    that group uniformly. Groups whose char-coverage came back empty are
    dropped at resolve time, so every group here is guaranteed renderable.

    Tag is a literal "multi"; if you need per-sample provenance, enable
    generate.py's `--metadata` flag which writes the picked source into a
    sidecar JSON (see `render_one_with_meta` below).
    """
    def __init__(self, groups: list, weights: list[float]):
        self.groups = groups
        w = np.array(weights, dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(groups), dtype=float)
        self.weights = w / w.sum()
        self.last_picked = None       # full tag of picked source (e.g. "malgun-0")
        self.last_kind = None         # canonical kind (e.g. "font", "ehanja_median")

    def render_mask(self, char: str, rng: np.random.Generator):
        g_idx = int(rng.choice(len(self.groups), p=self.weights))
        group = self.groups[g_idx]
        s_idx = int(rng.integers(0, len(group)))
        picked = group[s_idx]
        self.last_picked = picked.tag()
        self.last_kind = getattr(picked, "kind", "")
        return picked.render_mask(char, rng)

    @property
    def kind(self) -> str:
        # Best-effort: return last picked kind if available, else "multi".
        return self.last_kind or "multi"

    def tag(self) -> str:
        return "multi"


def resolve_base_sources(spec: dict, char: str, fonts_dir: Path):
    """Turn base_source spec into a list of source instances."""
    kind = spec.get("kind", "font")
    if kind == "multi":
        sub_specs = spec.get("sources", []) or []
        groups = []
        weights = []
        for sub in sub_specs:
            if not isinstance(sub, dict):
                continue
            w = float(sub.get("weight", 1.0))
            sub_clean = {k: v for k, v in sub.items() if k != "weight"}
            child_sources = resolve_base_sources(sub_clean, char, fonts_dir)
            if child_sources:
                groups.append(child_sources)
                weights.append(w)
        if not groups:
            fb_kind = spec.get("fallback")
            if fb_kind:
                fb_spec = {"kind": fb_kind, "filter": "all"}
                return resolve_base_sources(fb_spec, char, fonts_dir)
            return []
        return [MultiSource(groups, weights)]
    if kind == "font":
        filt = spec.get("filter", "all")
        all_sources = base_source.discover_font_sources(fonts_dir, char_filter=char)
        if filt == "all" or not filt:
            return all_sources
        wanted = [w.strip().lower() for w in str(filt).split(",") if w.strip()]
        return [s for s in all_sources
                if any(w in s.font_path.stem.lower() for w in wanted)]
    if kind == "svg_stroke":
        jsonl_path = Path(spec.get("canonical_jsonl", DEFAULT_CANONICAL_JSONL))
        stroke_ops = spec.get("stroke_ops", [])
        base_width = float(spec.get("base_width", 48.0))
        return svg_stroke.discover_svg_stroke_sources(
            char=char,
            canonical_jsonl=jsonl_path,
            stroke_ops=stroke_ops,
            base_width=base_width,
        )
    if kind == "ehanja_stroke":
        manifest_path = spec.get("manifest")
        manifest_path = Path(manifest_path) if manifest_path else None
        stroke_ops = spec.get("stroke_ops", [])
        return outline_stroke.discover_ehanja_outline_sources(
            char=char,
            manifest_path=manifest_path,
            stroke_ops=stroke_ops,
        )
    if kind == "mmh_stroke":
        graphics_path = spec.get("graphics")
        graphics_path = Path(graphics_path) if graphics_path else None
        stroke_ops = spec.get("stroke_ops", [])
        return outline_stroke.discover_mmh_outline_sources(
            char=char,
            graphics_path=graphics_path,
            stroke_ops=stroke_ops,
        )
    if kind == "ehanja_median":
        jsonl_path = spec.get("manifest")
        jsonl_path = Path(jsonl_path) if jsonl_path else None
        stroke_ops = spec.get("stroke_ops", [])
        return svg_stroke.discover_ehanja_median_sources(
            char=char,
            jsonl_path=jsonl_path,
            stroke_ops=stroke_ops,
            width_scale=float(spec.get("width_scale", 1.0)),
            width_override=spec.get("width_override"),
        )
    if kind == "kanjivg_median":
        jsonl_path = spec.get("manifest")
        jsonl_path = Path(jsonl_path) if jsonl_path else None
        stroke_ops = spec.get("stroke_ops", [])
        return svg_stroke.discover_kanjivg_median_sources(
            char=char,
            jsonl_path=jsonl_path,
            stroke_ops=stroke_ops,
            width_scale=float(spec.get("width_scale", 1.0)),
            width_override=spec.get("width_override"),
        )
    raise SystemExit(
        f"unknown base_source.kind: {kind!r} "
        f"(known: font, svg_stroke, ehanja_stroke, mmh_stroke, "
        f"ehanja_median, kanjivg_median, multi)"
    )


def render_one(char: str, config: dict, src,
                rng: np.random.Generator) -> Image.Image:
    mask = src.render_mask(char, rng)
    if mask is None:
        return None
    # For `MultiSource` the kind is set by render_mask (picked child); for
    # direct sources it's a fixed attribute. `ctx.source_kind` lets style/
    # augment steps gate themselves (`skip_if_kinds`, `only_if_kinds`).
    kind = getattr(src, "kind", "") or ""
    ctx = Context(canvas=fresh_canvas(), mask=mask, rng=rng, char=char,
                   source_kind=kind)
    ctx = run_pipeline(ctx, config)
    return finalize(ctx.canvas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("char", help="single character literal, e.g. 鑑")
    p.add_argument("--config", required=True, help="YAML pipeline config path")
    p.add_argument("--count", type=int, default=1,
                   help="samples per (source, config) — useful when config has random ranges")
    p.add_argument("--sources", default=None,
                   help="override base_source.filter in config")
    p.add_argument("--fonts-dir", default=str(DEFAULT_FONTS_DIR))
    p.add_argument("--out", default=None,
                   help="output dir, default: out/<notation>/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metadata", action="store_true",
                   help="write sidecar JSON per sample with config + seed")
    args = p.parse_args()

    char = args.char
    if len(char) != 1:
        raise SystemExit(f"expected single character, got {char!r}")
    notation = f"U+{ord(char):04X}"

    config_path = Path(args.config)
    config = load_config(config_path)
    config_tag = config_path.stem

    if args.sources is not None:
        config.setdefault("base_source", {})["filter"] = args.sources

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else script_dir.parent / "out" / notation
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = resolve_base_sources(config.get("base_source", {}), char, Path(args.fonts_dir))

    uname = unicodedata.name(char, "?")
    n_layers_style = len(config.get("style", []) or [])
    n_layers_aug = len(config.get("augment", []) or [])
    print(f"char={char} ({notation}) {uname}")
    print(f"config={config_path.name}  style_layers={n_layers_style}  augment_layers={n_layers_aug}")
    print(f"sources={len(sources)}  count_per_source={args.count}  planned={len(sources)*args.count}")
    print(f"out: {out_dir}")

    written = 0
    for src in sources:
        for i in range(args.count):
            # fresh RNG per sample so ranges actually vary across count
            rng = np.random.default_rng(args.seed + i + hash(src.tag()) % (2**32))
            img = render_one(char, config, src, rng)
            if img is None:
                print(f"  [skip] {src.tag()}: render failed", file=sys.stderr)
                continue
            base_name = f"{src.tag()}.{config_tag}"
            suffix = f".{i:03d}" if args.count > 1 else ""
            out_path = out_dir / f"{base_name}{suffix}.png"
            img.save(out_path)
            written += 1
            if args.metadata:
                picked = getattr(src, "last_picked", None)
                meta = {
                    "char": char, "notation": notation,
                    "source": src.tag(), "config": config_tag,
                    "picked_source": picked,  # filled for MultiSource
                    "sample_idx": i, "seed": args.seed,
                    "config_content": config,
                }
                meta_path = out_path.with_suffix(".json")
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"wrote {written} files")
    print(f"registered layers: {len(REGISTRY)}")


if __name__ == "__main__":
    main()
