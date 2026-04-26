"""
Unified glyph image generator.

One pipeline, three pluggable axes:
    base_source        → glyph mask
    effect stack       → stylized RGB
    augment pipeline   → varied RGB (not wired in v1 CLI yet)

v1 CLI defaults: enumerate every CJK font face that covers the literal,
and render every registered effect stack for each face. The "clean" effect
is just one entry in the stack registry, so plain unstyled output is a
regular special case rather than a separate pipeline.

Examples
  python generate.py 鑑
    → all covering fonts × all effects

  python generate.py 鑑 --sources font:malgun --effects clean,neon_cyan
    → malgun only × (clean, neon_cyan)

  python generate.py 鑑 --effects clean
    → recreate former `render_systemfonts.py` output
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from base_sources import FontSource, discover_font_sources
from effects import REGISTRY as EFFECT_REGISTRY


DEFAULT_FONTS_DIR = Path("C:/Windows/Fonts")


def resolve_sources(spec: str, char: str, fonts_dir: Path) -> list[FontSource]:
    """Parse --sources spec into concrete FontSource list.

    v1 accepts 'font:<filter>':
      font:all         → every CJK font face that covers the literal
      font:malgun      → substring match on font file stem
      font:malgun,noto → multiple substrings (any match)
    """
    if not spec.startswith("font:"):
        raise SystemExit(f"v1 only supports 'font:' source kind, got {spec!r}")
    filt = spec[len("font:"):].strip()
    all_sources = discover_font_sources(fonts_dir, char_filter=char)
    if filt == "all" or filt == "":
        return all_sources
    wanted = [w.strip().lower() for w in filt.split(",") if w.strip()]
    return [s for s in all_sources
            if any(w in s.font_path.stem.lower() for w in wanted)]


def resolve_effects(spec: str) -> list[str]:
    if spec == "all" or spec == "":
        return list(EFFECT_REGISTRY.keys())
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in EFFECT_REGISTRY]
    if unknown:
        known = ", ".join(EFFECT_REGISTRY.keys())
        raise SystemExit(f"unknown effects: {unknown}. known: {known}")
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("char")
    p.add_argument("--sources", default="font:all",
                   help="source spec. v1: 'font:<filter>'. Default: font:all")
    p.add_argument("--effects", default="all",
                   help="comma-separated effect names, or 'all'")
    p.add_argument("--fonts-dir", default=str(DEFAULT_FONTS_DIR))
    p.add_argument("--out", default=None,
                   help="output dir. Default: <repo>/out/<notation>/")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    char = args.char
    if len(char) != 1:
        raise SystemExit(f"expected a single character, got {char!r}")
    notation = f"U+{ord(char):04X}"

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else script_dir.parent / "out" / notation
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = resolve_sources(args.sources, char, Path(args.fonts_dir))
    effects = resolve_effects(args.effects)

    uname = unicodedata.name(char, "?")
    print(f"char={char} ({notation}) {uname}")
    print(f"sources: {len(sources)}  effects: {len(effects)}  "
          f"planned: {len(sources) * len(effects)}")
    print(f"out: {out_dir}")

    rng = np.random.default_rng(args.seed)
    written = 0
    for src in sources:
        mask = src.render_mask(char)
        if mask is None:
            print(f"  [skip] {src.tag()}: render failed", file=sys.stderr)
            continue
        for eff_name in effects:
            fn = EFFECT_REGISTRY[eff_name]
            img = fn(mask, rng)
            out_path = out_dir / f"{src.tag()}.{eff_name}.png"
            img.save(out_path)
            written += 1
    print(f"wrote {written} files")


if __name__ == "__main__":
    main()
