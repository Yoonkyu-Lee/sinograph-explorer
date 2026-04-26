"""
Phase 1 — scan every downloaded e-hanja SVG and classify it.

Two kinds live in the same pool:
  animated  →  id="U{CP}ani", class="ani-svg", one <path class="stroke-normal|
               stroke-radical"> per stroke. Stroke-separable; path id carries
               order (U9451d7 = 7th stroke of 鑑).
  static    →  id="U{CP}", class="svg", single monolithic <path class="path-normal">.
               Stroke boundaries unavailable.

We process animated only (per user scope); static entries are still recorded
in the manifest so future work can decide whether to use them in a fallback
renderer.

Input:  <repo>/db_mining/RE_e-hanja_online/data/svg/{HEX}.svg
Output: <repo>/db_src/e-hanja_online/strokes_manifest.jsonl
        One JSON record per codepoint. Example:
            {"cp": 38481, "hex": "9451", "char": "鑑",
             "type": "animated", "stroke_count": 22,
             "viewbox": [1024, 1152]}

The manifest is what Phase 2 (extract) and downstream code (coverage report,
engine discovery) read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "data" / "svg"
OUT_DIR = ROOT.parent.parent / "db_src" / "e-hanja_online"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "strokes_manifest.jsonl"

# Animated signature — both markers must match a stroke-separable file.
# We look for the class attribute because id prefix alone is ambiguous.
ANIMATED_CLASS = re.compile(rb'class="ani-svg"')
STROKE_CLASS = re.compile(rb'class="(stroke-normal|stroke-radical)"')
VIEWBOX = re.compile(rb'viewBox="([\d\s.]+)"')


def classify(svg_bytes: bytes) -> tuple[str, int | None, tuple[int, int] | None]:
    """Return (type, stroke_count, (vb_w, vb_h))."""
    vb = None
    m = VIEWBOX.search(svg_bytes)
    if m:
        parts = m.group(1).split()
        if len(parts) == 4:
            try:
                vb = (int(float(parts[2])), int(float(parts[3])))
            except ValueError:
                vb = None

    if ANIMATED_CLASS.search(svg_bytes):
        count = len(STROKE_CLASS.findall(svg_bytes))
        return "animated", count, vb
    return "static", None, vb


def iter_svgs():
    """Yield (cp, hex, Path) for every .svg file, sorted by codepoint."""
    entries = []
    for p in SVG_DIR.glob("*.svg"):
        stem = p.stem
        try:
            cp = int(stem, 16)
        except ValueError:
            continue
        entries.append((cp, stem, p))
    entries.sort(key=lambda x: x[0])
    yield from entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress-every", type=int, default=5000)
    args = ap.parse_args()

    all_entries = list(iter_svgs())
    total = len(all_entries)
    print(f"scanning {total} SVGs -> {MANIFEST_PATH}")

    t0 = time.time()
    type_counts = {"animated": 0, "static": 0}
    stroke_stats: list[int] = []

    with MANIFEST_PATH.open("w", encoding="utf-8") as out:
        for i, (cp, hex_cp, p) in enumerate(all_entries, 1):
            kind, n_strokes, vb = classify(p.read_bytes())
            type_counts[kind] += 1
            if n_strokes is not None:
                stroke_stats.append(n_strokes)
            rec = {
                "cp": cp,
                "hex": hex_cp,
                "char": chr(cp),
                "type": kind,
                "stroke_count": n_strokes,
                "viewbox": list(vb) if vb else None,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % args.progress_every == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.01)
                print(f"  {i}/{total}  "
                      f"animated={type_counts['animated']}  "
                      f"static={type_counts['static']}  "
                      f"rate={rate:.0f}/s")

    print()
    print(f"done  elapsed={(time.time()-t0):.1f}s")
    print(f"  animated: {type_counts['animated']:,}")
    print(f"  static:   {type_counts['static']:,}")
    if stroke_stats:
        stroke_stats.sort()
        n = len(stroke_stats)
        print(f"  stroke count over animated: "
              f"min={stroke_stats[0]}  "
              f"median={stroke_stats[n//2]}  "
              f"max={stroke_stats[-1]}  "
              f"mean={sum(stroke_stats)/n:.1f}")


if __name__ == "__main__":
    main()
