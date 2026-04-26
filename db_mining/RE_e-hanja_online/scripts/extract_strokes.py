"""
Phase 2 — extract per-stroke outline paths from animated SVGs.

Reads Phase 1 manifest, filters to `type == "animated"`, parses each SVG for
its `<path class="stroke-normal|stroke-radical">` elements, and writes one
compact JSONL record per character.

Input
-----
    db_src/e-hanja_online/strokes_manifest.jsonl     (Phase 1 output)
    db_mining/RE_e-hanja_online/data/svg/{HEX}.svg   (raw SVGs)

Output
------
    db_src/e-hanja_online/strokes_animated.jsonl
        One record per char. `strokes` sorted by `order` ascending.
        ```
        {"cp": 38481, "hex": "9451", "char": "鑑",
         "viewbox": [1024, 1152],
         "transform": "scale(1,-1) translate(0, -879)",
         "strokes": [
           {"order": 7, "kind": "normal", "d": "M602 738L447..."},
           ...
         ]}
        ```

    db_src/e-hanja_online/extract_anomalies.jsonl
        Records for characters where the parse is suspicious:
          - stroke_count mismatch (manifest vs actual extracted)
          - stroke_count == 1 (flagged for manual review)
          - missing viewbox / transform
          - any path missing id/d/class

Approach
--------
Format is regular (Classic ASP template output), so regex is sufficient —
no need for a real XML parser. We extract:
  - the outer <g transform="..."> (for coordinate system alignment at render)
  - each <path id="U{CP}d{N}" d="..." class="stroke-normal|stroke-radical">
    (attributes may appear in any order; we extract each independently)

The stroke `order` is the `d{N}` integer; this matches e-hanja's
first-to-last animation sequence.
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
DB_DIR = ROOT.parent.parent / "db_src" / "e-hanja_online"
MANIFEST_PATH = DB_DIR / "strokes_manifest.jsonl"
OUT_PATH = DB_DIR / "strokes_animated.jsonl"
ANOMALY_PATH = DB_DIR / "extract_anomalies.jsonl"

# Outer group transform (we want the character coordinate system)
G_TRANSFORM = re.compile(rb'<g\s+transform="([^"]+)"')

# Each stroke path. Attributes can appear in any order — we match the whole
# element then extract each attribute individually.
STROKE_PATH = re.compile(rb'<path\b([^>]*?)\bclass="(stroke-normal|stroke-radical)"([^>]*?)/?>')
ID_ATTR = re.compile(rb'id="U[0-9A-Fa-f]+d(\d+)"')
D_ATTR = re.compile(rb'\bd="([^"]+)"')


def extract_from_svg(svg_bytes: bytes) -> tuple[str | None, list[dict], list[str]]:
    """Parse one animated SVG. Return (transform, strokes, warnings).

    `transform` may be None if the outer <g> is absent — unusual but possible.
    `strokes` is a list of {order, kind, d}. Sorted by order at the caller.
    `warnings` collects non-fatal issues seen while parsing.
    """
    warnings: list[str] = []

    m = G_TRANSFORM.search(svg_bytes)
    transform = m.group(1).decode("ascii", errors="replace") if m else None
    if transform is None:
        warnings.append("missing_outer_g_transform")

    strokes: list[dict] = []
    seen_orders: set[int] = set()
    for m in STROKE_PATH.finditer(svg_bytes):
        pre, kind, post = m.group(1), m.group(2), m.group(3)
        attrs_blob = pre + post
        id_m = ID_ATTR.search(attrs_blob)
        d_m = D_ATTR.search(attrs_blob)
        if not id_m:
            warnings.append("stroke_missing_id")
            continue
        if not d_m:
            warnings.append("stroke_missing_d")
            continue
        order = int(id_m.group(1))
        if order in seen_orders:
            warnings.append(f"duplicate_order_{order}")
            continue
        seen_orders.add(order)
        strokes.append({
            "order": order,
            "kind": kind.decode("ascii").split("-", 1)[1],  # "normal" or "radical"
            "d": d_m.group(1).decode("ascii", errors="replace"),
        })

    strokes.sort(key=lambda s: s["order"])
    return transform, strokes, warnings


def iter_animated_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "animated":
                yield rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process first N (smoke test)")
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    manifest_entries = list(iter_animated_manifest())
    if args.limit:
        manifest_entries = manifest_entries[:args.limit]
    total = len(manifest_entries)
    print(f"Phase 2: extracting {total:,} animated SVGs")
    print(f"  input : {SVG_DIR}")
    print(f"  output: {OUT_PATH}")
    print(f"  anomaly: {ANOMALY_PATH}")
    print()

    t0 = time.time()
    written = 0
    anomaly_count = 0
    mismatch_count = 0
    one_stroke_count = 0

    with OUT_PATH.open("w", encoding="utf-8") as out_f, \
         ANOMALY_PATH.open("w", encoding="utf-8") as anom_f:
        for i, rec in enumerate(manifest_entries, 1):
            hex_cp = rec["hex"]
            svg_path = SVG_DIR / f"{hex_cp}.svg"
            if not svg_path.exists():
                anom_f.write(json.dumps({
                    "cp": rec["cp"], "hex": hex_cp, "char": rec["char"],
                    "issue": "svg_file_missing"
                }, ensure_ascii=False) + "\n")
                anomaly_count += 1
                continue

            transform, strokes, warnings = extract_from_svg(svg_path.read_bytes())

            out_rec = {
                "cp": rec["cp"],
                "hex": hex_cp,
                "char": rec["char"],
                "viewbox": rec["viewbox"],
                "transform": transform,
                "strokes": strokes,
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            written += 1

            # Anomaly detection
            actual_n = len(strokes)
            manifest_n = rec.get("stroke_count")
            issues: list[str] = list(warnings)
            if manifest_n is not None and actual_n != manifest_n:
                issues.append(f"stroke_count_mismatch(manifest={manifest_n},actual={actual_n})")
                mismatch_count += 1
            if actual_n == 1:
                issues.append("only_one_stroke")
                one_stroke_count += 1
            if actual_n == 0:
                issues.append("no_strokes_extracted")
            if issues:
                anom_f.write(json.dumps({
                    "cp": rec["cp"], "hex": hex_cp, "char": rec["char"],
                    "stroke_count": actual_n,
                    "issues": issues,
                }, ensure_ascii=False) + "\n")
                anomaly_count += 1

            if i % args.progress_every == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.01)
                eta = (total - i) / max(rate, 0.01)
                print(f"  {i:,}/{total:,}  written={written:,}  "
                      f"anomalies={anomaly_count:,}  "
                      f"rate={rate:.0f}/s  eta={eta:.0f}s")

    print()
    print(f"done  elapsed={(time.time()-t0):.1f}s")
    print(f"  written:              {written:,}")
    print(f"  anomalies total:      {anomaly_count:,}")
    print(f"    - stroke mismatch:  {mismatch_count:,}")
    print(f"    - single-stroke:    {one_stroke_count:,}")
    print(f"  anomaly detail:       {ANOMALY_PATH}")


if __name__ == "__main__":
    main()
