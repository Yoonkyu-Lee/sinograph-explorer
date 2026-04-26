"""
Extract centerline medians from KanjiVG SVG files.

KanjiVG paths are already stroke centerlines (rendered with CSS stroke-width
styling, not closed outlines). No skeletonization needed — we just parse the
SVG path `d`, flatten curves to polyline vertices, flip y to match the
math-up convention, and write one record per base character.

Schema matches `strokes_medianized.jsonl` (e-hanja, Phase 5) so the same
svg_stroke loader can consume both via a thin wrapper.

Input
-----
    db_src/KanjiVG/kanji/{CP5}.svg          # base (non-variant)
    db_src/KanjiVG/kanji/{CP5}-{tag}.svg    # variant (Kaisho etc.) — SKIPPED

Output
------
    db_src/KanjiVG/strokes_kanjivg.jsonl
    Each line:
        {"char": "鑑", "cp": 38481,
         "viewbox": [109, 109], "y_pivot": 109,
         "strokes": [{"order": 1, "kind": "㇒",
                       "median": [[x, y], ...], "width": 3.0}, ...]}

Design notes
------------
- Coordinate system: KanjiVG uses viewBox (0 0 109 109), y-down. We flip y
  (y_math = 109 − y_svg) so svg_stroke's math-up rasterizer draws correctly
  with `box=109, y_pivot=109`.
- Width: all strokes get `width=3.0` (the KanjiVG CSS stroke-width). The
  rasterizer's `width_scale` config option can boost thickness at render
  time if needed.
- `kvg:type` (e.g. "㇒", "㇐", "㇔/㇏") is preserved as `kind` — richer
  metadata than MMH's implicit stroke ordering.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path("d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3")
SRC_DIR = ROOT / "db_src" / "KanjiVG" / "kanji"
OUT_PATH = ROOT / "db_src" / "KanjiVG" / "strokes_kanjivg.jsonl"
ANOMALY_PATH = ROOT / "db_src" / "KanjiVG" / "extract_anomalies.jsonl"

VIEWBOX = 109
MAX_WAYPOINTS = 7
KVG_STROKE_WIDTH = 3.0

# CJK ideograph whitelist. KanjiVG also ships glyphs for Hiragana, Katakana,
# ASCII digits/letters/punctuation, and fullwidth symbols — none of which are
# ideographic characters. For a CJK-ideograph OCR dataset these act as noise
# classes (kana is phonetic, ASCII entirely out of scope), so we drop them
# at extraction time. Keep this in sync with coverage_report.py buckets.
CJK_IDEOGRAPH_RANGES = (
    (0x4E00,  0x9FFF),     # CJK Unified Ideographs
    (0x3400,  0x4DBF),     # CJK Extension A
    (0x20000, 0x2A6DF),    # CJK Extension B (SMP)
    (0x2A700, 0x2B73F),    # CJK Extension C (SMP)
    (0x2B740, 0x2B81F),    # CJK Extension D (SMP)
    (0x2B820, 0x2CEAF),    # CJK Extension E (SMP)
    (0x2CEB0, 0x2EBEF),    # CJK Extension F (SMP)
    (0x30000, 0x3134F),    # CJK Extension G (SIP)
    (0xF900,  0xFAFF),     # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),    # CJK Compatibility Supplement
    (0x2E80,  0x2EFF),     # CJK Radicals Supplement
    (0x2F00,  0x2FDF),     # Kangxi Radicals
)


def _is_cjk_ideograph(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in CJK_IDEOGRAPH_RANGES)

# Regex for <path ...> elements — attributes can appear in any order.
# Attr *values* may contain `/` (e.g. `kvg:type="㇔/㇏"`), so we can NOT exclude
# `/` from the capture class. Non-greedy `[^>]+?` stops at the closing `>`
# while `\s*/?` absorbs the optional self-closing slash just before it.
PATH_ELEMENT = re.compile(r"<path\s+([^>]+?)\s*/?>", re.DOTALL)
ID_RE_TMPL = r'id="kvg:{cp}-s(\d+)"'
KVG_TYPE_RE = re.compile(r'kvg:type="([^"]*)"')
D_RE = re.compile(r'\bd="([^"]+)"')
ELEMENT_RE_TMPL = r'<g\s+id="kvg:{cp}"\s+kvg:element="([^"]*)"'


def flatten_svg_path(d, samples_per_curve=8):
    from svgpathtools import parse_path, Line
    path = parse_path(d)
    pts = []
    for seg in path:
        if isinstance(seg, Line):
            if not pts:
                pts.append((seg.start.real, seg.start.imag))
            pts.append((seg.end.real, seg.end.imag))
        else:
            for i in range(samples_per_curve + 1):
                t = i / samples_per_curve
                p = seg.point(t)
                if pts and abs(p.real - pts[-1][0]) < 1e-6 \
                        and abs(p.imag - pts[-1][1]) < 1e-6:
                    continue
                pts.append((p.real, p.imag))
    return np.asarray(pts, dtype=np.float64)


def downsample_polyline(pts, max_n):
    if len(pts) <= max_n:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_n).astype(int)
    return pts[idx]


def process_svg(svg_path):
    """Returns dict (record) on success, None if skipped, or
    (None, warnings_list) on partial parse."""
    stem = svg_path.stem
    # Only base files — variants have "{cp}-{tag}" format
    if "-" in stem:
        return None
    if len(stem) != 5:
        return None
    try:
        cp = int(stem, 16)
    except ValueError:
        return None

    # Drop non-ideograph entries (kana, ASCII, punctuation, fullwidth, etc.)
    # KanjiVG ships these as stroke-order training material but they are not
    # ideographic characters — unwanted noise classes for a CJK OCR dataset.
    if not _is_cjk_ideograph(cp):
        return None

    try:
        text = svg_path.read_text(encoding="utf-8")
    except Exception:
        return None

    pad_cp = f"{cp:05x}"
    id_re = re.compile(ID_RE_TMPL.format(cp=pad_cp))
    element_re = re.compile(ELEMENT_RE_TMPL.format(cp=pad_cp))

    elem_m = element_re.search(text)
    char = elem_m.group(1) if elem_m else chr(cp)

    strokes = []
    warnings = []
    for m in PATH_ELEMENT.finditer(text):
        attrs = m.group(1)
        id_m = id_re.search(attrs)
        if not id_m:
            continue  # path belongs to another char's scope (shouldn't happen)
        order = int(id_m.group(1))
        d_m = D_RE.search(attrs)
        if not d_m:
            warnings.append(f"order={order}_missing_d")
            continue
        kvg_type_m = KVG_TYPE_RE.search(attrs)
        kind = kvg_type_m.group(1) if kvg_type_m else ""

        try:
            pts = flatten_svg_path(d_m.group(1))
        except Exception as e:
            warnings.append(f"order={order}_exception={type(e).__name__}")
            continue
        if len(pts) < 2:
            warnings.append(f"order={order}_degenerate")
            continue

        pts = downsample_polyline(pts, MAX_WAYPOINTS)
        # Flip y: SVG viewBox is y-down; math-up = VIEWBOX - y_svg
        pts_math = np.column_stack([pts[:, 0], VIEWBOX - pts[:, 1]])
        strokes.append({
            "order": order,
            "kind": kind,
            "median": [[float(x), float(y)] for x, y in pts_math],
            "width": KVG_STROKE_WIDTH,
        })

    if not strokes:
        return {"__error": "no_strokes", "cp": cp, "char": char,
                "warnings": warnings}

    strokes.sort(key=lambda s: s["order"])
    return {
        "char": char,
        "cp": cp,
        "viewbox": [VIEWBOX, VIEWBOX],
        "y_pivot": VIEWBOX,
        "strokes": strokes,
        "__warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    svgs = sorted(SRC_DIR.glob("*.svg"))
    # Filter to base files only
    svgs = [p for p in svgs if "-" not in p.stem]
    if args.limit:
        svgs = svgs[:args.limit]
    total = len(svgs)
    print(f"KanjiVG median extract: {total:,} base SVGs")
    print(f"  output: {OUT_PATH}")
    print(f"  workers: {args.workers}")

    t0 = time.time()
    out_f = OUT_PATH.open("w", encoding="utf-8")
    anom_f = ANOMALY_PATH.open("w", encoding="utf-8")
    written = 0
    skipped = 0
    anomalies = 0
    stroke_counts = []
    try:
        if args.workers <= 1:
            results = (process_svg(p) for p in svgs)
        else:
            pool = mp.Pool(processes=args.workers)
            results = pool.imap_unordered(process_svg, svgs, chunksize=40)

        for i, rec in enumerate(results, 1):
            if rec is None:
                skipped += 1
            elif "__error" in rec:
                anomalies += 1
                anom_f.write(json.dumps({
                    "cp": rec["cp"], "char": rec["char"],
                    "issues": [rec["__error"]] + rec.get("warnings", []),
                }, ensure_ascii=False) + "\n")
            else:
                warnings = rec.pop("__warnings", None)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                stroke_counts.append(len(rec["strokes"]))
                if warnings:
                    anomalies += 1
                    anom_f.write(json.dumps({
                        "cp": rec["cp"], "char": rec["char"],
                        "issues": warnings,
                    }, ensure_ascii=False) + "\n")

            if i % args.progress_every == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.01)
                eta = (total - i) / max(rate, 0.01)
                print(f"  {i:,}/{total:,}  written={written:,}  "
                      f"skipped={skipped:,}  anomalies={anomalies:,}  "
                      f"rate={rate:.0f}/s  eta={eta:.0f}s")
    finally:
        out_f.close()
        anom_f.close()
        if args.workers > 1:
            pool.close()
            pool.join()

    print()
    print(f"done  elapsed={(time.time()-t0):.1f}s")
    print(f"  written:    {written:,}")
    print(f"  skipped:    {skipped:,}  (non-base or non-hex filename)")
    print(f"  anomalies:  {anomalies:,}")
    if stroke_counts:
        n = len(stroke_counts)
        stroke_counts.sort()
        print(f"  strokes/char: min={stroke_counts[0]}  "
              f"median={stroke_counts[n//2]}  "
              f"max={stroke_counts[-1]}  "
              f"mean={sum(stroke_counts)/n:.1f}")


if __name__ == "__main__":
    main()
