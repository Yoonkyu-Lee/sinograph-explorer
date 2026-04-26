"""
Phase 5 — convert e-hanja outline polygons to MMH-style (median + width).

Motivation
----------
The outline-based `outline_stroke` backend produces visually correct canonical
shapes, but per-vertex jitter on dense outline polygons looks like graphic
noise rather than handwriting variation. The median-based `svg_stroke`
backend (originally for MakeMeAHanzi) uses a stroke's *centerline* + a single
width, and jitters operate on stroke semantics (endpoints, control waypoints,
stroke-level width). That yields natural-looking variation.

This script skeletonizes each e-hanja stroke outline to recover a median +
width, producing a MakeMeAHanzi-compatible record that the `svg_stroke`
backend can consume — so e-hanja characters get the same handwriting feel.

Pipeline per stroke
-------------------
1. Parse SVG `d` → vertex array, apply outer `<g transform>`.
2. Flip y-axis and shift so the stroke lives in a math-up frame matching MMH
   (y-up, with glyph bounds roughly in [0, viewbox_h]; higher y = top).
3. Rasterize the polygon into a local bbox-fitted binary mask (ample pad).
4. `skimage.morphology.skeletonize` — 1-px-wide centerline.
5. Trace skeleton → ordered polyline. BFS from a terminus (or bbox-extreme
   pixel for closed loops) to the graph-farthest pixel gives the backbone.
6. Estimate width = 2 × mean distance-transform value along the skeleton.
7. Downsample to `max_pts` waypoints (uniform along polyline) and
   transform back to MMH-like math-up coords.

Output
------
`db_src/e-hanja_online/strokes_medianized.jsonl`, one record per char:

    {"char": "鑑",
     "viewbox": [1024, 1152],
     "y_pivot": 1152,
     "strokes": [
       {"order": 1, "kind": "normal",
        "median": [[x0, y0], ...],         # math-up, same frame as MMH
        "width": 48.0},
       ...
     ]}

Coord frame: x in [0, 1024], y math-up in [0, viewbox_h]. When rendered,
`svg_stroke`-style rasterizer uses `y_pivot=viewbox_h=1152` and `box=1152`.

Usage
-----
    python medianize_outlines.py                 # full dataset, multiproc
    python medianize_outlines.py --limit 50      # smoke test, single-proc
    python medianize_outlines.py --workers 8     # override worker count
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT.parent.parent / "db_src" / "e-hanja_online"
INPUT_PATH = DB_DIR / "strokes_animated.jsonl"
OUTPUT_PATH = DB_DIR / "strokes_medianized.jsonl"
ANOMALY_PATH = DB_DIR / "medianize_anomalies.jsonl"

# Rasterization resolution per stroke bbox — larger = more accurate skeleton,
# slower. 256 is a solid middle. Strokes are small relative to full char so
# we raster into a bbox-fitted local canvas, not a 256×256 full canvas.
RASTER_PAD_PX = 6
RASTER_SCALE = 0.35      # viewBox units → bbox raster px
# MMH medians average 4~7 waypoints per stroke. More than that creates
# high-frequency jitter artifacts when per-vertex noise is applied. We
# downsample aggressively (RDP-lite via uniform-index sampling).
MAX_WAYPOINTS = 7        # per median polyline
MIN_WAYPOINTS = 2        # skip degenerate
MIN_STROKE_AREA_PX = 4   # rasterized polygon area below this = drop


# ---------- SVG path parsing (copy of outline_stroke helpers) ---------------


_TRANSFORM_RE = re.compile(
    r"scale\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s+"
    r"translate\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)"
)


def parse_transform(s):
    if not s:
        return (1.0, 1.0, 0.0, 0.0)
    m = _TRANSFORM_RE.match(s)
    if not m:
        return (1.0, 1.0, 0.0, 0.0)
    return tuple(float(x) for x in m.groups())


def apply_transform(pts, sx, sy, tx, ty):
    out = np.empty_like(pts)
    out[:, 0] = sx * (pts[:, 0] + tx)
    out[:, 1] = sy * (pts[:, 1] + ty)
    return out


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


# ---------- skeleton tracing -----------------------------------------------


_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbors(p, pix_set):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in _NB if (r + dr, c + dc) in pix_set]


def _bfs_farthest(start, pix_set):
    """Return (farthest_node, parent_map) from BFS starting at `start`."""
    parent = {start: None}
    queue = deque([start])
    farthest = start
    while queue:
        cur = queue.popleft()
        for nb in _neighbors(cur, pix_set):
            if nb not in parent:
                parent[nb] = cur
                queue.append(nb)
                farthest = nb
    return farthest, parent


def _reconstruct(parent, target):
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def trace_skeleton(skel_mask):
    """Return an ordered list of (row, col) following the skeleton backbone.

    Strategy:
      - If the skeleton has ≥1 endpoints (degree-1 pixels), pick one and BFS
        twice to find the two diameter endpoints (standard tree diameter
        trick). The path between them is the backbone.
      - If no endpoints (closed loop), pick the topmost-leftmost pixel as a
        break point and walk one direction via DFS.
    """
    pix = np.argwhere(skel_mask)
    if len(pix) == 0:
        return []
    if len(pix) == 1:
        r, c = pix[0]
        return [(int(r), int(c))]
    pix_set = set(map(tuple, pix.tolist()))

    # Endpoints = skeleton pixels with exactly one 8-connected neighbor.
    endpoints = [p for p in pix_set if len(_neighbors(p, pix_set)) == 1]

    if endpoints:
        anchor = endpoints[0]
        far1, _ = _bfs_farthest(anchor, pix_set)
        far2, parent = _bfs_farthest(far1, pix_set)
        return _reconstruct(parent, far2)

    # Closed loop: DFS walk one direction until stuck.
    start = min(pix_set, key=lambda p: (p[0], p[1]))
    visited = {start}
    path = [start]
    cur = start
    while True:
        nxt = None
        for nb in _neighbors(cur, pix_set):
            if nb not in visited:
                nxt = nb
                break
        if nxt is None:
            break
        visited.add(nxt)
        path.append(nxt)
        cur = nxt
    return path


def downsample_polyline(path_rc, max_pts):
    if len(path_rc) <= max_pts:
        return np.asarray(path_rc, dtype=float)
    idx = np.linspace(0, len(path_rc) - 1, max_pts).astype(int)
    return np.asarray([path_rc[i] for i in idx], dtype=float)


# ---------- per-stroke pipeline --------------------------------------------


def rasterize_polygon_bbox(pts_vb, pad_px=RASTER_PAD_PX, scale=RASTER_SCALE):
    """Rasterize polygon into a bbox-fitted local mask.

    Returns (mask, offset, scale) where offset shifts local pixel coords
    back to viewBox coords:
        x_vb = col / scale - offset[0]
        y_vb = row / scale - offset[1]
    """
    mn = pts_vb.min(axis=0)
    mx = pts_vb.max(axis=0)
    w_px = int(np.ceil((mx[0] - mn[0]) * scale)) + 2 * pad_px
    h_px = int(np.ceil((mx[1] - mn[1]) * scale)) + 2 * pad_px
    w_px = max(w_px, 3)
    h_px = max(h_px, 3)

    off_x = -mn[0] * scale + pad_px
    off_y = -mn[1] * scale + pad_px
    local_pts = [(float(p[0] * scale + off_x), float(p[1] * scale + off_y))
                 for p in pts_vb]

    img = Image.new("L", (w_px, h_px), 0)
    ImageDraw.Draw(img).polygon(local_pts, fill=255)
    mask = np.asarray(img) > 0
    return mask, (off_x, off_y), scale


def medianize_one_stroke(d_str, sx, sy, tx, ty):
    """Return (median_vb_yup, width_vb) or None if skeleton degenerates.

    median_vb_yup: (N, 2) array in viewBox coords but y-flipped to math-up
                   (high y = character top). Matches the format other code
                   uses when storing.
    width_vb: estimated stroke width in viewBox units.
    """
    from skimage.morphology import skeletonize
    from scipy.ndimage import distance_transform_edt

    pts_raw = flatten_svg_path(d_str)
    if len(pts_raw) < 3:
        return None
    pts_vb = apply_transform(pts_raw, sx, sy, tx, ty)

    mask, (off_x, off_y), scale = rasterize_polygon_bbox(pts_vb)
    if mask.sum() < MIN_STROKE_AREA_PX:
        return None

    skel = skeletonize(mask)
    if not skel.any():
        return None

    path_rc = trace_skeleton(skel)
    if len(path_rc) < MIN_WAYPOINTS:
        return None
    path_rc = downsample_polyline(path_rc, MAX_WAYPOINTS)

    # Distance transform for width estimate
    dist = distance_transform_edt(mask)
    rows = np.clip(path_rc[:, 0].astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(path_rc[:, 1].astype(int), 0, mask.shape[1] - 1)
    radii_px = dist[rows, cols]
    radius_px = float(radii_px.mean())
    width_vb = 2.0 * radius_px / scale  # back to viewBox units

    # Convert (row, col) → (x_vb_ydown, y_vb_ydown)
    xy_ydown = np.column_stack([
        (path_rc[:, 1] - off_x) / scale,
        (path_rc[:, 0] - off_y) / scale,
    ])
    return xy_ydown, width_vb


# ---------- per-record worker -----------------------------------------------


def process_record(rec):
    """Worker: input record (dict from strokes_animated.jsonl), return
    (out_rec, warnings) or None on total failure."""
    char = rec["char"]
    vb = rec.get("viewbox") or (1024, 1152)
    vb_w, vb_h = int(vb[0]), int(vb[1])
    sx, sy, tx, ty = parse_transform(rec.get("transform"))

    warnings = []
    stroke_out = []
    for s in rec.get("strokes", []):
        try:
            res = medianize_one_stroke(s["d"], sx, sy, tx, ty)
        except Exception as e:
            warnings.append(f"order={s.get('order')}_exception={type(e).__name__}")
            continue
        if res is None:
            warnings.append(f"order={s.get('order')}_degenerate")
            continue
        median_ydown, width_vb = res

        # Flip y to math-up so rasterizer can use `y_pivot = vb_h`.
        med_yup = median_ydown.copy()
        med_yup[:, 1] = vb_h - med_yup[:, 1]

        stroke_out.append({
            "order": int(s["order"]),
            "kind": str(s["kind"]),
            "median": [[float(x), float(y)] for x, y in med_yup],
            "width": float(width_vb),
        })

    if not stroke_out:
        return None, warnings + ["no_strokes_extracted"]

    stroke_out.sort(key=lambda x: x["order"])
    out_rec = {
        "char": char,
        "cp": rec["cp"],
        "viewbox": [vb_w, vb_h],
        "y_pivot": vb_h,
        "strokes": stroke_out,
    }
    return out_rec, warnings


# ---------- driver ----------------------------------------------------------


def iter_input():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="process first N records (smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1),
                    help="worker processes (default: cpu_count - 1)")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    records = list(iter_input())
    if args.limit:
        records = records[:args.limit]
    total = len(records)
    print(f"medianize {total:,} animated e-hanja records")
    print(f"  input : {INPUT_PATH}")
    print(f"  output: {OUTPUT_PATH}")
    print(f"  workers: {args.workers}")
    print()

    t0 = time.time()
    out_f = OUTPUT_PATH.open("w", encoding="utf-8")
    anom_f = ANOMALY_PATH.open("w", encoding="utf-8")
    written = 0
    anomalies = 0
    all_stroke_counts = []

    try:
        if args.workers <= 1:
            results = (process_record(r) for r in records)
        else:
            pool = mp.Pool(processes=args.workers)
            results = pool.imap_unordered(process_record, records, chunksize=20)

        for i, res in enumerate(results, 1):
            if res is None:
                anomalies += 1
            else:
                out_rec, warnings = res
                if out_rec is not None:
                    out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    written += 1
                    all_stroke_counts.append(len(out_rec["strokes"]))
                if warnings:
                    anomalies += 1
                    # Minimal anomaly record
                    cp = out_rec["cp"] if out_rec else "?"
                    char = out_rec["char"] if out_rec else "?"
                    anom_f.write(json.dumps({
                        "cp": cp, "char": char,
                        "issues": warnings,
                    }, ensure_ascii=False) + "\n")
            if i % args.progress_every == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.01)
                eta = (total - i) / max(rate, 0.01)
                print(f"  {i:,}/{total:,}  written={written:,}  "
                      f"anomalies={anomalies:,}  "
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
    print(f"  anomalies:  {anomalies:,}")
    if all_stroke_counts:
        n = len(all_stroke_counts)
        all_stroke_counts.sort()
        print(f"  strokes/char: min={all_stroke_counts[0]}  "
              f"median={all_stroke_counts[n//2]}  "
              f"max={all_stroke_counts[-1]}  "
              f"mean={sum(all_stroke_counts)/n:.1f}")
    print(f"  output:      {OUTPUT_PATH}")
    print(f"  anomalies:   {ANOMALY_PATH}")


if __name__ == "__main__":
    main()
