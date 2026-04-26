"""Export Level A aux-labels sidecar from canonical_v3 → aux_labels.npz.

Reads `ids_merged.sqlite` (characters_ids + characters_structure) and a
synth-corpus `class_index.json`; writes a compact npz next to the corpus
that train_engine_v3 can load at startup. See
`doc/19_TRAIN_ENGINE_V3_PLAN.md` §4 for the selection rules.

Output keys (all indexed by class_idx 0..N_class-1):
    radical_idx        int16   0–214,  -1 missing
    total_strokes      int16   0–84,   -1 missing
    residual_strokes   int16   clamp(total - radical_stroke_count, 0..)
    ids_top_idc        int8    0–11  (⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻), -1 missing
    valid_mask         uint8   (N, 4)  per-head present-flag
    class_index_hash   uint64  blake2b-8 of sorted class_index.json

Usage:
    python 50_export_aux_labels.py \
        --class-index PATH/class_index.json \
        --db sinograph_canonical_v3/out/ids_merged.sqlite \
        --out PATH/aux_labels.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DB = Path(__file__).resolve().parents[1] / "out" / "ids_merged.sqlite"

# IDC char → int (doc/19 §4.2 IDC_MAP). Order = Unicode U+2FF0..U+2FFB.
IDC_CHARS = "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻"
IDC_MAP: dict[str, int] = {c: i for i, c in enumerate(IDC_CHARS)}


def class_index_hash(class_index: dict[str, int]) -> int:
    ordered = sorted(class_index.items(), key=lambda kv: kv[1])
    blob = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "big", signed=False)


def notation_to_cp(notation: str) -> int:
    # "U+XXXX" → int
    s = notation.strip()
    if s.startswith("U+"):
        s = s[2:]
    return int(s, 16)


def fetch_structure_rows(db: Path) -> dict[int, tuple]:
    """codepoint (int) → (radical_idx, total_strokes, residual_strokes)."""
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute(
            "SELECT codepoint, radical_idx, total_strokes, residual_strokes "
            "FROM characters_structure"
        )
        out: dict[int, tuple] = {}
        for cp_s, r, t, res in cur:
            try:
                cp = notation_to_cp(cp_s)
            except Exception:
                continue
            out[cp] = (r, t, res)
    finally:
        con.close()
    return out


def fetch_idc_rows(db: Path) -> dict[int, str]:
    """codepoint (int) → ids_top_idc char (1 char) or None."""
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute(
            "SELECT codepoint, ids_top_idc FROM characters_ids "
            "WHERE ids_top_idc IS NOT NULL"
        )
        out: dict[int, str] = {}
        for cp_s, idc in cur:
            try:
                cp = notation_to_cp(cp_s)
            except Exception:
                continue
            if idc:
                out[cp] = idc
    finally:
        con.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export Level A aux-labels sidecar for train_engine_v3."
    )
    ap.add_argument("--class-index", required=True,
                    help="Path to corpus class_index.json")
    ap.add_argument("--db", default=str(DEFAULT_DB),
                    help="Path to canonical_v3 ids_merged.sqlite")
    ap.add_argument("--out", default=None,
                    help="Output aux_labels.npz (default: alongside class_index)")
    args = ap.parse_args()

    class_index_path = Path(args.class_index)
    with open(class_index_path, "r", encoding="utf-8") as f:
        class_index: dict[str, int] = json.load(f)
    # sanity: class indices are contiguous 0..N-1
    n_class = len(class_index)
    values = sorted(class_index.values())
    if values != list(range(n_class)):
        raise ValueError("class_index values must be a contiguous 0..N-1 set")

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")

    out_path = Path(args.out) if args.out else class_index_path.parent / "aux_labels.npz"
    print(f"[export] n_class = {n_class}")
    print(f"[export] db      = {db_path}")
    print(f"[export] out     = {out_path}")

    # load lookup tables once
    struct_tbl = fetch_structure_rows(db_path)
    idc_tbl = fetch_idc_rows(db_path)
    print(f"[export] characters_structure rows   : {len(struct_tbl):,}")
    print(f"[export] characters_ids (idc) rows   : {len(idc_tbl):,}")

    radical = np.full(n_class, -1, dtype=np.int16)
    total = np.full(n_class, -1, dtype=np.int16)
    residual = np.full(n_class, -1, dtype=np.int16)
    idc = np.full(n_class, -1, dtype=np.int8)
    valid = np.zeros((n_class, 4), dtype=np.uint8)

    # Coverage counters
    cov = {"radical": 0, "total": 0, "residual": 0, "idc": 0}

    for notation, idx in class_index.items():
        try:
            cp = notation_to_cp(notation)
        except Exception:
            continue
        s = struct_tbl.get(cp)
        if s is not None:
            r, t, res = s
            # DB stores Kangxi radical 1..214 (1-indexed). Convert to 0..213
            # so it aligns with `nn.Linear(512 → 214)` output indices. −1 remains
            # the sentinel for missing.
            if r is not None and r > 0:
                radical[idx] = int(r) - 1; valid[idx, 0] = 1; cov["radical"] += 1
            if t is not None and t >= 0:
                total[idx] = int(t); valid[idx, 1] = 1; cov["total"] += 1
            if res is not None and res >= 0:
                residual[idx] = int(res); valid[idx, 2] = 1; cov["residual"] += 1
        idc_c = idc_tbl.get(cp)
        if idc_c and idc_c in IDC_MAP:
            idc[idx] = int(IDC_MAP[idc_c]); valid[idx, 3] = 1; cov["idc"] += 1

    # Sanity: residual must be 0..84; if DB produced a negative value, clamp
    neg = int((residual < 0).sum()) - int((residual == -1).sum())
    if neg > 0:
        # -1 is sentinel; anything else negative is a DB anomaly → clamp to 0 + still valid
        bad = (residual < 0) & (residual != -1)
        residual[bad] = 0
        print(f"[export] clamped {neg} residual<0 rows to 0")

    hv = class_index_hash(class_index)

    np.savez(
        out_path,
        radical_idx=radical,
        total_strokes=total,
        residual_strokes=residual,
        ids_top_idc=idc,
        valid_mask=valid,
        class_index_hash=np.array(hv, dtype=np.uint64),
    )

    # Report
    print(f"[export] wrote {out_path}")
    print(f"[export] coverage (of {n_class:,}):")
    for k, v in cov.items():
        pct = 100.0 * v / n_class
        print(f"           {k:>10s}: {v:>7,}  ({pct:5.2f} %)")
    print(f"[export] class_index_hash = 0x{hv:016x}")


if __name__ == "__main__":
    main()
