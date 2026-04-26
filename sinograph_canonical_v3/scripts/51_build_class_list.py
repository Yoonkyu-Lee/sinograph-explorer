"""Build a class_list.jsonl for synth_engine_v3 corpus generation.

Coverage choices (doc/19 §4, clarified 2026-04-23):
    practical   3 structure labels (radical + total + residual_strokes) fully
                non-null; IDC allowed to be "(leaf)" for atomic chars (635
                glyphs like 金 / 水 / 永 / 一 — these genuinely have no
                decomposition, so their IDC head is masked at train time).
                → 102,944 classes. **default.**
    strict      3 structure + IDC ∈ 12 standard codes. Excludes atomic +
                extended IDCs → 102,309.
    universe    Every row in characters_structure (no filtering) → 103,046.
                Some labels null → train_loop masks them out.

Output format (one JSON per line, sorted by integer codepoint):
    {"codepoint": "U+XXXX", "char": "X",
     "target_samples": N,
     "block": "CJK_Unified" | "Ext_B_SMP" | ...,
     "coverage": "practical" | "strict" | "universe"}

Usage:
    python 51_build_class_list.py \
      --coverage practical \
      --samples-per-class 500 \
      --out sinograph_canonical_v3/out/class_list_practical.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DB = Path(__file__).resolve().parents[1] / "out" / "ids_merged.sqlite"
STD_IDC = "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻"


def block_of(cp: int) -> str:
    if 0x3400 <= cp <= 0x4DBF:
        return "CJK_Ext_A"
    if 0x4E00 <= cp <= 0x9FFF:
        return "CJK_Unified"
    if 0xF900 <= cp <= 0xFAFF:
        return "CJK_Compat"
    if 0x2E80 <= cp <= 0x2FDF:
        return "Radicals_Supp"
    if 0x20000 <= cp <= 0x2A6DF:
        return "Ext_B_SMP"
    if 0x2A700 <= cp <= 0x2B73F:
        return "Ext_C_SMP"
    if 0x2B740 <= cp <= 0x2B81F:
        return "Ext_D_SMP"
    if 0x2B820 <= cp <= 0x2CEAF:
        return "Ext_E_SMP"
    if 0x2CEB0 <= cp <= 0x2EBEF:
        return "Ext_F_SMP"
    if 0x2EBF0 <= cp <= 0x2EE5F:
        return "Ext_I_SMP"
    if 0x2F800 <= cp <= 0x2FA1F:
        return "Compat_Supp"
    if 0x30000 <= cp <= 0x3134F:
        return "Ext_G_SMP"
    if 0x31350 <= cp <= 0x323AF:
        return "Ext_H_SMP"
    if 0x323B0 <= cp <= 0x33479:
        return "Ext_J_SMP"
    return "Other"


def query_rows(db: Path, coverage: str) -> list[str]:
    """Return list of codepoint strings ("U+XXXX") in integer order."""
    con = sqlite3.connect(str(db))
    try:
        if coverage == "universe":
            q = "SELECT codepoint FROM characters_structure"
            rows = con.execute(q).fetchall()
        elif coverage == "strict":
            q = f"""
                SELECT s.codepoint
                FROM characters_structure s
                JOIN characters_ids i ON s.codepoint = i.codepoint
                WHERE s.radical_idx IS NOT NULL
                  AND s.total_strokes IS NOT NULL
                  AND s.residual_strokes IS NOT NULL
                  AND i.ids_top_idc IN ({','.join('?' * 12)})
            """
            rows = con.execute(q, tuple(STD_IDC)).fetchall()
        elif coverage == "practical":
            q = """
                SELECT s.codepoint
                FROM characters_structure s
                WHERE s.radical_idx IS NOT NULL
                  AND s.total_strokes IS NOT NULL
                  AND s.residual_strokes IS NOT NULL
            """
            rows = con.execute(q).fetchall()
        else:
            raise ValueError(f"unknown coverage: {coverage!r}")
    finally:
        con.close()
    return [r[0] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", choices=["practical", "strict", "universe"],
                    default="practical")
    ap.add_argument("--samples-per-class", type=int, default=500)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise FileNotFoundError(f"db not found: {db}")

    cps = query_rows(db, args.coverage)
    cps.sort(key=lambda s: int(s[2:], 16))
    print(f"[class-list] coverage={args.coverage}  n_class={len(cps):,}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    block_counts: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for cp_s in cps:
            cp = int(cp_s[2:], 16)
            ch = chr(cp)
            b = block_of(cp)
            block_counts[b] = block_counts.get(b, 0) + 1
            rec = {
                "codepoint": cp_s,
                "char": ch,
                "target_samples": args.samples_per_class,
                "block": b,
                "coverage": args.coverage,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(cps) * args.samples_per_class
    print(f"[class-list] wrote {out_path}  ({len(cps):,} rows)")
    print(f"[class-list] target total samples: {total:,}")
    print("[class-list] block distribution:")
    for b in sorted(block_counts, key=lambda k: -block_counts[k]):
        print(f"    {b:20s}  {block_counts[b]:>7,}")


if __name__ == "__main__":
    main()
