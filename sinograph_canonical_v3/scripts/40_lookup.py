"""canonical_v3 IDS lookup — a single codepoint or literal char → full row
with alternates pretty-printed.

Usage:
  python sinograph_canonical_v3/scripts/40_lookup.py --char 鑑
  python sinograph_canonical_v3/scripts/40_lookup.py --cp U+9451
  python sinograph_canonical_v3/scripts/40_lookup.py --cp 24A12
  python sinograph_canonical_v3/scripts/40_lookup.py --char 鑑 --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DB = (Path(__file__).resolve().parents[1] / "out" / "ids_merged.sqlite")

COLS = ["codepoint", "primary_ids", "primary_source", "ids_top_idc",
        "ids_sources_bitmask", "ids_alternates_json", "agreement_level",
        "has_struct_conflict", "has_cdp_ref",
        "ehanja_components_json", "ehanja_agreement", "ehanja_aligned_sources"]


def normalize_cp(s: str) -> str:
    s = s.strip().upper()
    if s.startswith("U+"):
        s = s[2:]
    int(s, 16)
    return f"U+{int(s, 16):04X}"


def cp_to_char(cp: str) -> str:
    try:
        return chr(int(cp[2:], 16))
    except Exception:
        return "?"


def decode_mask(mask: int) -> list[str]:
    bits = []
    if mask & 1: bits.append("chise")
    if mask & 2: bits.append("cjkvi")
    if mask & 4: bits.append("babelstone")
    return bits


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cp", help="codepoint (U+XXXX or raw hex)")
    g.add_argument("--char", help="literal character")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--json", action="store_true",
                    help="raw JSON output for piping")
    args = ap.parse_args()

    cp = (f"U+{ord(args.char):04X}" if args.char else normalize_cp(args.cp))
    con = sqlite3.connect(args.db)
    row = con.execute(
        "SELECT " + ",".join(COLS) + " FROM characters_ids WHERE codepoint=?",
        (cp,)
    ).fetchone()
    con.close()
    if not row:
        print(f"[lookup] {cp} not found in {args.db}")
        sys.exit(1)

    d = dict(zip(COLS, row))
    d["ids_alternates"] = json.loads(d.pop("ids_alternates_json"))
    d["ids_sources"] = decode_mask(d["ids_sources_bitmask"])
    d["char"] = cp_to_char(cp)
    eh_json = d.pop("ehanja_components_json", None)
    d["ehanja_components"] = json.loads(eh_json) if eh_json else None
    al_json = d.pop("ehanja_aligned_sources", None)
    d["ehanja_aligned_sources"] = json.loads(al_json) if al_json else []

    if args.json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
        return

    # Pretty format
    print("=" * 60)
    print(f"  {d['char']}   {d['codepoint']}")
    print("=" * 60)
    print(f"  primary_ids    : {d['primary_ids']}   (top-IDC: {d['ids_top_idc']})")
    print(f"  primary_source : {d['primary_source']}")
    print(f"  sources        : {', '.join(d['ids_sources'])}  (bitmask={d['ids_sources_bitmask']})")
    print(f"  agreement_level: {d['agreement_level']}")
    flags = []
    if d["has_struct_conflict"]: flags.append("struct_conflict")
    if d["has_cdp_ref"]:         flags.append("cdp_ref")
    if flags: print(f"  flags          : {', '.join(flags)}")
    if d["ids_alternates"]:
        print(f"  alternates     :")
        for a in d["ids_alternates"]:
            regs = ""
            if "regions" in a and a["regions"]:
                regs = f"  regions={a['regions']}"
            print(f"    [{a['source']:<10}] {a['ids']}{regs}")
    if d.get("ehanja_components"):
        print(f"  e-hanja flat   : {d['ehanja_components']}")
        print(f"  e-hanja align  : {d['ehanja_agreement']}"
              f"  (matched sources: {d['ehanja_aligned_sources']})")
    elif d.get("ehanja_agreement"):
        print(f"  e-hanja        : (absent)")

    # Structure table (Phase 1 aux labels)
    try:
        con = sqlite3.connect(args.db)
        srow = con.execute(
            "SELECT radical_idx, total_strokes, residual_strokes, "
            "sources_json, alt_sources_json FROM characters_structure "
            "WHERE codepoint=?", (cp,)
        ).fetchone()
        con.close()
    except sqlite3.OperationalError:
        srow = None
    if srow:
        rad, total, resid, src, alt = srow
        print(f"  radical_idx    : {rad}   total_strokes: {total}   "
              f"residual: {resid}")
        if alt:
            print(f"  struct alternates: {alt}")


if __name__ == "__main__":
    main()
