"""IDS source coverage comparison — 3 IDS DBs vs e-hanja / T1 / custom set.

Two output modes:
  (1) Summary — per-source codepoint coverage vs chosen universe.
  (2) Detail  — for picked codepoints, show the actual IDS string each
      source provides (or '—' if missing). This is the "concrete example"
      needed to see which fields survive.

Examples:
  # summary only, e-hanja 76k universe
  python db_src/ids_coverage_compare.py --universe ehanja

  # summary + 10 random codepoints from universe (sampled from missing-set)
  python db_src/ids_coverage_compare.py --universe ehanja --sample 10

  # summary + specific codepoints
  python db_src/ids_coverage_compare.py --cps U+4E00 U+9451 U+24A12 U+5AA4 U+7553

  # T1 10,932 universe, focus on chars missing from BabelStone
  python db_src/ids_coverage_compare.py --universe t1 --missing-in babelstone --sample 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DB_SRC = ROOT / "db_src"

SOURCES = {
    "babelstone": DB_SRC / "babelstone_ids" / "ids_primary.jsonl",
    "chise":      DB_SRC / "chise_ids"      / "ids_primary.jsonl",
    "cjkvi":      DB_SRC / "cjkvi_ids"      / "ids_primary.jsonl",
    "mmh":        ROOT   / "db_src" / "MAKEMEAHANZI" / "dictionary.txt",
}

EHANJA_MANIFEST = DB_SRC / "e-hanja_online" / "strokes_manifest.jsonl"
T1_CLASS_INDEX  = ROOT / "train_engine_v2" / "out" / "03_v3r_prod_t1" / "class_index.json"


def load_source(name: str) -> dict[str, str]:
    """Returns {codepoint: ids_string}. MMH has its own format (dictionary.txt
    entries with `decomposition` field)."""
    path = SOURCES[name]
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if name == "mmh":
                ch = d.get("character")
                if ch and len(ch) == 1:
                    cp = f"U+{ord(ch):04X}"
                    ids = d.get("decomposition")
                    if ids and ids != "？":
                        out[cp] = ids
            else:
                out[d["codepoint"]] = d["ids"]
    return out


def load_universe(name: str) -> set[str]:
    if name == "ehanja":
        cps: set[str] = set()
        with open(EHANJA_MANIFEST, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                cps.add(f"U+{d['cp']:04X}")
        return cps
    if name == "t1":
        ci = json.load(open(T1_CLASS_INDEX, encoding="utf-8"))
        return set(ci.keys())
    raise SystemExit(f"unknown universe: {name}")


def fmt_char(cp: str) -> str:
    try:
        return chr(int(cp[2:], 16))
    except Exception:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["ehanja", "t1"], default="ehanja",
                    help="coverage baseline universe (default: ehanja 76k)")
    ap.add_argument("--cps", nargs="*", default=None,
                    help="explicit codepoints to show detail for (U+XXXX form)")
    ap.add_argument("--sample", type=int, default=0,
                    help="random sample N codepoints from universe for detail")
    ap.add_argument("--missing-in", default=None,
                    choices=list(SOURCES.keys()),
                    help="sample only from codepoints missing in given source")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 78)
    print("IDS source coverage comparison")
    print("=" * 78)
    print(f"Universe  : {args.universe}")

    # Load everything
    print("\n[load] sources...")
    src_data: dict[str, dict[str, str]] = {}
    for name in SOURCES:
        m = load_source(name)
        src_data[name] = m
        print(f"  {name:<12} {len(m):>7} entries")

    uni = load_universe(args.universe)
    print(f"\n[load] universe '{args.universe}': {len(uni):,} codepoints")

    # --- summary ---
    print("\n" + "=" * 78)
    print(f"Coverage vs universe ({len(uni):,} codepoints)")
    print("=" * 78)
    print(f"{'source':<12} {'∩ universe':>12} {'%':>8} {'missing':>10}")
    print("-" * 78)
    keysets: dict[str, set[str]] = {}
    for name, m in src_data.items():
        cps = set(m.keys())
        keysets[name] = cps
        inter = uni & cps
        miss = uni - cps
        pct = 100 * len(inter) / len(uni) if uni else 0
        print(f"{name:<12} {len(inter):>12,} {pct:>7.1f}% {len(miss):>10,}")
    union_all = set().union(*keysets.values())
    inter_all = uni & union_all
    miss_all = uni - union_all
    pct = 100 * len(inter_all) / len(uni) if uni else 0
    print(f"{'union':<12} {len(inter_all):>12,} {pct:>7.1f}% {len(miss_all):>10,}")

    # Pairwise intersection matrix
    print("\n" + "-" * 78)
    print("Pairwise intersection of source codepoint sets:")
    names = list(SOURCES.keys())
    header = "             " + " ".join(f"{n:>10}" for n in names)
    print(header)
    for a in names:
        row = f"{a:<12} "
        for b in names:
            inter = keysets[a] & keysets[b]
            row += f"{len(inter):>10,} "
        print(row.rstrip())

    # --- detail ---
    picks: list[str] = []
    if args.cps:
        for cp in args.cps:
            cp_norm = cp.upper()
            if not cp_norm.startswith("U+"):
                cp_norm = "U+" + cp_norm
            picks.append(cp_norm)

    if args.sample:
        rng = random.Random(args.seed)
        if args.missing_in:
            pool = list(uni - keysets[args.missing_in])
            # actually: "missing_in" means we want cps that are NOT in that
            # source but ARE in universe → those are the failure cases
            pool = list(uni - keysets[args.missing_in])
            print(f"\n[sample] {args.sample} random cps from universe "
                  f"NOT in '{args.missing_in}' "
                  f"(pool size: {len(pool)})")
        else:
            pool = list(uni)
            print(f"\n[sample] {args.sample} random cps from universe")
        if pool:
            picks.extend(rng.sample(pool, min(args.sample, len(pool))))

    if picks:
        print("\n" + "=" * 78)
        print(f"Detail — IDS string per source for {len(picks)} codepoint(s)")
        print("=" * 78)
        w_cp = 10
        w_ch = 4
        w_src = max(len(n) for n in SOURCES) + 2
        for cp in picks:
            ch = fmt_char(cp)
            in_uni = "✓" if cp in uni else "✗"
            print(f"\n{cp}  '{ch}'  (in {args.universe}: {in_uni})")
            for name in SOURCES:
                ids = src_data[name].get(cp)
                if ids is None:
                    print(f"  {name:<{w_src}} —  (missing)")
                else:
                    print(f"  {name:<{w_src}} {ids}")

    # --- e-hanja specifically missing from all IDS sources ---
    if args.universe == "ehanja":
        truly_missing = uni - union_all
        if truly_missing:
            print("\n" + "=" * 78)
            print(f"Codepoints in e-hanja but MISSING from all 3 IDS sources: "
                  f"{len(truly_missing)}")
            print("=" * 78)
            sample = sorted(truly_missing)[:20]
            for cp in sample:
                print(f"  {cp}  '{fmt_char(cp)}'")
            if len(truly_missing) > 20:
                print(f"  ... ({len(truly_missing) - 20} more)")


if __name__ == "__main__":
    main()
