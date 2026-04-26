"""Build canonical_v3.characters_ids table by merging 3 IDS sources.

Input:
  db_src/babelstone_ids/ids.jsonl     (97,649 cps, multi-alternate with regions)
  db_src/chise_ids/ids_primary.jsonl  (102,892 cps)
  db_src/cjkvi_ids/ids_primary.jsonl  (88,937 cps)

Output:
  sinograph_canonical_v3/out/ids_merged.sqlite
    table: characters_ids (see schema in doc/17 Section 6.3)
  sinograph_canonical_v3/out/ids_merged_stats.json

Merge rules (doc/17 Section 6.4):
  1. singleton (1 source only) → primary = that source's IDS
  2. unanimous (all-source agree) → primary = that IDS
  3. component-only diff (same top-IDC, different component chars) →
     canonical form preferred: prefer CHISE/cjkvi over BabelStone, majority
     rule, tag 'structure_only'
  4. structural diff (different top-IDC) → majority else priority CHISE >
     cjkvi > BabelStone, flag has_struct_conflict=1
  5. CDP references (&CDP-XXXX;) in IDS → flag has_cdp_ref=1
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3")
DB_SRC = ROOT / "db_src"
OUT_DIR = ROOT / "sinograph_canonical_v3" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SQLITE = OUT_DIR / "ids_merged.sqlite"
OUT_STATS = OUT_DIR / "ids_merged_stats.json"

# Source bitmask bits
BIT_CHISE = 1
BIT_CJKVI = 2
BIT_BABEL = 4

IDC_RANGE = range(0x2FF0, 0x3000)  # ⿰..⿻ and extended


def is_idc(ch: str) -> bool:
    return len(ch) == 1 and ord(ch) in IDC_RANGE


def top_idc(ids: str) -> str:
    if not ids:
        return "(empty)"
    first = ids[0]
    return first if is_idc(first) else "(leaf)"


def has_cdp(ids: str) -> bool:
    return "&CDP-" in ids


def normalize_first_char(s: str) -> str:
    """Extract first 'logical unit' of an IDS: either IDC or first component
    literal. Returns '(leaf)' if string is atomic (single char, no IDC)."""
    if not s:
        return ""
    return s[0]


def load_babelstone() -> dict[str, dict]:
    """Returns {cp: {"primary": str, "alternates": [{"ids","regions"}]}}"""
    out: dict[str, dict] = {}
    with open(DB_SRC / "babelstone_ids" / "ids.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cp = d["codepoint"]
            ids_list = d.get("ids") or []
            regions = d.get("regions") or []
            if not ids_list:
                continue
            out[cp] = {
                "primary": ids_list[0],
                "alternates": [
                    {"ids": ids, "regions": regions[i] if i < len(regions) else []}
                    for i, ids in enumerate(ids_list)
                ],
            }
    return out


def load_simple(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["codepoint"]] = d["ids"]
    return out


def decide_primary(cp: str, chise: str | None, cjkvi: str | None,
                    babel_info: dict | None) -> tuple[str, str, str, bool]:
    """Return (primary_ids, primary_source, agreement_level, has_struct_conflict)."""
    babel = babel_info["primary"] if babel_info else None
    vals = {}
    if chise: vals["chise"] = chise
    if cjkvi: vals["cjkvi"] = cjkvi
    if babel: vals["babelstone"] = babel

    if len(vals) == 1:
        src, v = next(iter(vals.items()))
        return v, src, "singleton", False

    unique_vals = set(vals.values())
    if len(unique_vals) == 1:
        # Unanimous
        if "chise" in vals:
            return chise, "chise", "unanimous", False
        if "cjkvi" in vals:
            return cjkvi, "cjkvi", "unanimous", False
        return babel, "babelstone", "unanimous", False

    # Some disagreement. Check top-IDC.
    top_idcs = {src: top_idc(v) for src, v in vals.items()}
    structure_agree = len(set(top_idcs.values())) == 1

    # Majority rule: find most common IDS string
    counts = Counter(vals.values())
    top_val, top_count = counts.most_common(1)[0]

    if structure_agree:
        # component-only diff → prefer CHISE/cjkvi canonical form
        if chise and chise == top_val and top_count >= 2:
            return chise, "chise", "structure_only", False
        if cjkvi and cjkvi == top_val and top_count >= 2:
            return cjkvi, "cjkvi", "structure_only", False
        # 2/3 agree (chise+cjkvi typically)
        if top_count >= 2:
            # pick the CHISE/cjkvi one first
            for src in ("chise", "cjkvi", "babelstone"):
                if vals.get(src) == top_val:
                    return top_val, src, "structure_only", False
        # all three differ — priority CHISE > cjkvi > BabelStone
        for src in ("chise", "cjkvi", "babelstone"):
            if src in vals:
                return vals[src], src, "structure_only", False

    # Structural diff
    if top_count >= 2:
        for src in ("chise", "cjkvi", "babelstone"):
            if vals.get(src) == top_val:
                return top_val, src, "disagree_atomic", True
    # all three differ in structure
    for src in ("chise", "cjkvi", "babelstone"):
        if src in vals:
            return vals[src], src, "disagree_atomic", True

    raise RuntimeError(f"unreachable at {cp}")


def main() -> None:
    print("[load] BabelStone...")
    t0 = time.perf_counter()
    babel = load_babelstone()
    print(f"  {len(babel):,} cps  ({time.perf_counter()-t0:.1f}s)")

    print("[load] CHISE...")
    t0 = time.perf_counter()
    chise = load_simple(DB_SRC / "chise_ids" / "ids_primary.jsonl")
    print(f"  {len(chise):,} cps  ({time.perf_counter()-t0:.1f}s)")

    print("[load] cjkvi-ids...")
    t0 = time.perf_counter()
    cjkvi = load_simple(DB_SRC / "cjkvi_ids" / "ids_primary.jsonl")
    print(f"  {len(cjkvi):,} cps  ({time.perf_counter()-t0:.1f}s)")

    all_cps = set(babel) | set(chise) | set(cjkvi)
    print(f"[merge] union = {len(all_cps):,} cps")

    # Merge
    if OUT_SQLITE.exists():
        OUT_SQLITE.unlink()
    con = sqlite3.connect(OUT_SQLITE)
    con.execute("""
        CREATE TABLE characters_ids (
            codepoint TEXT PRIMARY KEY,
            primary_ids TEXT NOT NULL,
            primary_source TEXT NOT NULL,
            ids_top_idc TEXT NOT NULL,
            ids_sources_bitmask INTEGER NOT NULL,
            ids_alternates_json TEXT NOT NULL,
            agreement_level TEXT NOT NULL,
            has_struct_conflict INTEGER NOT NULL DEFAULT 0,
            has_cdp_ref INTEGER NOT NULL DEFAULT 0
        )
    """)

    stats: dict[str, int] = Counter()
    stats_by_idc: Counter = Counter()
    stats_by_level: Counter = Counter()
    n_cdp = n_struct_conflict = 0

    rows = []
    t0 = time.perf_counter()
    for cp in all_cps:
        c_val = chise.get(cp)
        j_val = cjkvi.get(cp)
        b_info = babel.get(cp)

        primary, src, level, has_conflict = decide_primary(cp, c_val, j_val, b_info)
        idc = top_idc(primary)
        cdp = has_cdp(primary)

        mask = 0
        if c_val: mask |= BIT_CHISE
        if j_val: mask |= BIT_CJKVI
        if b_info: mask |= BIT_BABEL

        alternates = []
        if c_val and c_val != primary:
            alternates.append({"source": "chise", "ids": c_val})
        if j_val and j_val != primary:
            alternates.append({"source": "cjkvi", "ids": j_val})
        if b_info:
            for alt in b_info["alternates"]:
                if alt["ids"] != primary:
                    alternates.append({
                        "source": "babelstone",
                        "ids": alt["ids"],
                        "regions": alt["regions"],
                    })

        rows.append((
            cp, primary, src, idc, mask,
            json.dumps(alternates, ensure_ascii=False),
            level, 1 if has_conflict else 0, 1 if cdp else 0,
        ))

        stats[f"level:{level}"] += 1
        stats_by_idc[idc] += 1
        stats_by_level[level] += 1
        if has_conflict: n_struct_conflict += 1
        if cdp: n_cdp += 1

    con.executemany(
        "INSERT INTO characters_ids VALUES (?,?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    print(f"[merge] inserted {len(rows):,} rows ({time.perf_counter()-t0:.1f}s)")

    # Index for fast queries
    con.execute("CREATE INDEX idx_ids_top_idc ON characters_ids(ids_top_idc)")
    con.execute("CREATE INDEX idx_ids_agreement ON characters_ids(agreement_level)")
    con.commit()
    con.close()

    # Stats JSON
    out_stats = {
        "total_codepoints": len(rows),
        "by_agreement_level": dict(stats_by_level),
        "by_top_idc": dict(stats_by_idc.most_common()),
        "has_struct_conflict": n_struct_conflict,
        "has_cdp_ref": n_cdp,
    }
    OUT_STATS.write_text(json.dumps(out_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {OUT_SQLITE}")
    print(f"[write] {OUT_STATS}")
    print()
    print(json.dumps(out_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
